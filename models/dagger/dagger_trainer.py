import torch
import torch.nn as nn
import copy
from datetime import datetime
import numpy as np,cv2
import os
import time
import yaml
from ..common.buffer import ReplayBuffer
from ..common.optimizer import build_optimizer
from utils.utils import build_logger,build_writer
import tqdm
from collections import defaultdict, OrderedDict
from utils.utils import is_main_proc
from torch.nn.parallel import DistributedDataParallel as DDP
from ..amp.amp_network import AMP_NETWORK
from .vla_network import VLANetwork
import torchvision

class DaggerTrainer:
    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        self.env = env
        self.cfg.env = env.cfg
        
        cfg.teacher_network.num_amp_obs = cfg.env.num_ref_obs_frames * cfg.env.num_ref_obs_per_frame
        cfg.teacher_network.obs_space = env.obs_space # teacher的obs_space是环境整个的obs_space，会使用'obs'部分
        self.num_action = cfg.student_network.num_action = cfg.teacher_network.num_action = cfg.env.num_action
        self.num_last_imgs = cfg.student_network.num_last_imgs = cfg.teacher_network.num_last_imgs = cfg.env.num_last_imgs

        self.num_envs = cfg.env.num_envs
        self.action_clip = cfg.action_clip
        self.obs_clip = cfg.obs_clip
        self.device = cfg.device
        self.auto_mixed_precision = self.cfg.auto_mixed_precision
        self.logger = build_logger(
            verbose=is_main_proc(cfg),
            filepath=os.path.join(cfg.root,f'_log_rk{cfg.rank}.txt')
        )
        if is_main_proc(cfg):
            self.writer = build_writer(
                log_dir=os.path.join(cfg.root,f'tb')
            )
        
        self.teacher_network = AMP_NETWORK(cfg.teacher_network).to(self.device).eval()
        teacher_ckpt = os.path.join(cfg.data_prefix, cfg.teacher_ckpt)
        teacher_ckpt = torch.load(teacher_ckpt, map_location='cpu')
        self.teacher_network.load_state_dict(teacher_ckpt['weight'])

        self.prop_dim = cfg.student_network.prop_dim = self.env.num_prop_obs
        #self.text_dim = cfg.student_network.text_dim = self.env.num_text_obs
        self.student_network = VLANetwork(cfg.student_network).to(self.device)
        if self.cfg.ddp: # 判断配置中是否启用分布式训练（DDP）
            # 2.1 同步BatchNorm层（分布式训练必需） 
            self.student_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.student_network)
            # 2.2 用DDP包装模型，开启分布式训练
            self.ddp_network = DDP(self.student_network,device_ids=[self.cfg.rank])
        else:
            # 2.3 不启用分布式：直接用基础模型作为调用入口
            self.ddp_network = self.student_network
            
        self.img_h, self.img_w, self.image_transform = self.student_network.build_transform()
        self.logger.info('===============Image Transform=================')
        self.logger.info(self.image_transform)
        self.logger.info('===============Teacher Network=================')
        self.logger.info(self.teacher_network)
        self.logger.info('===============Student Network=================')
        self.logger.info(self.student_network)

        # 把全局缓冲区大小(cfg里设置的buffer_size)按进程数(world_size)均分
        self.buffer_size = int(np.ceil(self.cfg.buffer_size / self.cfg.world_size))
        self.bz = np.ceil(self.cfg.bz / self.cfg.world_size)
        
        if self.cfg.debug:
            self.buffer_size = self.buffer_size // 100
            self.bz = self.bz // 50
        buffer_info_dict = {
            'image'     :   dict(shape = (self.buffer_size, self.img_h, self.img_w, 3), dtype = torch.uint8), 
            'prop'      :   dict(shape = (self.buffer_size, self.prop_dim)), 
            #'text'      :   dict(shape = (self.buffer_size, self.text_dim)), 
            'teacher_action'    :   dict(shape = (self.buffer_size, self.num_action)), 
            'last_action'    :   dict(shape = (self.buffer_size, self.num_action)), 
            'image_feat' :   dict(shape = (self.buffer_size, self.student_network.vl_dim)),
        }
        self.data_buffer = ReplayBuffer(buffer_info_dict, self.device, default_dtype = torch.float32,
                                        num_last_imgs=self.cfg.env.num_last_imgs,
                                        last_img_interval=self.cfg.env.last_img_interval,
                                        num_envs=self.cfg.env.num_envs)
        self.logger.info('===============ReplayBuffer=================')
        self.logger.info(self.data_buffer)

        self.optimizer = build_optimizer(cfg.optimizer, self.ddp_network.parameters())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.auto_mixed_precision)

    def env_reset(self):
        obs = self.env.reset()      
        assert obs['image'].dtype == torch.uint8

        for k in ['obs', 'bps', 'prop']:
            obs[k] = torch.clamp(obs[k], - self.obs_clip, self.obs_clip)

        return obs
    
    def env_step(self,action):
        action = torch.clamp(action, - self.action_clip, self.action_clip)
        obs, reward, termination, timeout, info = self.env.step(action)
        for k in ['obs', 'bps', 'prop']:
            if k in obs:
                obs[k] = torch.clamp(obs[k], - self.obs_clip, self.obs_clip)
        return obs, reward, termination, timeout, info

    def set_eval(self):
        self.ddp_network.eval()     

    def set_train(self):
        self.ddp_network.train()    

    
    def get_teacher_action(self,obs):
        with torch.no_grad():
            #print("get teacher action: ", obs.keys())
            result = self.teacher_network.get_action(obs)
            result = result['mu']
            result = torch.clamp(result, -self.action_clip, self.action_clip)
        return result

    def get_student_action(self, obs):
        with torch.no_grad():
            result = self.student_network.get_action(obs)
            result = torch.clamp(result, -self.action_clip, self.action_clip)
        return result
    

    def run(self):
        
        beta_init = self.cfg.beta
        for ep in range(1, self.cfg.max_epoch + 1):
            ####### collect data
            curr_beta = beta_init ** ep
            #curr_beta = beta_init
            self.set_eval()
            rewards = []
            env_step_start = time.time()
            for j in range(self.cfg.num_step_iters):
                obs = self.env_reset()
                teacher_action = self.get_teacher_action(obs) #根据amp网络输出的概率分布进行一个采样
                
                active_rendering_action, active_rendering_index = self.env.compute_active_rendering_action()
                teacher_action[:, active_rendering_index] = \
                    self.cfg.active_rendering_weight * active_rendering_action + \
                    (1-self.cfg.active_rendering_weight) * teacher_action[:, active_rendering_index]

                prop = obs['prop']
                raw_images = obs['image']
                transform_images = self.image_transform(raw_images.float().permute(0,3,1,2)/255.)
                obs['image'] = transform_images
                obs['last_imgs'] = self.data_buffer.get_last_imgs() # 获取上 num_last_imgs 帧视觉特征作为 obs 输入
                
                student_action = self.get_student_action(obs)
                #texts = obs['text']
                if np.random.rand() < curr_beta:
                    step_action = teacher_action
                else:
                    step_action = student_action
                #step_action = curr_beta * teacher_action + (1 - curr_beta) * student_action
                last_action = obs['last_action']
                
                image_feat = self.student_network.image_pre(self.student_network.image_backbone(obs['image']))
                self.data_buffer.store({
                    'image'     : raw_images,
                    #'text'      : texts,
                    'prop'      : prop,
                    'last_action'    : last_action,
                    'image_feat' : image_feat.detach(),
                    'teacher_action' : teacher_action.detach()
                })
                next_obs, reward, _, _ ,_ = self.env_step(step_action)
                rewards.append(reward.mean().item())
            rewards = np.mean(rewards)
            env_time = time.time() - env_step_start

            train_start_time = time.time()
            self.set_train()
            ####### train
            losses = []
            for j in range(self.cfg.num_train_iters):
                
                data = self.data_buffer.sample(self.bz)
                prop = data['prop']
                image = data['image']
                image = self.image_transform(image.permute(0,3,1,2)/255.)
                #text = data['text']
                last_action = data['last_action']
                last_imgs = data['last_imgs']
                #last_imgs[-1] = self.image_transform(image.permute(0,3,1,2)/255.)
                teacher_action = data['teacher_action']
                prop = self.student_network.normalize_prop(prop)
                if self.cfg.ddp:
                    self.student_network.sync_stats()

                with torch.cuda.amp.autocast(enabled=self.auto_mixed_precision):
                    action = self.ddp_network(prop, image, last_action, last_imgs)
                    loss = torch.nn.functional.mse_loss(action, teacher_action)
                
                for k, v in data.items(): # 防止梯度传递到数据采集环节
                    assert data[k].requires_grad == False, f"{k} requires grad"
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                if self.cfg.truncate_grads:
                    nn.utils.clip_grad_norm_(self.ddp_network.parameters(), self.cfg.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                losses.append(np.round(loss.item(),4)) #保留四位小数
            losses = np.mean(losses)
            
            train_time = time.time() - train_start_time

            if ep % self.cfg.report_epoch == 0:
                ########################### logger
                report = ''
                for k,v in self.env.export_logging_stats().items():
                    report += f'{k} {v:.3f}. '
                self.logger.info(f'Time [ENV {env_time:.3f}. TR {train_time:.3f}]. Epoch {ep}. Beta {curr_beta:.4f}. {report}Reward {rewards:.4f}. Loss {losses}')

                ########################### writer
                if is_main_proc(self.cfg):
                    self.writer.add_scalar(f'info/beta',        curr_beta, ep)
                    self.writer.add_scalar(f'info/reward',      rewards, ep)
                    self.writer.add_scalar(f'info/lr',          self.optimizer.param_groups[0]['lr'], ep)
                    self.writer.add_scalar(f'time/env_time',    env_time, ep)
                    self.writer.add_scalar(f'time/train_time',  train_time, ep)
                    self.writer.add_scalar(f'loss/total_loss',   np.mean(losses), ep)
            
            ########################### save weight
            if ep % self.cfg.save_epoch == 0:
                path = os.path.join(self.cfg.root, f'epoch_{ep}.pth')
                self.logger.info(f'Save ckpt to === > {path}')
                if is_main_proc(self.cfg):
                    self.save_ckpt(path)
    
    def save_ckpt(self,path):
        torch.save({
            'weight'        : self.student_network.state_dict(),
            'optimizer'     : self.optimizer.state_dict(),
        },path)


    def load_ckpt(self,path):
        self.logger.info(f'Load Ckpt from <== {path}')
        ckpt = torch.load(path, map_location='cpu')
        self.student_network.load_state_dict(ckpt['weight'])
        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(ckpt['optimizer'])
        