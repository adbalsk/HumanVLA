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
from ..common.base_network import BaseNetwork
from ..common.runningmeanstd import RunningMeanStd
import torch.distributed as dist

class DaggerRLTrainer:
    def __init__(self, cfg,  env) -> None:
        self.cfg = cfg
        self.env = env
        
        cfg.teacher_network.num_amp_obs = cfg.env.num_ref_obs_frames * cfg.env.num_ref_obs_per_frame
        cfg.teacher_network.obs_space = env.obs_space
        self.num_action = cfg.student_network.num_action = cfg.teacher_network.num_action = cfg.env.num_action
        self.num_last_imgs = cfg.student_network.num_last_imgs = cfg.teacher_network.num_last_imgs = cfg.env.num_last_imgs
        self.num_goal_obs = cfg.student_network.num_goal_obs = cfg.teacher_network.num_goal_obs = self.env.num_goal_obs

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
        if self.cfg.ddp:    
            self.student_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.student_network)
            self.ddp_network = DDP(self.student_network,device_ids=[self.cfg.rank])
        else:
            self.ddp_network = self.student_network
            
        self.img_h, self.img_w, self.image_transform = self.student_network.build_transform()
        self.logger.info('===============Image Transform=================')
        self.logger.info(self.image_transform)
        self.logger.info('===============Teacher Network=================')
        self.logger.info(self.teacher_network)
        self.logger.info('===============Student Network=================')
        self.logger.info(self.student_network)

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
            'last_imgs'     : dict(shape = (self.buffer_size, self.num_last_imgs, 128), dtype = torch.uint8),
            'goal'    :   dict(shape = (self.buffer_size, self.num_goal_obs)),
        }
        self.data_buffer = ReplayBuffer(buffer_info_dict, self.device)
        self.logger.info('===============ReplayBuffer=================')
        self.logger.info(self.data_buffer)

        self.optimizer = build_optimizer(cfg.optimizer, self.ddp_network.parameters())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.auto_mixed_precision)

        num_obs = self.prop_dim + self.num_action + self.num_goal_obs
        self.critic_mlp = BaseNetwork.build_mlp(BaseNetwork, num_obs, cfg.student_network.critic.hidden, last_activation=False).to(self.device)
        self.val_normalizer     = RunningMeanStd(1).to(self.device)  if self.cfg.teacher_network.normalize_value  else nn.Identity()
        self.gamma  = cfg.gamma
        self.tau = cfg.tau
        self.sigma = nn.Parameter(torch.zeros(self.env.num_action, device=self.device, requires_grad=True, dtype=torch.float32), requires_grad=False)

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
            #curr_beta = beta_init ** ep
            curr_beta = beta_init
            self.set_eval()
            rewards = []
            env_step_start = time.time()
            on_policy_buffer = defaultdict(list)
            obs = self.env_reset()
            for j in range(self.cfg.num_step_iters):
                teacher_action = self.get_teacher_action(obs)
                active_rendering_action, active_rendering_index = self.env.compute_active_rendering_action()
                teacher_action[:, active_rendering_index] = \
                    self.cfg.active_rendering_weight * active_rendering_action + \
                    (1-self.cfg.active_rendering_weight) * teacher_action[:, active_rendering_index]

                prop = obs['prop']
                raw_images = obs['image']
                transform_images = self.image_transform(raw_images.float().permute(0,3,1,2)/255.)
                obs['image'] = transform_images
                
                student_action_mu = self.get_student_action(obs)
                student_action_dict = self.get_action(student_action_mu, obs)
                student_action = student_action_dict['action']
                #texts = obs['text']
                if np.random.rand() < curr_beta:
                    step_action = teacher_action
                else:
                    step_action = student_action
                #step_action = curr_beta * teacher_action + (1 - curr_beta) * student_action

                next_obs, reward, termination, timeout,_ = self.env_step(step_action)
                rewards.append(reward.mean().item())
                val = self.critic_mlp(torch.cat([obs['prop'], obs['last_action'], obs['goal']], axis=-1))
                next_val = self.critic_mlp(torch.cat([next_obs['prop'], next_obs['last_action'], next_obs['goal']], axis=-1)).detach() #是否要计算梯度？
                self.data_buffer.store({
                    'image'     : raw_images,
                    #'text'      : texts,
                    'prop'      : prop,
                    'last_action'    : obs['last_action'],
                    'last_imgs' : obs['last_imgs'],
                    'teacher_action' : teacher_action.detach(),
                    'goal' : obs['goal'],
                })
                info = {
                    'prop'      : prop,
                    'last_action'    : obs['last_action'],
                    'goal' : obs['goal'],
                    'value' : val,
                    'next_value' : next_val,
                    'reward' : reward,
                    'termination' : termination,
                    'timeout' : timeout,
                    'action_dict' : student_action_dict,
                    'last_imgs': obs['last_imgs'],
                    'image': raw_images, 
                    }
                for k, v in info.items():
                    on_policy_buffer[k].append(v)
                
                obs = next_obs
            rewards = np.mean(rewards)
            env_time = time.time() - env_step_start

            train_start_time = time.time()
            self.set_train()
            ####### train
            #losses = []
            train_info = defaultdict(list)
            for j in reversed(range(self.cfg.num_train_iters)):
                
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
                    dagger_loss = torch.nn.functional.mse_loss(action, teacher_action)
                    
                    action_dict = on_policy_buffer['action_dict']
                    on_policy_buffer["neglogp"] = action_dict[j]['neglogp']
                    advantage = self.calc_adv(
                        termination = on_policy_buffer['termination'],
                        timeout     = on_policy_buffer['timeout'],
                        values      = on_policy_buffer['value'],
                        rewards     = on_policy_buffer['reward'],
                        next_values = on_policy_buffer['next_value'],
                    )
                    if self.cfg.normalize_adv:
                        normalize_advantage = self.dist_normalize_adv(advantage)
                    else:
                        normalize_advantage = advantage
                    on_policy_buffer["advantage"] = normalize_advantage

                rl_loss_dict = self.compute_rl_loss(on_policy_buffer)
                rl_loss = rl_loss_dict['rl_total_loss']
                loss = dagger_loss + rl_loss
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                if self.cfg.truncate_grads:
                    nn.utils.clip_grad_norm_(self.ddp_network.parameters(), self.cfg.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_info['total_loss'].append(loss.item())
                train_info['dagger_loss'].append(dagger_loss.item())
                for k, v in rl_loss_dict.items():
                    train_info[k].append(v.item())

            # losses = np.mean(losses)
            for k, v in train_info.items():
                train_info[k] = np.mean(v)
            train_time = time.time() - train_start_time

            if ep % self.cfg.report_epoch == 0:
                ########################### logger
                report = ''
                for k,v in self.env.export_logging_stats().items():
                    report += f'{k} {v:.3f}. '
                loss_report = f"Dagger Loss: {train_info['dagger_loss']:.4f}, Actor Loss: {train_info['actor_loss']:.4f}, Critic Loss: {train_info['critic_loss']:.4f}"
                self.logger.info(f'Time [ENV {env_time:.3f}. TR {train_time:.3f}]. Epoch {ep}. Beta {curr_beta:.4f}. {report}Reward {rewards:.4f}. {loss_report}')

                ########################### writer
                if is_main_proc(self.cfg):
                    self.writer.add_scalar(f'info/beta',        curr_beta, ep)
                    self.writer.add_scalar(f'info/reward',      rewards, ep)
                    self.writer.add_scalar(f'info/lr',          self.optimizer.param_groups[0]['lr'], ep)
                    self.writer.add_scalar(f'time/env_time',    env_time, ep)
                    self.writer.add_scalar(f'time/train_time',  train_time, ep)
                    #self.writer.add_scalar(f'loss/toal_loss',   np.mean(losses), ep)
                    for k, v in train_info.items():
                        self.writer.add_scalar(f'loss/{k}', v, ep)

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
    
    def get_action(self, mu, obs, need_normalize=True): #from get_action() in amp_network
        logstd = self.sigma.expand_as(mu)
        std = torch.exp(logstd)
        distribution = torch.distributions.Normal(mu, std)
        action = distribution.sample()
        neglogp = self.neglogp(action, mu, std, logstd)

        norm_value = self.critic_mlp(torch.cat([obs['prop'], obs['last_action'], obs['goal']], axis=-1)).squeeze(-1)
        if self.cfg.teacher_network.normalize_value:
            value = self.val_normalizer(norm_value.detach(), unnorm = True)
            #value = self.val_normalizer.unnormalize(norm_value)
        else:
            value = norm_value
        return {
            'obs'           : obs,
            'action'        : action,
            'mu'            : mu,
            'sigma'         : std,
            'neglogp'       : neglogp,
            'value'         : value.squeeze(-1),
            'norm_value'    : norm_value.squeeze(-1)
        }
    
    def calc_adv(self, termination, timeout, values, rewards, next_values): #删除了termination和timeout
        lastgaelam = 0
        advs = torch.zeros_like(torch.stack(rewards))
        for t in reversed(range(self.cfg.num_step_iters)):
            delta = rewards[t] + self.gamma * (1.0 - termination[t]) * next_values[t].squeeze(-1) - values[t].squeeze(-1)
            lastgaelam = delta + self.gamma * self.tau * (1.0 - timeout[t]) * lastgaelam
            advs[t] = lastgaelam
        return advs

    def compute_rl_loss(self, on_policy_buffer):
        with torch.cuda.amp.autocast(enabled=self.auto_mixed_precision):
            # amp_obs_pos = batch_dict['amp_demo_obs']
            # amp_obs_neg = batch_dict['amp_obs']
            # amp_obs_pos.requires_grad_(True)

            prop = torch.cat(on_policy_buffer["prop"], dim=0)
            last_action = torch.cat(on_policy_buffer["last_action"], dim=0)
            raw_images = torch.cat(on_policy_buffer["image"], dim=0)
            images = self.image_transform(raw_images.float().permute(0, 3, 1, 2) / 255.)
            last_imgs = torch.cat(on_policy_buffer["last_imgs"], dim=0)
            # prop = on_policy_buffer["prop"]
            # last_action = on_policy_buffer["last_action"]
            # images = on_policy_buffer["image"]
            # last_imgs = on_policy_buffer["last_imgs"]
            
            # train_meta  = self.ddp_network(obs, batch_dict['action'], amp_obs_pos, amp_obs_neg)
            
            train_meta = self.ddp_network(prop, images, last_action, last_imgs)

            entropy = train_meta['entropy'].mean(0)
            #################################################################################################### actor loss
            ratio = torch.exp(on_policy_buffer['neglogp'] - train_meta['neglogp'])
            surr1 = on_policy_buffer['advantage'] * ratio
            surr2 = on_policy_buffer['advantage'] * torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip)
            a_loss = torch.max(-surr1, -surr2)
            a_loss = a_loss.mean()
            a_clip_ratio = torch.abs(ratio - 1) > self.epsilon_clip
            a_clip_ratio = a_clip_ratio.float().mean()
            
            #################################################################################################### critic loss
            c_loss = (on_policy_buffer['norm_return'] - train_meta['norm_value']).square()
            c_loss = c_loss.mean()
            
            #################################################################################################### bound mu loss
            mu_bound = self.action_clip
            mu_loss_high = torch.clamp_min(train_meta['mu'] - mu_bound,    0) ** 2
            mu_loss_low  = torch.clamp_max(train_meta['mu'] + mu_bound,    0) ** 2
            mu_loss  = mu_loss_high + mu_loss_low
            mu_loss  = mu_loss.mean()

            ################################################################
            # amp_logit_pos = train_meta['amp_logit_pos']
            # amp_logit_neg = train_meta['amp_logit_neg']
            
            ############################### adv. prediction
            # disc_loss_pos = torch.nn.functional.binary_cross_entropy_with_logits(
            #     amp_logit_pos, torch.ones_like(amp_logit_pos)
            # )
            # disc_loss_neg = torch.nn.functional.binary_cross_entropy_with_logits(
            #     amp_logit_neg, torch.zeros_like(amp_logit_neg)
            # )
            # disc_prediction_loss = 0.5 * (disc_loss_pos + disc_loss_neg)

            # ############################### adv. logit reg
            # last_linear = self.network.disc_mlp[-1]
            # assert isinstance(last_linear, torch.nn.Linear)
            # disc_logit_loss = last_linear.weight.square().sum()
            
            ############################### adv. grad penalty
            # disc_pos_grad = torch.autograd.grad(
            #     amp_logit_pos, amp_obs_pos, grad_outputs=torch.ones_like(amp_logit_pos), create_graph=True, retain_graph=True, only_inputs=True)
            # disc_pos_grad = disc_pos_grad[0].square().sum(-1)
            # disc_grad_penalty = disc_pos_grad.mean()

            ################################ adv. weight_decay
            # disc_weights = []
            # for m in self.network.disc_mlp.modules():
            #     if isinstance(m, nn.Linear):
            #         disc_weights.append(m.weight.flatten())
            # disc_weights = torch.cat(disc_weights)
            # disc_weight_decay = disc_weights.square().sum()

            # disc_items    = [disc_prediction_loss,  disc_logit_loss,        disc_grad_penalty,      disc_weight_decay]
            # disc_items_w  = [self.loss.disc_pred,   self.loss.disc_logit,   self.loss.disc_grad,    self.loss.disc_wd]
            # disc_loss = sum([a * b for a,b in zip(disc_items,disc_items_w)])

            #################################################################################################### total loss
            #loss_items   = [a_loss,             c_loss,             mu_loss,            -entropy,           disc_loss]
            loss_items   = [a_loss,             c_loss,             mu_loss,            -entropy]
            loss_items_w = [self.loss.actor,    self.loss.critic,   self.loss.bound_mu, self.loss.entropy]
            total_loss = sum([a * b for a,b in zip(loss_items,   loss_items_w)])
        
        return {
            'rl_total_loss': total_loss,
            'actor_loss': a_loss,
            'critic_loss': c_loss,
            'entropy': -entropy, # 熵是负号，所以这里取反
        }
    
    def neglogp(self, x,mean,std,logstd):
        neglogp = 0.5 * (((x - mean) / std)**2)  + 0.5 * np.log(2.0 * np.pi)  + logstd
        neglogp = neglogp.sum(-1)
        return neglogp
    
    def dist_normalize_adv(self, values):
        # 1. 计算当前进程（GPU）上优势张量 'values' 的均值和平方均值
        mean = values.mean()
        sqaure_mean = (values ** 2).mean()

        # 2. 如果使用了分布式数据并行 (DDP)
        if self.cfg.ddp:
            # a. 将均值和平方均值打包成一个张量
            mean_info = torch.tensor([mean.item(), sqaure_mean.item()], device=self.device, dtype=torch.float32)
            
            # b. 在所有进程（GPU）之间进行 all_reduce 操作，计算全局平均值
            #    dist.all_reduce 会将所有进程的 mean_info 相加，然后除以进程总数
            dist.all_reduce(mean_info, op=dist.ReduceOp.AVG)
            
            # c. 用计算出的全局均值和全局平方均值覆盖局部值
            mean, sqaure_mean = mean_info

        # 3. 计算标准差 (std)
        #    std = sqrt(E[X^2] - (E[X])^2)
        std = torch.sqrt(sqaure_mean - mean ** 2)

        # 4. 标准化优势函数
        #    (value - mean) / (std + epsilon)
        #    加上一个很小的数 1e-8 是为了防止除以零
        values = (values - mean) / (std + 1e-8)
        
        return values