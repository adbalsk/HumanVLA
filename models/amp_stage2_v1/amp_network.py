import torch
import torch.nn as nn
from ..common.runningmeanstd import RunningMeanStd
from ..common.base_network import BaseNetwork
import numpy as np
import torchvision

import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.ERROR)

class ResBlock(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.layers(x) + x

class AMP_NETWORK(BaseNetwork):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        assert cfg.actor.hidden[-1] == cfg.num_action
        assert cfg.critic.hidden[-1] == 1
        assert cfg.disc.hidden[-1] == 1

        # Align with DAgger VLANetwork obs layout: prop, last_action, last_imgs, image features, bps
        # prop dim
        prop_dim = 15 * (3 + 6 + 3 + 3) - 2
        num_action = 28
        last_imgs_dim = self.cfg.image_pre.hidden[-1] * self.cfg.num_last_imgs
        # bps
        # self.bps_pts, _bps_dim = cfg.obs_space['bps']
        # assert _bps_dim == 3
        # self.bps_dim = self.bps_pts * _bps_dim

        # image backbone and preproc (use same backbone as DAgger)
        self.image_backbone = torchvision.models.efficientnet_b0()
        # remove classifier so backbone outputs feature vector
        self.image_backbone.classifier = nn.Sequential()
        image_feat_dim = 1280
        # image_pre: map image features to configured hidden dim
        self.image_pre = self.build_mlp(image_feat_dim, self.cfg.image_pre.hidden)
        
        # last_imgs: preserved as in DAgger (cfg.num_last_imgs * image_pre.hidden[-1])
        if hasattr(self.cfg, 'num_last_imgs') and self.cfg.num_last_imgs > 0:
            last_imgs_dim = self.cfg.image_pre.hidden[-1] * self.cfg.num_last_imgs
        else:
            last_imgs_dim = 0

        # last_action dim
        last_action_dim = cfg.obs_space.get('last_action', 0)

        # VL dim is image_pre output dim
        vl_dim = self.cfg.image_pre.hidden[-1]

        # final concatenated obs for actor/critic: prop + last_action + last_imgs + image_feat + bps_feat
        num_obs = prop_dim + last_action_dim + last_imgs_dim + vl_dim
        #223 28 1280 128 total: 1659

        self.actor_mlp  = nn.Sequential(
            nn.Linear(vl_dim + prop_dim + num_action + last_imgs_dim, 1024),
            ResBlock(1024),
            ResBlock(1024),
            nn.Linear(1024,28)
        )
        self.critic_mlp = self.build_mlp(num_obs, cfg.critic.hidden, last_activation=False)
        self.sigma = nn.Parameter(torch.zeros(cfg.num_action, requires_grad=True, dtype=torch.float32), requires_grad=False)

        amp_dim = cfg.num_amp_obs
        self.disc_mlp = self.build_mlp(amp_dim, cfg.disc.hidden, last_activation = False, last_bias=False)

        nn.init.constant_(self.sigma, self.cfg.sigma.init)
        for m in self.modules():
            if getattr(m, "bias", None) is not None:
                torch.nn.init.zeros_(m.bias)

        # try to init disc last layer uniformly
        try:
            torch.nn.init.uniform_(self.disc_mlp[-1].weight, -1, 1)
        except Exception:
            pass

        self.prop_normalizer    = RunningMeanStd(prop_dim)          if self.cfg.normalize_prop      else nn.Identity()
        self.bps_normalizer     = RunningMeanStd(3)                 if self.cfg.normalize_bps       else nn.Identity()
        self.val_normalizer     = RunningMeanStd(1)                 if self.cfg.normalize_value     else nn.Identity()
        self.disc_normalizer    = RunningMeanStd(amp_dim)           if self.cfg.normalize_amp       else nn.Identity()

    def normalize_obs(self, key, val):
        if key == 'obs' or key == 'prop':
            return self.prop_normalizer(val)
        elif key == 'bps':
            return self.bps_normalizer(val)
        else:
            raise NotImplementedError

    def normalize_value(self, val):
        return self.val_normalizer(val)

    def normalize_disc(self, val):
        return self.disc_normalizer(val)

    def compute_obs(self, obs, need_normalize = True):
        if(need_normalize):
            prop    = self.prop_normalizer(obs['prop'])
        else: prop = obs['prop']
        raw_img = obs['image']
        img = raw_img.float().permute(0,3,1,2)/255.
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.cfg.camera_height, self.cfg.camera_width)),
            torchvision.transforms.Normalize(
                mean = (0.5, 0.5, 0.5),
                std  = (0.5, 0.5, 0.5),
            )
        ])
        img = (transform(img))
        img = self.image_pre(self.image_backbone(img))

        last_action=obs['last_action']
        last_imgs = obs['last_imgs']
        last_imgs = last_imgs.flatten(1, 2)

        obs_cat = torch.cat([prop, last_action, last_imgs, img], dim=-1)
        return obs_cat

    def forward(self, obs, action, amp_obs_pos, amp_obs_neg):
        ### train
        obs_tensor = self.compute_obs(obs, need_normalize=False)
        mu = self.actor_mlp(obs_tensor)
        logstd = self.sigma.expand_as(mu)
        std = torch.exp(logstd)
        distribution = torch.distributions.Normal(mu, std)
        entropy = distribution.entropy().sum(-1)
        neglogp = self.neglogp(action, mu, std, logstd)

        norm_value = self.critic_mlp(obs_tensor)
        amp_logit_pos = self.disc_mlp(amp_obs_pos)
        amp_logit_neg = self.disc_mlp(amp_obs_neg)
        return {
            'mu'            : mu, #Actor输出的动作分布均值，对应高斯分布的均值参数。
            'sigma'         : std, #Actor输出的动作分布标准差，对应高斯分布的标准差参数，一个可学习的量。
            'neglogp'       : neglogp, #给定动作下的负对数概率密度，用于计算动作的概率权重。
            'entropy'       : entropy, #动作分布的熵，表示策略的随机性，熵越大表示策略越随机。鼓励策略保持探索性，避免过早收敛到局部最优。
            'norm_value'    : norm_value.squeeze(-1), #Critic输出的归一化状态值估计
            'amp_logit_pos' : amp_logit_pos.squeeze(-1), #AMP判别器对正样本的logits输出
            'amp_logit_neg' : amp_logit_neg.squeeze(-1), #AMP判别器对负样本的logits输出
        }


    def get_action(self,obs,need_normalize=True):
        obs_tensor = self.compute_obs(obs, need_normalize=need_normalize)
        mu = self.actor_mlp(obs_tensor)
        logstd = self.sigma.expand_as(mu)
        std = torch.exp(logstd)
        distribution = torch.distributions.Normal(mu, std)
        action = distribution.sample()
        neglogp = self.neglogp(action, mu, std, logstd)

        norm_value = self.critic_mlp(obs_tensor)
        if self.cfg.normalize_value:
            value = self.val_normalizer(norm_value.detach(), unnorm = True)
        else:
            value = norm_value
        return {
            'obs'           : obs_tensor,
            'action'        : action,
            'mu'            : mu,
            'sigma'         : std,
            'neglogp'       : neglogp,
            'value'         : value.squeeze(-1),
            'norm_value'    : norm_value.squeeze(-1)
        }

    def eval_critic(self,obs, need_normalize = True):
        obs_tensor = self.compute_obs(obs, need_normalize=need_normalize)
        norm_value = self.critic_mlp(obs_tensor)
        if self.cfg.normalize_value:
            value = self.val_normalizer(norm_value.detach(), unnorm = True)
        else:
            value = norm_value
        return {
            'value'   : value.squeeze(-1),
            'norm_value' : norm_value.squeeze(-1)
        }

    def eval_disc_reward(self, feat, need_normalize = True):
        if need_normalize:
            feat = self.disc_normalizer(feat)
        logits = self.disc_mlp(feat).squeeze(-1)
        prob = torch.sigmoid(logits)
        disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.01, device=prob.device)))
        return disc_r