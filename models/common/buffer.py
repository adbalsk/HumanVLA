import torch
import numpy as np

class ExperienceBuffer():
    def __init__(self, info_dict, device, default_dtype = torch.float32) -> None:
        self.device = device
        self.tensor_dict = {}
        for k,info in info_dict.items():
            shape = info['shape']
            dtype = info.get('dtype', default_dtype)
            self.tensor_dict[k] = torch.zeros(
                shape,dtype=dtype,device=device
            )

    def update(self, key, idx, val):
        assert val.dtype == self.tensor_dict[key].dtype, (key, val.dtype, self.tensor_dict[key].dtype)
        assert val.shape == self.tensor_dict[key][idx].shape, (key, val.shape, self.tensor_dict[key].shape)
        self.tensor_dict[key][idx] = val.detach()

    
    def __str__(self) -> str:
        size = sum([v.numel() * v.element_size() / 4 for k,v in self.tensor_dict.items()])
        items = ''
        for k,v in self.tensor_dict.items():
            items += f'\t{k:20s} shape={str(v.shape):30s} dtype={v.dtype}\n'
        return f'Experience Buffer: device={self.device} size={size}@FP32\n{items}'

    def export(self,):
        return self.tensor_dict


class ReplayBuffer():
    def __init__(self, info_dict, device, default_dtype = torch.float32, num_last_imgs=5, last_img_interval=1, num_envs=1) -> None:
        self.device = device
        self.store_device = torch.device('cpu')
        # self.store_device = device
        self.tensor_dict = {}
        for k,info in info_dict.items():
            shape = info['shape']
            dtype = info.get('dtype', default_dtype)
            self.tensor_dict[k] = torch.zeros(
                shape,dtype=dtype,device=self.store_device
            )
        
        self.head = 0
        self.count = 0
        self.size = info['shape'][0]
        self.sample_idx = torch.randperm(self.size).to(self.store_device)
        self.sample_head = 0
        for k,info in info_dict.items():
            assert self.size == info['shape'][0]
        
        self.num_last_imgs = num_last_imgs
        self.last_img_interval = last_img_interval
        self.num_envs = num_envs

    def store(self, key_val_dict):
        key, val = list(key_val_dict.items())[0]
        new_len = val.shape[0] # 每次存入num_envs条数据
        #print('Buffer store new_len:', new_len)
        self.num_envs = new_len
        
        assert key_val_dict.keys() == self.tensor_dict.keys()
        for key,val in key_val_dict.items():
            assert val.shape[0] == new_len
            assert val.dtype == self.tensor_dict[key].dtype

        if new_len > self.size:
            rand_idx = torch.randperm(val.shape[0])
            rand_idx = rand_idx[:self.size]
            for key in key_val_dict:
                key_val_dict[key] = key_val_dict[key][rand_idx]
            new_len = self.size

        store_n = min(new_len, self.size - self.head)
        remind_n = new_len - store_n
        for key,val in key_val_dict.items():
            self.tensor_dict[key][self.head : self.head + store_n] = val[ : store_n].to(self.store_device)
            if remind_n > 0:
                self.tensor_dict[key][: remind_n] = val[store_n: ].to(self.store_device)
        self.head = (self.head + new_len) % self.size
        self.count = self.count + new_len
        
    
    def sample(self, n):
        idx = torch.arange(self.sample_head, self.sample_head + n, dtype = torch.long, device = self.store_device)
        idx = idx % self.size
        idx = self.sample_idx[idx]
        if self.count < self.size:
            idx = idx % self.count

        self.sample_head += n
        if self.sample_head > self.size:
            self.sample_idx = torch.randperm(self.size).to(self.store_device)
            self.sample_head = 0
        sample_dict = {
            key : val[idx].to(self.device)
            for key, val in self.tensor_dict.items()
        }

        batch_last_images = self.get_last_imgs(pos=idx)
        if( batch_last_images is not None):
            sample_dict['last_imgs'] = batch_last_images.to(self.device)

        return sample_dict


    def __str__(self) -> str:
        size = sum([v.numel() * v.element_size() / 4 for k,v in self.tensor_dict.items()])
        items = ''
        for k,v in self.tensor_dict.items():
            items += f'\t{k:20s} shape={str(v.shape):30s} dtype={v.dtype}\n'
        return f'Replay Buffer: store_device={self.store_device} data_device={self.device} size={size}@FP32\n{items}'


    def get_last_imgs(self, pos=None):
        if pos is None:
            # 如果没指定位置，默认取刚刚存入的那一批数据（最新的 num_envs 个数据）
            start = self.head - self.num_envs
            pos_tensor = torch.arange(start, start + self.num_envs, dtype=torch.long, device=self.store_device)
            pos_tensor = pos_tensor % self.size
        else:
            # 如果指定了位置，转成 tensor
            if isinstance(pos, (list, tuple, np.ndarray)):
                pos_tensor = torch.tensor(pos, dtype=torch.long, device=self.store_device)
            elif isinstance(pos, torch.Tensor):
                pos_tensor = pos.to(self.store_device).long()
            else:
                pos_tensor = torch.tensor([int(pos)], dtype=torch.long, device=self.store_device)
    
        n = pos_tensor.shape[0]
    
        #使用的图像缓冲区的键
        key = 'image_feat'
        if key not in self.tensor_dict:
            return None
        
        buf = self.tensor_dict[key]
        buf_ndim = buf.ndim
        
        stride = self.last_img_interval * self.num_envs

        #情况1：缓冲区是二维的，即 (buffer_size, feat_dim)
        if buf_ndim == 2:
            #print('get_last_imgs: buffer ndim=2, buffer shape=', buf.shape)
            feat_dim = buf.shape[1]
    
            if self.count >= self.size: #缓冲区已满，可以直接计算索引
                starts = (pos_tensor - (self.num_last_imgs - 1) * stride) % self.size
                offsets = torch.arange(self.num_last_imgs, device=self.store_device) * stride
                idx_matrix = (starts.unsqueeze(1) + offsets.unsqueeze(0)) % self.size
                flat_idxs = idx_matrix.reshape(-1)
                gathered = buf[flat_idxs]
                gathered = gathered.view(n, self.num_last_imgs, feat_dim)
                return gathered
    
            results = []
            for p in pos_tensor.tolist():
                p = int(p)
                if p < 0 or p >= self.count:
                    results.append(torch.zeros((self.num_last_imgs, feat_dim), dtype=buf.dtype, device=self.store_device))
                    continue
                
                idxs = []
                cur = p
                for _ in range(self.num_last_imgs):
                    if cur < 0:
                        break
                    idxs.append(cur)
                    cur -= stride
                
                idxs = list(reversed(idxs))
                if len(idxs) < self.num_last_imgs:
                    pad_n = self.num_last_imgs - len(idxs)
                    pad = torch.zeros((pad_n, feat_dim), dtype=buf.dtype, device=self.store_device)
                    if len(idxs) > 0:
                        fetched = buf[torch.tensor(idxs, device=self.store_device, dtype=torch.long)]
                        merged = torch.cat([pad, fetched], dim=0)
                    else:
                        merged = pad
                else:
                    fetched = buf[torch.tensor(idxs, device=self.store_device, dtype=torch.long)]
                    merged = fetched
                results.append(merged)
            stacked = torch.stack(results, dim=0)
            #print('get_last_imgs: stacked shape=', stacked.shape)
            return stacked
        
        #情况2：缓冲区是三维的，即 (buffer_size, N, feat_dim)
        if buf_ndim == 3:
            #print('get_last_imgs: buffer ndim=3, buffer shape=', buf.shape)
            N = buf.shape[1]
            feat_dim = buf.shape[2]
            results = []
            for p in pos_tensor.tolist():
                if self.count < self.size and (p < 0 or p >= self.count):
                    results.append(torch.zeros((self.num_last_imgs, feat_dim), dtype=buf.dtype, device=self.store_device))
                    continue
                entry = buf[p]
                if N < self.num_last_imgs:
                    pad_n = self.num_last_imgs - N
                    pad = torch.zeros((pad_n, feat_dim), dtype=buf.dtype, device=self.store_device)
                    merged = torch.cat([pad, entry[-N:]], dim=0)
                else:
                    merged = entry[-self.num_last_imgs:]
                results.append(merged)
            stacked = torch.stack(results, dim=0)
            return stacked
    
        raise ValueError('Unsupported buffer layout for get_last_imgs')
