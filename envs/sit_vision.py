from isaacgym import gymapi, gymtorch
import torch,os,json
import numpy as np
from utils import torch_utils
import pickle as pkl
from .sit_obs import SitEnv
from collections import OrderedDict
import open3d as o3d
import trimesh
import imageio

class SitVisionEnv(SitEnv):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
    
    def create_buffer(self):
        super().create_buffer()
        self.image_buf = torch.zeros((self.num_envs, self.cfg.camera_height, self.cfg.camera_width, 3), device=self.device, dtype=torch.uint8)
        
    def render_camera_image(self, env_ids = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) > 0:
            self.gym.step_graphics(self.sim)
            #self.gym.render_all_camera_sensors(self.sim)
            if self.enable_camera_tensor:
                self.gym.start_access_image_tensors(self.sim)
                for env_id in env_ids:
                    env_id = env_id.item()
                    self.image_buf[env_id] = self.camera_tensors[env_id][:,:,:3]
                # fname = os.path.join(f"TEST.png")
                # cam_img = self.image_buf[-1].cpu().numpy().astype(np.uint8)
                # imageio.imwrite(fname, cam_img)
                self.gym.end_access_image_tensors(self.sim)            
            else:
                for env_id in env_ids:
                    env_id = env_id.item()
                    cam_img = self.gym.get_camera_image(self.sim, self.env_handle[env_id], self.camera_handle[env_id], gymapi.IMAGE_COLOR)
                    cam_img = cam_img.reshape(self.cfg.camera_height, self.cfg.camera_width, 4)
                    self.image_buf[env_id] = torch.from_numpy(cam_img).to(self.device)[:,:,:3]
    
    @property
    def obs_space(self):
        return {
            'obs' : self.num_prop_obs + self.num_goal_obs,
            'bps' : (self.num_bps, 3),
            'last_action' : self.num_action,
            'prop' : self.num_prop_obs,
            'image': (self.cfg.camera_height, self.cfg.camera_width, 3)
        }

    def reset_output(self):
        obs = torch.cat([self.prop_buf, self.goal_buf], dim=-1)
        bps = self.asset_bps[self.object2asset[self.task_objectid]]
        self.render_camera_image()
        obs_buf = {
            'image' : self.image_buf,
            'prop': self.prop_buf,
            'obs' : obs,
            'bps' : bps,
            'last_action' : self.last_action,
        }
        return obs_buf
     
    def step_output(self):
        obs = torch.cat([self.prop_buf, self.goal_buf], dim=-1)
        bps = self.asset_bps[self.object2asset[self.task_objectid]]
        obs_buf = {
            'obs' : obs,
            'bps' : bps
        }

        return obs_buf, self.reward_buf, self.reset_termination_buf, self.reset_timeout_buf, {}

    def reset_env(self, env_ids):
        if env_ids is not None and len(env_ids) > 0:
            n = len(env_ids)
            rootmask = torch.isin(self.root2env, env_ids) #返回一个和self.num_root一样长的布尔向量，表示哪些root属于要reset的env
            objmask = torch.isin(self.object2env, env_ids)
            # reset root states
            self._root_states[rootmask, 0:3] = self.init_trans[rootmask]
            self._root_states[rootmask, 3:7] = self.init_rot[rootmask]
            self._root_states[rootmask, 7:] = 0.

            ############# 设置amp参考动作
            motion_ids = self.sample_motion_ids(n, samp=True)
            ref_motion_end_frame = torch.randint(90,size=env_ids.shape).to(self.device).float()
            
            # 对 env_id % 10 == 0 的环境，强制设为最后一帧
            mask1 = (env_ids % 11 == 0)
            mask2 = (env_ids % 5 == 1)
            mask = torch.logical_or(mask1, mask2)
            if mask.any() and not self.cfg.eval:
                # 这里假设 motion 长度 >= 400，可以直接取最后一帧索引 (num_frame-1)
                num_frames = self.motion['num_frame'][motion_ids[mask1]]
                ref_motion_end_frame[mask1] = (num_frames - 1).float().to(self.device)
                num_frames = self.motion['num_frame'][motion_ids[mask2]]
                ref_motion_end_frame[mask2] = (num_frames-num_frames+2).float().to(self.device)

            ref_motion_end_time = ref_motion_end_frame / self.query_motion_fps(motion_ids)
            ref_motion_time = ref_motion_end_time.unsqueeze(1).tile(self.num_ref_obs_frames) - torch.arange(self.num_ref_obs_frames).to(self.device) * self.dt
                #生成倒叙的参考动作的时间点，shape (n, num_ref_obs_frames)
            ref_motion_time = ref_motion_time.flatten()
            motion_ids = torch.repeat_interleave(motion_ids, self.num_ref_obs_frames,)
            state_info = self.query_motion_state(motion_ids, ref_motion_time)
            
            ## transform state #把参考动作对齐到仿真环境
            state_info['rigid_body_pos'] = state_info['rigid_body_pos'].view(n, self.num_ref_obs_frames, self.num_body, 3)
            state_info['rigid_body_rot'] = state_info['rigid_body_rot'].view(n, self.num_ref_obs_frames, self.num_body, 4)
            state_info['rigid_body_vel'] = state_info['rigid_body_vel'].view(n, self.num_ref_obs_frames, self.num_body, 3)
            state_info['rigid_body_anv'] = state_info['rigid_body_anv'].view(n, self.num_ref_obs_frames, self.num_body, 3)
            now_pos = self._root_states[self.robot2root[env_ids], :3]
            # goal_trans = self.goal_trans[self.task_rootid][env_ids]
            #print("nowpos, goaltrans", now_pos.shape, self.goal_trans.shape, self.goal_trans[self.task_rootid].shape)
            now_pos[mask1] = self.goal_trans[self.task_rootid[env_ids]][mask1] #added
            now_pos[mask2] = self.goal_trans[self.task_rootid[env_ids]][mask2]
            #print(now_pos[mask2].shape)
            now_pos[mask2, 0] += 0.5
            #print(now_pos.shape)
            now_pos[~mask1,2] = state_info['rigid_body_pos'][~mask1,0,0,2]
            delta_rot = torch_utils.quat_mul( #把 motion 的参考根姿态旋转到当前环境的朝向
                self._root_states[self.robot2root[env_ids], 3:7], 
                torch_utils.calc_heading_quat_inv(state_info['rigid_body_rot'][:,0,0,:])
                )
            
            obj_rot = self._root_states[self.task_rootid[env_ids], 3:7]
            #print(delta_rot.shape, obj_rot.shape)
            delta_rot[mask] = torch_utils.quat_mul( #把 motion 的参考根姿态旋转到当前环境的朝向
                obj_rot[mask], torch_utils.calc_heading_quat_inv(state_info['rigid_body_rot'][mask][:,0,0,:])
            )
            tensor90 = torch.tensor([-0.5,-0.5,-0.5,0.5],dtype=torch.float32,device=self.device).unsqueeze(0).expand(delta_rot.shape[0], -1)
            delta_rot[mask] = torch_utils.quat_mul(tensor90[mask], delta_rot[mask])
            
            #delta_rot[mask] = obj_rot[mask]
            #(0.7071, 0, 0.7071, 0)
            now_pos[:,2][mask1] += 0.1

            #记录初始状态
            now_rot = torch_utils.quat_mul(delta_rot, state_info['rigid_body_rot'][:,0,0,:])
            now_vel = torch_utils.quat_apply(delta_rot, state_info['rigid_body_vel'][:,0,0,:])
            now_anv = torch_utils.quat_apply(delta_rot, state_info['rigid_body_anv'][:,0,0,:])
            #用delta rot变换整个身体
            delta_rot_exp = delta_rot.unsqueeze(1).unsqueeze(1).tile(1,self.num_ref_obs_frames, self.num_body,1)
            state_info['rigid_body_pos'] = torch_utils.quat_rotate(delta_rot_exp, state_info['rigid_body_pos'])
            state_info['rigid_body_rot'] = torch_utils.quat_mul(delta_rot_exp, state_info['rigid_body_rot'])
            state_info['rigid_body_vel'] = torch_utils.quat_rotate(delta_rot_exp, state_info['rigid_body_vel'])
            state_info['rigid_body_anv'] = torch_utils.quat_rotate(delta_rot_exp, state_info['rigid_body_anv'])
            delta_pos = now_pos - state_info['rigid_body_pos'][:,0,0,:]
            state_info['rigid_body_pos'] = state_info['rigid_body_pos'] + delta_pos.unsqueeze(1).unsqueeze(1)
            
            ## reset sim state
            self._root_states[self.robot2root[env_ids], 0:3] = now_pos
            self._root_states[self.robot2root[env_ids], 3:7] = now_rot
            self._root_states[self.robot2root[env_ids[~mask2]], 7:10] = now_vel[~mask2]
            self._root_states[self.robot2root[env_ids[mask2]], 7:10] = 0.
            self._root_states[self.robot2root[env_ids], 10:13] = now_anv
            self._robot_dof_pos[env_ids,:] = state_info['dof_pos'].view(n, self.num_ref_obs_frames, self.num_dof)[:, 0, :]
            self._robot_dof_vel[env_ids,:] = state_info['dof_vel'].view(n, self.num_ref_obs_frames, self.num_dof)[:, 0, :]
            #把状态更新到物理引擎
            self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self._root_states))
            self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(self._dof_state))
            self.gym.fetch_results(self.sim, True)
            self._refresh_sim_tensors()


            ############# reset tensors
            self.progress_buf[env_ids]  = 0 
            self.reset_termination_buf[env_ids] = 0.
            self.reset_timeout_buf[env_ids] = 0.
            self.timeout_limit[env_ids] = self.cfg.max_episode_length
            self.consecutive_success[env_ids] = 0.

            # self.movenow_guideidx[env_ids] = 0.
            # self.movenow_preguidecomplete[env_ids] = False
            # self.movenow_guide[env_ids] = self.preguide_full[env_ids, 0,]

            ## Note: RB buffer is not flushed without phy step
            rb_state = torch.cat([
                state_info['rigid_body_pos'][:,0],
                state_info['rigid_body_rot'][:,0],
                state_info['rigid_body_vel'][:,0],
                state_info['rigid_body_anv'][:,0],
            ], dim=-1)
            self._rigid_body_state[self.robot2rb[env_ids]] = rb_state
            self.compute_observation(env_ids)

    def update_amp_obs(self):
        pass


    def compute_active_rendering_action(self):
        robot_rb_state = self._rigid_body_state[self.robot2rb].view(self.num_envs, self.num_body, 13)
        
        object_state = self._root_states[self.task_rootid]
        geom_object_pcd = self.asset_pcd[self.object2asset[self.task_objectid]] * self.object2scale[self.task_objectid].unsqueeze(-1).unsqueeze(-1)
        geom_object_pcd = torch_utils.transform_pcd(geom_object_pcd, pos = object_state[:,:3], rot = object_state[:,3:7])
        geom_object_pos = geom_object_pcd.mean(1)

        head_pos = robot_rb_state[:, self.body2index['head'], 0:3]
        head_rot = robot_rb_state[:, self.body2index['head'], 3:7]
        torso_pos = robot_rb_state[:, self.body2index['torso'], 0:3]
        torso_rot = robot_rb_state[:, self.body2index['torso'], 3:7]

        camera_pos = torch_utils.quat_rotate(
            torch_utils.quat_mul(head_rot, torch.tensor(self.camera_rot_offset,device=self.device).unsqueeze(0).tile(self.num_envs,1)), 
            torch.tensor(self.camera_pos_offset,device=self.device).unsqueeze(0).tile(self.num_envs,1)
            ) + head_pos        
        target_camera_dir = torch_utils.normalize(geom_object_pos - camera_pos)

        ############## quat-axis-delta
        # now_camera_dir = torch_utils.normalize(torch_utils.quat_rotate(
        #     torch_utils.quat_mul(head_rot, torch.tensor(self.camera_rot_offset,device=self.device).tile(self.num_envs,1)), 
        #     torch.tensor([1.,0.,0.],device=self.device).unsqueeze(0).tile(self.num_envs,1),
        #     ))
        # axis = torch.linalg.cross(now_camera_dir, target_camera_dir) 
        # angle = torch.arccos(torch.sum(now_camera_dir * target_camera_dir, dim=-1))
        # delta_q = torch_utils.quat_from_angle_axis(angle, axis)
        # head_target_rot = torch_utils.quat_mul(delta_q, head_rot)

        ############## up regularization
        forward_dir = target_camera_dir
        up_dir      = torch_utils.normalize(head_pos - torso_pos)
        side_dir    = torch_utils.normalize(torch.linalg.cross(up_dir, forward_dir))
        up_dir      = torch_utils.normalize(torch.linalg.cross(forward_dir, side_dir))
        rotation_matrix = torch.stack([forward_dir, side_dir, up_dir], dim=-1)
        head_target_rot = torch_utils.matrix_to_quat(rotation_matrix)
        
        neck_rot = torch_utils.quat_mul(torch_utils.quat_conjugate(torso_rot), head_target_rot)
        neck_dof = torch_utils.quat_to_exp_map(neck_rot)
        neck_index = [self.dof2index['neck_x'], self.dof2index['neck_y'], self.dof2index['neck_z']]
        neck_dof_upper = self.dof_limit_upper[neck_index]
        neck_dof_lower = self.dof_limit_lower[neck_index]
        neck_dof = torch.clamp(neck_dof, neck_dof_lower, neck_dof_upper)
        neck_action = (neck_dof - (neck_dof_lower+neck_dof_upper)/2) / ((neck_dof_upper-neck_dof_lower)/2)
        return neck_action, neck_index
    

