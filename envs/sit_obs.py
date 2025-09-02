from isaacgym import gymapi, gymtorch
import torch,os,json
import numpy as np
from utils import torch_utils
import pickle as pkl
from .humanoid import HumanoidEnv
from collections import OrderedDict
import open3d as o3d
import trimesh
import imageio

class SitNewEnv(HumanoidEnv):
    def __init__(self, cfg) -> None:
        self.num_pcd = cfg.num_pcd

        # added：calculate obs size - 修正观测空间计算
        # 机器人身体观测：root_h(1) + body_pos(14*3) + body_rot(15*6) + body_vel(15*3) + body_anv(15*3)  
        self.num_prop_obs = 1 + 14*3 + 15*6 + 15*3 + 15*3  # = 1 + 42 + 90 + 45 + 45 = 223
        self.num_goal_obs = 9 + 2  # target sit position + chair facing dir (2D)
        #self.num_guidance_obs = 3
        self.num_obs = self.num_prop_obs + self.num_goal_obs  # simplified as num_obj_obs = 0

        self.num_ref_obs_frames = cfg.num_ref_obs_frames
        self.num_ref_obs_per_frame = cfg.num_ref_obs_per_frame
        self.num_bps = cfg.num_bps
        
        self.tasks = json.load(open(os.path.join(cfg.data_prefix, cfg.task_json)))
        if cfg.debug:
            self.tasks = self.tasks[:5]
        
        self.eval = cfg.eval
        self.test = cfg.test
        if self.eval:
            cfg.graphics_device_id = -1
            cfg.num_envs = len(self.tasks)
            cfg.enable_early_termination = False


        env_ids = range(cfg.num_envs*cfg.rank//cfg.world_size, cfg.num_envs*(cfg.rank+1)//cfg.world_size)
        self.num_envs = cfg.num_envs = len(env_ids)    
        self.env2task = []
        for env_id in env_ids:
            task_id = env_id % len(self.tasks)
            self.env2task.append(task_id)        
        self.eval_success_thresh = cfg.eval_success_thresh

        self.max_guide = 8
        self.num_guidance_obs = 3
        self.guide_proceed_dist = cfg.guide_proceed_dist

        self.enable_camera = cfg.get('enable_camera', False)
        self.enable_camera_tensor = cfg.get('enable_camera_tensor', False)
        super().__init__(cfg)
        self._robot_dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[:, :, 0]
        self._robot_dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[:, :, 1]
        
        self.hand_index = [self.body2index['left_hand'], self.body2index['right_hand']]
        self.foot_index = [self.body2index['left_foot'], self.body2index['right_foot']]

        self.consecutive_success_thresh = self.cfg.consecutive_success_thresh
        if self.test:
            self.consecutive_success_thresh = 5
        self.enable_early_termination = cfg.enable_early_termination


        print(f'#Num Tasks {len(self.tasks)}')
        print(f'#Num Envs {self.num_envs}')
        print(f'#Num Roots {self.num_root}')
        print(f'#Num Objects {self.num_object}')
        print(f'#Num RigidBodys {self.num_rigid_body}')

        #from HITR_carry
        self.fall_thresh = cfg.fall_thresh
        self.ignore_contact_name = cfg.ignore_contact_name
        self.ignore_contact_idx = [self.body2index[n] for n in self.ignore_contact_name]
        self.stats_step = {}

        self.amp_body_name = cfg.amp_body_name
        self.amp_body_idx = [self.body2index[n] for n in self.amp_body_name]

        #from tokenhsi
        self.sit_vel_penalty = cfg["sit_vel_penalty"]
        self.sit_vel_pen_coeff = cfg["sit_vel_pen_coeff"]
        self.sit_vel_pen_thre = cfg["sit_vel_pen_threshold"]
        self.sit_ang_vel_pen_coeff = cfg["sit_ang_vel_pen_coeff"]
        self.sit_ang_vel_pen_thre = cfg["sit_ang_vel_pen_threshold"]

        self._power_reward = cfg["power_reward"]
        self._power_coefficient = cfg["power_coefficient"]

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_root_rot = torch.zeros([self.num_envs, 4], device=self.device, dtype=torch.float)

    def create_sim(self):
        if self.enable_camera:
            self.graphics_device_id = self.sim_device_id
        return super().create_sim()

    @property
    def obs_space(self): #todo: 设置obs_space
        return {
            'obs' : self.num_prop_obs + self.num_goal_obs,
            'bps' : (self.num_bps, 3)
        }
    
    def create_buffer(self):
        self.last_action = torch.zeros((self.num_envs, self.num_action), device=self.device, dtype=torch.float32)
        self.reward_buf     = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.progress_buf   = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_termination_buf  = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.reset_timeout_buf      = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.timeout_limit          = torch.zeros(self.num_envs, device=self.device, dtype=torch.long).fill_(self.cfg.max_episode_length)
        self.success_steps          = torch.zeros(self.num_envs, device=self.device, dtype=torch.long).fill_(self.cfg.max_episode_length)
        self.guide_buf  = torch.zeros((self.num_envs, self.num_guidance_obs), device=self.device, dtype=torch.float32)

        #from HITR_carry
        self.prop_buf   = torch.zeros((self.num_envs, self.num_prop_obs), device=self.device, dtype=torch.float32)
        self.goal_buf   = torch.zeros((self.num_envs, self.num_goal_obs), device=self.device, dtype=torch.float32)
        self.amp_obs_buf = torch.zeros((self.num_envs, self.num_ref_obs_frames, self.num_ref_obs_per_frame), device=self.device,dtype=torch.float32)
        
        self.consecutive_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.prev_root_rot = torch.zeros([self.num_envs, 4], device=self.device, dtype=torch.float)

    def load_asset(self): #todo: load object assets
        super().load_asset() #load humanoid asset
        filefix_keys = []
        self.asset_files = []
        self.file2asset = {}

        for env_id, task_id in enumerate(self.env2task):
            task = self.tasks[task_id]
            for name, obj in task['object'].items():
                file = obj['file']
                fix = obj['fix_base_link']
                key = (file, fix)
                if file in self.file2asset:
                    pass
                else:
                    self.file2asset[file] = len(self.asset_files)
                    self.asset_files.append(file)
                if key not in filefix_keys:
                    filefix_keys.append(key)
        
        self.object_assets = {} #加载urdf
        for file, fix in filefix_keys:
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.use_mesh_materials = True
            object_asset_options.fix_base_link = fix
            object_asset_options.armature = 0.01
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            #object_asset_options.vhacd_params.resolution = 3000000 if file in self.high_vhacd_resolution_assets else 100000
            object_asset_options.vhacd_params.max_num_vertices_per_ch = 512
            object_asset_root = os.path.join(self.cfg.data_prefix, self.cfg.object.asset_root, file)
            object_asset_file = f'{file}.urdf'
            object_asset = self.gym.load_asset(self.sim, object_asset_root, object_asset_file, object_asset_options)
            self.object_assets[(file,fix)] = object_asset
    
        self.asset_pcd = [] #加载点云
        for idx, file in enumerate(self.asset_files):
            pcdpath = os.path.join(self.cfg.data_prefix, self.cfg.object.asset_root, file, f'{file}_pcd1000.xyz')
            pcd = np.loadtxt(pcdpath)
            pcd = torch.from_numpy(pcd).float().to(self.device)
            self.asset_pcd.append(pcd)
        self.asset_pcd = torch.stack(self.asset_pcd, 0)
        centrioids, self.asset_pcd = torch_utils.farthest_point_sample(self.asset_pcd, self.num_pcd)

        #from HITR_carry #加载bps
        bps_path = os.path.join(self.cfg.data_prefix, self.cfg.object.asset_root, f'bps_{self.num_bps:d}.pkl')
        bps_dict = pkl.load(open(bps_path, 'rb'))
        self.bps_points = torch.from_numpy(bps_dict['bps_points']).to(self.device)
        self.asset_bps = []

        for idx, file in enumerate(self.asset_files):
            bps = bps_dict[file]
            bps = torch.from_numpy(bps).float().to(self.device)
            self.asset_bps.append(bps)
        self.asset_bps = torch.stack(self.asset_bps, 0)
        

    def create_scene(self): #todo: create scene
        lower = gymapi.Vec3(-self.cfg.spacing, -self.cfg.spacing, 0.0)
        upper = gymapi.Vec3(self.cfg.spacing, self.cfg.spacing, self.cfg.spacing)
        
        self.num_rigid_body = 0
        self.num_root = 0
        self.num_object = 0
        self.num_robot = 0

        self.env_handle     = [] # [<handle>, ...] (num_envs)
        self.robot_handle   = [] # [<handle>, ...] (num_envs)
        self.object_handle  = [] # [{'bed': <handle>, ...} ,...]
        self.camera = []
        self.camera_handle  = []
        self.camera_tensors = []

        self.init_trans = [] # (num_root, 3) 
        self.init_rot   = [] # (num_root, 4)
        self.goal_trans = [] # (num_root, 3)
        self.goal_rot   = [] # (num_root, 4)

        self.root2env           = [] # (num_root)
        self.env2rootlist       = [] # [[...], [...] , ...]
        self.env2objectsize     = [] # (num_env)
        self.robot2root         = [] # (num_robot) 
        self.robot2rb           = [] # (num_robot, 15) 
        self.object2env         = [] # (num_object)
        self.object2name        = [] # (num_object)
        self.object2root        = [] # (num_object)
        self.object2fix         = [] # (num_object)
        self.object2asset       = [] # (num_object)
        self.object2scale       = [] # (num_object)
        
        for env_id in range(self.num_envs):
            ##### load task
            task_id = self.env2task[env_id]
            task = self.tasks[task_id]

            ##### spawn env
            env_ptr = self.gym.create_env(self.sim, lower, upper, round(self.num_envs ** 0.5))
            self.env_handle.append(env_ptr)
            self.env2rootlist.append([])
            self.object_handle.append({})

            ##### spawn robot
            init_trans = task['robot']['init_pos']
            init_rot = task['robot']['init_rot']
                
            init_robot_pose = gymapi.Transform(p=gymapi.Vec3(*init_trans), r=gymapi.Quat(*init_rot))
            robot = self.gym.create_actor(env_ptr, self.humanoid_asset, init_robot_pose, "robot", env_id, 0, 0)
            for j in range(self.num_body):
                self.gym.set_rigid_body_color(env_ptr, robot, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))
            dof_prop = self.gym.get_asset_dof_properties(self.humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, robot, dof_prop)
        
            self.init_trans.append(init_trans)
            self.init_rot.append(init_rot)
            # not used goal trans/rot for robot
            self.goal_trans.append(init_trans)
            self.goal_rot.append(init_rot)
            
            self.robot_handle.append(robot)
            self.robot2root.append(self.num_root)
            self.env2rootlist[env_id].append(self.num_root)
            self.root2env.append(env_id)

            self.num_root += 1
            self.num_robot += 1
            for i in range(self.num_body):
                self.robot2rb.append(self.num_rigid_body)
                self.num_rigid_body += 1

            if self.enable_camera:
                cam_props = gymapi.CameraProperties()
                cam_props.horizontal_fov = self.cfg.horizontal_fov
                cam_props.width = self.cfg.camera_width
                cam_props.height = self.cfg.camera_height
                cam_props.enable_tensors = self.enable_camera_tensor
                self.camera_pos_offset = self.cfg.camera_pos
                self.camera_rot_offset = self.cfg.camera_rot
                pos = gymapi.Vec3(*self.camera_pos_offset)
                rot = gymapi.Quat(*self.camera_rot_offset)

                camera_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                self.gym.attach_camera_to_body(
                    camera_handle, env_ptr, self.gym.find_actor_rigid_body_handle(env_ptr, robot, 'head'), 
                    gymapi.Transform(p=pos, r=rot), gymapi.FOLLOW_TRANSFORM)
                self.camera_handle.append(camera_handle)

                if self.enable_camera_tensor:
                    cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
                    torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
                    self.camera_tensors.append(torch_cam_tensor)
                
            ##### spawn objects
            self.env2objectsize.append(len(task['object']))
            for name in sorted(task['object'].keys()):
                object_info = task['object'][name]
                
                init_trans  = object_info['init_pos']
                init_rot    = object_info['init_rot']
                goal_trans  = object_info['goal_pos']
                goal_rot    = object_info['goal_rot']
                self.init_trans.append(init_trans)
                self.init_rot.append(init_rot)
                self.goal_trans.append(goal_trans)
                self.goal_rot.append(goal_rot)
                
                init_object_pose = gymapi.Transform(p = gymapi.Vec3(*init_trans), r = gymapi.Quat(*init_rot))
                object_handle = self.gym.create_actor(
                    env_ptr, self.object_assets[(object_info['file'],object_info['fix_base_link'])], init_object_pose, name, env_id, 0, segmentationId = 0) # todo segmentationId
                # self.gym.set_actor_scale(env_ptr, object_handle, object_info['scale'])
                self.object_handle[env_id][name] = object_handle
                
                self.root2env.append(env_id)
                self.env2rootlist[env_id].append(self.num_root)
                self.object2name.append(name)
                self.object2root.append(self.num_root)
                self.object2env.append(env_id)
                self.object2fix.append(object_info['fix_base_link'] )
                self.object2asset.append(self.file2asset[object_info['file']])
                self.object2scale.append(object_info['scale'])


                self.num_root += 1
                self.num_rigid_body += 1 
                self.num_object += 1

        assert self.num_robot == self.num_envs
        assert self.num_robot + self.num_object == self.num_root
        assert self.num_robot * self.num_body + self.num_object == self.num_rigid_body

        assert len(self.env_handle) == self.num_envs
        assert len(self.robot_handle) == self.num_envs
        assert len(self.object_handle) == self.num_envs
        

        self.init_trans = torch.tensor(self.init_trans).to(self.device)
        self.init_rot   = torch.tensor(self.init_rot).to(self.device)
        self.goal_trans = torch.tensor(self.goal_trans).to(self.device)
        self.goal_rot   = torch.tensor(self.goal_rot).to(self.device)
        self.root2env   = torch.tensor(self.root2env).to(self.device)

        self.env2objectsize = torch.tensor(self.env2objectsize).to(self.device)
        self.robot2root     = torch.tensor(self.robot2root, device=self.device) 
        self.robot2rb       = torch.tensor(self.robot2rb, device=self.device).view(self.num_robot, self.num_body)  
        self.object2root    = torch.tensor(self.object2root, device=self.device) 
        self.object2env     = torch.tensor(self.object2env, device=self.device) 
        self.object2fix     = torch.tensor(self.object2fix).to(self.device)
        self.object2asset   = torch.tensor(self.object2asset).to(self.device)
        self.object2scale   = torch.tensor(self.object2scale).to(self.device)
        
        assert len(self.env_handle)     == self.num_envs
        assert len(self.robot_handle)   == self.num_envs
        assert len(self.object_handle)  == self.num_envs
        assert self.init_trans.shape    == (self.num_root, 3)
        assert self.init_rot.shape      == (self.num_root, 4)
        assert self.goal_trans.shape    == (self.num_root, 3)
        assert self.goal_rot.shape      == (self.num_root, 4)
        assert self.root2env.shape      == (self.num_root,)  
        assert len(self.env2rootlist)   == self.num_envs
        assert self.env2objectsize.shape== (self.num_envs,)  
        assert self.robot2root.shape    == (self.num_robot,)  
        assert self.robot2rb.shape      == (self.num_robot, 15)  
        assert len(self.object2name)    == self.num_object
        assert self.object2root.shape   == (self.num_object,)  
        assert self.object2env.shape    == (self.num_object,)  
        assert self.object2fix.shape    == (self.num_object,)  
        assert self.object2asset.shape  == (self.num_object,)  
        assert self.object2scale.shape  == (self.num_object,)      

        ##################### todo tasks
        self.envname2object = {(env_id.item(), name) : obj_id for obj_id, (env_id, name) in enumerate(zip(self.object2env, self.object2name))}

        self.task_objectid = torch.zeros((self.num_envs,), dtype=torch.long,device=self.device)
        self.task_rootid = torch.zeros((self.num_envs,), dtype=torch.long,device=self.device)

        self.preguide_full = torch.zeros((self.num_envs, self.max_guide, 3), dtype=torch.float32, device=self.device)        
        self.preguide_length = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)
        self.postguide_full = torch.zeros((self.num_envs, self.max_guide, 3), dtype=torch.float32, device=self.device)
        self.postguide_length = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)
        
        for env_id in range(self.num_envs):
            task_id = self.env2task[env_id]
            task = self.tasks[task_id]
            
            object_name = task['move'] #全是chair
            object_id = self.envname2object[(env_id, object_name)]
            root_id = self.object2root[object_id].item()
            
            self.task_objectid[env_id] = object_id
            self.task_rootid[env_id] = root_id

            #plan = task['plan']
            
            init_z = self.init_trans[root_id,2].item()
            goal_z = self.goal_trans[root_id,2].item()
            #guide_z = max(init_z, goal_z) + self.cfg.carry_height if init_z > 0.1 or goal_z > 0.1 else goal_z
            guide_z = goal_z + 0.1

            # for guide_id, p in enumerate(plan['pre_waypoint']):
            #     p[2] = guide_z
            #     self.preguide_full[env_id, guide_id] = torch.tensor(p, dtype=torch.float32, device=self.device)
            #     self.preguide_length[env_id] += 1
                
            # for guide_id, p in enumerate(plan['post_waypoint']):
            #     p[2] = guide_z
            #     self.postguide_full[env_id, guide_id] = torch.tensor(p, dtype=torch.float32, device=self.device)
            #     self.postguide_length[env_id] += 1
        
        ########### running time buffer
        self.movenow_preguidecomplete = torch.zeros(self.num_envs, dtype=torch.bool,device=self.device)
        self.movenow_guideidx = torch.zeros(self.num_envs, dtype=torch.long,device=self.device)
        self.movenow_guide = self.preguide_full[:,0,:].clone()

        #added # position the camera
        # if self.up_axis == 'z':
        #     cam_pos = gymapi.Vec3(6.0, 8.0, 3.0) #(20.0, 25.0, 3.0)
        #     cam_target = gymapi.Vec3(0, 0, 2.0) # (10.0, 15.0, 2.0)
        # else:
        #     cam_pos = gymapi.Vec3(6.0, 3.0, 8.0) #(20.0, 3.0, 25.0)
        #     cam_target = gymapi.Vec3(0, 2.0, 0) # (10.0, 2.0, 15.0)
        cam_pos = gymapi.Vec3(*self.cfg.camera.pos)
        cam_target = gymapi.Vec3(*self.cfg.camera.tgt)
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1000
        camera_properties.height = 750
        # position the camera
        for i in range(4):
            self.camera.append(self.gym.create_camera_sensor(self.env_handle[i], camera_properties))
            self.gym.set_camera_location(self.camera[i], self.env_handle[i], cam_pos, cam_target)

        self.gym.viewer_camera_look_at(
            self.viewer, None, cam_pos, cam_target)
        
    def reset(self):
        reset = torch.logical_or(self.reset_termination_buf == 1, self.reset_timeout_buf == 1)
        reset_env_ids = torch.where(reset)[0]
        self.eval_last_runs(reset_env_ids)
        self.reset_env(reset_env_ids)
        self.success_steps[reset_env_ids] = self.cfg.max_episode_length
        self.compute_guidance()
        # self.guide_buf[:]=self.compute_guide_obs(
        #     self._root_states[self.robot2root, 0:3],
        #     self._root_states[self.robot2root, 3:7],
        #     self.movenow_guide,
        # )

        #todo:按照tokenhsi，这边需要计算坐下来的target位置
        return self.reset_output()

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
            # 修改参考动作的时间选择，从较早的帧开始
            ref_motion_end_frame = torch.randint(30, 90, size=env_ids.shape).to(self.device).float()  # 从30-90帧中选择
            ref_motion_end_time = ref_motion_end_frame / self.query_motion_fps(motion_ids)
            ref_motion_time = ref_motion_end_time.unsqueeze(1).tile(self.num_ref_obs_frames) - torch.arange(self.num_ref_obs_frames).to(self.device) * self.dt
                #生成倒叙的参考动作的时间点，shape (n, num_ref_obs_frames)
            ref_motion_time = torch.clamp(ref_motion_time, 0.0, float('inf'))  # 确保时间不会小于0
            ref_motion_time = ref_motion_time.flatten()
            motion_ids = torch.repeat_interleave(motion_ids, self.num_ref_obs_frames,)
            state_info = self.query_motion_state(motion_ids, ref_motion_time)
            
            ## transform state #把参考动作对齐到仿真环境
            state_info['rigid_body_pos'] = state_info['rigid_body_pos'].view(n, self.num_ref_obs_frames, self.num_body, 3)
            state_info['rigid_body_rot'] = state_info['rigid_body_rot'].view(n, self.num_ref_obs_frames, self.num_body, 4)
            state_info['rigid_body_vel'] = state_info['rigid_body_vel'].view(n, self.num_ref_obs_frames, self.num_body, 3)
            state_info['rigid_body_anv'] = state_info['rigid_body_anv'].view(n, self.num_ref_obs_frames, self.num_body, 3)
            now_pos = self._root_states[self.robot2root[env_ids], :3]
            now_pos[:,2] = state_info['rigid_body_pos'][:,0,0,2]
            delta_rot = torch_utils.quat_mul( #把 motion 的参考根姿态旋转到当前环境的朝向
                self._root_states[self.robot2root[env_ids], 3:7], 
                torch_utils.calc_heading_quat_inv(state_info['rigid_body_rot'][:,0,0,:])
                )
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
            self._root_states[self.robot2root[env_ids], 7:10] = now_vel
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

            self.movenow_guideidx[env_ids] = 0.
            self.movenow_preguidecomplete[env_ids] = False
            self.movenow_guide[env_ids] = self.preguide_full[env_ids, 0,]

            ## Note: RB buffer is not flushed without phy step
            rb_state = torch.cat([
                state_info['rigid_body_pos'][:,0],
                state_info['rigid_body_rot'][:,0],
                state_info['rigid_body_vel'][:,0],
                state_info['rigid_body_anv'][:,0],
            ], dim=-1)
            self._rigid_body_state[self.robot2rb[env_ids]] = rb_state
            self.compute_observation(env_ids)
            

            ############# reset AMP #生成基础的amp obs
            obj_pos = torch.repeat_interleave(self._root_states[self.task_rootid[env_ids], 0:3], self.num_ref_obs_frames, dim=0)
            amp_demo_obs = self.compute_ref_frame_obs(
                root_pos=state_info['rigid_body_pos'][:,:,0,:].view(-1,3),
                root_rot=state_info['rigid_body_rot'][:,:,0,:].view(-1,4),
                root_vel=state_info['rigid_body_vel'][:,:,0,:].view(-1,3),
                root_anv=state_info['rigid_body_anv'][:,:,0,:].view(-1,3),
                dof_pos=state_info['dof_pos'],
                dof_vel=state_info['dof_vel'],
                key_body_pos=state_info['rigid_body_pos'].view(-1,self.num_body,3)[:,self.amp_body_idx,:],
                obj_pos=obj_pos
            ).view(n, self.num_ref_obs_frames, self.num_ref_obs_per_frame)
            self.amp_obs_buf[env_ids] = amp_demo_obs

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._root_states[self.robot2root, 0:3]
        self._prev_root_rot[:] = self._root_states[self.robot2root, 3:7]
        return
        
    def post_physics_step(self):
        self.progress_buf[:] += 1
        self._refresh_sim_tensors()
        self.compute_observation()
        self.compute_reward()
        self.compute_termination()
        self.compute_success_steps()

        #from HITR_carry
        self.update_amp_obs()
    
    #from HITR_carry
    def update_amp_obs(self):
        for j in range(self.num_ref_obs_frames - 1, 0, - 1):
            self.amp_obs_buf[:, j, :] = self.amp_obs_buf[:, j - 1, :]
        robot_rb_state = self._rigid_body_state[self.robot2rb].view(self.num_envs, self.num_body, 13)
        object_state = self._root_states[self.task_rootid]
        amp_obs = self.compute_ref_frame_obs(
            root_pos=robot_rb_state[:, 0, 0:3],
            root_rot=robot_rb_state[:, 0, 3:7],
            root_vel=robot_rb_state[:, 0, 7:10],
            root_anv=robot_rb_state[:, 0, 10:13],
            dof_pos=self._robot_dof_pos,
            dof_vel=self._robot_dof_vel,
            key_body_pos=robot_rb_state[:,self.amp_body_idx,:3],
            obj_pos     =object_state[:, 0:3],
        )
        self.amp_obs_buf[:, 0, :] = amp_obs

    def compute_success_steps(self):
        dist = torch.norm(
            self._root_states[self.task_rootid,0:3] - self.goal_trans[self.task_rootid], #goal_trans即目标位置
            p=2, dim=-1)
        success_mask = dist < self.eval_success_thresh
        self.success_steps[success_mask] = torch.min(
            self.progress_buf[success_mask],
            self.success_steps[success_mask]
            )
    

    def calc_diff_pos(self, p1 , p2):
        return torch.norm(p1-p2,dim=-1)
    
    def calc_diff_rot(self, r1, r2):
        diff_ang, _ = torch_utils.quat_to_angle_axis(torch_utils.quat_mul(
            r1, torch_utils.quat_conjugate(r2)))
        diff_ang = diff_ang.abs()
        return diff_ang    


    def compute_guidance(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # 计算椅子正面的目标位置
        object_state = self._root_states[self.task_rootid[env_ids]]
        obj_pos = object_state[:, 0:3]
        obj_rot = object_state[:, 3:7]
        chair_forward_vec = torch.tensor([0.0, 1.0, 0.0], device=obj_rot.device, dtype=obj_rot.dtype)
        chair_forward_vec = chair_forward_vec.unsqueeze(0).expand(obj_rot.shape[0], -1)
        chair_forward = torch_utils.quat_rotate(obj_rot, chair_forward_vec)

        # 在椅子正面0.5米处设置引导点，机器人站在这里背对椅子
        approach_distance = 0.5
        front_target = obj_pos - chair_forward * approach_distance
        front_target[:, 2] = obj_pos[:, 2]  # 保持相同高度
        
        self.movenow_guide[env_ids] = front_target
        return

        # n = len(env_ids)
        # robot_rb_state = self._rigid_body_state[self.robot2rb[env_ids]].view(n, self.num_body, 13)

        # foot_pos = robot_rb_state[:,self.foot_index,:2].mean(1)
        # guide_dist = torch.norm(foot_pos - self.movenow_guide[env_ids,:2], dim=-1)
        # guide_ok = guide_dist < self.guide_proceed_dist

        # guide_proceed_envs = env_ids[guide_ok]
        # self.movenow_guideidx[guide_proceed_envs] += 1
        # postguide_alldone_mask = torch.logical_and(self.movenow_preguidecomplete[env_ids], self.movenow_guideidx[env_ids]>=self.postguide_length[env_ids])
        # postguide_alldone_envs = env_ids[postguide_alldone_mask]
        # self.movenow_guideidx[postguide_alldone_envs] = self.postguide_length[postguide_alldone_envs]
        # preguide_alldone_mask = torch.logical_and(~self.movenow_preguidecomplete[env_ids], self.movenow_guideidx[env_ids]==self.preguide_length[env_ids])
        # preguide_alldone_envs = env_ids[preguide_alldone_mask]
        # self.movenow_guideidx[preguide_alldone_envs] = 0.
        # self.movenow_preguidecomplete[preguide_alldone_envs] = True

        # self.movenow_guide[env_ids] = torch.where(
        #     self.movenow_preguidecomplete[env_ids].unsqueeze(-1).tile(3),
        #     self.postguide_full[env_ids, self.movenow_guideidx[env_ids]],
        #     self.preguide_full[env_ids, self.movenow_guideidx[env_ids]]
        # )
        # self.movenow_guide[postguide_alldone_envs] = self.goal_trans[self.task_rootid[postguide_alldone_envs]]
    
    def eval_last_runs(self,env_ids): #modified
        if env_ids is not None and len(env_ids) > 0:
            root_pos = self._root_states[self.robot2root, 0:3],
            root_rot = self._root_states[self.robot2root, 3:7],
            tar_pos_sit = self.goal_trans[self.task_rootid]

            root_pos = root_pos[0] #这样使两者都是torch.Size([4, 3])的tensor

            pos_diff = self.calc_diff_pos(root_pos, tar_pos_sit)
            success = pos_diff <= self.eval_success_thresh
            #print(root_pos.shape, tar_pos_sit.shape, pos_diff.shape, success.shape)
            self.last_success[env_ids] = success[env_ids].float()

        return
    
    def reset_output(self): #modified
        obs = torch.cat([self.prop_buf, self.goal_buf], dim=-1)
        bps = self.asset_bps[self.object2asset[self.task_objectid]] #todo: 这里的物体点云是什么
        obs_buf = {
            'obs' : obs,
            'bps' : bps
        }
        return obs_buf

    def step_output(self): #from HITR_carry
        obs = torch.cat([self.prop_buf, self.goal_buf], dim=-1)
        bps = self.asset_bps[self.object2asset[self.task_objectid]] #todo: 这里的物体点云是什么
        obs_buf = {
            'obs' : obs,
            'bps' : bps
        }
        extra = {
            'amp_obs' : self.amp_obs_buf.reshape(self.num_envs, self.num_ref_obs_frames * self.num_ref_obs_per_frame)
        }

        return obs_buf, self.reward_buf, self.reset_termination_buf, self.reset_timeout_buf, extra
    
    def compute_observation(self,env_ids = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        n = len(env_ids)
        robot_rb_state = self._rigid_body_state[self.robot2rb[env_ids]].view(n, self.num_body, 13)
        object_state = self._root_states[self.task_rootid[env_ids]]
        self.prop_buf[env_ids] = self.compute_prop_obs(
            body_pos    =robot_rb_state[:, :, 0:3],
            body_rot    =robot_rb_state[:, :, 3:7],
            body_vel    =robot_rb_state[:, :, 7:10],
            body_ang_vel=robot_rb_state[:, :, 10:13],
        )
        self.goal_buf[env_ids]=self.compute_goal_obs(
            root_pos    =robot_rb_state[:, 0, 0:3],
            root_rot    =robot_rb_state[:, 0, 3:7],
            goal_pos    =self.goal_trans[self.task_rootid[env_ids]],
            goal_rot    =self.goal_rot[self.task_rootid[env_ids]],
            obj_rot     =object_state[:, 3:7],
        )
        self.guide_buf[env_ids]=self.compute_guide_obs(
            root_pos    =robot_rb_state[:, 0, 0:3],
            root_rot    =robot_rb_state[:, 0, 3:7],
            guide_pos   =self.movenow_guide[env_ids]
        )
    
    def compute_termination(self): #finish modified
        if self.enable_early_termination:
            force       = self._contact_force_state[self.robot2rb].view(self.num_envs, self.num_body, 3)
            fall_force  = torch.any(torch.abs(force) > 0.1, dim = -1)
            height      = self._rigid_body_state[self.robot2rb].view(self.num_envs, self.num_body, 13)[:,:,2]
            fall        = torch.logical_and(fall_force, height < self.fall_thresh)
            fall[:, self.ignore_contact_idx] = False
            fall = torch.any(fall, dim= -1)
            
            # 添加根部高度检查 - 如果机器人根部过低说明倒地了
            root_height = self._rigid_body_state[self.robot2rb].view(self.num_envs, self.num_body, 13)[:, 0, 2]
            root_fall = root_height < (self.fall_thresh + 0.3)  # 根部应该保持在合理高度
            
            terminate = torch.logical_or(fall, root_fall)
        else:
            terminate = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        terminate[self.progress_buf <= 2] = False
        timeout = self.progress_buf >= self.timeout_limit
        
        #判断任务是否提前完成
        root_pos = self._root_states[self.robot2root, 0:3]
        root_pos = root_pos[0] #打补丁
        root_rot = self._root_states[self.robot2root, 3:7]
        goal_trans = self.goal_trans[self.task_rootid]
        pos_diff = self.calc_diff_pos(root_pos, goal_trans)    
        movenow_success = pos_diff < (self.eval_success_thresh * 0.5)   
        self.consecutive_success[~movenow_success] = 0
        self.consecutive_success[movenow_success] += 1

        if self.enable_early_termination:
            timeout = torch.logical_or(timeout, self.consecutive_success > self.consecutive_success_thresh)

        
        self.reset_timeout_buf[:] = timeout[:].float()
        self.reset_termination_buf[:] = terminate[:].float()

    def compute_reward(self):
        robot_rb_state = self._rigid_body_state[self.robot2rb].view(self.num_envs, self.num_body, 13)
        #root_pos = self._root_states[self.robot2root, 0:3]
        #root_pos = root_pos[0] #打补丁
        root_rot = self._root_states[self.robot2root, 3:7]
        object_state = self._root_states[self.task_rootid] #todo：object state是什么
        object_pos = object_state[:,0:3]
        object_rot = object_state[:,3:7]

        goal_pos = self.goal_trans[self.task_rootid]
        goal_rot = self.goal_rot[self.task_rootid]   

        # robot2guide_dir = self.movenow_guide[:, :2] - robot_rb_state[:, 0, :2]
        # robot2guide_dist = torch.norm(robot2guide_dir, dim=-1)
        robot2object_dir = object_pos[:,:2] - robot_rb_state[:, 0, :2]
        robot2object_dist = torch.norm(robot2object_dir, dim=-1)

        geom_object_pcd = self.asset_pcd[self.object2asset[self.task_objectid]] * self.object2scale[self.task_objectid].unsqueeze(-1).unsqueeze(-1)
        global_object_pcd = torch_utils.transform_pcd(geom_object_pcd, pos = object_pos, rot = object_rot)
        # init_object_pcd = torch_utils.transform_pcd(geom_object_pcd, pos = init_pos, rot = init_rot)
        # init_z_pt = init_object_pcd[..., -1].min(1)[0]
        # goal_object_pcd = torch_utils.transform_pcd(geom_object_pcd, pos = goal_pos, rot = goal_rot)
        # goal_z_pt = goal_object_pcd[..., -1].min(1)[0]
        # object_z_pt = global_object_pcd[...,-1].min(1)[0]
        
        root_pos = robot_rb_state[:, 0, :3]
        #print(root_pos.shape, global_object_pcd.shape)
        root2object_dist = root_pos.unsqueeze(1) - global_object_pcd
        root2object_dist = torch.norm(root2object_dist, p=2, dim = -1)
        root2object_dist = torch.min(root2object_dist, dim=-1)[0]

        thresh_robot2object = 0.5

        ######## approach object
        robot_vel_xy = robot_rb_state[:, 0, 7:9]
        robot2target_dir = robot2object_dir #torch_utils.normalize(torch.where(self.movenow_preguidecomplete.unsqueeze(-1).tile(2), robot2object_dir, robot2guide_dir))
        robot2target_speed = 1.5
        reward_robot2object_vel = torch.exp(-2 * torch.square(robot2target_speed - torch.sum(robot2target_dir * robot_vel_xy, dim=-1)))
        reward_robot2object_pos = torch.exp(-0.5 * robot2object_dist)

        reward_robot2object_vel[robot2object_dist < thresh_robot2object] = 1
        reward_robot2object_pos[robot2object_dist < thresh_robot2object] = 1

        ######## sit down
        reward_root2object = torch.exp(- 5 *  root2object_dist)#.mean(-1)
        reward_root2object[robot2object_dist > 0.5] = 0.
        
        ######## 添加稳定性奖励
        # 鼓励机器人保持直立
        root_height = robot_rb_state[:, 0, 2]
        reward_height = torch.exp(-5.0 * torch.clamp(0.8 - root_height, 0.0, float('inf')))
        
        # 鼓励机器人保持平衡（根部姿态接近竖直）
        root_up = torch_utils.quat_rotate(root_rot, torch.tensor([0.0, 0.0, 1.0], device=root_rot.device).unsqueeze(0).repeat(root_rot.shape[0], 1))
        up_reward = torch.clamp(root_up[:, 2], 0.0, 1.0) ** 2
        
        # 惩罚过大的角速度
        root_ang_vel = robot_rb_state[:, 0, 10:13]
        ang_vel_penalty = torch.exp(-0.5 * torch.sum(root_ang_vel ** 2, dim=-1))
        
        # reward_vel = 1 - (torch.clamp(torch.norm(object_vel, dim=-1),0,1) / 1 - 1) ** 2
        # reward_vel[reward_height_pt > 0.3] = 1.

        reward_items = [
            [0.25,  reward_robot2object_vel],
            [0.20,  reward_robot2object_pos], 
            [0.20,  reward_root2object],
            [0.15,  reward_height],
            [0.10,  up_reward],
            [0.10,  ang_vel_penalty],
        ]

        reward = sum([a * b for a,b in reward_items])
        self.reward_buf[:] = reward

        self.stats_step['robot2object_dist'] = robot2object_dist
        self.stats_step['reward_robot2object_vel'] = reward_robot2object_vel
        self.stats_step['reward_robot2object_pos'] = reward_robot2object_pos
        self.stats_step['reward_root2object'] = reward_root2object
        self.stats_step['reward_height'] = reward_height
        self.stats_step['up_reward'] = up_reward
        self.stats_step['ang_vel_penalty'] = ang_vel_penalty


        # location_reward = self.compute_location_reward(root_pos, self.prev_root_pos, root_rot, self.prev_root_rot, object_pos, goal_pos, 1.5, self.dt,
        #                                     self.sit_vel_penalty, self.sit_vel_pen_coeff, self.sit_vel_pen_thre, self.sit_ang_vel_pen_coeff, self.sit_ang_vel_pen_thre)

        # power = torch.abs(torch.multiply(self.dof_force_tensor, self._robot_dof_vel)).sum(dim = -1)
        # power_reward = -self._power_coefficient * power

        # if self._power_reward:
        #     self.reward_buf[:] = location_reward + power_reward
        # else:
        #     self.reward_buf[:] = location_reward

    def export_stats(self):
        stats = {}
        stats['env/progress'] = self.progress_buf.float().mean().item()
        stats['env/termination'] = self.reset_termination_buf.float().mean().item()
        stats['env/success'] = self.last_success.float().mean().item()
        for k,v in self.stats_step.items():
            stats[f'env/{k}'] = v.mean().item()
        return stats
    
    def export_logging_stats(self):
        stats = {}
        stats['progress'] = self.progress_buf.float().mean().item()
        stats['success'] = self.last_success.float().mean().item()
        return stats
    
    def export_evaluation(self,):
        object_state = self._root_states[self.task_rootid]
        object_pos = object_state[:,0:3]
        goal_pos = self.goal_trans[self.task_rootid]
        dist = torch.norm(goal_pos - object_pos, p=2, dim=-1)
        result = {
            'l2_dist' : dist.mean().item(),
            'success' : (dist < self.eval_success_thresh).float().mean().item(),
            'success_steps' : self.success_steps.float().mean().item() * self.dt
        }
        # result = {
        #     'l2_dist' : dist,
        #     'success' : (dist < self.eval_success_thresh).float(),
        #     'success_steps' : self.success_steps.float() * self.dt
        # }
        return result
    

    ######################### prepare feats
    def dof_to_obs(self, dof_pos):
        dof_obs = []
        for offset in self.revolute_dof_offset: # 1D dof
            angle = dof_pos[:, offset]
            axis = torch.tensor([0.0, 1.0, 0.0], dtype=angle.dtype, device=self.device)
            q = torch_utils.quat_from_angle_axis(angle, axis)
            obs  = torch_utils.quat_to_tan_norm(q) # 归一化，防止pai和-pai不一致
            dof_obs.append(obs)
        for offset in self.sphere_dof_offset: # 3D dof
            angle = dof_pos[:,offset : offset + 3]
            q = torch_utils.exp_map_to_quat(angle)
            obs = torch_utils.quat_to_tan_norm(q)
            dof_obs.append(obs)
        dof_obs = torch.cat(dof_obs,1) # 在1维上拼接成为 (bz, ndof*2)
        return dof_obs
    
    def compute_ref_frame_obs( #把环境中的物理状态转换成一个以机器人朝向为参考系的观测向量 obs
            self, root_pos, root_rot, root_vel, root_anv, dof_pos, dof_vel, key_body_pos, obj_pos):
        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

        root_rot_obs = torch_utils.quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
        
        root_h_obs = root_h
        
        local_root_vel = torch_utils.quat_rotate(heading_rot, root_vel)
        local_root_anv = torch_utils.quat_rotate(heading_rot, root_anv)

        bz, nkb, _ = key_body_pos.shape
        root_pos_expand = root_pos.unsqueeze(-2)
        local_key_body_pos = key_body_pos - root_pos_expand
        heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, nkb, 1)).view(bz * nkb, 4)
        local_key_body_pos = local_key_body_pos.view(bz * nkb, 3)
        local_key_body_pos = torch_utils.quat_rotate(heading_rot_expand, local_key_body_pos)
        local_key_body_pos = local_key_body_pos.view(bz, nkb * 3)
        
        dof_obs = self.dof_to_obs(dof_pos)

        local_obj_pos = torch_utils.quat_rotate(heading_rot, obj_pos - root_pos)
        local_obj_pos[:,2] = 0.
        obs = torch.cat(( #todo：我们需要什么obs
            root_h_obs, 
            root_rot_obs, 
            local_root_vel, 
            local_root_anv, 
            dof_obs, 
            dof_vel, 
            local_key_body_pos,
            local_obj_pos), dim=-1)
        
        return obs
    
    def compute_guide_obs(self, root_pos, root_rot, guide_pos):
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        local_guide_pos = torch_utils.quat_rotate(heading_rot, guide_pos - root_pos)
        return local_guide_pos
    
    def compute_goal_obs(self, root_pos, root_rot, goal_pos, goal_rot, obj_rot):
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        local_goal_pos = torch_utils.quat_rotate(heading_rot, goal_pos - root_pos)
        local_goal_rot = torch_utils.quat_mul(heading_rot, goal_rot)
        local_goal_rot_tannorm = torch_utils.quat_to_tan_norm(local_goal_rot)
        
        # 添加椅子朝向观测 - 椅子正面朝向+Y方向
        # 基于数据分析，椅子前向向量是[0, 1, 0]
        chair_forward_vec = torch.tensor([0.0, 1.0, 0.0], device=obj_rot.device, dtype=obj_rot.dtype)
        chair_forward_vec = chair_forward_vec.unsqueeze(0).expand(obj_rot.shape[0], -1)
        chair_forward_global = torch_utils.quat_rotate(obj_rot, chair_forward_vec)
        chair_forward_local = torch_utils.quat_rotate(heading_rot, chair_forward_global)
        chair_facing_2d = torch.nn.functional.normalize(chair_forward_local[:, :2], dim=-1)
        
        obs = torch.cat((
            local_goal_pos,
            local_goal_rot_tannorm,
            chair_facing_2d,  # 整合椅子朝向的2D方向向量
            ),    
        dim = -1)
        return obs
        
    def compute_prop_obs(self, body_pos, body_rot, body_vel, body_ang_vel):
        bz, nbody, _ = body_pos.shape
        assert nbody == self.num_body
        
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        root_h = root_pos[:, 2:3]
        root_h_obs = root_h
        
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        flat_heading_rot = heading_rot.unsqueeze(-2).repeat((1, nbody, 1)).reshape(bz * nbody, 4)
        
        local_body_pos = body_pos - root_pos.unsqueeze(-2)
        local_body_pos = local_body_pos.reshape(bz * nbody, 3)
        local_body_pos = torch_utils.quat_rotate(flat_heading_rot, local_body_pos)
        local_body_pos = local_body_pos.reshape(bz, nbody * 3)
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        local_body_rot = body_rot.reshape(bz * nbody, 4)
        local_body_rot = torch_utils.quat_mul(flat_heading_rot, local_body_rot)
        local_body_rot_tannorm = torch_utils.quat_to_tan_norm(local_body_rot)
        local_body_rot_tannorm = local_body_rot_tannorm.reshape(bz, nbody * 6)
        
        local_body_vel = body_vel.reshape(bz * nbody,3)
        local_body_vel = torch_utils.quat_rotate(flat_heading_rot, local_body_vel)
        local_body_vel = local_body_vel.reshape(bz, nbody * 3)
        
        local_body_anv = body_ang_vel.reshape(bz * nbody,3)
        local_body_anv = torch_utils.quat_rotate(flat_heading_rot, local_body_anv)
        local_body_anv = local_body_anv.reshape(bz, nbody * 3)


        obs = torch.cat((
            root_h_obs, 
            local_body_pos, 
            local_body_rot_tannorm, 
            local_body_vel, 
            local_body_anv,
            ),    
        dim = -1)
        return obs
    
    ####################### added from HITR_carry
    def fetch_amp_demo(self,n):
        assert self.motion is not None
        motion_ids = self.sample_motion_ids(n)

        motion_times = self.sample_motion_time(motion_ids = motion_ids)
        motion_times = motion_times.unsqueeze(1).tile(self.num_ref_obs_frames)
        motion_times -= torch.arange(self.num_ref_obs_frames).to(motion_times.device) * self.dt # 根据 num_ref_obs_frames（比如说 4 帧），往前/往后取一段动作片段
        motion_times = motion_times.flatten()
        motion_ids = torch.repeat_interleave(motion_ids, self.num_ref_obs_frames,)
        
        state_info = self.query_motion_state(motion_ids, motion_times)
        amp_demo_obs = self.compute_ref_frame_obs(
            root_pos=state_info['rigid_body_pos'][:,0,:],
            root_rot=state_info['rigid_body_rot'][:,0,:],
            root_vel=state_info['rigid_body_vel'][:,0,:],
            root_anv=state_info['rigid_body_anv'][:,0,:],
            dof_pos=state_info['dof_pos'],
            dof_vel=state_info['dof_vel'],
            key_body_pos=state_info['rigid_body_pos'][:,self.amp_body_idx,:],
            obj_pos=state_info['object_pos'],
        )
        amp_demo_obs = amp_demo_obs.reshape((n, self.num_ref_obs_frames * self.num_ref_obs_per_frame))
        return amp_demo_obs

    ######################## added from tokenhsi
    
    #@torch.jit.script #编译成 TorchScript，从而在 PyTorch 运行时之外加速或导出模型
    def compute_location_reward(self, root_pos, prev_root_pos, root_rot, prev_root_rot, object_root_pos, tar_pos, tar_speed, dt,
                                sit_vel_penalty, sit_vel_pen_coeff, sit_vel_penalty_thre, sit_ang_vel_pen_coeff, sit_ang_vel_penalty_thre):
        # type: (torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, float, float, bool, float, float, float, float) -> torch.tensor
        # tar_speed: desired speed toward the target
        # sit_vel_penalty: whether to enable the velocity penalty when close to the object

        # when humanoid is far from the object
        # 靠近奖励
        d_obj2human_xy = torch.sum((object_root_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1) #机器人和物体在 XY平面 的平方距离
        reward_far_pos = torch.exp(-0.5 * d_obj2human_xy)

        # 方向奖励 - 引导到椅子正面位置，准备背对椅子坐下
        delta_root_pos = root_pos - prev_root_pos
        root_vel = delta_root_pos / dt
        
        # 获取椅子状态用于计算朝向
        object_state = self._root_states[self.task_rootid]
        obj_rot = object_state[:, 3:7]
        
        # 计算椅子正面位置
        chair_forward_vec = torch.tensor([0.0, 1.0, 0.0], device=obj_rot.device, dtype=obj_rot.dtype)
        chair_forward_vec = chair_forward_vec.unsqueeze(0).expand(obj_rot.shape[0], -1)  # [num_envs, 3]
        chair_forward = torch_utils.quat_rotate(obj_rot, chair_forward_vec)
        approach_distance = 0.5
        front_target_pos = object_root_pos - chair_forward * approach_distance
        
        # 计算朝向椅子正面位置的方向
        tar_dir = front_target_pos[..., 0:2] - root_pos[..., 0:2]
        tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
        tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
        tar_vel_err = tar_speed - tar_dir_speed
        reward_far_vel = torch.exp(-2.0 * tar_vel_err * tar_vel_err)

        reward_far_final = 0.0 * reward_far_pos + 1.0 * reward_far_vel
        dist_mask = (d_obj2human_xy <= 0.5 ** 2)
        reward_far_final[dist_mask] = 1.0 #如果机器人离物体非常近（0.5 米以内），奖励直接设为 1

        # when humanoid is close to the object
        # 靠近奖励 - 奖励接近椅子正面位置（准备坐下的位置）
        reward_near = torch.exp(-10.0 * torch.sum((front_target_pos - root_pos) ** 2, dim=-1))
        
        # 朝向奖励 - 奖励机器人背对椅子（与椅子朝向相反）
        robot_forward_vec = torch.tensor([1.0, 0.0, 0.0], device=root_rot.device, dtype=root_rot.dtype)
        robot_forward_vec = robot_forward_vec.unsqueeze(0).expand(root_rot.shape[0], -1)  # [num_envs, 3]
        robot_forward = torch_utils.quat_rotate(root_rot, robot_forward_vec)
        # 机器人应该背对椅子，所以机器人朝向应该与椅子朝向相反
        chair_back_dir = -chair_forward  # 椅子的背面方向
        facing_reward = torch.sum(robot_forward * chair_back_dir, dim=-1)
        facing_reward = torch.clamp(facing_reward, 0.0, 1.0)  # 只奖励正向朝向

        reward = 0.5 * reward_near + 0.3 * reward_far_final + 0.2 * facing_reward

        if sit_vel_penalty:
            # 平移速度惩罚
            min_speed_penalty = sit_vel_penalty_thre
            root_vel_norm = torch.norm(root_vel, p=2, dim=-1)
            root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
            root_vel_err = min_speed_penalty - root_vel_norm
            root_vel_penalty = -1 * sit_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))
            dist_mask = (d_obj2human_xy <= 1.5 ** 2)
            root_vel_penalty[~dist_mask] = 0.0
            reward += root_vel_penalty
            
            # Z轴角速度惩罚
            root_z_ang_vel = torch.abs((self.get_euler_xyz(root_rot)[2] - self.get_euler_xyz(prev_root_rot)[2]) / dt)
            root_z_ang_vel = torch.clamp_min(root_z_ang_vel, sit_ang_vel_penalty_thre)
            root_z_ang_vel_err = sit_ang_vel_penalty_thre - root_z_ang_vel
            root_z_ang_vel_penalty = -1 * sit_ang_vel_pen_coeff * (1 - torch.exp(-0.5 * (root_z_ang_vel_err ** 2)))
            root_z_ang_vel_penalty[~dist_mask] = 0.0
            reward += root_z_ang_vel_penalty
        
        self.stats_step['robot2object_dist'] = d_obj2human_xy

        return reward
    
    #@torch.jit.script
    def get_euler_xyz(self, q):
        qx, qy, qz, qw = 0, 1, 2, 3
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
        cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
            q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
        pitch = torch.where(torch.abs(sinp) >= 1, self.copysign(
            np.pi / 2.0, sinp), torch.asin(sinp))

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
        cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
            q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)
    
    #@torch.jit.script
    def copysign(self, a, b):
        # type: (float, torch.tensor) -> torch.tensor
        a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
        return torch.abs(a) * torch.sign(b)