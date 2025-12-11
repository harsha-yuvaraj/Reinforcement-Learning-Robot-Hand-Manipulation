import os
import numpy as np
import mujoco
import mujoco.viewer as mjv
import gymnasium as gym
from gymnasium import spaces

from simulation import Simulation

MAX_EPISODE_STEPS = 300 

class CanRotateEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super(CanRotateEnv, self).__init__()
        
        # Initialize simulation
        self.sim = Simulation(
            scene_path=os.path.join(os.path.dirname(__file__), "scene.xml"),
            output_dir="rl_output"
        )
        self.sim.load()
        
        # IDs and Helpers
        self.obj_body_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "obj1")
        self.site_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.sim.ids_by_name(["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], mujoco.mjtObj.mjOBJ_JOINT, 'arm')
        self.sim.ids_by_name(["1", "0", "2", "3", "5", "4", "6", "7", "9", "8", "10", "11", "12", "13", "14", "15"], mujoco.mjtObj.mjOBJ_JOINT, 'hand')
        self.sim.actuators_for_joints('arm')
        self.sim.actuators_for_joints('hand')
        self.can_geom_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "obj1")
        self.fingertip_geom_ids = {
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip"),
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_2"),
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "thumb_fingertip"),
        }
        
        # Spaces
        self.action_space = spaces.Box(low=-0.03, high=0.03, shape=(16,), dtype=np.float32) 
        obs_size = len(self.sim.hand_joint_ids) + 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.render_mode = render_mode
        if render_mode == 'human':
            self.viewer = mjv.launch_passive(self.sim.model, self.sim.data) 
        else:
            self.viewer = None
            
        self.step_count = 0
        self.cumulative_rotation = 0.0

    def _get_obs(self):
        finger_qpos = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[j]] for j in self.sim.hand_joint_ids])
        obj_jnt_adr = self.sim.model.body_jntadr[self.obj_body_id]
        obj_qpos_adr = self.sim.model.jnt_qposadr[obj_jnt_adr]
        object_pose = self.sim.data.qpos[obj_qpos_adr : obj_qpos_adr + 7]
        return np.concatenate([finger_qpos, object_pose])

    def _calculate_reward(self):
        # 1. Get Velocity
        obj_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.sim.model, self.sim.data, mujoco.mjtObj.mjOBJ_BODY, self.obj_body_id, obj_vel, 0)
        
        # [vx, vy, vz, wx, wy, wz] -> Indices 3,4,5 are angular velocity
        angular_velocity_x = obj_vel[3]
        angular_velocity_y = obj_vel[4]
        angular_velocity_z = obj_vel[5] 

        # 2. Track Cumulative Rotation (Z-Axis)
        dt = self.sim.model.opt.timestep 
        self.cumulative_rotation += angular_velocity_z * dt
        abs_rot = abs(self.cumulative_rotation)

        # 3. Base Rewards
        # Stronger Penalty on X/Y to prevent "Flipping"
        rotation_reward = (angular_velocity_z * 5.0) - (abs(angular_velocity_x) + abs(angular_velocity_y)) * 2.0
        
        # Survival (Keep close to palm)
        can_pos = self.sim.data.xpos[self.obj_body_id]
        palm_pos = self.sim.data.site_xpos[self.site_id]
        distance_from_palm = np.linalg.norm(can_pos - palm_pos)
        survival_reward = 0.1 - distance_from_palm

        # Contact Bonus
        contact_reward = 0.0
        fingers_in_contact = set()
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            if (geom1 in self.fingertip_geom_ids and geom2 == self.can_geom_id): fingers_in_contact.add(geom1)
            elif (geom2 in self.fingertip_geom_ids and geom1 == self.can_geom_id): fingers_in_contact.add(geom2)
        
        if len(fingers_in_contact) >= 3: contact_reward = 1.0
        elif len(fingers_in_contact) > 0: contact_reward = 0.2 * len(fingers_in_contact)
        
        # 4. STRICT BOUNDED SUCCESS REWARD (90 - 95 degrees)
        # 1.57 rad = 90 deg | 1.66 rad = 95 deg
        success_bonus = 0.0
        
        if 1.57 <= abs_rot <= 1.66:
            success_bonus = 100.0 # Jackpot
        elif abs_rot > 1.66:
            success_bonus = -50.0 # OVERSHOOT PENALTY (Went past 95 degrees)

        return rotation_reward + survival_reward + contact_reward + success_bonus

    def _is_terminated(self):
        can_z_pos = self.sim.data.xpos[self.obj_body_id][2] 
        palm_z_pos = self.sim.data.site_xpos[self.site_id][2] 
        
        # 1. Dropped Check
        dropped = can_z_pos < (palm_z_pos - 0.05)
        
        # 2. Strict Success Check (Between 90 and 95)
        abs_rot = abs(self.cumulative_rotation)
        succeeded = (1.57 <= abs_rot <= 1.66)
        
        # 3. Fail Check (Overshoot > 95)
        failed_overshoot = abs_rot > 1.66

        return dropped or succeeded or failed_overshoot or self.step_count > MAX_EPISODE_STEPS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        mujoco.mj_resetData(self.sim.model, self.sim.data)

        # Reset Arm
        target_pos_up = np.array([0.4, 0.0, .5]) 
        target_euler_up = np.array([0, 0, 0]) 
        q_palm_up = self.sim.desired_qpos_from_ik(self.site_id, target_pos_up, target_euler_up) 
        self.sim.set_joint_positions(self.sim.arm_joint_ids, q_palm_up) 
        for i, act_id in enumerate(self.sim.arm_act_ids):
            self.sim.data.ctrl[act_id] = q_palm_up[i] 
        mujoco.mj_forward(self.sim.model, self.sim.data)

        # --- OBJECT RANDOMIZATION ---
        palm_surface_pos = self.sim.data.site_xpos[self.site_id].copy() 
        
        # Noise (+/- 2mm)
        noise_x = np.random.uniform(-0.002, 0.002) 
        noise_y = np.random.uniform(-0.002, 0.002)
        noise_z = np.random.uniform(0.0, 0.002)

        object_start_pos = palm_surface_pos + np.array([0.011 + noise_x, -0.03 + noise_y, 0.075 + noise_z]) 

        obj_jnt_adr = self.sim.model.body_jntadr[self.obj_body_id] 
        obj_qpos_adr = self.sim.model.jnt_qposadr[obj_jnt_adr] 
        self.sim.data.qpos[obj_qpos_adr : obj_qpos_adr + 3] = object_start_pos 
        self.sim.data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7] = [1, 0, 0, 0] 

        mujoco.mj_forward(self.sim.model, self.sim.data)

        # Settle
        for _ in range(20): mujoco.mj_step(self.sim.model, self.sim.data) 

        # Reset Hand
        q_open_angles = np.array([1.0, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.3, 1.0, 1.3, 1.0, 0.8, 1.3, 0.8, 0.5]) 
        self.sim.set_joint_positions(self.sim.hand_joint_ids, q_open_angles) 
        for i, act_id in enumerate(self.sim.hand_act_ids):
            self.sim.data.ctrl[act_id] = q_open_angles[i] 
        mujoco.mj_forward(self.sim.model, self.sim.data)

        self.cumulative_rotation = 0.0

        if self.render_mode != "headless":
            self.viewer.sync()
        
        return self._get_obs(), {}
        
    def step(self, action):
        target_angles = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[j]] for j in self.sim.hand_joint_ids]) + action
        self.sim.move_gripper_to_angles(target_angles, 0.5) 

        if self.render_mode != "headless":
            self.viewer.sync()

        self.step_count += 1
        return self._get_obs(), self._calculate_reward(), self._is_terminated(), self.step_count >= MAX_EPISODE_STEPS, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None: self.viewer = mujoco.viewer.launch(self.sim.model, self.sim.data)
            if self.viewer.is_running(): self.viewer.sync()
            else: self.close()
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None