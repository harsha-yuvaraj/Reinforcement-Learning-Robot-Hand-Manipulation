import numpy as np
from scipy.spatial.transform import Rotation as R

class DiscreteTranslator:
    def __init__(self, num_bins=36):
        self.num_bins = num_bins
        self.bin_step = 360 / num_bins
        self.num_actions = 4 
        
        # ACTION MACROS
        # 0: Open 
        self.pose_open = np.array([
            1.0, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 
            1.3, 1.0, 1.3, 1.0, 0.8, 1.3, 0.8, 0.5
        ])
        
        # 1: Grasp 
        self.pose_close = np.array([
            1.0, 2.0, 2.0, 1.0, 2.0, 0.0, 2.0, 0.0, 
            2.0, 1.0, 2.0, 1.0, 0.8, 2.0, 0.8, 0.5
        ])

        # 2: Spin Clockwise 
        self.pose_spin_cw = self.pose_close.copy()
        self.pose_spin_cw[0] += 0.35   
        self.pose_spin_cw[12] -= 0.35 

        # 3: Spin Counter-Clockwise 
        self.pose_spin_ccw = self.pose_close.copy()
        self.pose_spin_ccw[0] -= 0.35  
        self.pose_spin_ccw[12] += 0.35 

        self.action_lookup = {
            0: self.pose_open,
            1: self.pose_close,
            2: self.pose_spin_cw,
            3: self.pose_spin_ccw
        }

    def get_state_index(self, observation, last_action_idx=0):
        """
        Returns (Bin * 4) + Last_Action
        """
        # Extract Z-Rotation Bin
        quat_wxyz = observation[-4:]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        
        try:
            r = R.from_quat(quat_xyzw)
            z_rot = r.as_euler('xyz', degrees=True)[2]
        except:
            z_rot = 0.0

        if z_rot < 0: z_rot += 360
        bin_idx = int(z_rot / self.bin_step)
        bin_idx = max(0, min(bin_idx, self.num_bins - 1))
        
        if last_action_idx is None: last_action_idx = 0
            
        return (bin_idx * self.num_actions) + last_action_idx

    def get_continuous_action(self, discrete_action_idx):
        return self.action_lookup.get(discrete_action_idx, self.pose_open)

    def calculate_delta_action(self, current_joints, target_joints, max_step=0.1):
        diff = target_joints - current_joints
        return np.clip(diff, -max_step, max_step)