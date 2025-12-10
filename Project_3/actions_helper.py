import numpy as np

# --- The Function to Verify ---
# 1. ADJUST THESE VALUES based on what you see in the simulation window.
def discrete_to_continuous_action(action_idx):
    """
    Maps 4 discrete actions to the 16 continuous joint motors.
    Based on ur5e_leap.xml:
    - Indices 0-11: Three Fingers (Index, Middle, Ring)
    - Indices 12-15: Thumb
    """
    act = np.zeros(16)
    
    # "Magnitude": Step size per update. 
    # Start small (0.03). If movement is invisible, try 0.1.
    mag = 0.03 
    
    if action_idx == 0: # OPEN HAND
        # Negative values typically extend (open) the joints in this UR5e model
        act[:] = -mag 
        
    elif action_idx == 1: # GRASP / CLOSE
        # Positive values typically flex (close) the joints
        act[:] = mag 
        
    elif action_idx == 2: # ROTATE CLOCKWISE
        # THUMB (12-15) pushes against FINGERS (0-11)
        # Thumb moves Positive (Flex/Push In)
        act[12:16] = mag  
        
        # Fingers move Negative (Extend/Move Away) or opposing
        act[0:12]  = -mag 
        
    elif action_idx == 3: # ROTATE COUNTER-CLOCKWISE
        # The Inverse of CW
        act[12:16] = -mag
        act[0:12]  = mag
        
    return act