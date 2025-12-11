import numpy as np

def discrete_to_continuous_action(action_idx):
    """
    Maps 4 discrete actions to the 16 continuous joint motors.
    Understanding:
    - Indices 0-11: Three Fingers (Index, Middle, Ring)
    - Indices 12-15: Thumb
    """
    act = np.zeros(16)
    mag = 0.05
    
    if action_idx == 0: # OPEN HAND
        act[:] = -mag 
        
    elif action_idx == 1: # GRASP / CLOSE
        act[:] = mag 
        
    elif action_idx == 2: # ROTATE CLOCKWISE
        act[12:16] = mag  
        act[0:12]  = -mag 
        
    elif action_idx == 3: # ROTATE COUNTER-CLOCKWISE
        act[12:16] = -mag
        act[0:12]  = mag
        
    return act