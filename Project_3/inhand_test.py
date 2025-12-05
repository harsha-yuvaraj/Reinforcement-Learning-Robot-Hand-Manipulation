import numpy as np
import time
import os
from inhand_env import CanRotateEnv
from q_utils import DiscreteTranslator

MODEL_PATH = "rl_results/q_table_best.npy"
TEST_EPISODES = 200
RENDER_MODE = "human"

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please change MODEL_PATH to rl_results/q_table_final.npy")
        return

    q_table = np.load(MODEL_PATH)
    env = CanRotateEnv(render_mode=RENDER_MODE)
    translator = DiscreteTranslator(num_bins=36)
    
    print(f"Loaded Q-Table {q_table.shape}. Starting tests...")

    success_count = 0
    time_taken_list = [] 
    
    for ep in range(TEST_EPISODES):
        obs, info = env.reset()
        last_action = 0 
        
        start_full_state = translator.get_state_index(obs, 0)
        start_bin = start_full_state // 4 
        
        terminated = False
        truncated = False
        steps = 0
        success = False
        
        start_time = time.time() 
        
        while not (terminated or truncated) and steps < 200:
            state_idx = translator.get_state_index(obs, last_action)
            action_idx = np.argmax(q_table[state_idx])
            
            macro_target = translator.get_continuous_action(action_idx)
            current_joints = obs[:16]
            action_delta = translator.calculate_delta_action(current_joints, macro_target)
            
            obs, reward, terminated, truncated, info = env.step(action_delta)
            
            last_action = action_idx
            steps += 1
            
            current_bin = translator.get_state_index(obs, last_action) // 4
            bin_diff = abs(current_bin - start_bin)
            if bin_diff > 18: bin_diff = 36 - bin_diff 
            
            if bin_diff >= 9: 
                success = True
                terminated = True
        
        end_time = time.time()
        
        if success:
            success_count += 1
            elapsed_time = end_time - start_time
            time_taken_list.append(elapsed_time) 
        
        if (ep+1) % 50 == 0:
            print(f"Tested {ep+1}/{TEST_EPISODES}...")

    env.close()

    print("\nEVALUATION RESULTS:")

    avg_time = np.mean(time_taken_list)
    print(f"Success Rate: {success_count}/{TEST_EPISODES} ({(success_count/TEST_EPISODES)*100:.1f}%)")
    print(f"Avg Time to 90°: {avg_time:.4f} seconds")

if __name__ == "__main__":
    evaluate()