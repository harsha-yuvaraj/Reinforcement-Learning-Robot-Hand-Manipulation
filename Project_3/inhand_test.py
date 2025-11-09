# inhand_test.py (Student Skeleton)
import time
from inhand_env import CanRotateEnv
from hand_movement_test import get_object_z_rotation, print_object_status
import mujoco  
import numpy as np 
from scipy.spatial.transform import Rotation as R

# --- TODO: Import your agent class ---
# from agent import MyRLAgent 

# --- Configuration ---
MODEL_PATH = "my_agent_final.pth" # Path to your saved student model
EPISODES_TO_RUN = 20

# --- TODO: Load the environment ---
env = CanRotateEnv(render_mode="human")

# --- TODO: Load your trained agent ---
# agent = MyRLAgent(
#     obs_space_shape=env.observation_space.shape,
#     action_space_shape=env.action_space.shape,
#     device='cpu'
# )
# try:
#     agent.load_model(MODEL_PATH)
#     print(f"Successfully loaded model from {MODEL_PATH}")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()


# --- Run the evaluation ---
for episode in range(EPISODES_TO_RUN):
    print(f"--- Starting Episode {episode + 1}/{EPISODES_TO_RUN} ---")
    
    # --- TODO: Reset the environment ---
    obs, info = env.reset()
    
    terminated = False
    truncated = False
    total_reward = 0
    step = 0
    
    while not (terminated or truncated):
        
        # --- TODO: Get a deterministic action from your agent ---
        # The 'deterministic=True' part is key for testing
        # action = agent.get_action(obs, deterministic=True)
        action = env.action_space.sample() # Placeholder: Replace with your agent's action
        
        # --- Print-out of the action taken at each time step ---
        print(f"  STEP {step} | ACTION: {np.array2string(action, precision=4, suppress_small=True, max_line_width=np.inf)}")

        # --- TODO: Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Print-out of the location and orientation ---
        print_object_status(env.sim, env.obj_body_id)
        
        total_reward += reward
        step += 1
        
        # Render/sleep is handled by the environment's step/render methods
        time.sleep(1/60) # Keep visualization smooth
        
    # print(f"Episode {episode + 1} finished. Total Reward: {total_reward:.2f}")
    print(f"Episode {episode + 1} finished.")

# Clean up
env.close()
print("\nEvaluation finished.")