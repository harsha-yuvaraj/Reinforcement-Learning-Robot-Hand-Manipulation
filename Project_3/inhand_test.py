import time
import numpy as np
from inhand_env import CanRotateEnv
from dqn_agent import DQNAgent 
from actions_helper import discrete_to_continuous_action

MODEL_PATH = "training_logs/best_agent.pth" 
EPISODES_TO_RUN = 100 

# 90 degrees
TARGET_ROTATION = 1.57  

env = CanRotateEnv(render_mode="human")
agent = DQNAgent(env.observation_space.shape[0], 4, device='cpu')

# simulation timestep (dt) to calculate real time
dt = env.sim.model.opt.timestep

try:
    agent.load_model(MODEL_PATH)
    print(f"Loaded {MODEL_PATH}")
except:
    print("Model not found. Please ensure 'best_agent.pth' exists.")
    exit()

success_count = 0
success_durations = [] # store time (seconds) for successful episodes only

print(f"Simulation Timestep (dt): {dt:.4f} seconds")
print(f"Starting {EPISODES_TO_RUN} test runs...")

for episode in range(EPISODES_TO_RUN):
    obs, info = env.reset()
    terminated = False
    truncated = False
    step = 0
    
    final_rotation = 0.0
    success = False

    while not (terminated or truncated):
        # exploitation (No random noise, epsilon=0)
        action_idx = agent.get_action(obs, epsilon=0.0)
        cont_action = discrete_to_continuous_action(action_idx)
        obs, reward, terminated, truncated, info = env.step(cont_action)
        step += 1
        
        current_rot = abs(env.cumulative_rotation)
        final_rotation = current_rot 
        
        if current_rot >= TARGET_ROTATION:
            success = True
            break 


    duration = step * dt # convert steps to seconds
    
    if success:
        success_count += 1
        success_durations.append(duration) 
        print(f"Episode {episode+1}: SUCCESS")
    else:
        print(f"Episode {episode+1}: FAILED")

avg_time = np.mean(success_durations) if success_durations else 0.0

print("\nRESULTS SUMMARY")
print(f"Success Rate: {success_count} out of {EPISODES_TO_RUN} trials")
print(f"Average Time needed to rotate 90 degrees: {avg_time:.4f} seconds")

env.close()