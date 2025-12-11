import time
import numpy as np
from inhand_env import CanRotateEnv
from dqn_agent import DQNAgent 
from actions_helper import discrete_to_continuous_action

# --- Configuration ---
MODEL_PATH = "training_logs/best_agent.pth" 
EPISODES_TO_RUN = 100 

# STRICT BOUNDS (90 - 95 degrees)
LOWER_BOUND = 1.57  # 90 degrees (1.57 rad)
UPPER_BOUND = 1.66  # 95 degrees (1.66 rad)

env = CanRotateEnv(render_mode="human")
agent = DQNAgent(env.observation_space.shape[0], 4, device='cpu')

# Get the simulation timestep (dt) to calculate real time
# Standard MuJoCo dt is usually 0.002s, but we fetch it dynamically to be exact.
dt = env.sim.model.opt.timestep

try:
    agent.load_model(MODEL_PATH)
    print(f"Loaded {MODEL_PATH}")
except:
    print("Model not found. Please ensure 'best_agent.pth' exists.")
    exit()

success_count = 0
success_durations = [] # Store time (seconds) for successful episodes only

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
        # Exploitation (No random noise)
        action_idx = agent.get_action(obs, epsilon=0.0)
        cont_action = discrete_to_continuous_action(action_idx)
        obs, reward, terminated, truncated, info = env.step(cont_action)
        step += 1
        
        current_rot = abs(env.cumulative_rotation)
        final_rotation = current_rot 
        
        # Check success based on Strict Bounds
        if terminated and (LOWER_BOUND <= current_rot <= UPPER_BOUND):
            success = True

    # Report
    duration = step * dt # Convert steps to seconds
    
    if success:
        success_count += 1
        success_durations.append(duration) # Only track time for successes
        print(f"Episode {episode+1}: SUCCESS (Rotation: {final_rotation:.3f} rad | Time: {duration:.3f}s)")
    else:
        print(f"Episode {episode+1}: FAILED (Rotation: {final_rotation:.3f} rad | Time: {duration:.3f}s)")

# --- Final Results ---
avg_time = np.mean(success_durations) if success_durations else 0.0

print("Results")
print(f"Success Rate: {success_count} out of {EPISODES_TO_RUN} trials\n")
print(f"Average Time needed to rotate 90 degrees (Successes Only): {avg_time:.4f} seconds")

env.close()