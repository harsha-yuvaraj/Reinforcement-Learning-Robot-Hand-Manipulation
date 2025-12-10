import os
import csv
import numpy as np
import time
from inhand_env import CanRotateEnv
from dqn_agent import DQNAgent
from actions_helper import discrete_to_continuous_action

# --- Configuration ---
TOTAL_TIMESTEPS = 100000  # Increase this for full training (e.g. 1M)
LEARNING_RATE = 3e-4
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995
TARGET_UPDATE_FREQ = 1000  # Steps between target net updates
DEVICE = 'cpu' 

# Logging Setup
log_dir = "training_logs/"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "training_log.csv")

# Initialize CSV Header
with open(log_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Steps", "Total_Reward", "Avg_Loss", "Epsilon"])

# --- Initialization ---
env = CanRotateEnv(render_mode="headless") # Use "human" to debug movements
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=4,  # We defined 4 discrete actions above
    lr=LEARNING_RATE,
    gamma=GAMMA,
    device=DEVICE
)

print("Deep Q-Learning Training Started...")
obs, _ = env.reset()
global_step = 0
epsilon = EPSILON_START
episode_num = 0

# Variables for episode tracking
curr_episode_reward = 0
curr_episode_steps = 0
episode_losses = []

while global_step < TOTAL_TIMESTEPS:
    
    # 1. Select Action (Discrete)
    action_idx = agent.get_action(obs, epsilon)
    
    # 2. Translate to Continuous for Env
    cont_action = discrete_to_continuous_action(action_idx)
    
    # 3. Step Environment
    next_obs, reward, terminated, truncated, info = env.step(cont_action)
    done = terminated or truncated
    
    # 4. Clip Reward (Crucial for stability in RL)
    # The env gives big rewards (velocity * 10). Clipping helps gradients.
    clipped_reward = np.clip(reward, -10.0, 10.0)
    
    # 5. Store in Buffer
    agent.memory.push(obs, action_idx, clipped_reward, next_obs, done)
    
    # 6. Train Agent
    loss = agent.learn(BATCH_SIZE)
    if loss != 0:
        episode_losses.append(loss)
    
    # 7. Update Target Network
    if global_step % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()
        
    # Update state and counters
    obs = next_obs
    curr_episode_reward += reward
    global_step += 1
    curr_episode_steps += 1
    
    # Decay Epsilon
    if epsilon > EPSILON_END:
        epsilon *= EPSILON_DECAY

    # --- Episode End Handling ---
    if done:
        episode_num += 1
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        
        # Log to Console
        print(f"Ep {episode_num} | Steps: {curr_episode_steps} | Reward: {curr_episode_reward:.2f} | Loss: {avg_loss:.4f} | Eps: {epsilon:.3f}")
        
        # Log to CSV
        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode_num, curr_episode_steps, curr_episode_reward, avg_loss, epsilon])
        
        # Save Model periodically
        if episode_num % 50 == 0:
            agent.save_model(f"{log_dir}/model_ep{episode_num}.pth")
            
        # Reset for next episode
        obs, _ = env.reset()
        curr_episode_reward = 0
        curr_episode_steps = 0
        episode_losses = []

# Final Save
agent.save_model("agent_final.pth")
env.close()
print("Training done.")