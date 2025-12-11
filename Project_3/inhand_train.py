import os
import numpy as np
from inhand_env import CanRotateEnv
from dqn_agent import DQNAgent
from actions_helper import discrete_to_continuous_action

MAX_EPISODES = 600       
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99
TARGET_UPDATE_FREQ = 1000 
DEVICE = 'cpu' 

log_dir = "training_logs/"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "training_log.txt")

env = CanRotateEnv(render_mode="headless") 
agent = DQNAgent(env.observation_space.shape[0], 4, lr=LEARNING_RATE, gamma=0.99, device=DEVICE)

print(f"Training Started (Total {MAX_EPISODES} Episodes)...")

obs, _ = env.reset()
epsilon = EPSILON_START
episode_num = 0
global_step = 0

curr_episode_reward = 0
curr_episode_steps = 0
episode_losses = []
best_reward = -float('inf') 

while episode_num < MAX_EPISODES:
    
    # Action
    action_idx = agent.get_action(obs, epsilon)
    cont_action = discrete_to_continuous_action(action_idx)
    
    # Step
    next_obs, reward, terminated, truncated, info = env.step(cont_action)
    done = terminated or truncated
    
    # Store
    scaled_reward = reward * 0.1 
    agent.memory.push(obs, action_idx, scaled_reward, next_obs, done)
    
    # Train
    loss = agent.learn(BATCH_SIZE)
    if loss != 0: episode_losses.append(loss)
    
    # Update Target
    if global_step % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()
        
    obs = next_obs
    curr_episode_reward += reward
    global_step += 1
    curr_episode_steps += 1

    # Episode End
    if done:
        episode_num += 1

        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY

        avg_loss = np.mean(episode_losses) if episode_losses else 0
        
        is_success = "YES" if curr_episode_reward > 50 else "NO" 
        
        print(f"Ep {episode_num} | Steps: {curr_episode_steps} | Reward: {curr_episode_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f} | Success: {is_success}")
        
        with open(log_file_path, mode='a') as file:
            file.write(f"Ep {episode_num} | Steps: {curr_episode_steps} | Reward: {curr_episode_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {epsilon:.3f} | Success: {is_success}\n")
        
        if curr_episode_reward > best_reward:
            best_reward = curr_episode_reward
            agent.save_model(f"{log_dir}/best_agent.pth")
            print(f"\tNew Best Model Saved (Reward: {best_reward:.2f})")
            
        if episode_num % 100 == 0:
            agent.save_model(f"{log_dir}/model_ep{episode_num}.pth")
            
        obs, _ = env.reset()
        curr_episode_reward = 0
        curr_episode_steps = 0
        episode_losses = []

agent.save_model(f"{log_dir}/agent_final.pth")
env.close()
print("Training Complete.")