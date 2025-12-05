import numpy as np
import time
import os
import matplotlib.pyplot as plt
from inhand_env import CanRotateEnv
from q_utils import DiscreteTranslator

# CONFIGURATION
EPISODES = 2000          
MAX_STEPS = 150         
LEARNING_RATE = 0.15     
DISCOUNT = 0.97        
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998   

LOG_DIR = "rl_results"
os.makedirs(LOG_DIR, exist_ok=True)

def train():
    env = CanRotateEnv(render_mode="headless") 
    translator = DiscreteTranslator(num_bins=36)

    num_states = translator.num_bins * translator.num_actions
    q_table = np.zeros((num_states, 4))
    print(f"Initialized Q-Table: {q_table.shape}")

    epsilon = EPSILON_START
    reward_history = []
    best_avg_reward = -float('inf') 

    for episode in range(EPISODES):
        obs, info = env.reset()
        last_action = 0
        state_idx = translator.get_state_index(obs, last_action)
        
        total_reward = 0
        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated) and step < MAX_STEPS:
            if np.random.random() < epsilon:
                action_idx = np.random.randint(0, 4)
            else:
                action_idx = np.argmax(q_table[state_idx])

            macro_target = translator.get_continuous_action(action_idx)
            current_joints = obs[:16]
            action_delta = translator.calculate_delta_action(current_joints, macro_target)
            
            next_obs, reward, terminated, truncated, info = env.step(action_delta)
            next_state_idx = translator.get_state_index(next_obs, action_idx)
            
            # Bellman Update
            current_q = q_table[state_idx, action_idx]
            max_next_q = np.max(q_table[next_state_idx])
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_next_q - current_q)
            q_table[state_idx, action_idx] = new_q

            state_idx = next_state_idx
            last_action = action_idx
            total_reward += reward
            step += 1
            obs = next_obs

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        reward_history.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_rew = np.mean(reward_history[-50:])
            print(f"Ep {episode+1} | Avg: {avg_rew:.2f} | Epsilon: {epsilon:.2f}")
            
            if avg_rew > best_avg_reward and avg_rew > 30.0: 
                best_avg_reward = avg_rew
                np.save(os.path.join(LOG_DIR, "q_table_best.npy"), q_table)
                print(f"New Best Model Saved (Avg: {best_avg_reward:.2f})")

    np.save(os.path.join(LOG_DIR, "q_table_final.npy"), q_table)
    print("Training Complete.")
    
    plt.plot(reward_history)
    plt.savefig(os.path.join(LOG_DIR, "training_plot.png"))
    env.close()

if __name__ == "__main__":
    train()