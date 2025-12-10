# inhand_test.py
import time
import numpy as np
from inhand_env import CanRotateEnv
from dqn_agent import DQNAgent 
from actions_helper import discrete_to_continuous_action

MODEL_PATH = "agent_final.pth" # Ensure this matches your save name
EPISODES_TO_RUN = 20

env = CanRotateEnv(render_mode="human") # Human mode to see the robot

# Initialize Agent
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=4,
    device='cpu' # Usually CPU is fine for inference
)

# Load Weights
try:
    agent.load_model(MODEL_PATH)
    print(f"Loaded {MODEL_PATH}")
except Exception as e:
    print(f"Error: {e}")
    exit()

for episode in range(EPISODES_TO_RUN):
    obs, info = env.reset()
    terminated = False
    truncated = False
    step = 0
    
    print(f"\nEpisode {episode+1}")
    while not (terminated or truncated):
        # 1. Get Best Action (Epsilon = 0 for pure exploitation)
        action_idx = agent.get_action(obs, epsilon=0.0)
        
        # 2. Translate
        cont_action = discrete_to_continuous_action(action_idx)
        
        # 3. Step
        obs, reward, terminated, truncated, info = env.step(cont_action)
        
        # Print action to debug
        print(f"\t Step {step} | Action: {action_idx}")
        step += 1
        time.sleep(0.02) # Slow down for visualization

env.close()