# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Standard MLP: Input -> 256 -> 256 -> Output (Q-values for each action)
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device

        # Double DQN requires two networks: Policy (online) and Target
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net is never trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=100000)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state, epsilon):
        # Epsilon-Greedy Strategy
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0.0 # Not enough data to train yet

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Double DQN Logic ---
        # 1. Get Q(s, a) from Policy Net
        current_q = self.policy_net(states).gather(1, actions)

        # 2. Get max action argmax_a' Q_policy(s', a')
        next_actions = self.policy_net(next_states).argmax(1, keepdim=True)

        # 3. Get value Q_target(s', next_action)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # 4. Compute Loss and Backpropagate
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient Clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())