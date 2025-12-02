import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


# 1. The Neural Network (The Brain)
class DQN(nn.Module):
    def __init__(self, n_states, n_actions, embedding_dim=64):
        super(DQN, self).__init__()
        # Embedding handles the discrete states (e.g., State 402)
        self.embedding = nn.Embedding(n_states, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, n_actions)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# 2. The Agent
class DQNAgent:
    def __init__(self, n_states, n_actions, alpha=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):

        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = "Deep Q-Network (DQN)"

        # Networks
        self.policy_net = DQN(n_states, n_actions).to(self.device)
        self.target_net = DQN(n_states, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.memory = deque(maxlen=2000)  # Replay Buffer

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.steps_done = 0

    def select_action(self, state, greedy=False):
        # Epsilon-Greedy Logic
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                # Convert state integer to tensor
                state_tensor = torch.tensor([state], device=self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()

    def update(self, state, action, reward, next_state, done):
        """
        Compatible with main.py API.
        Stores transition and attempts to learn.
        """
        # 1. Store memory
        self.memory.append((state, action, reward, next_state, done))

        # 2. Learn from batch (if we have enough data)
        self._learn()

    def _learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Randomly sample a batch
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # Convert to tensors
        state_batch = torch.tensor(state_batch, device=self.device)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device)
        next_state_batch = torch.tensor(next_state_batch, device=self.device)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float32)

        # Current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # Next Q values (from Target Net for stability)
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]

        # Target Q value
        target = reward_batch + (self.gamma * next_q * (1 - done_batch))

        # Compute Loss & Optimize
        loss = F.mse_loss(current_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        self.steps_done += 1
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)