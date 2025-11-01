import gymnasium as gym
import numpy as np
import time
from collections import defaultdict


# --- QLearningAgent class from before ---
class QLearningAgent:
    """
    Q-Learning: Off-Policy TD Control Algorithm
    Q(S, A) ← Q(S, A) + α[R + γ max_a Q(S', a) - Q(S, A)]
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.name = "Q-Learning (Off-Policy)"

    def select_action(self, state, greedy=False):
        """Select action using ε-greedy policy."""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """Update Q-value using Q-Learning rule (off-policy)."""
        current_q = self.q_table[state][action]
        if done:
            td_target = reward
        else:
            # Off-policy: use max Q-value (greedy action)
            max_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * max_next_q
        self.q_table[state][action] += self.alpha * (td_target - current_q)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# --- NEW: SARSA Agent Class ---
class SARSAAgent(QLearningAgent):
    """
    SARSA: On-Policy TD Control Algorithm
    Q(S, A) ← Q(S, A) + α[R + γ Q(S', A') - Q(S, A)]

    Inherits from QLearningAgent, just overrides the update method.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # Initialize the parent class
        super().__init__(n_states, n_actions, alpha, gamma,
                         epsilon, epsilon_decay, epsilon_min)
        self.name = "SARSA (On-Policy)"

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Update Q-value using SARSA rule (on-policy).
        Note the extra 'next_action' parameter!
        """
        current_q = self.q_table[state][action]

        if done:
            td_target = reward
        else:
            # On-policy: use Q-value of the actual next action (A')
            next_q = self.q_table[next_state][next_action]
            td_target = reward + self.gamma * next_q

        # SARSA update
        self.q_table[state][action] += self.alpha * (td_target - current_q)