# File: sarsaa.py

import numpy as np
from Q_Learning import QLearningAgent  # <-- IMPORTANT IMPORT


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