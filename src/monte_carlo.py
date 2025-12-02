import numpy as np
from Q_Learning import QLearningAgent


class MonteCarloAgent(QLearningAgent):
    """
    Monte Carlo Agent
    - Updates only at the end of the episode.
    - Uses Constant-Alpha update rule (compatible with your config).
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # Initialize parent (QLearningAgent) to get q_table, select_action, etc.
        super().__init__(n_states, n_actions, alpha, gamma, epsilon, epsilon_decay, epsilon_min)
        self.name = "Monte Carlo"
        self.episode_memory = []  # Stores (state, action, reward)

    def update(self, state, action, reward, next_state, done):
        """
        In main.py, this is called every step.
        For MC, we just store the data. We only 'learn' when done=True.
        """
        # 1. Record the step
        self.episode_memory.append((state, action, reward))

        # 2. Only update the Q-table if the episode is finished
        if done:
            G = 0
            visited_sa = set()

            # Iterate backwards through the episode to calculate Returns (G)
            for s, a, r in reversed(self.episode_memory):
                G = self.gamma * G + r

                # First-Visit Monte Carlo Logic
                if (s, a) not in visited_sa:
                    visited_sa.add((s, a))

                    # Update rule: Q(S,A) = Q(S,A) + alpha * (G - Q(S,A))
                    # This is "Constant-Alpha MC", which fits your alpha config perfectly.
                    current_q = self.q_table[s][a]
                    self.q_table[s][a] += self.alpha * (G - current_q)

            # 3. Clear memory for the next episode
            self.episode_memory = []