import gymnasium as gym
import numpy as np
import time
from collections import defaultdict
from SARSAA import QLearningAgent, SARSAAgent

class QLearningAgent:
    """
    Q-Learning: Off-Policy TD Control Algorithm

    Update Rule:
    Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """Initialize Q-Learning agent."""
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Use defaultdict for a cleaner Q-table
        # It automatically creates an entry with np.zeros(n_actions)
        # when a new state is accessed.
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.name = "Q-Learning (Off-Policy)"

    def select_action(self, state, greedy=False):
        """Select action using ε-greedy policy."""
        if not greedy and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)

        # Exploit: best action based on current Q-values
        # (q_table[state] will be created if it doesn't exist)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """Update Q-value using Q-Learning rule (off-policy)."""

        current_q = self.q_table[state][action]

        if done:
            td_target = reward
        else:
            # Off-policy: use max Q-value (greedy action)
            # (q_table[next_state] will be created if it doesn't exist)
            max_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * max_next_q

        # Q-Learning update
        self.q_table[state][action] += self.alpha * (td_target - current_q)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(env, agent, n_episodes=2000, max_steps=200):
    """
    Train an RL agent in the Taxi-v3 environment.
    This function now handles both Q-Learning and SARSA.
    """
    metrics = {
        'episode_rewards': [],
        'episode_steps': [],
    }
    start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"Training: {agent.name}")
    print(f"Episodes: {n_episodes} | Max Steps: {max_steps}")
    print(f"{'=' * 70}")

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        truncated = False

        # --- SARSA specific: select initial action ---
        if isinstance(agent, SARSAAgent):
            action = agent.select_action(state)

        while not (done or truncated) and episode_steps < max_steps:

            # --- Q-Learning: selects action inside the loop ---
            if isinstance(agent, QLearningAgent) and not isinstance(agent, SARSAAgent):
                action = agent.select_action(state)

            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1

            # --- Update agent based on its type ---
            if isinstance(agent, SARSAAgent):
                # SARSA: need next action from policy for the update
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
                action = next_action  # Use next_action in next iteration
            else:
                # Q-Learning: update immediately
                agent.update(state, action, reward, next_state, done)

            state = next_state

        # End of episode
        agent.decay_epsilon()
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_steps'].append(episode_steps)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-100:])
            avg_steps = np.mean(metrics['episode_steps'][-100:])
            print(f"Episode {episode + 1:4d}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Steps: {avg_steps:5.1f} | "
                  f"ε: {agent.epsilon:.3f}")

    training_time = time.time() - start_time
    print(f"{'=' * 70}")
    print(f"Training Complete! Total time: {training_time:.2f}s")
    print(f"{'=' * 70}\n")

    return metrics

def evaluate_agent(env, agent, n_episodes=100):
    """Evaluate trained agent using greedy policy."""
    total_rewards = []
    total_steps = []
    successes = 0

    print(f"\n{'=' * 70}")
    print(f"EVALUATION: {agent.name}")
    print(f"{'=' * 70}")

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        truncated = False

        while not (done or truncated) and steps < 200:
            # Use greedy policy (no exploration)
            action = agent.select_action(state, greedy=True)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

        total_rewards.append(episode_reward)
        total_steps.append(steps)
        if done:
            successes += 1

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = successes / n_episodes

    print(f"Episodes: {n_episodes}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Success Rate: {success_rate * 100:.1f}%")
    print(f"{'=' * 70}\n")

    return {
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'success_rate': success_rate
    }


def watch_agent(agent):
    """Run one episode to watch the trained agent play."""

    print(f"\n--- 🍿 WATCHING: {agent.name} ---")

    # Create a new environment in 'human' mode
    # This will open a new window to show the game
    env = gym.make('Taxi-v3', render_mode='human')

    state, _ = env.reset()
    done, truncated = False, False

    while not (done or truncated):
        # Select action greedily (no exploration)
        action = agent.select_action(state, greedy=True)

        # Take the step
        state, reward, done, truncated, _ = env.step(action)

        # Slow down the loop so we can see what's happening
        # (0.1 seconds between frames)
        time.sleep(0.1)

    env.close()
    print("Demo finished.")
def main():
    """
    Main execution function.
    Trains and evaluates Q-Learning and SARSA agents.
    """
    print("\n" + "=" * 70)
    print("REINFORCEMENT LEARNING: TAXI-V3 (Q-Learning vs. SARSA)")
    print("=" * 70)

    # 1. Create Taxi-v3 environment
    env = gym.make('Taxi-v3')

    # 2. Hyperparameter configuration
    config = {
        'n_states': 500,
        'n_actions': 6,
        'alpha': 0.1,  # Learning rate
        'gamma': 0.99,  # Discount factor
        'epsilon': 1.0,  # Initial exploration rate
        'epsilon_decay': 0.995,  # Epsilon decay rate
        'epsilon_min': 0.01,  # Minimum epsilon
        'n_episodes': 2000,  # Training episodes
        'eval_episodes': 100  # Evaluation episodes
    }

    # 3. Initialize agents
    agents = [
        QLearningAgent(
            config['n_states'], config['n_actions'],
            config['alpha'], config['gamma'],
            config['epsilon'], config['epsilon_decay'], config['epsilon_min']
        ),
        SARSAAgent(
            config['n_states'], config['n_actions'],
            config['alpha'], config['gamma'],  # <-- FIX 1: Corrected typo 'gamma'
            config['epsilon'], config['epsilon_decay'], config['epsilon_min']
        )
    ]

    # 4. Dictionaries to store results
    all_training_metrics = {}
    all_eval_metrics = {}

    # 5. --- CORRECTED LOOP ---
    # Loop through each agent to train and evaluate it
    for agent in agents:
        # Train
        train_metrics = train_agent(env, agent, n_episodes=config['n_episodes'])

        # Evaluate
        eval_metrics = evaluate_agent(env, agent, n_episodes=config['eval_episodes'])

        # --- FIX 2: These lines are NOW INSIDE the loop ---
        # Store the results for this agent
        all_training_metrics[agent.name] = train_metrics
        all_eval_metrics[agent.name] = eval_metrics

    # --- END OF LOOP ---

    # 6. Print summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Algorithm':<20} {'Avg Reward':<15} {'Success Rate':<15}")
    print("-" * 70)

    for name, metrics in all_eval_metrics.items():
        print(f"{name:<20} {metrics['avg_reward']:>6.2f}         "
              f"{metrics['success_rate'] * 100:>6.1f}%")
    print("=" * 70)


    # --- ADD THIS PART ---
    print("\n✅ Comparison run complete! Showing demos...")

    # 'agents' is the list of your trained agents
    # Watch the Q-Learning agent (the first one)
    watch_agent(agents[0])

    # Watch the SARSA agent (the second one)
    watch_agent(agents[1])
    # --- END OF ADDED PART ---

    print("\nAll tasks complete!")
    env.close()  # Now you can close the original 'headless' env
    print("\n✅ Comparison run complete!")
    env.close()


# This block ensures 'main()' is only called when you run the script directly
if __name__ == "__main__":
    main()