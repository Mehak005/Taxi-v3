"""
Reinforcement Learning Agents: Taxi-v3 Case Study
Learning Optimal Behavior with Reinforcement Learning Agents

Authors: Daniel Truax (dtruax@albany.edu), Mehak Seth (mseth3@albany.edu)
Date: November 2025
Reference: https://lilianweng.github.io/posts/2018-02-19-rl-overview/

This implementation compares Q-Learning, SARSA, and Monte Carlo methods
in the OpenAI Gym Taxi-v3 environment.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import json
import time
from datetime import datetime


class QLearningAgent:
    """
    Q-Learning: Off-Policy TD Control Algorithm

    Update Rule:
    Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]

    Key characteristic: Learns optimal Q* independent of the policy being followed.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-Learning agent.

        Args:
            n_states: Number of states in environment
            n_actions: Number of possible actions
            alpha: Learning rate (step size)
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Multiplicative decay factor for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table as regular dict (pickle-friendly)
        self.q_table = {}
        self.name = "Q-Learning (Off-Policy)"

        # Tracking statistics
        self.training_errors = []

    def select_action(self, state, greedy=False):
        """
        Select action using ε-greedy policy.

        Args:
            state: Current state
            greedy: If True, select best action (no exploration)

        Returns:
            Selected action index
        """
        if not greedy and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)

        # Initialize state if not seen before
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)

        # Exploit: best action based on current Q-values
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-Learning rule (off-policy).

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated

        Returns:
            TD error (for tracking learning progress)
        """
        # Initialize states if not seen before
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)

        current_q = self.q_table[state][action]

        if done:
            td_target = reward
        else:
            # Off-policy: use max Q-value (greedy action)
            max_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * max_next_q

        # Temporal Difference Error
        td_error = td_target - current_q

        # Q-Learning update
        self.q_table[state][action] += self.alpha * td_error

        return td_error

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self, state):
        """Get the greedy policy for a state."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return np.argmax(self.q_table[state])


class SARSAAgent(QLearningAgent):
    """
    SARSA: On-Policy TD Control Algorithm

    Update Rule:
    Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

    Key characteristic: Learns Q-values for the policy being followed.
    More conservative than Q-Learning in stochastic environments.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(n_states, n_actions, alpha, gamma, epsilon, epsilon_decay, epsilon_min)
        self.name = "SARSA (On-Policy)"

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Update Q-value using SARSA rule (on-policy).

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (from policy)
            done: Whether episode terminated

        Returns:
            TD error
        """
        # Initialize states if not seen before
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)

        current_q = self.q_table[state][action]

        if done:
            td_target = reward
        else:
            # On-policy: use Q-value of actual next action (from ε-greedy policy)
            next_q = self.q_table[next_state][next_action]
            td_target = reward + self.gamma * next_q

        # Temporal Difference Error
        td_error = td_target - current_q

        # SARSA update
        self.q_table[state][action] += self.alpha * td_error

        return td_error


class MonteCarloAgent:
    """
    First-Visit Monte Carlo Control

    Updates Q-values based on complete episode returns.
    Does not use bootstrapping - waits for actual returns.

    Q(s,a) = average of all returns following first visit to (s,a)
    """

    def __init__(self, n_states, n_actions, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """Initialize Monte Carlo agent."""
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = {}
        self.returns = {}  # Store returns for averaging (pickle-friendly)
        self.name = "Monte Carlo (First-Visit)"

    def select_action(self, state, greedy=False):
        """Select action using ε-greedy policy."""
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Initialize state if not seen before
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)

        return np.argmax(self.q_table[state])

    def update_episode(self, episode):
        """
        Update Q-values from complete episode.

        Args:
            episode: List of (state, action, reward) tuples
        """
        visited = set()
        G = 0  # Return

        # Traverse episode backwards to calculate returns
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]

            # Calculate discounted return G_t
            G = reward + self.gamma * G

            # First-visit MC: only update on first occurrence
            state_action = (state, action)
            if state_action not in visited:
                visited.add(state_action)

                # Initialize if not seen before
                if state not in self.q_table:
                    self.q_table[state] = np.zeros(self.n_actions)
                if state_action not in self.returns:
                    self.returns[state_action] = []

                # Store return
                self.returns[state_action].append(G)

                # Update Q-value as average of all returns
                self.q_table[state][action] = np.mean(self.returns[state_action])

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self, state):
        """Get the greedy policy for a state."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return np.argmax(self.q_table[state])


def train_agent(env, agent, n_episodes=2000, max_steps=200, verbose=True):
    """
    Train an RL agent in the Taxi-v3 environment.

    Args:
        env: Gymnasium environment
        agent: RL agent (Q-Learning, SARSA, or Monte Carlo)
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        verbose: Print training progress

    Returns:
        Dictionary containing training metrics
    """
    metrics = {
        'episode_rewards': [],
        'episode_steps': [],
        'success_rate': [],
        'epsilon_values': [],
        'td_errors': []
    }

    start_time = time.time()

    if verbose:
        print(f"\n{'='*70}")
        print(f"Training: {agent.name}")
        print(f"Episodes: {n_episodes} | Max Steps: {max_steps}")
        print(f"{'='*70}")

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        truncated = False

        # For Monte Carlo: store episode trajectory
        if isinstance(agent, MonteCarloAgent):
            episode_data = []

        # For SARSA: select initial action
        if isinstance(agent, SARSAAgent):
            action = agent.select_action(state)

        td_errors = []

        # Episode loop
        while not (done or truncated) and episode_steps < max_steps:
            # Select action (for Q-Learning and MC)
            if not isinstance(agent, SARSAAgent):
                action = agent.select_action(state)

            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1

            # Update agent
            if isinstance(agent, MonteCarloAgent):
                # Store transition for end-of-episode update
                episode_data.append((state, action, reward))
            elif isinstance(agent, SARSAAgent):
                # SARSA: need next action from policy
                next_action = agent.select_action(next_state)
                td_error = agent.update(state, action, reward, next_state, next_action, done)
                td_errors.append(abs(td_error))
                action = next_action  # Use next_action in next iteration
            else:
                # Q-Learning: update immediately
                td_error = agent.update(state, action, reward, next_state, done)
                td_errors.append(abs(td_error))

            state = next_state

        # Monte Carlo: update from complete episode
        if isinstance(agent, MonteCarloAgent) and episode_data:
            agent.update_episode(episode_data)

        # Decay exploration rate
        agent.decay_epsilon()

        # Store metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_steps'].append(episode_steps)
        metrics['success_rate'].append(1 if done else 0)
        metrics['epsilon_values'].append(agent.epsilon)

        if td_errors:
            metrics['td_errors'].append(np.mean(td_errors))

        # Progress reporting
        if verbose and (episode + 1) % 100 == 0:
            window = min(100, episode + 1)
            avg_reward = np.mean(metrics['episode_rewards'][-window:])
            avg_steps = np.mean(metrics['episode_steps'][-window:])
            success_rate = np.mean(metrics['success_rate'][-window:]) * 100

            elapsed = time.time() - start_time
            print(f"Episode {episode + 1:4d}/{n_episodes} | "
                  f"Reward: {avg_reward:6.2f} | "
                  f"Steps: {avg_steps:5.1f} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Time: {elapsed:.1f}s")

    training_time = time.time() - start_time

    if verbose:
        print(f"{'='*70}")
        print(f"Training Complete! Total time: {training_time:.2f}s")
        print(f"{'='*70}\n")

    metrics['training_time'] = training_time
    return metrics


def evaluate_agent(env, agent, n_episodes=100, verbose=True):
    """
    Evaluate trained agent using greedy policy (no exploration).

    Args:
        env: Gymnasium environment
        agent: Trained RL agent
        n_episodes: Number of evaluation episodes
        verbose: Print results

    Returns:
        Dictionary containing evaluation metrics
    """
    total_rewards = []
    total_steps = []
    successes = 0

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

    results = {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_steps': np.mean(total_steps),
        'std_steps': np.std(total_steps),
        'success_rate': successes / n_episodes,
        'rewards': total_rewards,
        'steps': total_steps
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATION: {agent.name}")
        print(f"{'='*70}")
        print(f"Episodes: {n_episodes}")
        print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Average Steps: {results['avg_steps']:.1f} ± {results['std_steps']:.1f}")
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
        print(f"{'='*70}\n")

    return results


def plot_comparison(results_dict, save_path='comparison_plots.png', show=True):
    """
    Create comparison plots for multiple algorithms.

    Args:
        results_dict: Dictionary mapping algorithm names to their metrics
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('RL Algorithm Comparison: Taxi-v3 Environment',
                 fontsize=16, fontweight='bold')

    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Professional color scheme

    def smooth(data, window=50):
        """Smooth data using moving average."""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Plot 1: Episode Rewards (Learning Curve)
    ax = axes[0, 0]
    for idx, (name, metrics) in enumerate(results_dict.items()):
        smoothed = smooth(metrics['episode_rewards'])
        x = np.arange(len(smoothed))
        ax.plot(x, smoothed, label=name, color=colors[idx], linewidth=2, alpha=0.9)
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
    ax.set_title('Learning Curve: Cumulative Reward per Episode', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Steps to Completion
    ax = axes[0, 1]
    for idx, (name, metrics) in enumerate(results_dict.items()):
        smoothed = smooth(metrics['episode_steps'])
        x = np.arange(len(smoothed))
        ax.plot(x, smoothed, label=name, color=colors[idx], linewidth=2, alpha=0.9)
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Steps to Complete', fontsize=11, fontweight='bold')
    ax.set_title('Efficiency: Steps per Episode', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Plot 3: Success Rate (Rolling Average)
    ax = axes[1, 0]
    for idx, (name, metrics) in enumerate(results_dict.items()):
        smoothed = smooth(metrics['success_rate'], window=100) * 100
        x = np.arange(len(smoothed))
        ax.plot(x, smoothed, label=name, color=colors[idx], linewidth=2, alpha=0.9)
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Reliability: Task Completion Rate', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])

    # Plot 4: Exploration Rate (Epsilon Decay)
    ax = axes[1, 1]
    for idx, (name, metrics) in enumerate(results_dict.items()):
        x = np.arange(len(metrics['epsilon_values']))
        ax.plot(x, metrics['epsilon_values'], label=name,
                color=colors[idx], linewidth=2, alpha=0.9)
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Epsilon (ε)', fontsize=11, fontweight='bold')
    ax.set_title('Exploration vs Exploitation: Epsilon Decay', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plots saved to: {save_path}")

    if show:
        plt.show()


def save_agent(agent, filename):
    """Save trained agent to file."""
    with open(filename, 'wb') as f:
        pickle.dump(agent, f)
    print(f"💾 Agent saved: {filename}")


def load_agent(filename):
    """Load trained agent from file."""
    with open(filename, 'rb') as f:
        agent = pickle.load(f)
    print(f"📂 Agent loaded: {filename}")
    return agent


def save_results(results_dict, eval_results, config, filename='results.json'):
    """Save comprehensive results to JSON file."""
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': config,
        'training_results': {},
        'evaluation_results': {}
    }

    for name, metrics in results_dict.items():
        output['training_results'][name] = {
            'final_avg_reward': float(np.mean(metrics['episode_rewards'][-100:])),
            'final_avg_steps': float(np.mean(metrics['episode_steps'][-100:])),
            'final_success_rate': float(np.mean(metrics['success_rate'][-100:])),
            'training_time': float(metrics.get('training_time', 0))
        }

    for name, results in eval_results.items():
        output['evaluation_results'][name] = {
            'avg_reward': float(results['avg_reward']),
            'std_reward': float(results['std_reward']),
            'avg_steps': float(results['avg_steps']),
            'std_steps': float(results['std_steps']),
            'success_rate': float(results['success_rate'])
        }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"📄 Results saved: {filename}")


def main():
    """
    Main execution function.
    Trains and evaluates Q-Learning, SARSA, and Monte Carlo agents.
    """
    print("\n" + "="*70)
    print("REINFORCEMENT LEARNING: TAXI-V3 CASE STUDY")
    print("Authors: Daniel Truax & Mehak Seth")
    print("="*70)

    # Create Taxi-v3 environment
    env = gym.make('Taxi-v3')

    # Hyperparameter configuration
    config = {
        'n_states': 500,
        'n_actions': 6,
        'alpha': 0.1,           # Learning rate
        'gamma': 0.99,          # Discount factor
        'epsilon': 1.0,         # Initial exploration rate
        'epsilon_decay': 0.995, # Epsilon decay rate
        'epsilon_min': 0.01,    # Minimum epsilon
        'n_episodes': 2000,     # Training episodes
        'eval_episodes': 100    # Evaluation episodes
    }

    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    # Initialize agents
    agents = {
        'Q-Learning': QLearningAgent(
            config['n_states'], config['n_actions'],
            config['alpha'], config['gamma'],
            config['epsilon'], config['epsilon_decay'], config['epsilon_min']
        ),
        'SARSA': SARSAAgent(
            config['n_states'], config['n_actions'],
            config['alpha'], config['gamma'],
            config['epsilon'], config['epsilon_decay'], config['epsilon_min']
        ),
        'Monte Carlo': MonteCarloAgent(
            config['n_states'], config['n_actions'],
            config['gamma'], config['epsilon'],
            config['epsilon_decay'], config['epsilon_min']
        )
    }

    # Train all agents
    training_results = {}
    eval_results = {}

    for name, agent in agents.items():
        # Train
        metrics = train_agent(env, agent,
                            n_episodes=config['n_episodes'],
                            max_steps=200,
                            verbose=True)
        training_results[name] = metrics

        # Evaluate
        eval_metrics = evaluate_agent(env, agent,
                                     n_episodes=config['eval_episodes'],
                                     verbose=True)
        eval_results[name] = eval_metrics

        # Save trained agent
        filename = f"{name.replace(' ', '_').lower()}_agent.pkl"
        save_agent(agent, filename)

    # Generate comparison plots
    plot_comparison(training_results, save_path='comparison_plots.png', show=False)

    # Save all results
    save_results(training_results, eval_results, config, filename='results.json')

    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Algorithm':<20} {'Avg Reward':<15} {'Success Rate':<15} {'Avg Steps':<15}")
    print("-"*70)
    for name, results in eval_results.items():
        print(f"{name:<20} {results['avg_reward']:>6.2f} ± {results['std_reward']:<5.2f} "
              f"{results['success_rate']*100:>6.1f}%         "
              f"{results['avg_steps']:>6.1f} ± {results['std_steps']:<5.1f}")
    print("="*70)

    print("\n✅ Training complete! Files generated:")
    print("   - comparison_plots.png (visualization)")
    print("   - results.json (detailed metrics)")
    print("   - *_agent.pkl (saved models)")
    print("\n")

    env.close()


if __name__ == "__main__":
    main()