# File: main.py

import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

# --- Import your agents ---
from Q_Learning import QLearningAgent
from SARSAA import SARSAAgent


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
            # We check if it's QLearningAgent AND not SARSAAgent
            # because SARSAAgent is ALSO an instance of QLearningAgent (inheritance)
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
    env = gym.make('Taxi-v3', render_mode='human')
    state, _ = env.reset()
    done, truncated = False, False

    while not (done or truncated):
        action = agent.select_action(state, greedy=True)
        state, reward, done, truncated, _ = env.step(action)
        time.sleep(0.1)

    env.close()
    print("Demo finished.")


def plot_comparison(results_dict, save_path='comparison_plots.png', show=True):
    """Create comparison plots for multiple algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('RL Algorithm Comparison: Taxi-v3 Environment',
                 fontsize=16, fontweight='bold')

    colors = ['#2E86AB', '#A23B72', '#F18F01']  # QL, SARSA, MC

    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode='valid')

    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    for idx, (name, metrics) in enumerate(results_dict.items()):
        if 'episode_rewards' in metrics:
            smoothed = smooth(metrics['episode_rewards'])
            x = np.arange(len(smoothed))
            ax.plot(x, smoothed, label=name, color=colors[idx], linewidth=2, alpha=0.9)
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
    ax.set_title('Learning Curve: Cumulative Reward per Episode', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Plot 2: Steps to Completion
    ax = axes[0, 1]
    for idx, (name, metrics) in enumerate(results_dict.items()):
        if 'episode_steps' in metrics:
            smoothed = smooth(metrics['episode_steps'])
            x = np.arange(len(smoothed))
            ax.plot(x, smoothed, label=name, color=colors[idx], linewidth=2, alpha=0.9)
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Steps to Complete', fontsize=11, fontweight='bold')
    ax.set_title('Efficiency: Steps per Episode', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Hide the other two empty plots
    axes[1, 0].set_visible(False)
    axes[1, 1].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plots saved to: {save_path}")

    if show:
        plt.show()


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("REINFORCEMENT LEARNING: TAXI-V3 (Q-Learning vs. SARSA)")
    print("=" * 70)

    # 1. Create *headless* environment for training
    env = gym.make('Taxi-v3')

    # 2. Hyperparameter configuration
    config = {
        'n_states': 500,
        'n_actions': 6,
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'n_episodes': 2000,
        'eval_episodes': 100
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
            config['alpha'], config['gamma'],
            config['epsilon'], config['epsilon_decay'], config['epsilon_min']
        )
    ]

    # 4. Dictionaries to store results
    all_training_metrics = {}
    all_eval_metrics = {}

    # 5. --- Loop to train and evaluate ---
    for agent in agents:
        train_metrics = train_agent(env, agent, n_episodes=config['n_episodes'])
        eval_metrics = evaluate_agent(env, agent, n_episodes=config['eval_episodes'])
        all_training_metrics[agent.name] = train_metrics
        all_eval_metrics[agent.name] = eval_metrics

    # 6. Close the training environment
    env.close()

    # 7. Print summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Algorithm':<20} {'Avg Reward':<15} {'Success Rate':<15}")
    print("-" * 70)
    for name, metrics in all_eval_metrics.items():
        print(f"{name:<20} {metrics['avg_reward']:>6.2f}         "
              f"{metrics['success_rate'] * 100:>6.1f}%")
    print("=" * 70)

    # 8. Show plot
    plot_comparison(all_training_metrics)

    # 9. Run demos
    print("\n✅ Comparison run complete! Showing demos...")
    for agent in agents:
        watch_agent(agent)

    print("\nAll tasks complete!")


if __name__ == "__main__":
    main()