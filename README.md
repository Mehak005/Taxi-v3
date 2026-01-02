
# Reinforcement Learning: Taxi-v3 Environment

A comprehensive comparison of Reinforcement Learning algorithms applied to OpenAI Gym's Taxi-v3 environment. This project implements and compares Q-Learning, SARSA, Monte Carlo, and Deep Q-Network (DQN) agents.

## Authors

Daniel Truax (dtruax@albany.edu)
Mehak Seth (mseth3@albany.edu)

##  Project Overview

This project trains and evaluates multiple RL agents to solve the classic Taxi navigation problem where an agent must:
- Navigate a n×n grid (typcially 5x5, expanded out further)
- Pick up a passenger from one location
- Drop them off at a destination
- Avoid illegal pickups/dropoffs and minimize steps

##  Features

- **Multiple RL Algorithms**: Q-Learning, SARSA, Monte Carlo, and DQN
- **Comprehensive Metrics**: Training curves, success rates, and performance comparisons
- **Visual Demonstrations**: Watch trained agents perform in real-time in either Jupyter of ASCII representation
- **Graph Representations**: A selection of the ran graphs contained within the Images folder
- **Scalable Environment**: Includes a larger 10×10 grid variant (`TaxiLarge-v3`)
- **Comparative Analysis**: Automatic plotting and performance benchmarking for easy analysis

## Requirements

```bash
gymnasium
numpy
torch
matplotlib
```

Install dependencies:
```bash
pip install gymnasium numpy torch matplotlib
```

##  Usage

### Basic Training

Run the main script to train and compare all agents:

```bash
python main.py
```
When the main script is called, you will see the following:

```bash
1. Standard Taxi-v3 (5x5 grid, 500 states)
2. Larger Taxi (10x10 grid, 2000 states)

Enter your choice (1 or 2):
```
Depending on which one you select is the environment the agent is trained on!


### Hyperparameter Configuration

Modify the `config` dictionary in `main.py`:

```python
config = {
    'n_states': n_states, # Depends on which state space is used
    'n_actions': 6,
    'alpha': 0.15,           # Learning rate
    'gamma': 0.97,           # Discount factor
    'epsilon': 1.0,          # Initial exploration rate
    'epsilon_decay': 0.997,  # Exploration decay
    'epsilon_min': 0.01,     # Minimum exploration
    'n_episodes': 3000 if use_large_taxi else 2500,      # Training episodes
    'eval_episodes': 150     # Evaluation episodes
}
```

##  Algorithms

### 1. Q-Learning (Off-Policy)
- **Update Rule**: `Q(S,A) ← Q(S,A) + α[R + γ max Q(S',a) - Q(S,A)]`
- **Type**: Off-policy TD control
- **Best for**: Optimal policy learning regardless of exploration strategy

### 2. SARSA (On-Policy)
- **Update Rule**: `Q(S,A) ← Q(S,A) + α[R + γ Q(S',A') - Q(S,A)]`
- **Type**: On-policy TD control
- **Best for**: Safe exploration and risk-averse policies

### 3. Monte Carlo
- **Update Rule**: `Q(S,A) ← Q(S,A) + α[G - Q(S,A)]`
- **Type**: Episode-based learning
- **Best for**: Episodic tasks without bootstrapping

### 4. Deep Q-Network (DQN)
- **Architecture**: Neural network with embedding layer
- **Features**: Experience replay, target network
- **Best for**: Large state spaces and deep learning integration

## 📊 Output

The program generates:
1. **Training Progress**: Real-time console output with metrics
2. **Comparison Plots**: Learning curves and efficiency graphs saved as `comparison_plots.png` (Some select graphs are stored in the Images folder)
3. **Evaluation Summary**: Final performance statistics
4. **Live Demonstrations**: Visual playback of trained agents

### Sample Output

```
======================================================================
Training: Q-Learning (Off-Policy)
Episodes: 2500 | Max Steps: 200
======================================================================
Episode  100/2500 | Avg Reward:  -671.61 | Avg Steps:  192.6 | ε: 0.740
Episode  200/2500 | Avg Reward:  -461.23 | Avg Steps:  165.5 | ε: 0.548
...
======================================================================
FINAL RESULTS SUMMARY
======================================================================

Algorithm            Avg Reward      Success Rate   
----------------------------------------------------------------------
Q-Learning (Off-Policy)    4.96         98.7%
SARSA (On-Policy)          2.31         97.3%
Monte Carlo                -1083.83         6.7%
Deep Q-Network (DQN)       -0.59         96.7%
======================================================================
```



### Using the Larger Environment

All you have to do to run the larger environment is press 2 when prompted in the training section!







##  Acknowledgments

- OpenAI Gym/Gymnasium for the Taxi-v3 environment
- https://www.gocoder.one/blog/rl-tutorial-with-openai-gym/ for basic Taxi-v3 tutorial
- https://lilianweng.github.io/posts/2018-02-19-rl-overview/ for learning DQ - Learning algorithms


---



