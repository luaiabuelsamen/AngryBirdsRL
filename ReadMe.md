# CartPole DQN Implementation

## Overview
This project implements a Deep Q-Network to solve the classic CartPole control problem. The agent learns to balance a pole attached to a cart by applying horizontal forces to the cart. This implementation includes visualization tools, model persistence, and interactive gameplay modes.

![](https://github.com/luaiabuelsamen/CartPoleRL/blob/master/demo.gif)

## System Architecture

### Physical System Dynamics
The CartPole system consists of a cart that can move horizontally and a pole that can rotate around a pivot point on the cart. The system's state is described by four variables:

- $x$: Cart position
- $\dot{x}$: Cart velocity
- $\theta$: Pole angle
- $\dot{\theta}$: Pole angular velocity

The equations of motion for the system are:

$\ddot{\theta} = \frac{g \sin(\theta) - \cos(\theta)[\frac{F + ml\dot{\theta}^2\sin(\theta)}{M + m}]}{l[\frac{4}{3} - \frac{m\cos^2(\theta)}{M + m}]}$

$\ddot{x} = \frac{F + ml[\dot{\theta}^2\sin(\theta) - \ddot{\theta}\cos(\theta)]}{M + m}$

Where:
- $g$: Gravity constant
- $F$: Applied force
- $m$: Pole mass
- $M$: Cart mass
- $l$: Pole length

### DQN Architecture
The DQN uses a neural network to approximate the Q-function:

$Q(s, a) \approx r + \gamma \max_{a'} Q(s', a')$

Network structure:
```
Input Layer (4) → Hidden Layer (64) → ReLU → Hidden Layer (64) → ReLU → Output Layer (2)
```

## Project Structure
```
.
├── main_game.py       # Interactive game environment
├── train_model.py     # DQN training implementation
├── visualize_rl.py    # Training visualization tools
├── models/            # Saved model checkpoints
└── logs/             # Training logs and metrics
```

## Key Components

### 1. State Space
- Normalized state vector: $[x/W, \dot{x}/5, \theta/(\pi/2), \dot{\theta}/2]$
- Where W is screen width

### 2. Action Space
- Binary action space: {left force (-0.2), right force (0.2)}

### 3. Reward Structure
- +1 for each timestep the pole remains upright
- 0 on episode termination

### 4. Training Parameters
```python
MEMORY_SIZE = 100000    # Experience replay buffer size
BATCH_SIZE = 64         # Training batch size
GAMMA = 0.99           # Discount factor
EPSILON_START = 1.0     # Initial exploration rate
EPSILON_END = 0.01      # Final exploration rate
EPSILON_DECAY = 0.995   # Exploration decay rate
```

## Running the Project

### Prerequisites
```bash
poetry install
```

### Training
```bash
python train_model.py
```
This will:
1. Initialize the DQN agent
2. Train for specified episodes
3. Save model checkpoints and logs

### Playing
```bash
python main_game.py
```
Features:
- Switch between AI and human control with 'M' key
- Use arrow keys for manual control
- Watch trained agent perform

### Visualization
```bash
python visualize_rl.py
```
Generates plots for:
- Training rewards
- Episode lengths
- Learning curves
- Q-value distributions

## Performance Metrics
The agent typically achieves:
- Convergence within 500-1000 episodes
- Average episode length >200 steps after training
- Stable pole balancing for extended periods

## Implementation Notes

### Double DQN
Uses two networks to reduce overestimation:
1. Policy network: Action selection
2. Target network: Value estimation

Update rule:
```python
target = reward + GAMMA * target_net(next_state).max()
loss = MSE(policy_net(state), target)
```

### Experience Replay
Stores transitions $(s, a, r, s')$ in circular buffer:
```python
self.memory.append((state, action, reward, next_state, done))
```

### Exploration Strategy
Epsilon-greedy with decay:
```python
ε = max(EPSILON_END, ε * EPSILON_DECAY)
```

## Future Improvements
- Prioritized Experience Replay
- Dueling DQN architecture
- Noisy Networks for exploration
- Multi-step returns
- Using wandb

