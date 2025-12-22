# Reinforcement Learning with DDPG + HER

A Deep Deterministic Policy Gradient (DDPG) implementation with Hindsight Experience Replay (HER) for continuous control tasks, specifically designed for parking environment challenges.

## Overview

This project implements a state-of-the-art reinforcement learning agent using DDPG combined with HER to solve goal-conditioned continuous control tasks. The architecture follows the original papers' specifications with modern best practices.

---

## Architecture & Component Walkthrough

### 1. **Actor Network** (`ddpg/actor_network.py`)

**Conceptual Role**: The policy network that learns to select actions.

**How It Works**:

- **Input**: Current state (including goal information) as a tensor of shape `(batch_size, state_dim)`

- **Forward Method**: `forward(self, state: torch.Tensor) -> torch.Tensor`

- **Architecture**:

- State → 400 hidden units (Layer 1)

- LayerNorm + ReLU activation

- 400 → 400 hidden units (Layer 2)

- LayerNorm + ReLU activation

- 400 → action_dim output

- Tanh activation (outputs in range [-1, 1])

- **Output Scaling**: Final action is multiplied by `max_action` to scale to actual action bounds

- **Purpose**: Maps states deterministically to continuous actions

- **Training**: Updated via policy gradient to maximise expected Q-value

**Key Design Choices**:

- LayerNorm instead of BatchNorm for stability with small batches

- 400-unit hidden layers (DDPG paper recommendation for complex tasks)

- Tanh ensures bounded outputs before scaling

---

### 2. **Critic Network** (`ddpg/critic_network.py`)

**Conceptual Role**: The Q-function approximator that estimates state-action values.

**How It Works**:

- **Input**: State tensor `(batch_size, state_dim)` and action tensor `(batch_size, action_dim)`

- **Forward Method**: `forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor`

- **Architecture**:

- **Layer 1**: Processes state only → 400 hidden units

- **Layer 2**: Concatenates state features with action → 400 hidden units

- **Layer 3**: Outputs scalar Q-value per sample

- **Purpose**: Learns Q(s,a) - the expected cumulative reward from taking action `a` in state `s`

- **Training**: Minimises TD-error using Bellman equation targets

**Key Design Choices**:

- Action is injected after the first layer (standard DDPG architecture)

- This allows the network to process state features before combining with actions

- LayerNorm for training stability

- Single Q-value output (unlike Double-DQN, DDPG uses single Q-network)

---

### 3. **Ornstein-Uhlenbeck Noise** (`ddpg/noise.py`)

**Conceptual Role**: Exploration noise process for continuous action spaces.

**How It Works**:

- **Mathematical Model**: OU process follows: `dx_t = θ(μ - x_t)dt + σ√dt·N(0,1)`

- `θ (theta)`: Mean reversion rate (0.15) - how quickly noise returns to mean

- `μ (mu)`: Long-term mean (0.0) - target value noise reverts to

- `σ (sigma)`: Volatility (0.2) - magnitude of random fluctuations

- `dt`: Time step (0.01)

- **State Evolution**: Maintains internal state that evolves over time

- **Reset**: Called at episode start to reinitialise to mean

- **Purpose**: Generates temporally correlated noise (better than pure random for physical systems)

**Why OU Noise?**:

- Unlike white noise, OU noise has temporal correlation

- Beneficial for tasks with momentum/inertia (like parking)

- Allows smooth exploration in action space

---

### 4. **DDPG Agent** (`ddpg/ddpg_agent.py`)

**Conceptual Role**: Central orchestrator of the DDPG algorithm.

**Components**:

1.  **Actor**: Current policy network

2.  **Actor Target**: Slowly-updated copy for stable targets

3.  **Critic**: Current Q-network

4.  **Critic Target**: Slowly-updated copy for stable targets

5.  **Noise**: OU exploration process

6.  **Optimisers**: Adam optimisers (1e-4 for actor, 1e-3 for critic)

**Constructor Parameters**:

- `state_dim`: Input dimensionality (state + goal)
- `action_dim`: Output action dimensionality
- `max_action`: Maximum action value for scaling
- `discount_factor`: Reward discount factor (default 0.99, stored as `self.discount_factor`)
- `soft_update_factor`: Target network update rate (default 0.005, stored as `self.soft_update_factor`)
- `device`: PyTorch device for computations

**Key Methods**:
) -> None:

- **Input**: Current state observation

- **Process**:

1. Convert state to tensor and add batch dimension

2. Set actor to eval mode (disable dropout/batchnorm training)

3. Forward pass through actor (deterministic action)

4. If noise=True, add OU noise for exploration

5. Clip to ensure action bounds are respected

- **Output**: Action array ready for environment

- **Usage**: Called during training (with noise) and evaluation (without noise)

#### `get_sample_batch(her_replay_buffer, batch_size)`

- **Purpose**: Extract and prepare a training batch

- **Process**:

1. Sample random batch from HER replay buffer

2. Convert all components to PyTorch tensors

3. Move to appropriate device (CPU/GPU)

4. Add necessary dimensions (rewards, dones need unsqueeze)

- **Output**: Tuple of (states, actions, rewards, next_states, dones)

#### `train(her_replay_buffer, batch_size)`

**The Core DDPG Training Algorithm**:

1.  **Sample Batch**: Get transitions from replay buffer

2.  **Compute Target Q-values** (no gradient):

```

Q_target = r + γ(1 - done) · Q'(s', μ'(s'))

```

- Use target actor to get next action: `μ'(s')`

- Use target critic to estimate: `Q'(s', μ'(s'))`

- Apply Bellman equation with discount `discount_factor` (γ)

- Multiply by (1 - done) to zero out terminal states

3.  **Update Critic**:

- Compute current Q-values: `Q(s, a)`

- Loss: MSE between current Q and target Q

- Backpropagate and update critic weights

- **Goal**: Make Q-network better at predicting returns

4.  **Update Actor**:

- Loss: `-Q(s, μ(s)).mean()`

- Negative because we want to maximise Q-value

- Backpropagate through both actor and critic

- Only actor weights are updated (critic frozen via optimiser)

- **Goal**: Improve policy to take actions with higher Q-values

5.  **Soft Update Targets**:

```

θ_target = τ·θ + (1-τ)·θ_target
```

- Slowly blend current networks into targets

- `soft_update_factor` (τ) = 0.005 (only 0.5% update per step)

- **Goal**: Stable learning targets without oscillation

#### `softUpdate(source_net, target_net)`

- **Purpose**: Polyak averaging for stable target networks

- **Process**: For each parameter, update target = `soft_update_factor`·source + (1-`soft_update_factor`)·target

- **Why**: Prevents targets from changing too rapidly, which would destabilise training

---

### Why This Works:

- **Off-Policy**: Learn from any experience, not just current policy

- **Deterministic Policy**: No need to sample during execution

- **Target Networks**: Stabilise learning by slowing target updates

- **Replay Buffer**: Decorrelate data and improve sample efficiency

- **Actor-Critic**: Combine value-based and policy-based methods

---

## Integration with HER (Hindsight Experience Replay)

**HER Enhancement**: The agent is designed to work with HER replay buffers:

- **Goal-Conditioned**: State includes goal information

- **Hindsight Relabeling**: Failed episodes can be relabeled with achieved goals

- **Sample Efficiency**: Learn from failures, not just successes

- **HER-Compatible Sampling**: `get_sample_batch()` works with HER buffer format

**Why HER?**:

- Sparse reward environments (like parking) benefit enormously

- Every trajectory provides learning signal

- Dramatically improves sample efficiency

---

## Hyperparameters & Design Choices

| Parameter                  | Value | Rationale                                |
| -------------------------- | ----- | ---------------------------------------- |
| **discount_factor (γ)**    | 0.99  | Standard discount for episodic tasks     |
| **soft_update_factor (τ)** | 0.005 | Very slow target updates for stability   |
| **Actor LR**               | 1e-4  | Lower than critic to prevent instability |
| **Critic LR**              | 1e-3  | Higher learning rate for value function  |
| **Hidden Dim**             | 400   | DDPG paper recommendation                |
| **OU θ**                   | 0.15  | Moderate mean reversion                  |
| **OU σ**                   | 0.2   | Moderate exploration noise               |

## Key Concepts

### Actor-Critic Methods

- **Actor**: Learns policy π(s) → a

- **Critic**: Learns value Q(s,a)

- **Synergy**: Critic guides actor toward better actions

### Deterministic Policy Gradient (DPG)

- Unlike stochastic policies, DDPG learns deterministic μ(s)

- Gradient: ∇*θ J = E[∇_a Q(s,a) · ∇*θ μ(s)]

- More sample efficient for continuous control

### Target Networks

- Prevent moving target problem

- TD target uses old network weights

- Slowly updated via soft updates

### Off-Policy Learning

- Learn from any experience

- Use replay buffer for decorrelation

- Higher sample efficiency than on-policy
