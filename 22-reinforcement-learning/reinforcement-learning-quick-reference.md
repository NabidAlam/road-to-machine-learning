# Reinforcement Learning Quick Reference

Quick reference guide for reinforcement learning algorithms, formulas, and code snippets.

## Key Formulas

### Return (Cumulative Reward)
```
Gt = Rt+1 + γRt+2 + γ²Rt+3 + ...
```

### Q-Learning Update
```
Q(s,a) ← Q(s,a) + α[r + γ max(a') Q(s',a') - Q(s,a)]
```

### Policy Gradient
```
∇θ J(θ) = Eπ[∇θ log π(a|s) * Q^π(s,a)]
```

### Bellman Equation for Q
```
Q^π(s,a) = Σ(s',r) P(s',r|s,a) [r + γΣ(a') π(a'|s') Q^π(s',a')]
```

## Algorithms Comparison

| Algorithm | Type | Action Space | Off-Policy | Notes |
|-----------|------|--------------|------------|-------|
| **Q-Learning** | Value-based | Discrete | Yes | Simple, tabular |
| **DQN** | Value-based | Discrete | Yes | Deep, experience replay |
| **Double DQN** | Value-based | Discrete | Yes | Reduces overestimation |
| **REINFORCE** | Policy-based | Discrete/Continuous | No | Monte Carlo |
| **Actor-Critic** | Both | Discrete/Continuous | Yes | Combines value & policy |
| **DDPG** | Actor-Critic | Continuous | Yes | Deterministic policy |
| **PPO** | Policy-based | Discrete/Continuous | Yes | Stable updates |

## Code Snippets

### Q-Learning
```python
Q[s, a] += alpha * (reward + gamma * np.max(Q[next_s]) - Q[s, a])
```

### Epsilon-Greedy
```python
if random.random() < epsilon:
    action = random.choice(actions)
else:
    action = np.argmax(Q[state])
```

### Experience Replay
```python
memory.append((state, action, reward, next_state, done))
batch = random.sample(memory, batch_size)
```

### Policy Gradient Loss
```python
loss = -log_prob * advantage
```

## Libraries

- **Gym**: `import gym; env = gym.make('CartPole-v1')`
- **Stable-Baselines3**: `from stable_baselines3 import DQN`
- **Ray RLlib**: `from ray.rllib.algorithms import ppo`

## Common Environments

- **CartPole-v1**: Balance pole
- **FrozenLake**: Grid world navigation
- **MountainCar**: Drive car up hill
- **Atari**: Game playing
- **MuJoCo**: Continuous control

