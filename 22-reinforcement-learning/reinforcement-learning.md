# Reinforcement Learning Complete Guide

Comprehensive guide to Reinforcement Learning (RL) - learning through interaction with an environment.

## Table of Contents

- [Introduction to Reinforcement Learning](#introduction-to-reinforcement-learning)
- [Markov Decision Processes (MDPs)](#markov-decision-processes-mdps)
- [Value-Based Methods](#value-based-methods)
- [Policy-Based Methods](#policy-based-methods)
- [Actor-Critic Methods](#actor-critic-methods)
- [Deep Reinforcement Learning](#deep-reinforcement-learning)
- [RL Libraries and Tools](#rl-libraries-and-tools)
- [Practice Exercises](#practice-exercises)

---

## Introduction to Reinforcement Learning

### What is Reinforcement Learning?

**Reinforcement Learning (RL)** is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize cumulative rewards over time.

**Key Characteristics:**
- **No labeled data**: Learns from experience (trial and error)
- **Delayed feedback**: Rewards may come after many actions
- **Sequential decisions**: Current actions affect future states
- **Exploration vs Exploitation**: Balance trying new actions vs using known good ones

### Key Components

1. **Agent**: The learner/decision maker
2. **Environment**: The world the agent interacts with
3. **State (s)**: Current situation/observation
4. **Action (a)**: What the agent can do
5. **Reward (r)**: Feedback from environment (positive/negative)
6. **Policy (π)**: Strategy for choosing actions

### Example: Game Playing

```python
# Example: Tic-Tac-Toe
Agent: AI player
Environment: Game board
States: Current board position (3x3 grid)
Actions: Place X or O in empty cell
Rewards: Win (+1), Lose (-1), Draw (0)
Policy: Which move to make in each position
```

### RL vs Other ML Types

| Aspect | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|--------|-------------------|---------------------|---------------------|
| **Data** | Labeled examples | Unlabeled examples | Experience from interaction |
| **Goal** | Learn input-output mapping | Find patterns | Maximize cumulative reward |
| **Feedback** | Correct answers provided | None | Rewards/penalties (delayed) |
| **Examples** | Classification, Regression | Clustering, PCA | Game playing, Robotics |

---

## Markov Decision Processes (MDPs)

### Definition

**Markov Decision Process (MDP)** is the mathematical framework for RL problems.

**Components:**
- **States (S)**: Set of possible situations
- **Actions (A)**: Set of possible actions
- **Transition Probabilities (P)**: P(s'|s,a) - probability of moving to state s' from state s by taking action a
- **Rewards (R)**: R(s,a,s') - immediate reward for state-action-next state
- **Discount Factor ($\gamma$)**: How much we value future rewards ($0 \leq \gamma \leq 1$)

### Markov Property

**Markov Property**: The future depends only on the current state, not the history.

```
P(St+1 | St, At, St-1, At-1, ...) = P(St+1 | St, At)
```

### Return (Cumulative Reward)

**Return ($G_t$)**: Total discounted reward from time $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**Discount Factor ($\gamma$):**
- **$\gamma = 0$**: Only care about immediate reward
- **$\gamma = 1$**: Care equally about all future rewards
- **$\gamma = 0.9$**: Typical value, values near future more

### Value Functions

#### State Value Function $V(s)$

**$V^{\pi}(s)$**: Expected return starting from state $s$, following policy $\pi$:

$$V^{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]$$

**Interpretation**: How good is it to be in state $s$ under policy $\pi$?

#### Action Value Function $Q(s,a)$

**$Q^{\pi}(s,a)$**: Expected return starting from state $s$, taking action $a$, then following policy $\pi$:

$$Q^{\pi}(s,a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]$$

**Interpretation**: How good is it to take action $a$ in state $s$ under policy $\pi$?

### Bellman Equations

#### Bellman Equation for $V(s)$

$$V^{\pi}(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a) [r + \gamma V^{\pi}(s')]$$

#### Bellman Equation for $Q(s,a)$

$$Q^{\pi}(s,a) = \sum_{s',r} P(s',r|s,a) [r + \gamma \sum_{a'} \pi(a'|s') Q^{\pi}(s',a')]$$

### Optimal Policy

**Optimal Policy ($\pi^*$)**: Policy that maximizes expected return:

$$\pi^* = \arg\max_{\pi} V^{\pi}(s) \text{ for all } s$$

**Optimal Value Functions:**
- **$V^*(s)$**: Maximum value achievable from state $s$
- **$Q^*(s,a)$**: Maximum value achievable from state $s$ taking action $a$

---

## Value-Based Methods

### Q-Learning

**Q-Learning** is an off-policy value-based algorithm that learns the optimal action-value function Q*(s,a).

#### Algorithm

```python
# Q-Learning Algorithm
Initialize Q(s,a) arbitrarily
For each episode:
    Initialize state s
    For each step in episode:
        Choose action a from s using policy derived from Q (e.g., ε-greedy)
        Take action a, observe reward r and next state s'
        Q(s,a) ← Q(s,a) + α[r + γ max(a') Q(s',a') - Q(s,a)]
        s ← s'
    Until s is terminal
```

#### Q-Learning Update Rule

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

Where:
- **$\alpha$**: Learning rate
- **$r$**: Immediate reward
- **$\gamma$**: Discount factor
- **$\max_{a'} Q(s',a')$**: Maximum Q-value in next state

#### Implementation

```python
import numpy as np
import random

class QLearning:
    def __init__(self, states, actions, learning_rate=0.1, 
                 discount=0.95, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.q_table = np.zeros((states, actions))
    
    def choose_action(self, state):
        # Epsilon-greedy: explore or exploit
        if random.random() < self.epsilon:
            return random.choice(range(self.actions))
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        # Q-learning update
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (
            reward + self.gamma * max_next_q - current_q
        )
        self.q_table[state, action] = new_q
```

### Deep Q-Network (DQN)

**DQN** uses a neural network to approximate $Q(s,a)$ instead of a Q-table.

#### Key Innovations

1. **Experience Replay**: Store and sample past experiences
2. **Target Network**: Separate network for stable Q-targets
3. **Neural Network**: Function approximator for Q-values

#### Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state))
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * ~dones)
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### Double DQN

**Double DQN** addresses overestimation bias in Q-learning by using two networks.

**Problem**: Q-learning overestimates Q-values because it uses max operation.

**Solution**: Use main network to select action, target network to evaluate it.

```python
class DoubleDQNAgent(DQNAgent):
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select, target to evaluate
        next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
        next_q = self.target_model(next_states).gather(1, next_actions).squeeze()
        
        target_q = rewards + (self.gamma * next_q * ~dones)
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### Dueling DQN

**Dueling DQN** separates value and advantage estimation.

**Key Insight**: 
- **Value V(s)**: How good is state s?
- **Advantage A(s,a)**: How much better is action a than average?

$$Q(s,a) = V(s) + (A(s,a) - \text{mean}(A(s,a)))$$

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Value stream
        self.value_stream = nn.Linear(128, 1)
        
        # Advantage stream
        self.advantage_stream = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))  # Mathematical: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
```

### Prioritized Experience Replay

**Prioritized Experience Replay** samples important experiences more frequently.

```python
import numpy as np
from collections import namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = Experience(state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Compute sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
```

### Rainbow DQN

**Rainbow DQN** combines multiple DQN improvements:
- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- Multi-step learning
- Distributional RL
- Noisy networks

---

## Policy-Based Methods

### Policy Gradient

**Policy Gradient** methods directly optimize the policy π(a|s) instead of learning value functions.

#### REINFORCE Algorithm

**REINFORCE** is a Monte Carlo policy gradient algorithm.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class REINFORCE:
    def __init__(self, state_size, action_size, lr=0.001):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.99
    
    def select_action(self, state):
        probs = self.policy(torch.FloatTensor(state))
        action = torch.multinomial(probs, 1)
        return action.item(), probs[action]
    
    def update(self, rewards, log_probs):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
```

---

## Actor-Critic Methods

**Actor-Critic** combines value-based and policy-based methods:
- **Actor**: Policy network (chooses actions)
- **Critic**: Value network (evaluates states)

### Advantage Actor-Critic (A2C)

**A2C** uses advantage function $A(s,a) = Q(s,a) - V(s)$ instead of Q-values.

**Advantage**: Reduces variance in policy gradient estimates.

```python
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Actor head (policy)
        self.actor = nn.Linear(128, action_size)
        
        # Critic head (value)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        
        return action_probs, value

class A2CAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99):
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, state):
        probs, value = self.model(torch.FloatTensor(state).unsqueeze(0))
        action = torch.multinomial(probs, 1)
        return action.item(), probs[action], value
    
    def update(self, states, actions, rewards, next_states, dones, values, log_probs):
        # Compute returns
        returns = []
        G = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        values = torch.cat(values)
        
        # Compute advantages
        advantages = returns - values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### Proximal Policy Optimization (PPO)

**PPO** is a popular policy gradient method with stable updates.

**Key Features:**
- Clipped objective to prevent large policy updates
- Multiple epochs on same data
- Works well in practice

```python
class PPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, epochs=10):
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon  # Clipping parameter
        self.epochs = epochs
    
    def update(self, states, actions, rewards, old_log_probs, advantages, returns):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        for _ in range(self.epochs):
            # Get current policy
            probs, values = self.model(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            # Critic loss
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
```

### Trust Region Policy Optimization (TRPO)

**TRPO** uses trust region method to ensure stable policy updates.

**Key Feature**: Constrains policy update to trust region.

```python
# TRPO is more complex, typically uses conjugate gradient
# This is a simplified version
class TRPOAgent:
    def __init__(self, state_size, action_size, max_kl=0.01):
        self.model = ActorCritic(state_size, action_size)
        self.max_kl = max_kl  # Maximum KL divergence
    
    def update(self, states, actions, advantages, old_log_probs):
        # Compute natural policy gradient
        # Use conjugate gradient to solve for update direction
        # Constrain update to trust region
        pass  # Full implementation is complex
```

---

## Deep Reinforcement Learning

### Key Techniques

1. **Experience Replay**: Store past experiences and sample randomly
2. **Target Networks**: Separate network for stable targets
3. **Gradient Clipping**: Prevent exploding gradients
4. **Reward Shaping**: Design rewards to guide learning
5. **Frame Stacking**: Use multiple frames for temporal information

### Using OpenAI Gym

**OpenAI Gym** provides standardized RL environments.

```python
import gym

# Create environment
env = gym.make('CartPole-v1')

# Environment properties
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Reset environment
state = env.reset()

# Run episode
done = False
total_reward = 0
step = 0
while not done:
    action = agent.act(state)  # Choose action
    next_state, reward, done, info = env.step(action)  # Take action
    agent.remember(state, action, reward, next_state, done)  # Store experience
    state = next_state
    total_reward += reward
    step += 1
    
    if step % 4 == 0:  # Train every 4 steps
        agent.replay()  # Train agent

print(f"Episode reward: {total_reward}, Steps: {step}")
```

### Popular Environments

**Classic Control:**
- `CartPole-v1`: Balance pole on cart
- `MountainCar-v0`: Drive car up hill
- `Pendulum-v1`: Swing pendulum

**Atari:**
- `Breakout-v4`: Breakout game
- `Pong-v4`: Pong game
- `SpaceInvaders-v4`: Space Invaders

**Box2D:**
- `LunarLander-v2`: Land on moon
- `BipedalWalker-v3`: Walk with two legs

**MuJoCo:**
- `HalfCheetah-v3`: Run with cheetah
- `Humanoid-v3`: Humanoid robot

### Custom Environments

```python
import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 4 actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # Initialize state
        self.state = None
    
    def reset(self):
        self.state = np.random.rand(4)
        return self.state
    
    def step(self, action):
        # Update state based on action
        self.state = self.state + 0.1 * np.random.rand(4)
        
        # Compute reward
        reward = -np.sum(self.state ** 2)
        
        # Check if done
        done = np.sum(self.state ** 2) > 10.0
        
        return self.state, reward, done, {}
```

---

## RL Libraries and Tools

### OpenAI Gym

Standard toolkit for RL environments.

```bash
pip install gym
```

### Stable-Baselines3

High-quality implementations of RL algorithms.

```bash
pip install stable-baselines3
```

```python
from stable_baselines3 import DQN
import gym

env = gym.make('CartPole-v1')
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
```

### Ray RLlib

Scalable RL library for production.

```bash
pip install ray[rllib]
```

---

## RL Applications

### Game Playing

**RL excels at game playing** - from simple games to complex strategy games.

#### Atari Games

```python
import gym
from stable_baselines3 import DQN

# Create Atari environment
env = gym.make('Breakout-v4')

# Train DQN agent
model = DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)

# Test agent
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

#### Chess and Go

**AlphaZero** uses self-play and MCTS (Monte Carlo Tree Search) with neural networks.

### Robotics and Control

**RL for Robotics** learns control policies for robots.

```python
# Example: Robot arm control
env = gym.make('FetchReach-v1')  # Fetch robot reaching task

# Use PPO for continuous control
from stable_baselines3 import PPO

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
```

### Autonomous Systems

**Self-Driving Cars**: RL for navigation, lane keeping, obstacle avoidance.

**Drones**: RL for flight control, path planning.

### Recommendation Systems

**RL for Recommendations** optimizes long-term user engagement.

```python
class RecommendationRL:
    def __init__(self, num_items):
        self.num_items = num_items
        # State: user history, context
        # Action: recommend item
        # Reward: user engagement (click, purchase, rating)
        self.agent = DQNAgent(state_size=100, action_size=num_items)
    
    def recommend(self, user_state):
        action = self.agent.act(user_state)
        return action  # Item to recommend
    
    def update(self, user_state, action, reward, next_state):
        self.agent.remember(user_state, action, reward, next_state, False)
        self.agent.replay()
```

### Finance and Trading

**RL for Trading** learns trading strategies.

```python
class TradingAgent:
    def __init__(self):
        # State: price history, indicators, portfolio
        # Action: buy, sell, hold
        # Reward: portfolio value change
        self.agent = PPOAgent(state_size=50, action_size=3)
    
    def trade(self, market_state):
        action = self.agent.act(market_state)
        return action  # 0: hold, 1: buy, 2: sell
```

### Healthcare

**Treatment Optimization**: RL for personalized treatment plans.

**Drug Discovery**: RL for molecular design.

## Practice Exercises

1. **Implement Q-Learning**: Solve FrozenLake environment
2. **Build DQN**: Train agent on CartPole
3. **Double DQN**: Implement Double DQN to reduce overestimation
4. **Dueling DQN**: Separate value and advantage estimation
5. **Prioritized Replay**: Implement prioritized experience replay
6. **Policy Gradient**: Implement REINFORCE
7. **Actor-Critic**: Build A2C agent
8. **PPO**: Implement Proximal Policy Optimization
9. **Custom Environment**: Create your own RL problem
10. **Atari Games**: Train agent to play Atari games
11. **Trading Bot**: Build RL-based trading strategy
12. **Recommendation System**: Build RL-based recommender

## Resources and Further Learning

### Books

1. **"Reinforcement Learning: An Introduction"** - Sutton & Barto
   - [Free Online](http://incompleteideas.net/book/)
   - Comprehensive RL textbook

2. **"Deep Reinforcement Learning"** - Arulkumaran et al., 2017
   - Survey of deep RL methods

### Important Papers

1. **"Playing Atari with Deep Reinforcement Learning"** - Mnih et al., 2013 (DQN)
2. **"Human-level control through deep reinforcement learning"** - Mnih et al., 2015
3. **"Mastering the game of Go with deep neural networks"** - Silver et al., 2016 (AlphaGo)
4. **"Proximal Policy Optimization Algorithms"** - Schulman et al., 2017 (PPO)
5. **"Trust Region Policy Optimization"** - Schulman et al., 2015 (TRPO)
6. **"Continuous control with deep reinforcement learning"** - Lillicrap et al., 2015 (DDPG)

### Online Courses

1. **Reinforcement Learning Specialization** - Coursera (University of Alberta)
   - [Course Link](https://www.coursera.org/specializations/reinforcement-learning)
   - Comprehensive RL course

2. **Deep Reinforcement Learning** - UC Berkeley CS285
   - [Course Website](http://rail.eecs.berkeley.edu/deeprlcourse/)
   - Advanced deep RL course

3. **Reinforcement Learning** - David Silver (UCL)
   - [Course Videos](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

### Datasets and Environments

1. **OpenAI Gym**: Standard RL environments
2. **Atari 2600**: Game environments
3. **MuJoCo**: Physics simulation for robotics
4. **PyBullet**: Physics simulation
5. **Unity ML-Agents**: Game-based RL environments

### Tools and Libraries

1. **OpenAI Gym**: Standard RL environments
2. **Stable-Baselines3**: High-quality RL algorithm implementations
3. **Ray RLlib**: Scalable RL for production
4. **Tianshou**: PyTorch-based RL library
5. **RLLib**: Research-focused RL library

## Key Takeaways

1. **RL is Different**: No labeled data, learns from experience (trial and error)
2. **Exploration vs Exploitation**: Balance trying new actions vs using known good ones
3. **Delayed Rewards**: Actions have long-term consequences
4. **Value vs Policy**: Value-based (Q-learning) vs Policy-based (REINFORCE) vs Actor-Critic
5. **Deep RL**: Neural networks enable RL on high-dimensional state spaces
6. **Start Simple**: Begin with simple environments (CartPole, FrozenLake)
7. **Use Libraries**: Leverage established libraries (Gym, Stable-Baselines3, Ray RLlib)
8. **Practice**: RL requires hands-on experience - implement algorithms yourself
9. **Applications**: Game playing, robotics, recommendations, finance, healthcare
10. **Challenges**: Sample efficiency, stability, exploration, safety

---

**Next Steps**: Explore [Advanced Topics](reinforcement-learning-advanced-topics.md) for multi-agent RL, hierarchical RL, imitation learning, and meta-learning.

