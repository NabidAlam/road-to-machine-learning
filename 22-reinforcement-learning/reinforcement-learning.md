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
- **Discount Factor (γ)**: How much we value future rewards (0 ≤ γ ≤ 1)

### Markov Property

**Markov Property**: The future depends only on the current state, not the history.

```
P(St+1 | St, At, St-1, At-1, ...) = P(St+1 | St, At)
```

### Return (Cumulative Reward)

**Return (Gt)**: Total discounted reward from time t:

```
Gt = Rt+1 + γRt+2 + γ²Rt+3 + ... = Σ(k=0 to ∞) γ^k * Rt+k+1
```

**Discount Factor (γ):**
- **γ = 0**: Only care about immediate reward
- **γ = 1**: Care equally about all future rewards
- **γ = 0.9**: Typical value, values near future more

### Value Functions

#### State Value Function V(s)

**V^π(s)**: Expected return starting from state s, following policy π:

```
V^π(s) = Eπ[Gt | St = s]
```

**Interpretation**: How good is it to be in state s under policy π?

#### Action Value Function Q(s,a)

**Q^π(s,a)**: Expected return starting from state s, taking action a, then following policy π:

```
Q^π(s,a) = Eπ[Gt | St = s, At = a]
```

**Interpretation**: How good is it to take action a in state s under policy π?

### Bellman Equations

#### Bellman Equation for V(s)

```
V^π(s) = Σ(a) π(a|s) Σ(s',r) P(s',r|s,a) [r + γV^π(s')]
```

#### Bellman Equation for Q(s,a)

```
Q^π(s,a) = Σ(s',r) P(s',r|s,a) [r + γΣ(a') π(a'|s') Q^π(s',a')]
```

### Optimal Policy

**Optimal Policy (π*)**: Policy that maximizes expected return:

```
π* = argmax(π) V^π(s) for all s
```

**Optimal Value Functions:**
- **V*(s)**: Maximum value achievable from state s
- **Q*(s,a)**: Maximum value achievable from state s taking action a

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

```
Q(s,a) ← Q(s,a) + α[r + γ max(a') Q(s',a') - Q(s,a)]
```

Where:
- **α**: Learning rate
- **r**: Immediate reward
- **γ**: Discount factor
- **max(a') Q(s',a')**: Maximum Q-value in next state

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

**DQN** uses a neural network to approximate Q(s,a) instead of a Q-table.

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

```python
# Double DQN uses main network to select action,
# but target network to evaluate it
next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
next_q = self.target_model(next_states).gather(1, next_actions).squeeze()
```

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

```python
import gym

# Create environment
env = gym.make('CartPole-v1')

# Reset environment
state = env.reset()

# Run episode
done = False
total_reward = 0
while not done:
    action = agent.act(state)  # Choose action
    next_state, reward, done, info = env.step(action)  # Take action
    agent.remember(state, action, reward, next_state, done)  # Store experience
    state = next_state
    total_reward += reward
    agent.replay()  # Train agent
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

## Practice Exercises

1. **Implement Q-Learning**: Solve FrozenLake environment
2. **Build DQN**: Train agent on CartPole
3. **Policy Gradient**: Implement REINFORCE
4. **Actor-Critic**: Build A2C agent
5. **Custom Environment**: Create your own RL problem

---

## Key Takeaways

1. **RL is Different**: No labeled data, learns from experience
2. **Exploration vs Exploitation**: Balance trying new things vs using what works
3. **Delayed Rewards**: Actions have long-term consequences
4. **Start Simple**: Begin with simple environments (CartPole, FrozenLake)
5. **Use Libraries**: Leverage established libraries (Gym, Stable-Baselines3)
6. **Practice**: RL requires hands-on experience

---

**Next Steps**: Explore [Advanced Topics](reinforcement-learning-advanced-topics.md) for multi-agent RL, hierarchical RL, and meta-learning.

