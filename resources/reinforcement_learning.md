# Reinforcement Learning Basics

Introduction to Reinforcement Learning (RL), a type of machine learning where agents learn to make decisions by interacting with an environment.

## Table of Contents

- [Introduction](#introduction)
- [Key Concepts](#key-concepts)
- [RL vs Other ML Types](#rl-vs-other-ml-types)
- [Basic Algorithms](#basic-algorithms)
- [Applications](#applications)
- [Getting Started](#getting-started)
- [Resources](#resources)

---

## Introduction

**Reinforcement Learning (RL)** is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize cumulative rewards over time.

### Key Characteristics

- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **Actions**: What the agent can do
- **States**: Current situation/observation
- **Rewards**: Feedback from environment (positive/negative)
- **Policy**: Strategy for choosing actions

### Example: Game Playing

- **Agent**: Game player
- **Environment**: Game board/rules
- **Actions**: Move pieces, make moves
- **States**: Current board position
- **Rewards**: Win (+1), Lose (-1), Draw (0)
- **Policy**: Which moves to make in each situation

---

## Key Concepts

### 1. Markov Decision Process (MDP)

Mathematical framework for RL:

- **States (S)**: Set of possible situations
- **Actions (A)**: Set of possible actions
- **Transition Probabilities (P)**: Probability of moving to next state
- **Rewards (R)**: Immediate reward for state-action pair
- **Discount Factor (γ)**: How much we value future rewards (0-1)

### 2. Policy (π)

Strategy for choosing actions:
- **π(a|s)**: Probability of taking action 'a' in state 's'
- **Deterministic**: Always same action for same state
- **Stochastic**: Probabilistic action selection

### 3. Value Functions

**State Value Function V(s)**: Expected cumulative reward from state s
- How good is it to be in this state?

**Action Value Function Q(s,a)**: Expected cumulative reward from taking action a in state s
- How good is this action in this state?

### 4. Exploration vs Exploitation

- **Exploration**: Try new actions to discover better strategies
- **Exploitation**: Use known good actions
- **Trade-off**: Balance between learning and earning

---

## RL vs Other ML Types

### Supervised Learning
- **Data**: Labeled examples (input-output pairs)
- **Goal**: Learn mapping from inputs to outputs
- **Feedback**: Correct answers provided

### Unsupervised Learning
- **Data**: Unlabeled examples
- **Goal**: Find hidden patterns
- **Feedback**: None

### Reinforcement Learning
- **Data**: Experience from interaction
- **Goal**: Maximize cumulative reward
- **Feedback**: Rewards/penalties (delayed, sparse)

---

## Basic Algorithms

### 1. Q-Learning

Value-based algorithm that learns action values.

**Key Idea**: Learn Q(s,a) - value of taking action 'a' in state 's'

**Update Rule**:
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- α: Learning rate
- r: Immediate reward
- γ: Discount factor
- s': Next state
- a': Best action in next state

**Example Implementation**:
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

### 2. Policy Gradient Methods

Directly optimize the policy (strategy).

**REINFORCE Algorithm**:
- Sample trajectory
- Calculate returns
- Update policy to increase probability of good actions

### 3. Actor-Critic Methods

Combine value-based and policy-based approaches.

- **Actor**: Policy network (chooses actions)
- **Critic**: Value network (evaluates states)

---

## Applications

### 1. Game Playing
- **Chess**: AlphaZero
- **Go**: AlphaGo
- **Atari Games**: DQN
- **Starcraft**: AlphaStar

### 2. Robotics
- Robot control
- Autonomous navigation
- Manipulation tasks

### 3. Autonomous Systems
- Self-driving cars
- Drone control
- Traffic optimization

### 4. Recommendation Systems
- Personalized recommendations
- Ad placement
- Content optimization

### 5. Finance
- Trading strategies
- Portfolio optimization
- Risk management

### 6. Healthcare
- Treatment optimization
- Drug discovery
- Resource allocation

---

## Getting Started

### Libraries

#### 1. OpenAI Gym
Standard toolkit for RL environments.

```bash
pip install gym
```

```python
import gym

# Create environment
env = gym.make('CartPole-v1')

# Reset environment
state = env.reset()

# Take action
action = env.action_space.sample()  # Random action
next_state, reward, done, info = env.step(action)

# Close environment
env.close()
```

#### 2. Stable-Baselines3
High-quality RL algorithm implementations.

```bash
pip install stable-baselines3
```

```python
from stable_baselines3 import PPO
import gym

# Create environment
env = gym.make('CartPole-v1')

# Create and train agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Test agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()
```

#### 3. Ray RLlib
Scalable RL library.

```bash
pip install ray[rllib]
```

### Simple Example: CartPole

```python
import gym
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(2))
    
    def discretize_state(self, state):
        # Simple discretization (for demonstration)
        return tuple(np.round(state, 1))
    
    def choose_action(self, state):
        state_key = self.discretize_state(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(2)  # Explore
        else:
            return np.argmax(self.q_table[state_key])  # Exploit
    
    def update(self, state, action, reward, next_state, done):
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])
        
        self.q_table[state_key][action] += self.alpha * (
            target - self.q_table[state_key][action]
        )

# Training
env = gym.make('CartPole-v1')
agent = QLearningAgent()

for episode in range(1000):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
```

---

## Resources

### Books

1. **"Reinforcement Learning: An Introduction"** by Sutton & Barto
   - [Online Book](http://incompleteideas.net/book/)
   - Classic textbook, comprehensive coverage

2. **"Deep Reinforcement Learning"** by Pieter Abbeel et al.
   - Advanced topics in deep RL

### Online Courses

1. **"Reinforcement Learning Specialization"** - Coursera (University of Alberta)
   - [Course](https://www.coursera.org/specializations/reinforcement-learning)

2. **"Deep Reinforcement Learning"** - UC Berkeley CS285
   - [Course](http://rail.eecs.berkeley.edu/deeprlcourse/)

### Libraries & Tools

- **OpenAI Gym**: [Documentation](https://www.gymlibrary.dev/)
- **Stable-Baselines3**: [Documentation](https://stable-baselines3.readthedocs.io/)
- **Ray RLlib**: [Documentation](https://docs.ray.io/en/latest/rllib/index.html)

### Papers

1. **"Playing Atari with Deep Reinforcement Learning"** (DQN)
   - Mnih et al., 2013

2. **"Mastering the game of Go with deep neural networks"** (AlphaGo)
   - Silver et al., 2016

3. **"Mastering Chess and Shogi by Self-Play"** (AlphaZero)
   - Silver et al., 2017

### Communities

- [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

---

## Key Takeaways

1. **RL is Different**: No labeled data, learns from experience
2. **Exploration vs Exploitation**: Balance trying new things vs using what works
3. **Delayed Rewards**: Actions have long-term consequences
4. **Start Simple**: Begin with simple environments (CartPole, FrozenLake)
5. **Use Libraries**: Leverage established libraries (Gym, Stable-Baselines3)
6. **Practice**: RL requires hands-on experience

---

**Note**: Reinforcement Learning is a complex field. This guide provides basics. For production applications, study advanced topics like Deep RL, Multi-Agent RL, and Imitation Learning.

