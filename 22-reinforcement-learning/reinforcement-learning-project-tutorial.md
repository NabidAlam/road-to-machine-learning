# Reinforcement Learning Project Tutorial

Step-by-step tutorial: Building a DQN agent for CartPole.

## Project: CartPole with Deep Q-Network (DQN)

### Objective

Train a DQN agent to balance a pole on a cart using OpenAI Gym.

### Step 1: Setup

```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
```

### Step 2: Define DQN Network

```python
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
```

### Step 3: Create Agent

```python
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

### Step 4: Training Loop

```python
env = gym.make('CartPole-v1')
agent = DQNAgent(state_size=4, action_size=2)

episodes = 500
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        agent.replay()
        
        if done:
            break
    
    if episode % 10 == 0:
        agent.update_target_model()
        print(f"Episode {episode}, Reward: {total_reward}")
```

### Step 5: Test Agent

```python
# Test trained agent
state = env.reset()
total_reward = 0
while True:
    action = agent.act(state)
    state, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()
    if done:
        break
print(f"Test Reward: {total_reward}")
```

## Extensions

1. **Add Double DQN**: Reduce overestimation
2. **Prioritized Experience Replay**: Learn from important experiences
3. **Dueling DQN**: Separate value and advantage
4. **Try Other Environments**: Atari, MuJoCo

