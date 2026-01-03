# Reinforcement Learning Advanced Topics

Advanced topics in reinforcement learning for production and research applications.

## Table of Contents

- [Multi-Agent Reinforcement Learning](#multi-agent-reinforcement-learning)
- [Hierarchical Reinforcement Learning](#hierarchical-reinforcement-learning)
- [Imitation Learning](#imitation-learning)
- [Transfer Learning in RL](#transfer-learning-in-rl)
- [Meta-Learning and Few-Shot RL](#meta-learning-and-few-shot-rl)
- [Continuous Action Spaces](#continuous-action-spaces)
- [Reward Shaping](#reward-shaping)
- [Safe Reinforcement Learning](#safe-reinforcement-learning)

---

## Multi-Agent Reinforcement Learning

### Introduction

**Multi-Agent RL (MARL)** involves multiple agents learning and interacting in the same environment.

**Challenges:**
- Non-stationary environment (other agents change)
- Coordination vs Competition
- Communication between agents
- Scalability

### Types of Multi-Agent Systems

1. **Cooperative**: Agents work together toward common goal
2. **Competitive**: Agents compete against each other
3. **Mixed**: Combination of cooperation and competition

### Algorithms

#### Independent Q-Learning (IQL)

Each agent learns independently:

```python
class IndependentQLearning:
    def __init__(self, num_agents, state_size, action_size):
        self.agents = [QLearning(state_size, action_size) 
                       for _ in range(num_agents)]
    
    def act(self, states):
        return [agent.act(state) for agent, state 
                in zip(self.agents, states)]
```

#### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

**MADDPG** extends DDPG to multi-agent settings with centralized training, decentralized execution.

```python
class MADDPGAgent:
    def __init__(self, state_size, action_size, num_agents):
        self.agents = []
        for _ in range(num_agents):
            agent = DDPGAgent(state_size, action_size)
            self.agents.append(agent)
    
    def act(self, states, add_noise=True):
        """Each agent acts based on its own state"""
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act(state, add_noise)
            actions.append(action)
        return actions
    
    def update(self, states, actions, rewards, next_states, dones):
        """Update all agents"""
        for i, agent in enumerate(self.agents):
            # Use all agents' states and actions for critic
            all_states = torch.cat(states, dim=1)
            all_actions = torch.cat(actions, dim=1)
            all_next_states = torch.cat(next_states, dim=1)
            
            agent.update(
                states[i], actions[i], rewards[i],
                next_states[i], dones[i],
                all_states, all_actions, all_next_states
            )
```

#### Mean Field Multi-Agent RL

**Mean Field RL** approximates other agents' policies for scalability.

---

## Hierarchical Reinforcement Learning

### Concept

**Hierarchical RL** learns at multiple levels of abstraction:
- **High-level**: Long-term goals and subgoals
- **Low-level**: Short-term actions to achieve subgoals

**Benefits:**
- Handles long-horizon tasks
- Reuses learned skills
- More sample efficient

### Options Framework

**Options** are temporally extended actions:
- **Initiation Set**: States where option can start
- **Policy**: How to execute option
- **Termination Condition**: When option ends

```python
class Option:
    def __init__(self, initiation_set, policy, termination_condition):
        self.initiation_set = initiation_set  # States where option starts
        self.policy = policy  # Policy to execute option
        self.termination = termination_condition  # When to terminate
    
    def can_start(self, state):
        return state in self.initiation_set
    
    def is_terminal(self, state):
        return self.termination(state)
    
    def act(self, state):
        return self.policy(state)
```

### Hindsight Experience Replay (HER)

**HER** learns from failed attempts by relabeling goals.

```python
class HER:
    def __init__(self, replay_buffer, goal_sampling_strategy='future'):
        self.replay_buffer = replay_buffer
        self.goal_sampling_strategy = goal_sampling_strategy
    
    def relabel_trajectory(self, trajectory):
        """Relabel trajectory with achieved goals"""
        relabeled = []
        for i, transition in enumerate(trajectory):
            # Sample goal from future states
            if self.goal_sampling_strategy == 'future':
                future_goals = [t['achieved_goal'] for t in trajectory[i:]]
                if future_goals:
                    goal = random.choice(future_goals)
                    # Relabel transition with new goal
                    new_transition = {
                        'state': transition['state'],
                        'action': transition['action'],
                        'reward': self.compute_reward(transition['achieved_goal'], goal),
                        'next_state': transition['next_state'],
                        'goal': goal
                    }
                    relabeled.append(new_transition)
        return relabeled
```

---

## Hierarchical Reinforcement Learning

### Concept

**Hierarchical RL** learns at multiple levels of abstraction:
- **High-level**: Long-term goals and subgoals
- **Low-level**: Short-term actions to achieve subgoals

### Options Framework

**Options** are temporally extended actions:
- **Initiation Set**: States where option can start
- **Policy**: How to execute option
- **Termination Condition**: When option ends

---

## Imitation Learning

### Overview

**Imitation Learning** learns from expert demonstrations rather than rewards.

**When to Use:**
- Rewards are hard to define
- Expert demonstrations available
- Faster than RL from scratch
- Safer for real-world applications

**Types:**
1. **Behavioral Cloning**: Supervised learning from demonstrations
2. **Inverse Reinforcement Learning**: Learn reward function from demonstrations
3. **Generative Adversarial Imitation Learning (GAIL)**: Use GANs for imitation

### Behavioral Cloning

**Behavioral Cloning** treats imitation as supervised learning.

```python
class BehavioralCloning:
    def __init__(self, state_size, action_size):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters())
    
    def train(self, expert_demos, epochs=100):
        # Expert demos: [(state, action), ...]
        states = torch.FloatTensor([s for s, a in expert_demos])
        actions = torch.LongTensor([a for s, a in expert_demos])
        
        dataset = torch.utils.data.TensorDataset(states, actions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(epochs):
            for batch_states, batch_actions in dataloader:
                predicted_actions = self.policy(batch_states)
                loss = nn.CrossEntropyLoss()(predicted_actions, batch_actions)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
```

**Limitations:**
- Distribution shift: Training states ≠ Test states
- No exploration: Only learns from demonstrations
- Compounding errors: Small errors accumulate

### Dataset Aggregation (DAgger)

**DAgger** addresses distribution shift by collecting new data.

```python
class DAgger:
    def __init__(self, expert_policy, learner_policy):
        self.expert = expert_policy
        self.learner = learner_policy
        self.dataset = []
    
    def collect_data(self, env, num_episodes=10):
        """Collect data using current learner policy"""
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                # Use learner policy
                action = self.learner.act(state)
                next_state, reward, done, info = env.step(action)
                
                # Get expert action for this state
                expert_action = self.expert.act(state)
                
                # Add to dataset
                self.dataset.append((state, expert_action))
                state = next_state
    
    def train(self):
        """Train learner on aggregated dataset"""
        states = torch.FloatTensor([s for s, a in self.dataset])
        actions = torch.LongTensor([a for s, a in self.dataset])
        # Train learner...
```

### Generative Adversarial Imitation Learning (GAIL)

**GAIL** uses GANs to learn from demonstrations.

```python
class GAIL:
    def __init__(self, state_size, action_size):
        self.policy = PolicyNetwork(state_size, action_size)
        self.discriminator = Discriminator(state_size, action_size)
        self.policy_optimizer = optim.Adam(self.policy.parameters())
        self.disc_optimizer = optim.Adam(self.discriminator.parameters())
    
    def train(self, expert_demos, env):
        # Train discriminator
        expert_states = torch.FloatTensor([s for s, a in expert_demos])
        expert_actions = torch.LongTensor([a for s, a in expert_demos])
        
        # Generate policy trajectories
        policy_states, policy_actions = self.generate_trajectories(env)
        
        # Discriminator loss
        expert_pred = self.discriminator(expert_states, expert_actions)
        policy_pred = self.discriminator(policy_states, policy_actions)
        
        disc_loss = -torch.mean(torch.log(expert_pred + 1e-8) + 
                                torch.log(1 - policy_pred + 1e-8))
        
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Policy loss (maximize discriminator's confusion)
        policy_pred = self.discriminator(policy_states, policy_actions)
        policy_loss = -torch.mean(torch.log(policy_pred + 1e-8))
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
```

---

## Transfer Learning in RL

### Concept

**Transfer Learning** applies knowledge from one task to another.

**Approaches:**
1. **Feature Transfer**: Share learned features
2. **Policy Transfer**: Initialize policy from similar task
3. **Reward Shaping**: Use knowledge to design rewards

---

## Meta-Learning and Few-Shot RL

### Model-Agnostic Meta-Learning (MAML)

**MAML** learns initial parameters that can be quickly adapted to new tasks.

**Key Idea**: Learn parameters that are good for many tasks, not just one.

```python
class MAML:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001):
        self.model = model
        self.lr_inner = lr_inner  # Inner loop learning rate
        self.lr_outer = lr_outer  # Outer loop learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_outer)
    
    def meta_update(self, tasks, k_shots=5):
        """Meta-update using multiple tasks"""
        meta_loss = 0
        
        for task in tasks:
            # Inner loop: adapt to task
            adapted_params = self.adapt_to_task(task, k_shots)
            
            # Compute loss on adapted parameters
            loss = self.compute_loss(task, adapted_params)
            meta_loss += loss
        
        # Outer loop: update initial parameters
        meta_loss /= len(tasks)
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
    
    def adapt_to_task(self, task, k_shots):
        """Adapt model to task with k gradient steps"""
        params = list(self.model.parameters())
        
        for _ in range(k_shots):
            # Compute gradient
            loss = self.compute_loss(task, params)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            
            # Update parameters
            params = [p - self.lr_inner * g for p, g in zip(params, grads)]
        
        return params
```

### Reptile

**Reptile** is a simpler meta-learning algorithm.

```python
class Reptile:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_outer)
    
    def meta_update(self, tasks):
        """Reptile meta-update"""
        initial_params = [p.clone() for p in self.model.parameters()]
        
        for task in tasks:
            # Adapt to task
            self.adapt_to_task(task)
        
        # Update: move towards adapted parameters
        for initial, current in zip(initial_params, self.model.parameters()):
            current.data += self.lr_outer * (initial - current)
```

---

## Continuous Action Spaces

### Deep Deterministic Policy Gradient (DDPG)

**DDPG** extends DQN to continuous action spaces using deterministic policy.

**Key Components:**
- **Actor**: Deterministic policy (outputs action directly)
- **Critic**: Q-function approximator
- **Target Networks**: For stable learning

```python
class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action=1.0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.max_action = max_action
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.max_action * x

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDPGAgent:
    def __init__(self, state_size, action_size, max_action=1.0):
        self.actor = Actor(state_size, action_size, max_action)
        self.critic = Critic(state_size, action_size)
        self.target_actor = Actor(state_size, action_size, max_action)
        self.target_critic = Critic(state_size, action_size)
        
        # Copy weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.tau = 0.005  # Soft update parameter
        self.noise = OrnsteinUhlenbeckNoise(action_size)
    
    def act(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy().flatten()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -self.actor.max_action, self.actor.max_action)
    
    def update(self, batch):
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Critic update
        next_actions = self.target_actor(next_states)
        next_q = self.target_critic(next_states, next_actions)
        target_q = rewards + (self.gamma * next_q.squeeze() * ~dones)
        
        current_q = self.critic(states, actions).squeeze()
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class OrnsteinUhlenbeckNoise:
    """OU noise for exploration in continuous action spaces"""
    def __init__(self, action_size, mu=0, theta=0.15, sigma=0.2):
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_size) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
```

### Twin Delayed DDPG (TD3)

**TD3** improves DDPG with:
- Twin critics (reduce overestimation)
- Delayed policy updates
- Target policy smoothing

```python
class TD3Agent(DDPGAgent):
    def __init__(self, state_size, action_size, max_action=1.0):
        super().__init__(state_size, action_size, max_action)
        # Twin critics
        self.critic1 = Critic(state_size, action_size)
        self.critic2 = Critic(state_size, action_size)
        self.target_critic1 = Critic(state_size, action_size)
        self.target_critic2 = Critic(state_size, action_size)
        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=1e-3)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=1e-3)
        
        self.policy_delay = 2  # Update policy every 2 critic updates
    
    def update(self, batch, step):
        # Update critics (same as DDPG but with both critics)
        # ...
        
        # Delayed policy update
        if step % self.policy_delay == 0:
            # Update actor
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            # ...
```

---

## Reward Shaping

### Concept

**Reward Shaping** designs rewards to guide learning.

**Principles:**
- Reward progress, not just final outcome
- Use potential-based shaping when possible
- Avoid reward hacking

---

## Safe Reinforcement Learning

### Constrained RL

**Safe RL** ensures agents satisfy safety constraints.

**Applications:**
- Autonomous vehicles (avoid collisions)
- Robotics (avoid dangerous actions)
- Healthcare (safe treatment policies)

### Constrained Policy Optimization (CPO)

**CPO** adds safety constraints to policy optimization.

```python
class CPO:
    def __init__(self, policy, cost_function, max_cost):
        self.policy = policy
        self.cost_function = cost_function
        self.max_cost = max_cost  # Maximum allowed cost
    
    def update(self, states, actions, rewards, costs):
        # Optimize policy subject to cost constraint
        # Use Lagrangian method or trust region
        pass
```

### Risk-Sensitive RL

**Risk-Sensitive RL** optimizes worst-case or risk-averse scenarios.

```python
class RiskSensitiveRL:
    def __init__(self, agent, risk_measure='CVaR'):
        self.agent = agent
        self.risk_measure = risk_measure  # CVaR, VaR, etc.
    
    def compute_risk(self, returns, alpha=0.1):
        """Compute Conditional Value at Risk (CVaR)"""
        sorted_returns = torch.sort(returns, descending=True)[0]
        tail_size = int(len(returns) * alpha)
        cvar = sorted_returns[:tail_size].mean()
        return cvar
```

### Shielded RL

**Shielded RL** uses external safety mechanisms.

```python
class ShieldedAgent:
    def __init__(self, agent, shield):
        self.agent = agent
        self.shield = shield  # Safety shield
    
    def act(self, state):
        action = self.agent.act(state)
        # Shield checks if action is safe
        safe_action = self.shield.filter(state, action)
        return safe_action
```

## Distributional RL

### Concept

**Distributional RL** models full return distribution instead of just mean.

**Benefits:**
- Better uncertainty estimation
- More stable learning
- Better exploration

### C51 (Categorical DQN)

**C51** models return distribution as categorical distribution.

```python
class C51DQN(nn.Module):
    def __init__(self, state_size, action_size, num_atoms=51, v_min=-10, v_max=10):
        super(C51DQN, self).__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size * num_atoms)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.action_size, self.num_atoms)
        return F.softmax(x, dim=2)  # Probability distribution
```

## Resources and Further Reading

### Important Papers

1. **"Playing Atari with Deep Reinforcement Learning"** - Mnih et al., 2013 (DQN)
2. **"Human-level control through deep reinforcement learning"** - Mnih et al., 2015
3. **"Mastering the game of Go with deep neural networks"** - Silver et al., 2016 (AlphaGo)
4. **"Mastering Chess and Shogi by Self-Play"** - Silver et al., 2017 (AlphaZero)
5. **"Proximal Policy Optimization Algorithms"** - Schulman et al., 2017 (PPO)
6. **"Trust Region Policy Optimization"** - Schulman et al., 2015 (TRPO)
7. **"Continuous control with deep reinforcement learning"** - Lillicrap et al., 2015 (DDPG)
8. **"Addressing Function Approximation Error in Actor-Critic Methods"** - Fujimoto et al., 2018 (TD3)
9. **"Hindsight Experience Replay"** - Andrychowicz et al., 2017 (HER)
10. **"Model-Agnostic Meta-Learning"** - Finn et al., 2017 (MAML)

### Books

1. **"Reinforcement Learning: An Introduction"** - Sutton & Barto
   - [Free Online](http://incompleteideas.net/book/)
   - Comprehensive RL textbook

2. **"Deep Reinforcement Learning"** - Arulkumaran et al., 2017
   - Survey paper

### Datasets and Environments

1. **OpenAI Gym**: Standard RL environments
2. **Atari 2600**: Game environments
3. **MuJoCo**: Physics simulation for robotics
4. **PyBullet**: Physics simulation
5. **Unity ML-Agents**: Game-based RL environments
6. **Procgen**: Procedurally generated environments

### Tools and Libraries

1. **OpenAI Gym/Gymnasium**: Standard RL environments
2. **Stable-Baselines3**: High-quality RL implementations
3. **Ray RLlib**: Scalable RL for production
4. **Tianshou**: PyTorch-based RL library
5. **RLLib**: Research-focused RL library
6. **Acme**: DeepMind's RL library

## Key Takeaways

1. **Multi-Agent RL**: Complex but powerful for real-world systems (MADDPG, Mean Field)
2. **Hierarchical RL**: Handles long-horizon tasks effectively (Options, HER)
3. **Imitation Learning**: Useful when rewards are hard to define (BC, DAgger, GAIL)
4. **Transfer Learning**: Accelerate learning on new tasks
5. **Continuous Actions**: Required for robotics and control (DDPG, TD3)
6. **Safe RL**: Critical for real-world deployment (CPO, Risk-Sensitive, Shielded)
7. **Meta-Learning**: Learn to learn quickly (MAML, Reptile)
8. **Distributional RL**: Model full return distribution (C51)
9. **Reward Shaping**: Design rewards to guide learning
10. **Exploration**: Critical for discovering good policies

---

**Next Steps**: Apply these concepts in [Project Tutorial](reinforcement-learning-project-tutorial.md) and build real-world RL applications.

