"""
Tutorial 16: Reinforcement Learning
===================================

This tutorial introduces reinforcement learning with PyTorch, covering
fundamental algorithms and modern deep RL techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import gym
from typing import Tuple, List, Optional
import math

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print()

# Example 1: Basic RL Environment
print("Example 1: Simple Grid World Environment")
print("=" * 50)

class GridWorld:
    """Simple grid world environment for RL"""
    def __init__(self, size=5):
        self.size = size
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        state = np.zeros((self.size, self.size))
        state[self.agent_pos[0], self.agent_pos[1]] = 1
        state[self.goal_pos[0], self.goal_pos[1]] = 2
        return state.flatten()
    
    def step(self, action):
        """Take action and return (next_state, reward, done)"""
        if self.done:
            return self._get_state(), 0, True
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        move = moves[action]
        
        # Update position
        new_pos = [self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]]
        
        # Check boundaries
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.agent_pos = new_pos
        
        # Check if goal reached
        if self.agent_pos == self.goal_pos:
            reward = 10
            self.done = True
        else:
            reward = -0.1  # Small negative reward for each step
        
        return self._get_state(), reward, self.done
    
    def render(self):
        """Visualize the environment"""
        grid = np.zeros((self.size, self.size))
        grid[self.agent_pos[0], self.agent_pos[1]] = 1
        grid[self.goal_pos[0], self.goal_pos[1]] = 2
        print(grid)
        print()

# Test the environment
env = GridWorld(size=5)
state = env.reset()
print("Initial state shape:", state.shape)
print("Initial grid:")
env.render()

# Take a few random actions
for i in range(3):
    action = np.random.randint(4)
    next_state, reward, done = env.step(action)
    print(f"Action {action}, Reward: {reward}, Done: {done}")
print()

# Example 2: Deep Q-Network (DQN)
print("Example 2: Deep Q-Network (DQN)")
print("=" * 50)

class DQN(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Experience replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent with experience replay and target network"""
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural networks
        self.q_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = ReplayBuffer()
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, next_state, reward, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, next_state, reward, done)
    
    def replay(self, batch_size=32):
        """Train the network on a batch of transitions"""
        if len(self.memory) < batch_size:
            return
        
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(device)
        action_batch = torch.LongTensor(batch.action).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(device)
        done_batch = torch.FloatTensor(batch.done).to(device)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

# Train DQN on GridWorld
env = GridWorld(size=5)
state_size = env.size * env.size
action_size = 4

agent = DQNAgent(state_size, action_size)

# Training loop
episodes = 200
scores = []
losses = []

print("Training DQN...")
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, next_state, reward, done)
        
        if len(agent.memory) > 32:
            loss = agent.replay()
            if loss:
                losses.append(loss)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    scores.append(total_reward)
    
    # Update target network
    if episode % 10 == 0:
        agent.update_target_network()
    
    if episode % 50 == 0:
        avg_score = np.mean(scores[-50:])
        print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

print()

# Example 3: Policy Gradient - REINFORCE
print("Example 3: REINFORCE Algorithm")
print("=" * 50)

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class REINFORCEAgent:
    """REINFORCE policy gradient agent"""
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.saved_log_probs = []
        self.rewards = []
        
    def act(self, state):
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.policy(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def store_reward(self, reward):
        """Store reward for current episode"""
        self.rewards.append(reward)
    
    def train(self):
        """Update policy using collected rewards"""
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate discounted returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.saved_log_probs = []
        self.rewards = []
        
        return policy_loss.item()

# Train REINFORCE
reinforce_agent = REINFORCEAgent(state_size, action_size)
scores = []

print("Training REINFORCE...")
for episode in range(200):
    state = env.reset()
    
    while True:
        action = reinforce_agent.act(state)
        next_state, reward, done = env.step(action)
        reinforce_agent.store_reward(reward)
        state = next_state
        
        if done:
            break
    
    loss = reinforce_agent.train()
    scores.append(sum(reinforce_agent.rewards))
    
    if episode % 50 == 0:
        avg_score = np.mean(scores[-50:])
        print(f"Episode {episode}, Average Score: {avg_score:.2f}")

print()

# Example 4: Actor-Critic
print("Example 4: Actor-Critic")
print("=" * 50)

class ActorCritic(nn.Module):
    """Combined Actor-Critic network"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor output (action probabilities)
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic output (state value)
        state_value = self.critic(x)
        
        return action_probs, state_value

class A2CAgent:
    """Advantage Actor-Critic agent"""
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.actor_critic = ActorCritic(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
    def act(self, state):
        """Select action and return value estimate"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_tensor)
        
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        
        return action.item()
    
    def train_step(self, state, action, reward, next_state, done):
        """Single step training update"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        # Get current predictions
        action_probs, value = self.actor_critic(state_tensor)
        
        # Get next state value
        with torch.no_grad():
            _, next_value = self.actor_critic(next_state_tensor)
            target_value = reward + self.gamma * next_value * (1 - done)
        
        # Calculate advantage
        advantage = target_value - value
        
        # Actor loss
        m = torch.distributions.Categorical(action_probs)
        actor_loss = -m.log_prob(torch.tensor(action).to(device)) * advantage.detach()
        
        # Critic loss
        critic_loss = F.mse_loss(value, target_value.detach())
        
        # Total loss
        loss = actor_loss + critic_loss
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Train A2C
a2c_agent = A2CAgent(state_size, action_size)
scores = []

print("Training A2C...")
for episode in range(200):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = a2c_agent.act(state)
        next_state, reward, done = env.step(action)
        
        loss = a2c_agent.train_step(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    scores.append(total_reward)
    
    if episode % 50 == 0:
        avg_score = np.mean(scores[-50:])
        print(f"Episode {episode}, Average Score: {avg_score:.2f}")

print()

# Example 5: Continuous Action Spaces with DDPG
print("Example 5: Deep Deterministic Policy Gradient (DDPG)")
print("=" * 50)

class Actor(nn.Module):
    """Actor network for continuous actions"""
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.max_action

class Critic(nn.Module):
    """Critic network for Q-value estimation"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDPGAgent:
    """DDPG agent for continuous control"""
    def __init__(self, state_dim, action_dim, max_action, lr_actor=3e-4, lr_critic=3e-4):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.max_action = max_action
        
    def select_action(self, state, noise=0.1):
        """Select action with exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).cpu().data.numpy()[0]
        
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -self.max_action, self.max_action)
            
        return action
    
    def train(self, replay_buffer, batch_size=64, gamma=0.99, tau=0.005):
        """Train actor and critic networks"""
        if len(replay_buffer) < batch_size:
            return
        
        # Sample batch
        transitions = replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        state = torch.FloatTensor(batch.state).to(device)
        action = torch.FloatTensor(batch.action).to(device)
        next_state = torch.FloatTensor(batch.next_state).to(device)
        reward = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(batch.done).unsqueeze(1).to(device)
        
        # Compute target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1 - done) * gamma * target_Q
        
        # Get current Q estimate
        current_Q = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q.detach())
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

print("DDPG implementation for continuous control tasks")
print("Suitable for environments like MuJoCo or robotic control")
print()

# Example 6: Proximal Policy Optimization (PPO)
print("Example 6: Proximal Policy Optimization (PPO)")
print("=" * 50)

class PPONetwork(nn.Module):
    """Network for PPO algorithm"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Action std (for continuous actions)
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        action_mean = self.action_head(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        value = self.value_head(x)
        
        return action_mean, action_std, value

class PPOAgent:
    """PPO agent implementation"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.policy = PPONetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = PPONetwork(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_mean, action_std, _ = self.policy_old(state)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        
        return action.cpu().numpy()[0], action_log_prob.cpu().numpy()[0]
    
    def update(self, states, actions, log_probs, rewards, dones):
        """Update policy using PPO algorithm"""
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        log_probs = torch.FloatTensor(log_probs).to(device)
        
        # Calculate discounted rewards
        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluate actions
            action_means, action_stds, values = self.policy(states)
            dist = torch.distributions.Normal(action_means, action_stds)
            
            new_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            
            # Calculate ratio
            ratios = torch.exp(new_log_probs - log_probs)
            
            # Calculate advantages
            advantages = discounted_rewards - values.squeeze()
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), discounted_rewards)
            entropy_loss = -0.01 * dist_entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss + entropy_loss
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

print("PPO implementation for robust policy optimization")
print("Widely used in modern RL applications")
print()

# Visualization of training progress
print("Training Progress Visualization")
print("=" * 50)

# Plot learning curves for different algorithms
plt.figure(figsize=(12, 8))

# Generate example learning curves
episodes = np.arange(200)
dqn_scores = 10 * (1 - np.exp(-episodes / 50)) + np.random.normal(0, 0.5, 200)
reinforce_scores = 8 * (1 - np.exp(-episodes / 70)) + np.random.normal(0, 0.8, 200)
a2c_scores = 9 * (1 - np.exp(-episodes / 40)) + np.random.normal(0, 0.3, 200)

plt.subplot(2, 2, 1)
plt.plot(episodes, dqn_scores, alpha=0.6)
plt.plot(episodes, np.convolve(dqn_scores, np.ones(20)/20, mode='valid'), linewidth=2)
plt.title('DQN Learning Curve')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(episodes, reinforce_scores, alpha=0.6)
plt.plot(episodes, np.convolve(reinforce_scores, np.ones(20)/20, mode='valid'), linewidth=2)
plt.title('REINFORCE Learning Curve')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(episodes, a2c_scores, alpha=0.6)
plt.plot(episodes, np.convolve(a2c_scores, np.ones(20)/20, mode='valid'), linewidth=2)
plt.title('A2C Learning Curve')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(episodes, np.convolve(dqn_scores, np.ones(20)/20, mode='valid'), label='DQN')
plt.plot(episodes, np.convolve(reinforce_scores, np.ones(20)/20, mode='valid'), label='REINFORCE')
plt.plot(episodes, np.convolve(a2c_scores, np.ones(20)/20, mode='valid'), label='A2C')
plt.title('Algorithm Comparison')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rl_learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print("Learning curves saved to 'rl_learning_curves.png'")
print()

# Summary of RL algorithms
print("Summary of RL Algorithms")
print("=" * 50)

rl_algorithms = {
    "DQN": {
        "Type": "Value-based",
        "Action Space": "Discrete",
        "Key Features": "Experience replay, target network",
        "Best For": "Atari games, discrete control"
    },
    "REINFORCE": {
        "Type": "Policy-based",
        "Action Space": "Discrete/Continuous",
        "Key Features": "Monte Carlo sampling, high variance",
        "Best For": "Simple tasks, baseline method"
    },
    "A2C": {
        "Type": "Actor-Critic",
        "Action Space": "Discrete/Continuous",
        "Key Features": "Lower variance, online learning",
        "Best For": "General purpose, faster learning"
    },
    "DDPG": {
        "Type": "Actor-Critic",
        "Action Space": "Continuous",
        "Key Features": "Deterministic policy, off-policy",
        "Best For": "Robotic control, continuous tasks"
    },
    "PPO": {
        "Type": "Policy-based",
        "Action Space": "Discrete/Continuous",
        "Key Features": "Clipped objective, stable training",
        "Best For": "Complex tasks, production use"
    }
}

for algo, details in rl_algorithms.items():
    print(f"\n{algo}:")
    for key, value in details.items():
        print(f"  {key}: {value}")

print("\nKey Takeaways:")
print("- Start with DQN for discrete actions or DDPG for continuous")
print("- Use PPO for robust performance in production")
print("- A2C provides good balance of simplicity and performance")
print("- Always consider the specific requirements of your task")