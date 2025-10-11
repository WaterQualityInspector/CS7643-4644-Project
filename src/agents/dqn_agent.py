"""
Deep Q-Network (DQN) agent implementation from scratch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from ..networks.networks import DQN
from ..utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent for learning optimal poker strategies.
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=[256, 256],
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        device='cpu'
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Frequency to update target network
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        
        # Initialize networks
        self.q_network = DQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training step counter
        self.train_step = 0
    
    def select_action(self, state, legal_actions=None, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            legal_actions: List of legal actions (None means all actions are legal)
            training: Whether in training mode
        
        Returns:
            Selected action
        """
        if legal_actions is None:
            legal_actions = list(range(self.action_dim))
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
            # Mask illegal actions
            masked_q_values = np.full(self.action_dim, -np.inf)
            masked_q_values[legal_actions] = q_values[legal_actions]
            
            return int(np.argmax(masked_q_values))
    
    def store_transition(self, state, action, next_state, reward, done):
        """
        Store a transition in replay buffer.
        """
        self.replay_buffer.push(state, action, next_state, reward, done)
    
    def train(self):
        """
        Perform one training step.
        
        Returns:
            Training loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q(s, a)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, filepath):
        """
        Save agent parameters.
        
        Args:
            filepath: Path to save file
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, filepath)
    
    def load(self, filepath):
        """
        Load agent parameters.
        
        Args:
            filepath: Path to saved file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
