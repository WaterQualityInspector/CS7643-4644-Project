"""
Experience replay buffer for DQN training.
"""
import random
from collections import deque, namedtuple
import numpy as np
import torch


Transition = namedtuple('Transition', 
                       ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    """
    Experience replay buffer for off-policy RL algorithms.
    """
    
    def __init__(self, capacity=10000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether episode ended
        """
        self.buffer.append(Transition(state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Batch of transitions
        """
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        
        states = torch.FloatTensor(np.array(batch.state))
        actions = torch.LongTensor(batch.action)
        next_states = torch.FloatTensor(np.array(batch.next_state))
        rewards = torch.FloatTensor(batch.reward)
        dones = torch.FloatTensor(batch.done)
        
        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    """
    
    def __init__(self, capacity=10000, alpha=0.6):
        """
        Args:
            capacity: Maximum number of transitions to store
            alpha: Prioritization exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def push(self, state, action, next_state, reward, done):
        """
        Add a transition to the buffer.
        """
        max_priority = max(self.priorities) if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, next_state, reward, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = Transition(state, action, next_state, reward, done)
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions with prioritization.
        
        Args:
            batch_size: Number of transitions to sample
            beta: Importance sampling exponent
        
        Returns:
            Batch of transitions, importance sampling weights, and indices
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        transitions = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = Transition(*zip(*transitions))
        states = torch.FloatTensor(np.array(batch.state))
        actions = torch.LongTensor(batch.action)
        next_states = torch.FloatTensor(np.array(batch.next_state))
        rewards = torch.FloatTensor(batch.reward)
        dones = torch.FloatTensor(batch.done)
        weights = torch.FloatTensor(weights)
        
        return states, actions, next_states, rewards, dones, weights, indices
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)
