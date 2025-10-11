"""
PyTorch neural network architectures for RL agents.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network for learning optimal poker strategies.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256]):
        """
        Args:
            input_dim: Dimension of state space
            output_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(DQN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: State tensor
        
        Returns:
            Q-values for each action
        """
        return self.network(x)


class PolicyNetwork(nn.Module):
    """
    Policy network for actor-critic algorithms.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256]):
        """
        Args:
            input_dim: Dimension of state space
            output_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(PolicyNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass through the policy network.
        
        Args:
            x: State tensor
        
        Returns:
            Action probabilities
        """
        features = self.shared_network(x)
        logits = self.policy_head(features)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """
    Value network for actor-critic algorithms.
    """
    def __init__(self, input_dim, hidden_dims=[256, 256]):
        """
        Args:
            input_dim: Dimension of state space
            hidden_dims: List of hidden layer dimensions
        """
        super(ValueNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the value network.
        
        Args:
            x: State tensor
        
        Returns:
            State value
        """
        return self.network(x)
