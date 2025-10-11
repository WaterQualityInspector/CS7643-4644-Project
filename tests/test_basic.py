"""
Basic tests for the DRL framework components.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np


def test_networks():
    """Test neural network architectures."""
    print("Testing neural networks...")
    from src.networks.networks import DQN, PolicyNetwork, ValueNetwork
    
    # Test DQN
    dqn = DQN(input_dim=10, output_dim=5, hidden_dims=[32, 32])
    x = torch.randn(4, 10)
    output = dqn(x)
    assert output.shape == (4, 5), "DQN output shape mismatch"
    print("  ✓ DQN network")
    
    # Test PolicyNetwork
    policy_net = PolicyNetwork(input_dim=10, output_dim=5, hidden_dims=[32, 32])
    output = policy_net(x)
    assert output.shape == (4, 5), "PolicyNetwork output shape mismatch"
    assert torch.allclose(output.sum(dim=1), torch.ones(4)), "PolicyNetwork output not normalized"
    print("  ✓ Policy network")
    
    # Test ValueNetwork
    value_net = ValueNetwork(input_dim=10, hidden_dims=[32, 32])
    output = value_net(x)
    assert output.shape == (4, 1), "ValueNetwork output shape mismatch"
    print("  ✓ Value network")


def test_replay_buffer():
    """Test replay buffer functionality."""
    print("Testing replay buffer...")
    from src.utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
    
    # Test standard replay buffer
    buffer = ReplayBuffer(capacity=100)
    for i in range(50):
        state = np.random.randn(10)
        action = i % 5
        next_state = np.random.randn(10)
        reward = np.random.randn()
        done = False
        buffer.push(state, action, next_state, reward, done)
    
    assert len(buffer) == 50, "ReplayBuffer size mismatch"
    
    states, actions, next_states, rewards, dones = buffer.sample(32)
    assert states.shape == (32, 10), "Sampled states shape mismatch"
    assert actions.shape == (32,), "Sampled actions shape mismatch"
    print("  ✓ Standard replay buffer")
    
    # Test prioritized replay buffer
    pri_buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
    for i in range(50):
        state = np.random.randn(10)
        action = i % 5
        next_state = np.random.randn(10)
        reward = np.random.randn()
        done = False
        pri_buffer.push(state, action, next_state, reward, done)
    
    assert len(pri_buffer) == 50, "PrioritizedReplayBuffer size mismatch"
    
    states, actions, next_states, rewards, dones, weights, indices = pri_buffer.sample(32, beta=0.4)
    assert states.shape == (32, 10), "Sampled states shape mismatch"
    assert weights.shape == (32,), "Importance weights shape mismatch"
    print("  ✓ Prioritized replay buffer")


def test_dqn_agent():
    """Test DQN agent functionality."""
    print("Testing DQN agent...")
    from src.agents.dqn_agent import DQNAgent
    
    agent = DQNAgent(
        state_dim=10,
        action_dim=5,
        hidden_dims=[32, 32],
        learning_rate=1e-3,
        gamma=0.99,
        buffer_size=100,
        batch_size=16,
        device='cpu'
    )
    
    # Test action selection
    state = np.random.randn(10)
    action = agent.select_action(state, legal_actions=[0, 1, 2], training=True)
    assert action in [0, 1, 2], "Selected action not in legal actions"
    print("  ✓ Action selection")
    
    # Test storing transitions
    for i in range(20):
        state = np.random.randn(10)
        action = i % 5
        next_state = np.random.randn(10)
        reward = np.random.randn()
        done = False
        agent.store_transition(state, action, next_state, reward, done)
    
    assert len(agent.replay_buffer) == 20, "Replay buffer size mismatch"
    print("  ✓ Transition storage")
    
    # Test training
    loss = agent.train()
    assert loss is not None, "Training should return loss"
    assert loss >= 0, "Loss should be non-negative"
    print("  ✓ Training step")


def test_environment():
    """Test environment wrapper (requires OpenSpiel)."""
    print("Testing environment wrapper...")
    try:
        import pyspiel
        from src.environment.poker_env import StudPokerEnv
        
        env = StudPokerEnv(game_name='kuhn_poker', num_players=2)
        
        # Test reset
        state = env.reset()
        assert state is not None or env.state.current_player() == pyspiel.PlayerId.CHANCE, \
            "Initial state should be valid or chance node"
        print("  ✓ Environment reset")
        
        # Test observation and action space
        obs_shape = env.get_observation_shape()
        action_size = env.get_action_space_size()
        assert len(obs_shape) == 1, "Observation should be 1D"
        assert action_size > 0, "Action space should be positive"
        print("  ✓ Observation and action space")
        
        # Test step
        legal_actions = env.get_legal_actions()
        if legal_actions:
            action = legal_actions[0]
            next_state, reward, done, info = env.step(action)
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            assert isinstance(done, bool), "Done should be boolean"
            print("  ✓ Environment step")
        
    except ImportError:
        print("  ⚠ OpenSpiel not installed, skipping environment test")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Basic Tests")
    print("="*60 + "\n")
    
    try:
        test_networks()
        print()
        test_replay_buffer()
        print()
        test_dqn_agent()
        print()
        test_environment()
        print()
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
        return True
    except Exception as e:
        print()
        print("="*60)
        print(f"Test failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
