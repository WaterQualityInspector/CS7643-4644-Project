"""
Simple example demonstrating the DQN agent on Kuhn Poker.
This script trains a DQN agent for a small number of episodes
and demonstrates the basic usage of the framework.
"""
import torch
from src.agents.dqn_agent import DQNAgent
from src.environment.poker_env import StudPokerEnv


def main():
    """Run a simple training example."""
    print("="*60)
    print("DQN Agent on Kuhn Poker - Simple Example")
    print("="*60)
    
    # Create environment
    env = StudPokerEnv(game_name='kuhn_poker', num_players=2)
    state_dim = env.get_observation_shape()[0]
    action_dim = env.get_action_space_size()
    
    print(f"\nEnvironment: Kuhn Poker")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 128],
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.99,
        buffer_size=1000,
        batch_size=32,
        target_update_freq=50,
        device=device
    )
    
    # Training loop
    num_episodes = 500
    print(f"\nTraining for {num_episodes} episodes...")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            # Select and perform action
            legal_actions = env.get_legal_actions()
            if state is None or not legal_actions:
                _, reward, done, info = env.step(0)
                episode_reward += reward
                if done:
                    break
                state = info.get('observation')
                continue
            
            action = agent.select_action(state, legal_actions, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            if next_state is not None:
                agent.store_transition(state, action, next_state, reward, done)
            
            # Train agent
            agent.train()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Avg Reward: {avg_reward:.3f} - "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Evaluation
    print("\n" + "="*60)
    print("Evaluating trained agent...")
    print("="*60)
    
    eval_episodes = 100
    eval_rewards = []
    
    for episode in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            legal_actions = env.get_legal_actions()
            if state is None or not legal_actions:
                _, reward, done, info = env.step(0)
                episode_reward += reward
                if done:
                    break
                state = info.get('observation')
                continue
            
            action = agent.select_action(state, legal_actions, training=False)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        eval_rewards.append(episode_reward)
    
    avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
    print(f"\nEvaluation over {eval_episodes} episodes:")
    print(f"Average Reward: {avg_eval_reward:.3f}")
    print("\nExample completed successfully!")


if __name__ == '__main__':
    main()
