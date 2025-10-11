"""
Evaluation script for trained DQN agent on Stud Poker.
"""
import argparse
import torch
import numpy as np
from tqdm import tqdm

from src.agents.dqn_agent import DQNAgent
from src.environment.poker_env import StudPokerEnv


def evaluate(args):
    """
    Evaluate trained DQN agent.
    
    Args:
        args: Command line arguments
    """
    # Create environment
    env = StudPokerEnv(game_name=args.game, num_players=args.num_players)
    state_dim = env.get_observation_shape()[0]
    action_dim = env.get_action_space_size()
    
    print(f"Environment: {args.game}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(f"Using device: {device}")
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        device=device
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    agent.load(args.checkpoint)
    
    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    
    for episode in tqdm(range(args.num_episodes), desc="Evaluating"):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select and perform action
            legal_actions = env.get_legal_actions()
            if state is None or not legal_actions:
                # Handle chance nodes or terminal states
                _, reward, done, info = env.step(0)
                episode_reward += reward
                if done:
                    break
                state = info.get('observation')
                continue
            
            action = agent.select_action(state, legal_actions, training=False)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if args.render and episode < args.render_episodes:
            print(f"\nEpisode {episode + 1}:")
            print(f"  Reward: {episode_reward}")
            print(f"  Length: {episode_length}")
    
    # Print statistics
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Episodes: {args.num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max Reward: {np.max(episode_rewards):.3f}")
    print(f"Min Reward: {np.min(episode_rewards):.3f}")
    print("="*50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate DQN agent on Stud Poker")
    
    # Environment arguments
    parser.add_argument('--game', type=str, default='kuhn_poker',
                       help='OpenSpiel game name')
    parser.add_argument('--num_players', type=int, default=2,
                       help='Number of players')
    
    # Agent arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256],
                       help='Hidden layer dimensions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to agent checkpoint')
    
    # Evaluation arguments
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes')
    parser.add_argument('--render_episodes', type=int, default=5,
                       help='Number of episodes to render')
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
