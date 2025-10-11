"""
Training script for DQN agent on Stud Poker.
"""
import argparse
import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.agents.dqn_agent import DQNAgent
from src.environment.poker_env import StudPokerEnv


def train(args):
    """
    Train DQN agent on poker environment.
    
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
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        device=device
    )
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    episode_rewards = []
    total_steps = 0
    
    for episode in tqdm(range(args.num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        losses = []
        
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
            
            action = agent.select_action(state, legal_actions, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            if next_state is not None:
                agent.store_transition(state, action, next_state, reward, done)
            
            # Train agent
            loss = agent.train()
            if loss is not None:
                losses.append(loss)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Logging
        if episode % args.log_interval == 0:
            avg_reward = sum(episode_rewards[-args.log_interval:]) / min(len(episode_rewards), args.log_interval)
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            writer.add_scalar('Train/EpisodeReward', episode_reward, episode)
            writer.add_scalar('Train/AvgReward', avg_reward, episode)
            writer.add_scalar('Train/Loss', avg_loss, episode)
            writer.add_scalar('Train/Epsilon', agent.epsilon, episode)
            writer.add_scalar('Train/EpisodeSteps', episode_steps, episode)
            
            print(f"\nEpisode {episode}/{args.num_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if episode % args.save_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"agent_episode_{episode}.pth")
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "agent_final.pth")
    agent.save(final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")
    
    writer.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Stud Poker")
    
    # Environment arguments
    parser.add_argument('--game', type=str, default='kuhn_poker',
                       help='OpenSpiel game name')
    parser.add_argument('--num_players', type=int, default=2,
                       help='Number of players')
    
    # Agent arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256],
                       help='Hidden layer dimensions')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='Initial epsilon')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                       help='Final epsilon')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                       help='Epsilon decay rate')
    parser.add_argument('--buffer_size', type=int, default=10000,
                       help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--target_update_freq', type=int, default=100,
                       help='Target network update frequency')
    
    # Training arguments
    parser.add_argument('--num_episodes', type=int, default=10000,
                       help='Number of training episodes')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Checkpoint save interval')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
