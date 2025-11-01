import time
import numpy as np
import argparse
from msstud_spiel_shim import MsStudSpielGame
from msstud_kisenwether_strategy import kisenwether_policy
from qlearning_tabular import TabularQAgent
from dqn_agent import DQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from ppo_agent import PPOAgent
from gpu_utils import get_device

def run_policy(game, policy_fn, num_hands=500):
    returns = []
    total_bet = 0.0
    hands_seen = set()
    for _ in range(num_hands):
        state = game.new_initial_state()
        
        # Track all unique hands for evaluation metrics
        obs = state._env.observation()
        hand_tuple = tuple(sorted(obs['hole'] + obs['community'] + getattr(state._env.state, 'hidden_community', [])))
        hands_seen.add(hand_tuple)
        
        hand_bet = 1.0
        while not state.is_terminal():
            action = policy_fn(state)
            if action in [1,2,3]:
                hand_bet += action
            state.apply_action(action)
        returns.append(state.returns()[0])
        total_bet += hand_bet
    avg_ev = np.mean(returns)
    house_edge = -avg_ev / 1.0 * 100.0
    avg_bet = total_bet / num_hands
    element_of_risk = house_edge / avg_bet
    unique_hands = len(hands_seen)
    return avg_ev, house_edge, avg_bet, element_of_risk, unique_hands

def analyze_model(name, agent, train_args, eval_policy_fn=None, eval_hands=10000):
    print(f"\n=== {name} ===")
    game = MsStudSpielGame(ante=1, seed=None)
    start = time.time()
    if hasattr(agent, 'train'):
        print(f"Training with {train_args.get('episodes', 'N/A')} episodes...")
        agent.train(**train_args)
    train_time = time.time() - start
    if eval_policy_fn is None:
        eval_policy_fn = agent.policy
    
    # Consistent evaluation with 10k hands for all agents
    metrics = run_policy(game, eval_policy_fn, num_hands=eval_hands)
    print(f"Train time: {train_time:.2f}s")
    print(f"Average EV per hand: {metrics[0]:.5f}")
    print(f"House edge (percent of ante): {metrics[1]:.2f}%")
    print(f"Average bet per hand: {metrics[2]:.2f}")
    print(f"Element of risk: {metrics[3]:.2f}%")
    print(f"Unique hands dealt: {metrics[4]}/{eval_hands}")
    return train_time, metrics

def main():
    parser = argparse.ArgumentParser(description="Compare Mississippi Stud RL agents and strategies.")
    parser.add_argument('--num_trials', type=int, default=10000,
                        help='Number of simulation rounds per agent (recommended: 10,000+)')
    args = parser.parse_args()
    num_trials = args.num_trials

    print(f"Running analysis with {num_trials} rounds per agent...")
    
    # Show device info for deep learning agents
    device = get_device()
    print(f"Deep learning agents will use: {device}")
    print()

    game = MsStudSpielGame(ante=1, seed=None)
    # Kisenwether (no training) - now consistent 10k hands
    print("\n=== Kisenwether Strategy ===")
    k_metrics = run_policy(game, kisenwether_policy, num_hands=10000)
    print(f"Average EV per hand: {k_metrics[0]:.5f}")
    print(f"House edge (percent of ante): {k_metrics[1]:.2f}%")
    print(f"Average bet per hand: {k_metrics[2]:.2f}")
    print(f"Element of risk: {k_metrics[3]:.2f}%")
    print(f"Unique hands dealt: {k_metrics[4]}/10000")
    assert k_metrics[4] > 0.9 * 10000, f"Too few unique hands: {k_metrics[4]}"

    # Tabular Q-Learning with very slow exploration decay for thorough learning
    tab_agent = TabularQAgent(game, alpha=0.2, gamma=0.95, epsilon=0.9, min_epsilon=0.005, epsilon_decay=0.9999)
    analyze_model("Tabular Q-Learning", tab_agent, {"episodes":100000})

    # DQN with aggressive exploration strategy
    dqn_agent = DQNAgent(game, lr=5e-4, gamma=1.0, epsilon=0.95, min_epsilon=0.005, epsilon_decay=0.9999, batch_size=128, double_dqn=True)
    analyze_model("DQN", dqn_agent, {"episodes":100000})

    # Dueling DQN with aggressive exploration strategy
    dueling_agent = DuelingDQNAgent(game, lr=5e-4, gamma=1.0, epsilon=0.95, min_epsilon=0.005, epsilon_decay=0.9999, batch_size=128, double_dqn=True)
    analyze_model("Dueling DQN", dueling_agent, {"episodes":100000})

    # PPO improvements with higher episodes
    ppo_agent = PPOAgent(game, lr=1e-3, gamma=1.0, clip=0.2, batch_size=256)
    analyze_model("PPO", ppo_agent, {"episodes":100000})

if __name__ == "__main__":
    main()
