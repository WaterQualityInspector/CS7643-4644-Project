import time
import numpy as np
import argparse
from msstud_spiel_shim import MsStudSpielGame
from msstud_kisenwether_strategy import kisenwether_policy
from qlearning_tabular import TabularQAgent
from dqn_agent import DQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from ppo_agent import PPOAgent

def run_policy(game, policy_fn, num_hands=500):
    returns = []
    total_bet = 0.0
    hands_seen = set()
    for _ in range(num_hands):
        state = game.new_initial_state()
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

def analyze_model(name, agent, train_args, eval_policy_fn=None):
    print(f"\n=== {name} ===")
    game = MsStudSpielGame(ante=1, seed=None)
    start = time.time()
    if hasattr(agent, 'train'):
        agent.train(**train_args)
    train_time = time.time() - start
    if eval_policy_fn is None:
        eval_policy_fn = agent.policy
    metrics = run_policy(game, eval_policy_fn, num_hands=500)
    print(f"Train time: {train_time:.2f}s")
    print(f"Average EV per hand: {metrics[0]:.5f}")
    print(f"House edge (percent of ante): {metrics[1]:.2f}%")
    print(f"Average bet per hand: {metrics[2]:.2f}")
    print(f"Element of risk: {metrics[3]:.2f}%")
    print(f"Unique hands dealt: {metrics[4]}/500")
    assert metrics[4] > 0.9 * 500, f"Too few unique hands: {metrics[4]}"
    return train_time, metrics

def main():
    parser = argparse.ArgumentParser(description="Compare Mississippi Stud RL agents and strategies.")
    parser.add_argument('--num_trials', type=int, default=10000,
                        help='Number of simulation rounds per agent (recommended: 10,000+)')
    args = parser.parse_args()
    num_trials = args.num_trials

    print(f"Running analysis with {num_trials} rounds per agent...")

    game = MsStudSpielGame(ante=1, seed=None)
    # Kisenwether (no training)
    print("\n=== Kisenwether Strategy ===")
    k_metrics = run_policy(game, kisenwether_policy, num_hands=num_trials)
    print(f"Average EV per hand: {k_metrics[0]:.5f}")
    print(f"House edge (percent of ante): {k_metrics[1]:.2f}%")
    print(f"Average bet per hand: {k_metrics[2]:.2f}")
    print(f"Element of risk: {k_metrics[3]:.2f}%")
    print(f"Unique hands dealt: {k_metrics[4]}/{num_trials}")
    assert k_metrics[4] > 0.9 * num_trials, f"Too few unique hands: {k_metrics[4]}"

    # Tabular Q-Learning improvements
    tab_agent = TabularQAgent(game, alpha=0.1, gamma=1.0, epsilon=0.2)
    analyze_model("Tabular Q-Learning", tab_agent, {"episodes":100000})

    # DQN improvements
    dqn_agent = DQNAgent(game, lr=5e-4, gamma=1.0, epsilon=0.2, batch_size=128, double_dqn=True, prioritized_replay=True)
    analyze_model("DQN", dqn_agent, {"episodes":50000})

    # Dueling DQN improvements
    dueling_agent = DuelingDQNAgent(game, lr=5e-4, gamma=1.0, epsilon=0.2, batch_size=128, double_dqn=True, prioritized_replay=True)
    analyze_model("Dueling DQN", dueling_agent, {"episodes":50000})

    # PPO improvements
    ppo_agent = PPOAgent(game, lr=1e-4, gamma=1.0, clip=0.2, batch_size=256)
    analyze_model("PPO", ppo_agent, {"episodes":20000})

if __name__ == "__main__":
    main()
