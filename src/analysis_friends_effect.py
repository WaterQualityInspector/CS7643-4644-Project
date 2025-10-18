import time
import numpy as np
from msstud_spiel_shim import MsStudSpielGame
from qlearning_tabular import TabularQAgent
from dqn_agent import DQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from tq2 import TabularQAgentV2

def run_policy(game, policy_fn, num_hands=2000):
    returns = []
    total_bet = 0.0
    for _ in range(num_hands):
        state = game.new_initial_state()
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
    return avg_ev, house_edge, avg_bet, element_of_risk

def analyze_friends_effect(num_trials=2000, max_friends=3):
    agents = [
        ("Tabular Q-Learning", TabularQAgent, {"alpha":0.1, "gamma":1.0, "epsilon":0.2, "episodes":100000}),
        ("Tabular Q-Learning V2", TabularQAgentV2, {}),
        ("DQN", DQNAgent, {"lr":5e-4, "gamma":1.0, "epsilon":0.2, "batch_size":128, "episodes":50000}),
        ("Dueling DQN", DuelingDQNAgent, {"lr":5e-4, "gamma":1.0, "epsilon":0.2, "batch_size":128, "episodes":50000}),
    ]
    for friends in range(0, max_friends+1):
        print(f"\n=== Results with {friends} friend(s) (see {2*friends} extra cards) ===")
        for name, AgentClass, params in agents:
            print(f"\n{name}:")
            game = MsStudSpielGame(ante=1, seed=None, friends=friends)
            if name == "Tabular Q-Learning V2":
                agent = AgentClass(game)
                agent.train(episodes=100000)
            else:
                agent = AgentClass(game, **{k:v for k,v in params.items() if k != 'episodes'})
                train_args = {k:v for k,v in params.items() if k == 'episodes'}
                if hasattr(agent, 'train'):
                    agent.train(**train_args)
            metrics = run_policy(game, agent.policy, num_hands=num_trials)
            print(f"Average EV per hand: {metrics[0]:.5f}")
            print(f"House edge (percent of ante): {metrics[1]:.2f}%")
            print(f"Average bet per hand: {metrics[2]:.2f}")
            print(f"Element of risk: {metrics[3]:.2f}%")

if __name__ == "__main__":
    analyze_friends_effect(num_trials=2000, max_friends=3)
