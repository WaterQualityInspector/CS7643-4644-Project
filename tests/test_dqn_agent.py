import pytest
from msstud_spiel_shim import MsStudSpielGame
from dqn_agent import DQNAgent
import numpy as np

def test_dqn_agent_runs():
    game = MsStudSpielGame(ante=1, seed=123)
    agent = DQNAgent(game, lr=1e-3, gamma=1.0, epsilon=0.2)
    agent.train(episodes=1000)
    returns = []
    for _ in range(50):
        state = game.new_initial_state()
        while not state.is_terminal():
            action = agent.policy(state)
            state.apply_action(action)
        returns.append(state.returns()[0])
    avg_ev = np.mean(returns)
    print(f"DQN agent test avg EV: {avg_ev:.4f}")
    # Should not crash and should produce a reasonable EV
    assert isinstance(avg_ev, float)
    assert -10.0 < avg_ev < 5.0
