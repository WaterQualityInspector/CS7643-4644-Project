import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from msstud_spiel_shim import MsStudSpielGame
from ppo_agent import PPOAgent
import numpy as np

def test_ppo_agent_runs():

    game = MsStudSpielGame(ante=1, seed=123)
    agent = PPOAgent(game, lr=3e-4, gamma=1.0, clip=0.2)
    agent.train(episodes=300)
    returns = []
    for _ in range(20):
        state = game.new_initial_state()
        while not state.is_terminal():
            action = agent.policy(state)
            state.apply_action(action)
        returns.append(state.returns()[0])
    avg_ev = np.mean(returns)
    print(f"PPO agent test avg EV: {avg_ev:.4f}")
    # Should not crash and should produce a reasonable EV
    assert isinstance(avg_ev, float)
    assert -10.0 < avg_ev < 5.0
