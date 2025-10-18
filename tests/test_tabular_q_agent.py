from msstud_spiel_shim import MsStudSpielGame
from qlearning_tabular import TabularQAgent
import numpy as np

def test_tabular_q_agent_runs():
    game = MsStudSpielGame(ante=1, seed=123)
    agent = TabularQAgent(game, alpha=0.1, gamma=1.0, epsilon=0.2)
    agent.train(episodes=5000)
    returns = []
    for _ in range(200):
        state = game.new_initial_state()
        while not state.is_terminal():
            action = agent.policy(state)
            state.apply_action(action)
        returns.append(state.returns()[0])
    avg_ev = np.mean(returns)
    print(f"Tabular Q agent test avg EV: {avg_ev:.4f}")
    # Should not crash and should produce a reasonable EV
    assert isinstance(avg_ev, float)
    assert -5.0 < avg_ev < 2.0
