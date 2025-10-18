from msstud_spiel_shim import MsStudSpielGame
from tq2 import TabularQAgentV2
import numpy as np

def test_tabular_q_agent_v2_with_friends():
    game = MsStudSpielGame(ante=1, seed=123, friends=1)
    agent = TabularQAgentV2(game)
    agent.train(episodes=500)
    state = game.new_initial_state()
    obs = state._env.observation()
    assert "friends_cards" in obs
    # Check that the agent can select a legal action
    action = agent.policy(state)
    assert action in state.legal_actions()
    # Check that the agent's Q-table is not empty
    assert len(agent.Q) > 0
