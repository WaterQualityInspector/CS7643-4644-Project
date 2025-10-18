import pytest
from msstud_spiel_shim import MsStudSpielGame
from qlearning_tabular import TabularQAgent
from dqn_agent import DQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from ppo_agent import PPOAgent
import numpy as np

def test_env_friends_cards():
    # Test that friends cards are visible and not in river or hole
    game = MsStudSpielGame(ante=1, seed=123, friends=2)
    state = game.new_initial_state()
    obs = state._env.observation()
    friends_cards = set(obs.get("friends_cards", []))
    assert len(friends_cards) == 4
    # None of the friends' cards should be in hole or community or hidden_community
    all_player_cards = set(obs["hole"] + obs["community"] + getattr(state._env.state, 'hidden_community', []))
    assert friends_cards.isdisjoint(all_player_cards)

def test_tabular_q_agent_with_friends():
    game = MsStudSpielGame(ante=1, seed=123, friends=1)
    agent = TabularQAgent(game, alpha=0.1, gamma=1.0, epsilon=0.2)
    agent.train(episodes=100)
    # Just check it runs and can see friends_cards in obs
    state = game.new_initial_state()
    obs = state._env.observation()
    assert "friends_cards" in obs

def test_dqn_agent_with_friends():
    game = MsStudSpielGame(ante=1, seed=123, friends=1)
    agent = DQNAgent(game, lr=1e-3, gamma=1.0, epsilon=0.2)
    agent.train(episodes=50)
    state = game.new_initial_state()
    obs = state._env.observation()
    assert "friends_cards" in obs

def test_dueling_dqn_agent_with_friends():
    game = MsStudSpielGame(ante=1, seed=123, friends=1)
    agent = DuelingDQNAgent(game, lr=1e-3, gamma=1.0, epsilon=0.2)
    agent.train(episodes=50)
    state = game.new_initial_state()
    obs = state._env.observation()
    assert "friends_cards" in obs

def test_ppo_agent_with_friends():
    game = MsStudSpielGame(ante=1, seed=123, friends=1)
    agent = PPOAgent(game, lr=3e-4, gamma=1.0, clip=0.2)
    agent.train(episodes=20)
    state = game.new_initial_state()
    obs = state._env.observation()
    assert "friends_cards" in obs
