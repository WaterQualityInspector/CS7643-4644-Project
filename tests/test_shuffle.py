from msstud_env import MississippiStudEnv
from msstud_spiel_shim import MsStudSpielGame
import numpy as np
import pytest

def test_env_hand_diversity():
    """Ensure MississippiStudEnv deals different hands each round."""
    env = MississippiStudEnv(ante=1, seed=123)
    hands = set()
    for _ in range(100):
        s = env.reset()
        hand = tuple(sorted(s.hole + s.hidden_community))
        hands.add(hand)
    # Should see high diversity (not all hands identical)
    assert len(hands) > 90, f"Too few unique hands: {len(hands)}"


def test_game_hand_diversity():
    """Ensure MsStudSpielGame deals different hands each round."""
    game = MsStudSpielGame(ante=1, seed=123)
    hands = set()
    for _ in range(100):
        s = game.new_initial_state()
        hand = tuple(sorted(s._env.state.hole + s._env.state.hidden_community))
        hands.add(hand)
    assert len(hands) > 90, f"Too few unique hands: {len(hands)}"


def test_simulate_optimal_hand_diversity():
    """Ensure simulate_optimal deals different hands each round."""
    pytest.skip("simulate_optimal.py not present in src, skipping test.")


def test_shuffle_randomness_quality():
    """Test that shuffling uses a high-quality RNG and hands are not repeated."""
    env = MississippiStudEnv(ante=1, seed=42)
    hands = set()
    for _ in range(200):
        s = env.reset()
        hand = tuple(sorted(s.hole + s.hidden_community))
        hands.add(hand)
    # Should see very high diversity
    assert len(hands) > 190, f"Randomness issue: only {len(hands)} unique hands out of 200 rounds."
