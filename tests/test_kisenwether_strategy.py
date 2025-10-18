from msstud_spiel_shim import MsStudSpielGame
from msstud_kisenwether_strategy import kisenwether_policy
import numpy as np

def test_kisenwether_ev_and_metrics():

    game = MsStudSpielGame(ante=1, seed=None)  # Use None for true randomness
    num_hands = 2000
    hands_seen = set()
    returns = []
    total_bet = 0.0
    for _ in range(num_hands):
        state = game.new_initial_state()
        # Track dealt hand for uniqueness
        obs = state._env.observation()
        hand_tuple = tuple(sorted(obs['hole'] + obs['community'] + getattr(state._env.state, 'hidden_community', [])))
        hands_seen.add(hand_tuple)
        hand_bet = 1.0  # ante
        while not state.is_terminal():
            action = kisenwether_policy(state)
            if action in [1,2,3]:
                hand_bet += action
            state.apply_action(action)
        returns.append(state.returns()[0])
        total_bet += hand_bet
    avg_ev = np.mean(returns)
    house_edge = -avg_ev / 1.0 * 100.0
    avg_bet = total_bet / num_hands
    element_of_risk = house_edge / avg_bet
    print(f"Kisenwether strategy over {num_hands} hands:")
    print(f"Average EV per hand: {avg_ev:.5f}")
    print(f"House edge (percent of ante): {house_edge:.2f}%")
    print(f"Average bet per hand: {avg_bet:.2f}")
    print(f"Element of risk: {element_of_risk:.2f}%")
    print(f"Unique hands dealt: {len(hands_seen)}")
    assert len(hands_seen) > 0.9 * num_hands, f"Too few unique hands: {len(hands_seen)}"
    assert 4.5 < house_edge < 5.5
    assert 3.3 < avg_bet < 3.9
    assert 1.2 < element_of_risk < 1.5
