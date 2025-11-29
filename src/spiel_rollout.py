# A super-simple rollout driver that mirrors OpenSpiel's example style.
from typing import Callable
from msstud_spiel_shim import MsStudSpielGame

Policy = Callable[[object], int]  # takes state, returns action

def random_policy(state) -> int:
    import random
    return random.choice(state.legal_actions())

def rollout(game: MsStudSpielGame, policy: Policy, episodes=1000):
    total = 0.0
    for _ in range(episodes):
        s = game.new_initial_state()
        while not s.is_terminal():
            a = policy(s)
            s.apply_action(a)
        total += s.returns()[0]
    return total / episodes

if __name__ == "__main__":
    g = MsStudSpielGame(ante=1, seed=123)
    print("Random policy EV ≈", rollout(g, random_policy, episodes=2000))
