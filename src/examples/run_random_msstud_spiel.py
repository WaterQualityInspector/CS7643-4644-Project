from msstud_spiel_shim import MsStudSpielGame
import random

if __name__ == "__main__":
    game = MsStudSpielGame(ante=1, seed=42)
    state = game.new_initial_state()
    ep_ret = 0.0

    while not state.is_terminal():
        acts = state.legal_actions()
        a = random.choice(acts)
        state.apply_action(a)

    ep_ret = state.returns()[0]
    print(state.history_str())
    print("episode return:", ep_ret)
