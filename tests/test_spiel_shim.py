from msstud_spiel_shim import MsStudSpielGame

def test_spiel_shim_rolls():
    g = MsStudSpielGame(ante=1, seed=0)
    s = g.new_initial_state()
    assert s.current_player() in (0, -2)
    steps = 0
    while not s.is_terminal() and steps < 10:
        a = s.legal_actions()[0]
        s.apply_action(a)
        steps += 1
    assert s.is_terminal()
    assert isinstance(s.returns(), list) and len(s.returns()) == 1
