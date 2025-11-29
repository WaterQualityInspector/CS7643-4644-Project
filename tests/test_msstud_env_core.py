# tests/test_msstud_env_core.py
from msstud_env import MississippiStudEnv

def new_env():
    return MississippiStudEnv(ante=1, seed=123)

def test_reset_and_round_flow():
    env = new_env()
    s = env.reset()
    assert s.round_ix == 0
    assert env.legal_actions() == [0,1,2,3]

    # 3rd street bet -> reveal first community
    s = env.step(1)
    assert s.round_ix == 1 and len(s.community) == 1 and not s.terminal

    # 4th street bet -> reveal second community
    s = env.step(2)
    assert s.round_ix == 2 and len(s.community) == 2 and not s.terminal

    # 5th street bet -> reveal final card & settle
    s = env.step(3)
    assert s.terminal and s.round_ix == 3 and len(s.community) == 3

def test_fold_forfeits_prior_wagers():
    env = new_env()
    s = env.reset()
    s = env.step(1)  # 3rd street bet 1x
    s = env.step(0)  # fold at 4th street
    assert s.terminal
    # Lose ante (1) + one 1x bet (1)
    assert s.last_reward == -(1 + 1)
