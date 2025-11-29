# tests/test_msstud_env_payouts.py
from msstud_env import MississippiStudEnv, PAYOUTS

def rig_and_run(env, hole, community3, bets):
    """
    Force a specific final hand:
      - set hole cards
      - load 3 community cards into hidden_community (revealed one per bet)
      - play the provided 3 betting actions (integers 1/2/3)
    """
    s = env.reset()
    st = env.state
    st.hole = hole[:]                # force player hole cards
    st.community = []                # nothing revealed yet
    st.hidden_community = community3[:]  # reveal via step()
    st.placed_bets = []
    st.round_ix = 0
    st.terminal = False
    for a in bets:
        s = env.step(a)
    return s

def test_royal_flush_500_to_1_on_all_wagers():
    env = MississippiStudEnv(ante=1, seed=77)
    s = rig_and_run(env, hole=["AS","KS"], community3=["QS","JS","TS"], bets=[3,3,3])
    total = 1 + 3 + 3 + 3
    assert s.terminal
    assert s.last_reward == PAYOUTS["ROYAL_FLUSH"] * total

def test_straight_flush_100_to_1():
    env = MississippiStudEnv(ante=2, seed=1)
    s = rig_and_run(env, hole=["9S","8S"], community3=["7S","6S","5S"], bets=[1,1,1])
    total = 2 + 2 + 2 + 2
    assert s.last_reward == PAYOUTS["STRAIGHT_FLUSH"] * total

def test_four_of_a_kind_40_to_1():
    env = MississippiStudEnv(ante=1, seed=1)
    s = rig_and_run(env, hole=["AH","AD"], community3=["AS","AC","2D"], bets=[1,1,1])
    total = 1 + 1 + 1 + 1
    assert s.last_reward == PAYOUTS["FOUR_KIND"] * total

def test_full_house_10_to_1():
    env = MississippiStudEnv(ante=1, seed=1)
    s = rig_and_run(env, hole=["AH","AD"], community3=["AS","KD","KC"], bets=[2,2,2])
    total = 1 + 2 + 2 + 2
    assert s.last_reward == PAYOUTS["FULL_HOUSE"] * total

def test_flush_6_to_1():
    env = MississippiStudEnv(ante=1, seed=1)
    s = rig_and_run(env, hole=["2S","8S"], community3=["4S","9S","KS"], bets=[1,2,3])
    total = 1 + 1 + 2 + 3
    assert s.last_reward == PAYOUTS["FLUSH"] * total

def test_straight_4_to_1():
    env = MississippiStudEnv(ante=1, seed=1)
    s = rig_and_run(env, hole=["9C","TC"], community3=["JD","QH","KS"], bets=[1,1,1])
    total = 1 + 1 + 1 + 1
    assert s.last_reward == PAYOUTS["STRAIGHT"] * total

def test_trips_3_to_1():
    env = MississippiStudEnv(ante=1, seed=1)
    s = rig_and_run(env, hole=["AH","AD"], community3=["AS","2C","3D"], bets=[1,1,1])
    total = 1 + 1 + 1 + 1
    assert s.last_reward == PAYOUTS["TRIPS"] * total

def test_two_pair_2_to_1():
    env = MississippiStudEnv(ante=2, seed=1)
    s = rig_and_run(env, hole=["AH","AD"], community3=["KC","KD","2S"], bets=[1,1,1])
    total = 2 + 2 + 2 + 2
    assert s.last_reward == PAYOUTS["TWO_PAIR"] * total

def test_pair_jacks_plus_1_to_1():
    env = MississippiStudEnv(ante=1, seed=1)
    s = rig_and_run(env, hole=["JH","2D"], community3=["JS","5C","9H"], bets=[1,1,1])
    total = 1 + 1 + 1 + 1
    assert s.last_reward == PAYOUTS["JACKS_PLUS"] * total

def test_push_pair_6_to_10_returns_all():
    env = MississippiStudEnv(ante=5, seed=1)
    s = rig_and_run(env, hole=["8H","2D"], community3=["8S","5C","9H"], bets=[3,1,2])
    # Push returns all wagers -> net 0
    assert s.last_reward == 0.0

def test_lose_high_card_loses_all_wagers():
    env = MississippiStudEnv(ante=2, seed=1)
    s = rig_and_run(env, hole=["2H","3D"], community3=["5C","9H","KD"], bets=[1,2,3])
    total = 2 + 2 + 4 + 6
    assert s.last_reward == -total
