# msstud_kisenwether_strategy.py
"""
Implements Joseph Kisenwether's optimal Mississippi Stud strategy as described in the prompt.
"""
from msstud_spiel_shim import MsStudSpielState
from collections import Counter
from hand_eval import card_str_to_tuple

# Helper functions for parsing and evaluating hands
RANKS = "23456789TJQKA"
RANK_TO_INT = {r: i for i, r in enumerate(RANKS, start=2)}

# Card value for strategy points
def card_value(card):
    rank = card_str_to_tuple(card)[0]
    if rank >= 11:  # J, Q, K, A
        return 2
    elif 6 <= rank <= 10:
        return 1
    else:
        return 0

def hand_points(cards):
    return sum(card_value(c) for c in cards)

def is_pair(cards):
    return card_str_to_tuple(cards[0])[0] == card_str_to_tuple(cards[1])[0]

def is_suited(cards):
    return card_str_to_tuple(cards[0])[1] == card_str_to_tuple(cards[1])[1]

def count_suits(cards):
    return len(set(card_str_to_tuple(c)[1] for c in cards))

def count_ranks(cards):
    return [card_str_to_tuple(c)[0] for c in cards]

# Helper for straight/flush draws
def is_made_hand(cards):
    ranks = [card_str_to_tuple(c)[0] for c in cards]
    c = Counter(ranks)
    # mid pair or higher
    for r, cnt in c.items():
        if cnt >= 2 and r >= 6:
            return True
    return False

def is_low_pair(cards):
    ranks = [card_str_to_tuple(c)[0] for c in cards]
    c = Counter(ranks)
    for r, cnt in c.items():
        if cnt >= 2 and 2 <= r <= 5:
            return True
    return False

def is_flush_draw(cards, need_count):
    suits = [card_str_to_tuple(c)[1] for c in cards]
    return any(suits.count(s) >= need_count for s in set(suits))

def is_straight_draw(cards, need_count, min_high=None, gaps_allowed=0):
    ranks = sorted(set(card_str_to_tuple(c)[0] for c in cards))
    for i in range(len(ranks) - need_count + 1):
        window = ranks[i:i+need_count]
        if len(window) == need_count:
            gaps = window[-1] - window[0] - (need_count - 1)
            if gaps <= gaps_allowed:
                if min_high is None or window[-1] >= min_high:
                    return True
    return False

def kisenwether_policy(state: MsStudSpielState) -> int:
    obs = state._env.observation()
    round_ix = obs['round']
    hole = obs['hole']
    comm = obs['community']
    cards = hole + comm
    points = hand_points(cards)
    # 2 cards
    if round_ix == 0:
        # Raise 3x with any pair
        if is_pair(hole):
            return 3
        # Raise 1x with at least two points
        if points >= 2:
            return 1
        # Raise 1x with 6/5 suited
        ranks = sorted([card_str_to_tuple(c)[0] for c in hole])
        if ranks == [5,6] and is_suited(hole):
            return 1
        # Fold all others
        return 0
    # 3 cards
    if round_ix == 1:
        # Raise 3x with any made hand (mid pair or higher)
        if is_made_hand(cards):
            return 3
        # Raise 3x with royal flush draw
        suits = [card_str_to_tuple(c)[1] for c in cards]
        ranks = [card_str_to_tuple(c)[0] for c in cards]
        if len(set(suits)) == 1 and set(ranks) >= set([10,11,12,13,14]):
            return 3
        # Raise 3x with straight flush draw, no gaps, 567 or higher
        if len(set(suits)) == 1 and is_straight_draw(cards, 3, min_high=7, gaps_allowed=0):
            return 3
        # Raise 3x with straight flush draw, one gap, and at least one high card
        if len(set(suits)) == 1 and is_straight_draw(cards, 3, gaps_allowed=1) and any(r >= 11 for r in ranks):
            return 3
        # Raise 3x with straight flush draw, two gaps, and at least two high cards
        if len(set(suits)) == 1 and is_straight_draw(cards, 3, gaps_allowed=2) and sum(r >= 11 for r in ranks) >= 2:
            return 3
        # Raise 1x with any other three suited cards
        if len(set(suits)) == 1:
            return 1
        # Raise 1x with a low pair
        if is_low_pair(cards):
            return 1
        # Raise 1x with at least three points
        if points >= 3:
            return 1
        # Raise 1x with a straight draw, no gaps, 456 or higher
        if is_straight_draw(cards, 3, min_high=6, gaps_allowed=0):
            return 1
        # Raise 1x with a straight draw, one gap, and two mid cards
        if is_straight_draw(cards, 3, gaps_allowed=1) and sum(6 <= r <= 10 for r in ranks) >= 2:
            return 1
        # Fold all others
        return 0
    # 4 cards
    if round_ix == 2:
        # Raise 3x with any made hand (mid pair or higher)
        if is_made_hand(cards):
            return 3
        suits = [card_str_to_tuple(c)[1] for c in cards]
        ranks = [card_str_to_tuple(c)[0] for c in cards]
        # Raise 3x with any four to a flush
        if any(suits.count(s) == 4 for s in set(suits)):
            return 3
        # Raise 3x with four to an outside straight, 8 high or better
        if is_straight_draw(cards, 4, min_high=8, gaps_allowed=0):
            return 3
        # Raise 1x with any other straight draw
        if is_straight_draw(cards, 4):
            return 1
        # Raise 1x with a low pair
        if is_low_pair(cards):
            return 1
        # Raise 1x with at least four points
        if points >= 4:
            return 1
        # Raise 1x with three mid cards and at least one previous 3x raise
        # (Not tracked: previous 3x raise, so skip this for now)
        # Fold all others
        return 0
    # Terminal
    return 0
