# hand_eval.py
from collections import Counter

RANKS = "23456789TJQKA"
RANK_TO_INT = {r: i for i, r in enumerate(RANKS, start=2)}  # 2..14
INT_TO_RANK = {v: k for k, v in RANK_TO_INT.items()}

def card_str_to_tuple(cs):
    """'AS' -> (14, 'S'); 'TD' -> (10, 'D')"""
    r, s = cs[0], cs[1].upper()
    return (RANK_TO_INT[r.upper()], s)

def is_straight(ranks_sorted_desc):
    """Return (True, high_rank) if 5 ranks form a straight (A can be low in A-2-3-4-5)."""
    r = sorted(set(ranks_sorted_desc), reverse=True)
    if len(r) != 5:
        return (False, None)
    # normal straight
    if r[0] - r[-1] == 4:
        return (True, r[0])
    # wheel: A-2-3-4-5
    if r == [14, 5, 4, 3, 2]:
        return (True, 5)
    return (False, None)

def eval_5card(cards):
    """
    cards: list of 5 tuples (rank_int, suit_char)
    Returns: (category, tiebreaker_tuple)
      category ordering (high -> low):
        9=Royal Flush, 8=Straight Flush, 7=Four Kind,
        6=Full House, 5=Flush, 4=Straight, 3=Trips,
        2=Two Pair, 1=Pair, 0=High Card
    """
    ranks = [r for (r, s) in cards]
    suits = [s for (r, s) in cards]
    ranks_sorted = sorted(ranks, reverse=True)
    flush = len(set(suits)) == 1
    straight, straight_high = is_straight(ranks_sorted)

    # Count kinds
    c = Counter(ranks)
    counts = sorted(c.values(), reverse=True)  # e.g., [4,1], [3,2], [2,2,1], ...
    by_count_then_rank = sorted(((cnt, r) for r, cnt in c.items()),
                                key=lambda x: (x[0], x[1]), reverse=True)
    primary_ranks = [r for cnt, r in by_count_then_rank]
    kickers = sorted([r for r in ranks if r not in primary_ranks], reverse=True)

    if flush and straight:
        if straight_high == 14:  # 10-J-Q-K-A
            return (9, (14,))  # Royal
        return (8, (straight_high,))  # Straight Flush
    if counts[0] == 4:
        # Four of a kind: rank of quad, then kicker
        quad = by_count_then_rank[0][1]
        kicker = max([r for r in ranks if r != quad])
        return (7, (quad, kicker))
    if counts[0] == 3 and counts[1] == 2:
        # Full house: trip rank, then pair rank
        trip = by_count_then_rank[0][1]
        pair = by_count_then_rank[1][1]
        return (6, (trip, pair))
    if flush:
        return (5, tuple(sorted(ranks, reverse=True)))
    if straight:
        return (4, (straight_high,))
    if counts[0] == 3:
        trip = by_count_then_rank[0][1]
        remain = sorted([r for r in ranks if r != trip], reverse=True)
        return (3, (trip, *remain))
    if counts[0] == 2 and counts[1] == 2:
        pair_high = max([r for r, cnt in c.items() if cnt == 2])
        pair_low  = min([r for r, cnt in c.items() if cnt == 2])
        kicker = max([r for r, cnt in c.items() if cnt == 1])
        return (2, (pair_high, pair_low, kicker))
    if counts[0] == 2:
        pair = by_count_then_rank[0][1]
        remain = sorted([r for r in ranks if r != pair], reverse=True)
        return (1, (pair, *remain))
    return (0, tuple(sorted(ranks, reverse=True)))
