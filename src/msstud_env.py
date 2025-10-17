# msstud_env.py
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from hand_eval import RANKS, card_str_to_tuple, eval_5card

# ---- Config per official rules (streets, push range, payouts) ----
# Payout odds apply equally to ante + each street bet; pairs 6–10 push; <=5 loses.
# Payouts: Jacks+ 1:1, TwoPair 2:1, Trips 3:1, Straight 4:1, Flush 6:1,
# FullHouse 10:1, Quads 40:1, StraightFlush 100:1, Royal 500:1
PAYOUTS = {
    "ROYAL_FLUSH": 500,
    "STRAIGHT_FLUSH": 100,
    "FOUR_KIND": 40,
    "FULL_HOUSE": 10,
    "FLUSH": 6,
    "STRAIGHT": 4,
    "TRIPS": 3,
    "TWO_PAIR": 2,
    "JACKS_PLUS": 1,  # Pair J/Q/K/A
    "PUSH_PAIR_6_10": 0,  # push
    "LOSE": -1,           # used internally
}

# Actions at each street:
# 0=FOLD, 1=BET_1x, 2=BET_2x, 3=BET_3x
LEGAL_ACTIONS = [0, 1, 2, 3]

def standard_52_deck():
    suits = "CDHS"
    return [r+s for s in suits for r in RANKS]

def shuffle_deck(rng: random.Random):
    deck = standard_52_deck()
    rng.shuffle(deck)
    return deck

def classify_final(cards5):
    """Return ('category_key', human_name) using evaluator category code."""
    cat, tb = eval_5card([card_str_to_tuple(c) for c in cards5])

    if cat == 9:  return ("ROYAL_FLUSH", "Royal Flush")
    if cat == 8:  return ("STRAIGHT_FLUSH", "Straight Flush")
    if cat == 7:  return ("FOUR_KIND", "Four of a Kind")
    if cat == 6:  return ("FULL_HOUSE", "Full House")
    if cat == 5:  return ("FLUSH", "Flush")
    if cat == 4:  return ("STRAIGHT", "Straight")
    if cat == 3:  return ("TRIPS", "Three of a Kind")
    if cat == 2:  return ("TWO_PAIR", "Two Pair")
    if cat == 1:
        pair_rank = tb[0]
        if 11 <= pair_rank <= 14:  # J Q K A
            return ("JACKS_PLUS", "Pair (Jacks+)")
        if 6 <= pair_rank <= 10:
            return ("PUSH_PAIR_6_10", "Pair (6–10)")
        return ("LOSE", "Pair (2–5)")
    return ("LOSE", "High Card / No Pair")

@dataclass
class MsStudState:
    rng_seed: int = 0
    ante: int = 1
    allow_bets: Tuple[int, int, int] = (1, 2, 3)

    deck: List[str] = field(default_factory=list)
    hole: List[str] = field(default_factory=list)           # 2 cards
    community: List[str] = field(default_factory=list)      # up to 3 revealed
    hidden_community: List[str] = field(default_factory=list)  # remaining face-down
    round_ix: int = 0   # 0=3rd street decision, 1=4th, 2=5th, 3=terminal
    placed_bets: List[int] = field(default_factory=list)  # chosen multipliers per street
    terminal: bool = False
    last_reward: float = 0.0

    def clone(self):
        return MsStudState(
            rng_seed=self.rng_seed,
            ante=self.ante,
            allow_bets=self.allow_bets,
            deck=self.deck[:],
            hole=self.hole[:],
            community=self.community[:],
            hidden_community=self.hidden_community[:],
            round_ix=self.round_ix,
            placed_bets=self.placed_bets[:],
            terminal=self.terminal,
            last_reward=self.last_reward,
        )

class MississippiStudEnv:
    """
    Single-player environment (casino rules: you vs house, no dealer hand).
    Reward defined as **net profit** for the round:
      - fold mid-round: reward = -(ante + all prior street bets already placed)
      - push (pair 6–10): reward = 0  (all wagers returned)
      - win with category C having odds o: reward = o * (ante + sum of all street bets)
      - otherwise (lose): reward = -(ante + sum bets)
    This matches table math where "pays X to 1" applies to each wager equally. 
    """

    def __init__(self, ante=1, seed: Optional[int]=None):
        self.ante = ante
        self.rng = random.Random(seed)
        self.state: Optional[MsStudState] = None

    # ---------- Game flow ----------
    def reset(self) -> MsStudState:
        st = MsStudState(rng_seed=self.rng.random(), ante=self.ante)
        st.deck = shuffle_deck(self.rng)
        # Deal: 2 hole, 3 community face-down
        st.hole = [st.deck.pop(), st.deck.pop()]
        hidden_three = [st.deck.pop(), st.deck.pop(), st.deck.pop()]
        st.hidden_community = hidden_three  # all face-down initially
        st.community = []  # revealed list grows
        st.round_ix = 0
        st.placed_bets = []
        st.terminal = False
        st.last_reward = 0.0
        self.state = st
        return self.clone_state()

    def step(self, action: int) -> MsStudState:
        st = self.state
        assert st is not None and not st.terminal
        assert action in LEGAL_ACTIONS, "Illegal action id"
        # fold
        if action == 0:
            # Forfeit ante + all prior bets
            total_bets = st.ante + sum(b * st.ante for b in st.placed_bets)
            st.last_reward = -float(total_bets)
            st.terminal = True
            st.round_ix = 3
            return self.clone_state()

        # place bet (1x/2x/3x)
        bet_mult = {1,2,3}
        if action not in {1,2,3}:
            raise ValueError("Unknown action")
        if action not in bet_mult:
            raise ValueError("Bet multiple must be 1,2,3")

        st.placed_bets.append(action)

        # Reveal a community card if we just committed at 3rd or 4th street
        if st.round_ix == 0:  # after 3rd street bet, reveal 1st community
            st.community.append(st.hidden_community.pop(0))
            st.round_ix = 1
        elif st.round_ix == 1:  # after 4th street bet, reveal 2nd community
            st.community.append(st.hidden_community.pop(0))
            st.round_ix = 2
        elif st.round_ix == 2:
            # 5th street bet → reveal final card and settle
            st.community.append(st.hidden_community.pop(0))
            st.round_ix = 3
            st.terminal = True
            st.last_reward = self._settle_reward(st)
        else:
            raise RuntimeError("Invalid round index")

        return self.clone_state()

    def legal_actions(self) -> List[int]:
        if self.state is None or self.state.terminal:
            return []
        # At each street you can FOLD or BET 1/2/3x
        return LEGAL_ACTIONS

    def _settle_reward(self, st: MsStudState) -> float:
        assert len(st.hole) == 2 and len(st.community) == 3
        cat_key, _name = classify_final(st.hole + st.community)
        total_wager = st.ante + sum(b * st.ante for b in st.placed_bets)

        if cat_key == "PUSH_PAIR_6_10":
            return 0.0
        if cat_key == "LOSE":
            return -float(total_wager)

        odds = PAYOUTS[cat_key]
        return float(odds * total_wager)

    # ---------- Observation / Encoding ----------
    def observation(self) -> Dict:
        st = self.state
        assert st is not None
        return {
            "round": st.round_ix,               # 0,1,2,3
            "hole": st.hole[:],
            "community": st.community[:],       # revealed
            "legal_actions": self.legal_actions(),
            "placed_bets": st.placed_bets[:],
            "terminal": st.terminal,
            "last_reward": st.last_reward,
        }

    def info_state_key(self) -> str:
        """Deterministic string for tabular Q keys (compact, order-invariant)."""
        st = self.state
        assert st is not None
        hole = "-".join(sorted(st.hole))
        comm = "-".join(sorted(st.community))
        bets = ",".join(map(str, st.placed_bets))
        return f"r{st.round_ix}|h:{hole}|c:{comm}|b:{bets}"

    def clone_state(self) -> MsStudState:
        return self.state.clone()
