# letitride_env.py
"""
Environment for the casino game *Let It Ride* using the same general style
as `msstud_env.MississippiStudEnv`.

Rules implemented are based on the CA Bureau of Gambling Control rules
document for *Let It Ride* and the "LIRX-01" base-game paytable, where the
minimum winning hand is a **pair of tens** and anything worse loses.

Key modelling choices
---------------------
- The player always starts with **three equal wagers** of size `ante`.
- On each of the first two decision rounds the player may:
    * action 0 = WITHDRAW the current bet (take it back), or
    * action 1 = LET_IT_RIDE (keep that bet in action).
- The third bet is never withdrawn (per rules).
- The hand is always played out to showdown; there is no "fold" action.
- Reward is **net profit** on the hand:
    * lose:   reward = - total_active_wager
    * win:    reward =  payout_odds[category] * total_active_wager
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from hand_eval import RANKS, card_str_to_tuple, eval_5card

# --------- Paytable (LIRX-01 from BGC rules) ---------
PAYOUTS_LIRX_01: Dict[str, int] = {
    "ROYAL_FLUSH": 1000,
    "STRAIGHT_FLUSH": 200,
    "FOUR_KIND": 50,
    "FULL_HOUSE": 11,
    "FLUSH": 8,
    "STRAIGHT": 5,
    "TRIPS": 3,
    "TWO_PAIR": 2,
    "PAIR_TENS_OR_BETTER": 1,
    "LOSE": -1,  # internal use only
}

# Actions:
# 0 = withdraw current bet ("take it back")
# 1 = let it ride (keep the bet out)
LEGAL_ACTIONS: List[int] = [0, 1]


def standard_52_deck() -> List[str]:
    suits = "CDHS"
    return [r + s for s in suits for r in RANKS]


def shuffle_deck(rng: random.Random) -> List[str]:
    deck = standard_52_deck()
    rng.shuffle(deck)
    return deck


def classify_final_letitride(cards5: List[str]) -> Tuple[str, str]:
    """
    Map a 5-card hand to a paytable key plus a human-readable name.
    Uses the same evaluator as Mississippi Stud but with Let It Ride thresholds.
    """
    cat, tb = eval_5card([card_str_to_tuple(c) for c in cards5])

    if cat == 9:
        return ("ROYAL_FLUSH", "Royal Flush")
    if cat == 8:
        return ("STRAIGHT_FLUSH", "Straight Flush")
    if cat == 7:
        return ("FOUR_KIND", "Four of a Kind")
    if cat == 6:
        return ("FULL_HOUSE", "Full House")
    if cat == 5:
        return ("FLUSH", "Flush")
    if cat == 4:
        return ("STRAIGHT", "Straight")
    if cat == 3:
        return ("TRIPS", "Three of a Kind")
    if cat == 2:
        return ("TWO_PAIR", "Two Pair")
    if cat == 1:
        # Pair: need to know which rank to decide 10s-or-better.
        pair_rank = tb[0]  # evaluator returns pair rank first in tie-breaker
        if pair_rank >= 10:
            return ("PAIR_TENS_OR_BETTER", "Pair (10s or Better)")
        return ("LOSE", "Pair (9s or lower)")
    # High-card and everything else lose.
    return ("LOSE", "High Card")


@dataclass
class LetItRideState:
    rng_seed: float = 0.0
    ante: float = 1.0  # size of each individual wager

    deck: List[str] = field(default_factory=list)
    hole: List[str] = field(default_factory=list)  # 3 cards
    community: List[str] = field(default_factory=list)  # up to 2 revealed
    hidden_community: List[str] = field(default_factory=list)  # remaining face-down

    # 0 = decision about Bet #1, 1 = decision about Bet #2, 2 = terminal
    round_ix: int = 0

    # active_bets[i] == True  <=> Bet i is still in action at the end
    # index 0 -> Bet #1, 1 -> Bet #2, 2 -> Bet #3 (never withdrawn by rules)
    active_bets: List[bool] = field(default_factory=lambda: [True, True, True])

    terminal: bool = False
    last_reward: float = 0.0

    def clone(self) -> "LetItRideState":
        return LetItRideState(
            rng_seed=self.rng_seed,
            ante=self.ante,
            deck=self.deck[:],
            hole=self.hole[:],
            community=self.community[:],
            hidden_community=self.hidden_community[:],
            round_ix=self.round_ix,
            active_bets=self.active_bets[:],
            terminal=self.terminal,
            last_reward=self.last_reward,
        )


class LetItRideEnv:
    """
    Single-player Let It Ride environment (player vs. house, no dealer hand).

    Rounds:
        round_ix = 0  -> player sees 3 hole cards, acts on Bet #1.
        round_ix = 1  -> reveal 1st community card, player acts on Bet #2.
        round_ix = 2  -> reveal 2nd community card, settle payouts (terminal).

    At each non-terminal round the legal actions are:
        - 0: withdraw current bet (take it back)
        - 1: let it ride (keep bet in action)

    Reward is the **net profit** at showdown (scalar float).
    """

    def __init__(
        self,
        ante: float = 1.0,
        seed: Optional[int] = None,
        paytable: Dict[str, int] = PAYOUTS_LIRX_01,
    ):
        self.ante = float(ante)
        self.rng = random.Random(seed)
        self.paytable = paytable
        self.state: Optional[LetItRideState] = None

    # ---------- Game flow ----------
    def reset(self) -> LetItRideState:
        """Shuffle, deal a new hand, and return the initial state."""
        st = LetItRideState(rng_seed=self.rng.random(), ante=self.ante)
        deck = shuffle_deck(self.rng)

        # Deal 3 hole cards to player, then 2 community cards face-down
        st.hole = [deck.pop(), deck.pop(), deck.pop()]
        hidden_two = [deck.pop(), deck.pop()]

        st.deck = deck
        st.hidden_community = hidden_two
        st.community = []
        st.round_ix = 0
        st.active_bets = [True, True, True]  # all three start active
        st.terminal = False
        st.last_reward = 0.0

        self.state = st
        return self.clone_state()

    def step(self, action: int) -> LetItRideState:
        """
        Apply an action for the current decision round.

        action == 0: withdraw Bet #round_ix+1
        action == 1: let that bet ride
        """
        st = self.state
        assert st is not None and not st.terminal, "Call reset() before step()"
        assert action in LEGAL_ACTIONS, "Illegal action id"

        # Determine which bet the player is acting on (Bet #1 or Bet #2).
        if st.round_ix not in (0, 1):
            raise RuntimeError("No decision available in terminal state")

        bet_index = st.round_ix  # 0 for Bet #1, 1 for Bet #2

        if action == 0:
            # Withdraw this bet: it will not count toward the total wager.
            st.active_bets[bet_index] = False
        elif action == 1:
            # Let it ride: leave the bet active (no state change needed).
            pass
        else:
            raise ValueError("Unknown action")

        # After the decision, reveal the next community card and maybe settle.
        if st.round_ix == 0:
            # Reveal first community card
            st.community.append(st.hidden_community.pop(0))
            st.round_ix = 1
        elif st.round_ix == 1:
            # Reveal second and final community card; hand ends here.
            st.community.append(st.hidden_community.pop(0))
            st.round_ix = 2
            st.terminal = True
            st.last_reward = self._settle_reward(st)

        self.state = st
        return self.clone_state()

    # ---------- Helpers ----------
    def legal_actions(self) -> List[int]:
        if self.state is None or self.state.terminal:
            return []
        # Decisions only happen on rounds 0 and 1.
        if self.state.round_ix in (0, 1):
            return LEGAL_ACTIONS
        return []

    def _settle_reward(self, st: LetItRideState) -> float:
        """Compute net profit at showdown."""
        assert len(st.hole) == 3 and len(st.community) == 2
        cat_key, _name = classify_final_letitride(st.hole + st.community)

        # Total amount actually left in action (after withdrawals).
        n_active = sum(1 for b in st.active_bets if b)
        total_wager = self.ante * n_active

        if cat_key == "LOSE":
            return -float(total_wager)

        odds = self.paytable[cat_key]
        return float(odds * total_wager)

    # ---------- Observation / encoding ----------
    def observation(self) -> Dict:
        st = self.state
        assert st is not None
        return {
            "round": st.round_ix,
            "hole": st.hole[:],
            "community": st.community[:],
            "hidden_community": st.hidden_community[:],
            "active_bets": st.active_bets[:],
            "legal_actions": self.legal_actions(),
            "terminal": st.terminal,
            "last_reward": st.last_reward,
        }

    def info_state_key(self) -> str:
        """
        Deterministic string for tabular Q-learning keys
        (similar spirit to MississippiStudEnv.info_state_key).
        """
        st = self.state
        assert st is not None
        hole = "-".join(sorted(st.hole))
        comm = "-".join(sorted(st.community))
        bets = ",".join("1" if b else "0" for b in st.active_bets)
        return f"r{st.round_ix}|h:{hole}|c:{comm}|b:{bets}"

    def clone_state(self) -> LetItRideState:
        assert self.state is not None
        return self.state.clone()


# ---------- Quick manual test ----------
if __name__ == "__main__":
    env = LetItRideEnv(ante=1.0, seed=42)
    n = 5
    for ep in range(n):
        s = env.reset()
        print(f"--- Hand {ep+1} ---")
        print("Hole:", s.hole)
        while not s.terminal:
            obs = env.observation()
            print(f"Round {obs['round']}, community={obs['community']}, active_bets={obs['active_bets']}")
            # Very dumb random policy: 50/50 withdraw vs let it ride
            a = random.choice(obs["legal_actions"])
            print("  -> taking action", a)
            s = env.step(a)
        final_obs = env.observation()
        print("Final hand:", final_obs["hole"], "+", final_obs["community"])
        print("Reward:", s.last_reward)
        print()
