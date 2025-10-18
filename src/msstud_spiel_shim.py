# mssud_spiel_shim.py
# A minimal OpenSpiel-style wrapper for Mississippi Stud.
# It mirrors OpenSpiel's Game/State API so you can plug agents in now,
# and later port to a true pyspiel PythonGame with minimal changes.

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

from msstud_env import MississippiStudEnv, LEGAL_ACTIONS  # your existing env

# Map our action ids to human names for readability (optional)
ACTION_NAME = {0: "FOLD", 1: "BET_1x", 2: "BET_2x", 3: "BET_3x"}

# ---------------- Game wrapper ----------------
class MsStudSpielGame:
    """Lightweight stand-in for pyspiel.Game."""
    def __init__(self, ante: int = 1, seed: Optional[int] = None, rng=None):
        self._ante = ante
        self._seed = seed
        # Use a dedicated RNG for hand generation
        if rng is not None:
            self._rng = rng
        else:
            self._rng = random.Random(seed)

    def new_initial_state(self):
        # Use a fresh random seed for each hand to ensure diversity
        hand_seed = self._rng.randint(0, 2**32 - 1)
        return MsStudSpielState(self._ante, hand_seed)

    # API mirrors pyspiel.Game minimally
    def num_players(self) -> int:
        return 1

    def max_chance_outcomes(self) -> int:
        # deck shuffles; we don't enumerate full space here
        return 52

    def __str__(self):
        return f"MsStudSpielGame(ante={self._ante})"


# ---------------- State wrapper ----------------
@dataclass
class MsStudSpielState:
    ante: int = 1
    seed: Optional[int] = None

    def __post_init__(self):
        self._env = MississippiStudEnv(ante=self.ante, seed=self.seed)
        self._state = self._env.reset()  # deals & hides community
        self._player = 0  # single-player game
        self._is_chance = False  # we fold chance into step() for simplicity

    # ----- Core OpenSpiel-like methods -----
    def current_player(self) -> int:
        """Return the current player id or -1 for chance, or -2 if terminal."""
        if self.is_terminal():
            return -2  # terminal per OpenSpiel convention
        return self._player if not self._is_chance else -1

    def is_chance_node(self) -> bool:
        return self._is_chance

    def chance_outcomes(self) -> List[Tuple[int, float]]:
        """Not enumerated; we keep shuffles internal. Return empty (no explicit chance action)."""
        return []

    def legal_actions(self, player: Optional[int] = None) -> List[int]:
        if self.is_terminal():
            return []
        # Always the same action set, per rules
        return self._env.legal_actions()

    def apply_action(self, action: int) -> None:
        """Apply a player action (Fold, 1x, 2x, 3x). Reveals as needed and settles at 5th street."""
        if self.is_terminal():
            return
        self._state = self._env.step(action)
        # In this simplified wrapper we do not expose explicit chance turns.
        # The environment handles revealing cards after bets.

    def child(self, action: int) -> "MsStudSpielState":
        """Functional-style step (clone + apply)."""
        import copy
        nxt = copy.deepcopy(self)
        nxt.apply_action(action)
        return nxt

    # ----- Episode status & returns -----
    def is_terminal(self) -> bool:
        return self._state.terminal

    def returns(self) -> List[float]:
        """OpenSpiel expects a list: one return per player (episodic)."""
        # We define reward as net profit at the end of the hand; 0 else.
        return [self._state.last_reward if self.is_terminal() else 0.0]

    # ----- Observations / information -----
    def observation_string(self, player: int = 0) -> str:
        o = self._env.observation()
        # Compact, deterministic string; good for logging / tabular keys
        hole = "-".join(sorted(o["hole"]))
        comm = "-".join(sorted(o["community"]))
        bets = ",".join(map(str, o["placed_bets"]))
        return f"r{o['round']}|h:{hole}|c:{comm}|b:{bets}"

    def information_state_string(self, player: int = 0) -> str:
        # Same as observation for a single-player perfect-information view
        return self.observation_string(player)

    def observation_tensor(self, player: int = 0) -> List[float]:
        """
        Very simple featureization to get you going:
        - round index one-hot (4)
        - ranks seen histogram 2..14 (13 bins)
        - suits seen histogram C,D,H,S (4 bins)
        - last bet choice one-hot (4)
        Total length: 25
        """
        o = self._env.observation()
        rnd = [0.0]*4
        rnd[o["round"]] = 1.0

        # Encode seen cards (hole + revealed community)
        ranks = [0.0]*13
        suits = {"C":0.0,"D":0.0,"H":0.0,"S":0.0}
        def rank_int(cs):
            r = cs[0]
            return "23456789TJQKA".index(r)  # 0..12
        for cs in o["hole"] + o["community"]:
            ranks[rank_int(cs)] += 1.0
            suits[cs[1]] += 1.0

        last = [0.0]*4
        if o["placed_bets"]:
            last[o["placed_bets"][-1]] = 1.0  # index matches {0,1,2,3}? We store 1/2/3; 0 for fold unused mid-hand
        return rnd + ranks + [suits["C"], suits["D"], suits["H"], suits["S"]]

    # ----- Debug helpers -----
    def history_str(self) -> str:
        o = self._env.observation()
        acts = [ACTION_NAME.get(a, str(a)) for a in o["placed_bets"]]
        return f"actions={acts} | cards={o['hole']} + {o['community']} | terminal={o['terminal']} | last_reward={o['last_reward']}"
