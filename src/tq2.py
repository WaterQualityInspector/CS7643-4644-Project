import numpy as np
from msstud_spiel_shim import MsStudSpielGame


def compact_state_key(state):
    """
    Build a compact, more scalable state abstraction.

    We use:
    - round index
    - sorted hole card ranks
    - sorted friend card ranks (NOT full card tuples)

    This still lets the agent react to friend information,
    but avoids exploding the state space over all exact card combinations.
    """
    obs = state._env.observation()

    round_ix = obs["round"]

    # Hole cards: store only ranks, sorted
    hole_ranks = tuple(sorted([c[0] for c in obs["hole"]]))

    # Friend cards: store only ranks, sorted
    friend_cards = obs.get("friends_cards", [])
    friend_ranks = tuple(sorted([c[0] for c in friend_cards]))

    return (round_ix, hole_ranks, friend_ranks)


class TabularQAgentV2:
    def __init__(
        self,
        game,
        alpha=0.1,
        gamma=1.0,
        epsilon=0.5,
        min_epsilon=0.05,
        epsilon_decay=0.9999,
        q_init=2.0,
    ):
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_init = q_init
        # Q[(compact_state_key, action)] = value
        self.Q = {}

    def get_key(self, state, action):
        return (compact_state_key(state), action)

    def select_action(self, state):
        legal = state.legal_actions()
        # ε-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal)
        qs = [self.Q.get(self.get_key(state, a), self.q_init) for a in legal]
        return legal[int(np.argmax(qs))]

    def update(self, state, action, reward, next_state):
        """
        One-step tabular Q-learning update with reward shaping.

        state: pre-action state
        next_state: post-action state
        """
        key = self.get_key(state, action)

        # Next-state max Q over legal actions
        if next_state.is_terminal():
            next_q = 0.0
        else:
            next_legal = next_state.legal_actions()
            next_q = max(
                [self.Q.get(self.get_key(next_state, a), self.q_init) for a in next_legal],
                default=0.0,
            )

        old_q = self.Q.get(key, self.q_init)

        # Reward shaping: bonus per round progressed (using pre-action state's round)
        obs = state._env.observation()
        round_bonus = 0.5 * obs["round"]
        shaped_reward = reward + round_bonus

        target = shaped_reward + self.gamma * next_q
        self.Q[key] = old_q + self.alpha * (target - old_q)

    def train(self, episodes=200000):
        for ep in range(episodes):
            state = self.game.new_initial_state()
            while not state.is_terminal():
                action = self.select_action(state)
                # Need prev_state because state is mutated by apply_action
                prev_state = state.child(action)
                state.apply_action(action)
                reward = state.returns()[0] if state.is_terminal() else 0.0
                self.update(prev_state, action, reward, state)

            # Decay epsilon per episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def policy(self, state):
        """
        Greedy policy (no exploration) for evaluation.
        """
        legal = state.legal_actions()
        qs = [self.Q.get(self.get_key(state, a), self.q_init) for a in legal]
        return legal[int(np.argmax(qs))]


if __name__ == "__main__":
    game = MsStudSpielGame(ante=1, seed=42)
    agent = TabularQAgentV2(game)
    agent.train(episodes=200000)
    # Evaluate learned policy
    returns = []
    for _ in range(1000):
        state = game.new_initial_state()
        while not state.is_terminal():
            action = agent.policy(state)
            state.apply_action(action)
        returns.append(state.returns()[0])
    avg_ev = np.mean(returns)
    print(f"Tabular Q v2 agent average EV per hand: {avg_ev:.4f}")
