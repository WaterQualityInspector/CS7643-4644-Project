# tq2.py
"""
Improved Tabular Q-learning agent for Mississippi Stud with:
- Compact state abstraction
- Optimistic Q initialization
- Stronger reward shaping
- Slower epsilon decay
"""
import numpy as np
from msstud_spiel_shim import MsStudSpielGame

def compact_state_key(state):
    obs = state._env.observation()
    # Use round, sorted hole card ranks, and number of friends as state abstraction
    round_ix = obs['round']
    hole_ranks = tuple(sorted([c[0] for c in obs['hole']]))
    friends = tuple(sorted(obs.get('friends_cards', [])))
    return (round_ix, hole_ranks, friends)

class TabularQAgentV2:
    def __init__(self, game, alpha=0.1, gamma=1.0, epsilon=0.5, min_epsilon=0.05, epsilon_decay=0.9999, q_init=2.0):
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_init = q_init
        self.Q = {}  # key: (compact_state, action), value: Q-value

    def get_key(self, state, action):
        return (compact_state_key(state), action)

    def select_action(self, state):
        legal = state.legal_actions()
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal)
        qs = [self.Q.get(self.get_key(state, a), self.q_init) for a in legal]
        return legal[int(np.argmax(qs))]

    def update(self, state, action, reward, next_state):
        key = self.get_key(state, action)
        next_legal = next_state.legal_actions() if not next_state.is_terminal() else []
        next_q = max([self.Q.get(self.get_key(next_state, a), self.q_init) for a in next_legal], default=0.0)
        old_q = self.Q.get(key, self.q_init)
        # Stronger reward shaping: +0.5 per round progressed
        obs = state._env.observation()
        round_bonus = 0.5 * obs['round']
        shaped_reward = reward + round_bonus
        self.Q[key] = old_q + self.alpha * (shaped_reward + self.gamma * next_q - old_q)

    def train(self, episodes=200000):
        for ep in range(episodes):
            state = self.game.new_initial_state()
            while not state.is_terminal():
                action = self.select_action(state)
                prev_state = state.child(action)
                state.apply_action(action)
                reward = state.returns()[0] if state.is_terminal() else 0.0
                self.update(prev_state, action, reward, state)
            # Decay epsilon after each episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def policy(self, state):
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
