# qlearning_tabular.py
"""
Tabular Q-learning agent for Mississippi Stud using the OpenSpiel-style environment.
"""
import numpy as np
from msstud_spiel_shim import MsStudSpielGame

class TabularQAgent:
    def __init__(self, game, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.game = game
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}  # key: (info_state, action), value: Q-value

    def get_key(self, state, action):
        return (state.information_state_string(), action)

    def select_action(self, state):
        legal = state.legal_actions()
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal)
        qs = [self.Q.get(self.get_key(state, a), 0.0) for a in legal]
        return legal[int(np.argmax(qs))]

    def update(self, state, action, reward, next_state):
        key = self.get_key(state, action)
        next_legal = next_state.legal_actions() if not next_state.is_terminal() else []
        next_q = max([self.Q.get(self.get_key(next_state, a), 0.0) for a in next_legal], default=0.0)
        old_q = self.Q.get(key, 0.0)
        self.Q[key] = old_q + self.alpha * (reward + self.gamma * next_q - old_q)

    def train(self, episodes=10000):
        for _ in range(episodes):
            state = self.game.new_initial_state()
            while not state.is_terminal():
                action = self.select_action(state)
                prev_state = state.child(action)
                state.apply_action(action)
                reward = state.returns()[0] if state.is_terminal() else 0.0
                self.update(prev_state, action, reward, state)

    def policy(self, state):
        legal = state.legal_actions()
        qs = [self.Q.get(self.get_key(state, a), 0.0) for a in legal]
        return legal[int(np.argmax(qs))]

if __name__ == "__main__":
    game = MsStudSpielGame(ante=1, seed=42)
    agent = TabularQAgent(game, alpha=0.1, gamma=1.0, epsilon=0.1)
    agent.train(episodes=20000)
    # Evaluate learned policy
    returns = []
    for _ in range(1000):
        state = game.new_initial_state()
        while not state.is_terminal():
            action = agent.policy(state)
            state.apply_action(action)
        returns.append(state.returns()[0])
    avg_ev = np.mean(returns)
    print(f"Tabular Q agent average EV per hand: {avg_ev:.4f}")
