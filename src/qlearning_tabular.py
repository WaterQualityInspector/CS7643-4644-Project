# qlearning_tabular.py
"""
Improved Tabular Q-learning agent using information_state_string with better exploration
"""
import numpy as np
from msstud_spiel_shim import MsStudSpielGame

class TabularQAgent:
    def __init__(self, game, alpha=0.1, gamma=0.99, epsilon=0.95, min_epsilon=0.01, epsilon_decay=0.9998):
        self.game = game
        self.alpha = alpha  # Lower learning rate for stable learning
        self.gamma = gamma  # Slightly lower discount for Mississippi Stud
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.Q = {}  # key: (information_state_string, action), value: Q-value
        
        # More conservative initialization - start neutral
        self.default_q_value = 0.0
        
    def get_key(self, state, action):
        """Use information_state_string for comprehensive state representation"""
        return (state.information_state_string(), action)
    
    def select_action(self, state):
        legal = state.legal_actions()
        
        # Higher exploration rate
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal)
            
        # Get Q-values
        qs = [self.Q.get(self.get_key(state, a), self.default_q_value) for a in legal]
        
        # Choose best action with tie-breaking
        max_q = max(qs)
        best_actions = [legal[i] for i, q in enumerate(qs) if abs(q - max_q) < 1e-6]
        return np.random.choice(best_actions)
    
    def update(self, state, action, reward, next_state):
        key = self.get_key(state, action)
        
        if not next_state.is_terminal():
            next_legal = next_state.legal_actions()
            next_q = max([self.Q.get(self.get_key(next_state, a), self.default_q_value) for a in next_legal])
        else:
            next_q = 0.0
            
        old_q = self.Q.get(key, self.default_q_value)
        
        # Standard Q-learning update - no reward shaping
        self.Q[key] = old_q + self.alpha * (reward + self.gamma * next_q - old_q)
    
    def train(self, episodes=100000):
        print(f"Training tabular Q-learning for {episodes} episodes...")
        
        for ep in range(episodes):
            state = self.game.new_initial_state()
            
            while not state.is_terminal():
                action = self.select_action(state)
                
                # Create a proper copy of the current state for Q-learning update
                prev_state_key = state.information_state_string()
                
                state.apply_action(action)
                
                # Simple sparse rewards - only at terminal states
                reward = state.returns()[0] if state.is_terminal() else 0.0
                
                # Manual Q-learning update using state keys instead of state objects
                key = (prev_state_key, action)
                
                if not state.is_terminal():
                    next_legal = state.legal_actions()
                    next_q = max([self.Q.get((state.information_state_string(), a), self.default_q_value) for a in next_legal])
                else:
                    next_q = 0.0
                    
                old_q = self.Q.get(key, self.default_q_value)
                self.Q[key] = old_q + self.alpha * (reward + self.gamma * next_q - old_q)
            
            # Proper epsilon decay every episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            if ep % 10000 == 0 and ep > 0:
                # Calculate some basic stats about learning
                sample_returns = []
                for _ in range(100):
                    test_state = self.game.new_initial_state()
                    while not test_state.is_terminal():
                        action = self.policy(test_state)  # Greedy policy
                        test_state.apply_action(action)
                    sample_returns.append(test_state.returns()[0])
                avg_return = sum(sample_returns) / len(sample_returns)
                
                #print(f"Episode {ep}, epsilon: {self.epsilon:.4f}, Q-table size: {len(self.Q)}, Avg return: {avg_return:.3f}")
    
    def policy(self, state):
        legal = state.legal_actions()
        qs = [self.Q.get(self.get_key(state, a), self.default_q_value) for a in legal]
        
        max_q = max(qs)
        best_actions = [legal[i] for i, q in enumerate(qs) if abs(q - max_q) < 1e-6]
        return np.random.choice(best_actions)

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
