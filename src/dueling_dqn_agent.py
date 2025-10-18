# dueling_dqn_agent.py
"""
Dueling Deep Q-Learning agent for Mississippi Stud using PyTorch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from msstud_spiel_shim import MsStudSpielGame

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals

class DuelingDQNAgent:
    def __init__(self, game, lr=5e-4, gamma=1.0, epsilon=0.2, batch_size=128, memory_size=20000, min_epsilon=0.01, epsilon_decay=0.9995, target_update=50, double_dqn=True):
        self.game = game
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = []
        self.memory_size = memory_size
        self.double_dqn = double_dqn
        self.target_update = target_update
        obs_len = len(game.new_initial_state().observation_tensor())
        self.model = DuelingDQN(obs_len, 4)
        self.target_model = DuelingDQN(obs_len, 4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.update_target()
    # (Add logic for double_dqn and prioritized_replay in future if desired)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        legal = state.legal_actions()
        obs = torch.FloatTensor(state.observation_tensor()).unsqueeze(0)
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal)
        with torch.no_grad():
            qvals = self.model(obs).cpu().numpy()[0]
        legal_qvals = [qvals[a] for a in legal]
        return legal[np.argmax(legal_qvals)]

    def store(self, s, a, r, s_next, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((s, a, r, s_next, done))

    def sample(self):
        idx = np.random.choice(len(self.memory), self.batch_size)
        batch = [self.memory[i] for i in idx]
        s = torch.FloatTensor([x[0] for x in batch])
        a = torch.LongTensor([x[1] for x in batch])
        r = torch.FloatTensor([x[2] for x in batch])
        s_next = torch.FloatTensor([x[3] for x in batch])
        done = torch.FloatTensor([x[4] for x in batch])
        return s, a, r, s_next, done

    def train(self, episodes=20000, target_update=None):
        if target_update is None:
            target_update = self.target_update
        for ep in range(episodes):
            state = self.game.new_initial_state()
            while not state.is_terminal():
                obs = state.observation_tensor()
                action = self.select_action(state)
                prev_state = state.child(action)
                state.apply_action(action)
                reward = state.returns()[0] if state.is_terminal() else 0.0
                next_obs = state.observation_tensor()
                done = float(state.is_terminal())
                self.store(obs, action, reward, next_obs, done)
                if len(self.memory) >= self.batch_size:
                    self.learn()
            if ep % target_update == 0:
                self.update_target()
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def learn(self):
        s, a, r, s_next, done = self.sample()
        qvals = self.model(s)
        qval = qvals.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.model(s_next).argmax(1)
                next_qvals = self.target_model(s_next)
                max_next_qval = next_qvals.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_qvals = self.target_model(s_next)
                max_next_qval = next_qvals.max(1)[0]
            target = r + self.gamma * max_next_qval * (1 - done)
        loss = self.loss_fn(qval, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def policy(self, state):
        legal = state.legal_actions()
        obs = torch.FloatTensor(state.observation_tensor()).unsqueeze(0)
        with torch.no_grad():
            qvals = self.model(obs).cpu().numpy()[0]
        legal_qvals = [qvals[a] for a in legal]
        return legal[np.argmax(legal_qvals)]

if __name__ == "__main__":
    game = MsStudSpielGame(ante=1, seed=42)
    agent = DuelingDQNAgent(game, lr=5e-4, gamma=1.0, epsilon=0.2)
    agent.train(episodes=20000)
    # Evaluate learned policy
    returns = []
    for _ in range(200):
        state = game.new_initial_state()
        while not state.is_terminal():
            action = agent.policy(state)
            state.apply_action(action)
        returns.append(state.returns()[0])
    avg_ev = np.mean(returns)
    print(f"Dueling DQN agent average EV per hand: {avg_ev:.4f}")
