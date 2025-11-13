import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class DistributionalDQN(nn.Module):
    def __init__(self, input_dim, action_dim, atom_size=51, v_min=-10, v_max=10):
        super().__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, atom_size)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim * atom_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.action_dim, self.atom_size)
        prob = torch.softmax(x, dim=2)
        return prob

class DistributionalDQNAgent:
    def __init__(self, game, lr=5e-4, gamma=0.99, epsilon=0.2, batch_size=128, memory_size=10000, atom_size=51, v_min=-10, v_max=10, min_epsilon=0.01, epsilon_decay=0.9995, target_update=50):
        self.game = game
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = []
        self.memory_size = memory_size
        self.target_update = target_update
        obs_len = len(game.new_initial_state().observation_tensor())
        self.model = DistributionalDQN(obs_len, 4, atom_size, v_min, v_max)
        self.target_model = DistributionalDQN(obs_len, 4, atom_size, v_min, v_max)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, atom_size)
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        legal = state.legal_actions()
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal)
        obs = torch.FloatTensor(state.observation_tensor()).unsqueeze(0)
        with torch.no_grad():
            prob = self.model(obs)
            qvals = torch.sum(prob * self.support, dim=2)
            qvals = qvals[0].cpu().numpy()
        legal_qvals = [qvals[a] for a in legal]
        return legal[int(np.argmax(legal_qvals))]

    def store(self, s, a, r, s_next, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((s, a, r, s_next, done))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        s = torch.FloatTensor([b[0] for b in batch])
        a = torch.LongTensor([b[1] for b in batch])
        r = torch.FloatTensor([b[2] for b in batch])
        s_next = torch.FloatTensor([b[3] for b in batch])
        done = torch.FloatTensor([b[4] for b in batch])
        return s, a, r, s_next, done

    def projection_distribution(self, next_prob, rewards, dones):
        batch_size = rewards.size(0)
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        support = torch.linspace(self.v_min, self.v_max, self.atom_size)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        Tz = rewards + self.gamma * support * (1 - dones)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        proj_dist = torch.zeros(next_prob.size())
        for i in range(batch_size):
            for j in range(self.atom_size):
                lj = l[i, j]
                uj = u[i, j]
                m = next_prob[i, :, j]
                if lj == uj:
                    proj_dist[i, :, lj] += m
                else:
                    proj_dist[i, :, lj] += m * (uj - b[i, j])
                    proj_dist[i, :, uj] += m * (b[i, j] - lj)
        return proj_dist

    def learn(self):
        s, a, r, s_next, done = self.sample()
        prob = self.model(s)
        prob_a = prob[range(self.batch_size), a]
        with torch.no_grad():
            next_prob = self.target_model(s_next)
            next_q = torch.sum(next_prob * self.support, dim=2)
            next_a = next_q.argmax(1)
            next_prob_a = next_prob[range(self.batch_size), next_a]
            target_prob = self.projection_distribution(next_prob, r, done)
        loss = -torch.sum(target_prob * torch.log(prob_a + 1e-8), dim=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes=10000):
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
            if ep % self.target_update == 0:
                self.update_target()
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def policy(self, state):
        legal = state.legal_actions()
        obs = torch.FloatTensor(state.observation_tensor()).unsqueeze(0)
        with torch.no_grad():
            prob = self.model(obs)
            qvals = torch.sum(prob * self.support, dim=2)
            qvals = qvals[0].cpu().numpy()
        legal_qvals = [qvals[a] for a in legal]
        return legal[int(np.argmax(legal_qvals))]
