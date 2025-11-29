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

        # Register as buffer so it moves with the model
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, atom_size)
        )

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim * atom_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.action_dim, self.atom_size)
        prob = torch.softmax(x, dim=2)  # (B, A, Z)
        return prob


class DistributionalDQNAgent:
    def __init__(
        self,
        game,
        lr=5e-4,
        gamma=0.99,
        epsilon=0.2,
        batch_size=128,
        memory_size=10000,
        atom_size=51,
        v_min=-10,
        v_max=10,
        min_epsilon=0.01,
        epsilon_decay=0.9995,
        target_update=50,
    ):
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

        # NOTE: do NOT redefine self.support here; use model.support
        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        legal = state.legal_actions()
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal)

        obs = torch.FloatTensor(state.observation_tensor()).unsqueeze(0)
        with torch.no_grad():
            prob = self.model(obs)  # (1, A, Z)
            device = prob.device
            support = self.model.support.to(device).view(1, 1, -1)  # (1,1,Z)
            qvals = torch.sum(prob * support, dim=2)  # (1, A)
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

    def projection_distribution(self, next_prob_a, r, done):
        """
        C51-style projection of the target distribution for the greedy next action.

        next_prob_a: (B, Z) – distribution over atoms for the greedy next action
        r:           (B,)
        done:        (B,)
        Returns:
            proj_dist: (B, Z)
        """
        device = next_prob_a.device
        batch_size, atom_size = next_prob_a.size()
        v_min, v_max = self.v_min, self.v_max

        support = self.model.support.to(device)           # (Z,)
        delta_z = (v_max - v_min) / (atom_size - 1)

        r = r.to(device).unsqueeze(1)                    # (B, 1)
        done = done.to(device).unsqueeze(1)              # (B, 1)

        # Tz_j = r + gamma * (1 - done) * z_j
        tz = r + (1.0 - done) * self.gamma * support.view(1, -1)  # (B, Z)
        tz = tz.clamp(v_min, v_max)

        b = (tz - v_min) / delta_z                       # (B, Z)
        l = b.floor().clamp(0, atom_size - 1).long()
        u = b.ceil().clamp(0, atom_size - 1).long()

        proj_dist = torch.zeros(batch_size, atom_size, device=device)

        # Flattened indices for scatter-add
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, atom_size)  # (B, Z)

        # Lower-atom contributions
        m_l = next_prob_a * (u.float() - b)             # (B, Z)
        index_l = (batch_idx * atom_size + l).view(-1)
        proj_dist.view(-1).index_add_(0, index_l, m_l.view(-1))

        # Upper-atom contributions
        m_u = next_prob_a * (b - l.float())             # (B, Z)
        index_u = (batch_idx * atom_size + u).view(-1)
        proj_dist.view(-1).index_add_(0, index_u, m_u.view(-1))

        proj_dist = proj_dist.clamp(min=1e-8)
        proj_dist = proj_dist / proj_dist.sum(dim=1, keepdim=True)

        return proj_dist  # (B, Z)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        s, a, r, s_next, done = self.sample()
        device = next(self.model.parameters()).device

        s = s.to(device)               # (B, input_dim)
        s_next = s_next.to(device)
        a = a.to(device).long()        # (B,)
        r = r.to(device).float()       # (B,)
        done = done.to(device).float() # (B,)
        batch_size = s.size(0)

        # Current policy distribution over atoms for all actions
        prob = self.model(s)  # (B, A, Z) – already softmaxed in model
        prob_a = prob[torch.arange(batch_size, device=device), a]  # (B, Z)

        with torch.no_grad():
            # Target network distribution for next states
            next_prob = self.target_model(s_next)  # (B, A, Z)
            support = self.model.support.to(device).view(1, 1, -1)  # (1, 1, Z)
            next_q = torch.sum(next_prob * support, dim=2)          # (B, A)
            next_a = next_q.argmax(dim=1)                           # (B,)
            next_prob_a = next_prob[torch.arange(batch_size, device=device), next_a]  # (B, Z)
            target_prob = self.projection_distribution(next_prob_a, r, done)          # (B, Z)

        # Cross-entropy loss between projected target and current policy for chosen action
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
            prob = self.model(obs)  # (1, A, Z)
            device = prob.device
            support = self.model.support.to(device).view(1, 1, -1)  # (1, 1, Z)
            qvals = torch.sum(prob * support, dim=2)                # (1, A)
            qvals = qvals[0].cpu().numpy()
        legal_qvals = [qvals[a] for a in legal]
        return legal[int(np.argmax(legal_qvals))]
