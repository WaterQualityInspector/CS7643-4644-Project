# ppo_agent.py
"""
Proximal Policy Optimization (PPO) agent for Mississippi Stud using PyTorch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from msstud_spiel_shim import MsStudSpielGame
from gpu_utils import get_device, move_to_device

__all__ = ["PPOAgent"]

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(128, output_dim)
        self.value_head = nn.Linear(128, 1)
    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x)

class PPOAgent:
    def __init__(self, game, lr=3e-4, gamma=1.0, clip=0.2, batch_size=64, epochs=4):
        self.game = game
        self.gamma = gamma
        self.clip = clip
        self.batch_size = batch_size
        self.epochs = epochs
        
        # GPU setup
        self.device = get_device()
        
        obs_len = len(game.new_initial_state().observation_tensor())
        self.model = ActorCritic(obs_len, 4).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []

    def select_action(self, state):
        legal = state.legal_actions()
        obs = torch.FloatTensor(state.observation_tensor()).unsqueeze(0).to(self.device)
        logits, _ = self.model(obs)
        logits = logits.detach().cpu().numpy()[0]
        legal_logits = np.array([logits[a] for a in legal])
        probs = np.exp(legal_logits - np.max(legal_logits))
        probs /= probs.sum()
        action = np.random.choice(legal, p=probs)
        return action, probs[legal.index(action)]

    def store(self, s, a, p, r, s_next, done):
        self.memory.append((s, a, p, r, s_next, done))

    def train(self, episodes=1000):
        for ep in range(episodes):
            state = self.game.new_initial_state()
            traj = []
            while not state.is_terminal():
                obs = state.observation_tensor()
                action, prob = self.select_action(state)
                prev_state = state.child(action)
                state.apply_action(action)
                reward = state.returns()[0] if state.is_terminal() else 0.0
                next_obs = state.observation_tensor()
                done = float(state.is_terminal())
                traj.append((obs, action, prob, reward, next_obs, done))
            self.memory.extend(traj)
            if len(self.memory) >= self.batch_size:
                self.learn()
                self.memory = []

    def learn(self):
        batch = self.memory
        s = torch.FloatTensor([x[0] for x in batch]).to(self.device)
        a = torch.LongTensor([x[1] for x in batch]).to(self.device)
        old_p = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        r = torch.FloatTensor([x[3] for x in batch]).to(self.device)
        s_next = torch.FloatTensor([x[4] for x in batch]).to(self.device)
        done = torch.FloatTensor([x[5] for x in batch]).to(self.device)
        _, values = self.model(s)
        _, next_values = self.model(s_next)
        returns = r + self.gamma * next_values.squeeze(1) * (1 - done)
        advantages = returns - values.squeeze(1)
        logits, _ = self.model(s)
        action_logits = logits.gather(1, a.unsqueeze(1)).squeeze(1)
        probs = torch.softmax(logits, dim=1)
        action_probs = probs.gather(1, a.unsqueeze(1)).squeeze(1)
        ratio = action_probs / old_p
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.functional.mse_loss(values.squeeze(1), returns)
        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def policy(self, state):
        legal = state.legal_actions()
        obs = torch.FloatTensor(state.observation_tensor()).unsqueeze(0).to(self.device)
        logits, _ = self.model(obs)
        logits = logits.detach().cpu().numpy()[0]
        legal_logits = np.array([logits[a] for a in legal])
        probs = np.exp(legal_logits - np.max(legal_logits))
        probs /= probs.sum()
        return legal[np.argmax(probs)]

if __name__ == "__main__":
    game = MsStudSpielGame(ante=1, seed=42)
    agent = PPOAgent(game, lr=3e-4, gamma=1.0, clip=0.2)
    agent.train(episodes=1000)
    # Evaluate learned policy
    returns = []
    for _ in range(100):
        state = game.new_initial_state()
        while not state.is_terminal():
            action = agent.policy(state)
            state.apply_action(action)
        returns.append(state.returns()[0])
    avg_ev = np.mean(returns)
    print(f"PPO agent average EV per hand: {avg_ev:.4f}")
