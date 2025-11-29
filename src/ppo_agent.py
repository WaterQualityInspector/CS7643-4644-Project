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
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, output_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x)


class PPOAgent:
    def __init__(
        self,
        game,
        lr=3e-4,
        gamma=1.0,
        clip=0.2,
        batch_size=64,
        epochs=4,
        entropy_coef=0.01,
    ):
        self.game = game
        self.gamma = gamma
        self.clip = clip
        self.batch_size = batch_size
        self.epochs = epochs
        self.entropy_coef = entropy_coef

        # GPU setup
        self.device = get_device()

        obs_len = len(game.new_initial_state().observation_tensor())
        self.model = ActorCritic(obs_len, 4).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []

    def select_action(self, state):
        """
        Returns:
            action (int)
            prob(a|s) under current policy (float)
        """
        legal = state.legal_actions()
        obs = (
            torch.FloatTensor(state.observation_tensor())
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            logits, _ = self.model(obs)  # (1, A)
            logits = logits[0]           # (A,)

        # Work in numpy for legal-action masking, keep probability as scalar
        logits_np = logits.cpu().numpy()
        legal_logits = np.array([logits_np[a] for a in legal])
        legal_logits = legal_logits - legal_logits.max()
        probs = np.exp(legal_logits)
        probs /= probs.sum()
        action = np.random.choice(legal, p=probs)
        # Probability of chosen action under legal-normalized distribution
        prob_chosen = float(probs[legal.index(action)])
        return int(action), prob_chosen

    def store(self, s, a, p, r, s_next, done):
        # s, s_next: observation tensors (numpy lists)
        # a: int, p: scalar prob(a|s), r: float, done: 0/1
        self.memory.append((s, a, p, r, s_next, done))

    def train(self, episodes=1000):
        for ep in range(episodes):
            state = self.game.new_initial_state()
            traj = []
            while not state.is_terminal():
                obs = state.observation_tensor()
                action, prob = self.select_action(state)
                # prev_state = state.child(action)  # unused, removed
                state.apply_action(action)
                reward = state.returns()[0] if state.is_terminal() else 0.0
                next_obs = state.observation_tensor()
                done = float(state.is_terminal())
                traj.append((obs, action, prob, reward, next_obs, done))

            # Append trajectory to memory
            self.memory.extend(traj)

            # When enough samples collected, run PPO update
            if len(self.memory) >= self.batch_size:
                self.learn()
                self.memory = []

    def learn(self):
        if len(self.memory) == 0:
            return

        # Convert memory to tensors
        s, a, old_p, r, s_next, done = zip(*self.memory)

        device = self.device
        states = torch.FloatTensor(s).to(device)         # (N, obs_dim)
        actions = torch.LongTensor(a).to(device)         # (N,)
        old_probs = torch.FloatTensor(old_p).to(device)  # (N,)
        rewards = torch.FloatTensor(r).to(device)        # (N,)
        next_states = torch.FloatTensor(s_next).to(device)  # (N,)
        dones = torch.FloatTensor(done).to(device)       # (N,)

        with torch.no_grad():
            _, values = self.model(states)         # (N, 1)
            _, next_values = self.model(next_states)

            values = values.squeeze(1)             # (N,)
            next_values = next_values.squeeze(1)   # (N,)

            # 1-step TD targets (you can upgrade to GAE later if desired)
            returns = rewards + self.gamma * next_values * (1.0 - dones)
            advantages = returns - values

            # Normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

            old_log_probs = torch.log(old_probs + 1e-8)  # (N,)

        # PPO epochs over same batch
        for _ in range(self.epochs):
            logits, values_pred = self.model(states)   # logits: (N, A), values_pred: (N,1)
            values_pred = values_pred.squeeze(1)

            # Categorical policy over full action space (no legality filtering here;
            # illegal actions never appear in 'actions', so gradients only flow to
            # those actually taken during sampling).
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)         # (N,)
            entropy = dist.entropy().mean()

            # Probability ratio
            ratio = torch.exp(log_probs - old_log_probs)  # (N,)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1.0 - self.clip, 1.0 + self.clip
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            value_loss = nn.functional.mse_loss(values_pred, returns)

            # Optional entropy bonus (for exploration)
            entropy_bonus = -self.entropy_coef * entropy

            loss = policy_loss + 0.5 * value_loss + entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def policy(self, state):
        """
        Greedy policy for evaluation (argmax over legal action probabilities).
        """
        legal = state.legal_actions()
        obs = (
            torch.FloatTensor(state.observation_tensor())
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            logits, _ = self.model(obs)  # (1, A)
            logits = logits[0].cpu().numpy()
        legal_logits = np.array([logits[a] for a in legal])
        legal_logits = legal_logits - legal_logits.max()
        probs = np.exp(legal_logits)
        probs /= probs.sum()
        return int(legal[np.argmax(probs)])


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
