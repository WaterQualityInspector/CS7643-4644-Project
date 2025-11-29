import numpy as np
import matplotlib.pyplot as plt
from msstud_spiel_shim import MsStudSpielGame
from qlearning_tabular import TabularQAgent
from dqn_agent import DQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from tq2 import TabularQAgentV2
from ppo_agent import PPOAgent
from scipy.stats import ttest_ind
# If you have a distributional DQN implementation, import it here:
try:
    from distributional_dqn_agent import DistributionalDQNAgent
    HAS_DIST_DQN = True
except ImportError:
    HAS_DIST_DQN = False

MODELS = [
    ("Tabular Q-Learning", TabularQAgent, {"alpha":0.1, "gamma":1.0, "epsilon":0.2}),
    ("Tabular Q-Learning V2", TabularQAgentV2, {}),
    ("DQN (default)", DQNAgent, {"lr":5e-4, "gamma":1.0, "epsilon":0.2, "batch_size":128}),
    ("DQN (low lr)", DQNAgent, {"lr":1e-4, "gamma":1.0, "epsilon":0.2, "batch_size":128}),
    ("DQN (high batch)", DQNAgent, {"lr":5e-4, "gamma":1.0, "epsilon":0.2, "batch_size":256}),
    ("Dueling DQN (default)", DuelingDQNAgent, {"lr":5e-4, "gamma":1.0, "epsilon":0.2, "batch_size":128}),
    ("Dueling DQN (low lr)", DuelingDQNAgent, {"lr":1e-4, "gamma":1.0, "epsilon":0.2, "batch_size":128}),
    ("Dueling DQN (high batch)", DuelingDQNAgent, {"lr":5e-4, "gamma":1.0, "epsilon":0.2, "batch_size":256}),
    ("PPO", PPOAgent, {"lr":3e-4, "gamma":1.0, "clip":0.2}),
]
if HAS_DIST_DQN:
    MODELS.append(("Distributional DQN", DistributionalDQNAgent, {"lr":5e-4, "gamma":1.0, "epsilon":0.2, "batch_size":128}))

TRAIN_EPISODES = 10000
TEST_EPISODES = 2000
FRIENDS_RANGE = [0, 1, 2, 3, 4]
SEED_TRAIN = 42
SEED_TEST = 12345

# Configurable parameters
import argparse
parser = argparse.ArgumentParser(description="Train and test models with different friends settings and DQN variations.")
parser.add_argument('--train_episodes', type=int, default=TRAIN_EPISODES)
parser.add_argument('--test_episodes', type=int, default=TEST_EPISODES)
parser.add_argument('--friends_min', type=int, default=0)
parser.add_argument('--friends_max', type=int, default=4)
args = parser.parse_args()
TRAIN_EPISODES = args.train_episodes
TEST_EPISODES = args.test_episodes
FRIENDS_RANGE = list(range(args.friends_min, args.friends_max+1))

results = {}

for friends in FRIENDS_RANGE:
    results[friends] = {}
    for name, AgentClass, params in MODELS:
        print(f"Training {name} with {friends} friends...")
        train_game = MsStudSpielGame(ante=1, seed=SEED_TRAIN, friends=friends)
        if name.startswith("Tabular Q-Learning V2"):
            agent = AgentClass(train_game)
            agent.train(episodes=TRAIN_EPISODES)
        else:
            agent = AgentClass(train_game, **params)
            if hasattr(agent, 'train'):
                agent.train(episodes=TRAIN_EPISODES)
        # Test with fixed seed
        test_game = MsStudSpielGame(ante=1, seed=SEED_TEST, friends=friends)
        returns = []
        for _ in range(TEST_EPISODES):
            state = test_game.new_initial_state()
            while not state.is_terminal():
                action = agent.policy(state)
                state.apply_action(action)
            returns.append(state.returns()[0])
        avg_ev = np.mean(returns)
        std_ev = np.std(returns)
        results[friends][name] = {
            "returns": returns,
            "avg_ev": avg_ev,
            "std_ev": std_ev,
        }
        print(f"{name} | Friends: {friends} | Avg EV: {avg_ev:.4f} | Std: {std_ev:.4f}")

# Statistical tests and graphs
for friends in FRIENDS_RANGE:
    print(f"\nStatistical comparison for {friends} friends:")
    model_names = list(results[friends].keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            r1 = results[friends][model_names[i]]["returns"]
            r2 = results[friends][model_names[j]]["returns"]
            stat, pval = ttest_ind(r1, r2)
            print(f"{model_names[i]} vs {model_names[j]}: p-value = {pval:.4g}")

# Plotting
for metric in ["avg_ev", "std_ev"]:
    plt.figure(figsize=(10,7))
    for name, _, _ in MODELS:
        ys = [results[f][name][metric] for f in FRIENDS_RANGE]
        plt.plot(FRIENDS_RANGE, ys, marker='o', label=name)
    plt.xlabel("Number of Friends")
    plt.ylabel(metric)
    plt.title(f"Model Comparison: {metric} vs Number of Friends (DQN Variations)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"model_{metric}_vs_friends_dqn_variations.png")
    plt.show()

print("Analysis complete. Graphs saved as PNG files.")
