# simulate.py
from msstud_env import MississippiStudEnv
import random

def random_episode(env, seed=None):
    rng = random.Random(seed)
    s = env.reset()
    total = 0.0
    while not s.terminal:
        obs = env.observation()
        a = rng.choice(obs["legal_actions"])  # random fold or 1x/2x/3x
        s = env.step(a)
        if s.terminal:
            total += s.last_reward
    return total

if __name__ == "__main__":
    env = MississippiStudEnv(ante=1, seed=42)
    N = 1000
    net = 0.0
    for i in range(N):
        net += random_episode(env, seed=i)
    print(f"Random policy avg EV per hand over {N}: {net/N:.4f}")
