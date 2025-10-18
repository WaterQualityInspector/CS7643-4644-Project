# Mississippi Stud RL Toolkit

This repository provides tools and simulation environments for Reinforcement Learning (RL) research on the Mississippi Stud poker game.

## Features

- Fast simulation of Mississippi Stud hands
- Optimal policy evaluation (Kisenwether strategy)
- Tabular Q-learning agent
- Deep Q-Network (DQN) agent
- Dueling DQN agent
- PPO agent
- RL environment wrappers
- Hand evaluation utilities
- Example scripts and tests
- Analysis script to compare all models

## Mississippi Stud Game Flow

- Each round, the player is dealt two hole cards and three community cards (face down).
- Bets are placed in up to three rounds (3rd, 4th, 5th street), with information revealed progressively:
  - After the first bet, one community card is revealed.
  - After the second bet, another community card is revealed.
  - After the third bet, the final card is revealed and the hand is settled.
- Legal actions at each street: FOLD, BET 1x, BET 2x, BET 3x.
- Rewards and payouts follow official Mississippi Stud rules.

## RL Environment Details

- The environment enforces progressive information revelation: only the correct number of community cards is visible at each round.
- Observations include only the player's hole cards and revealed community cards, matching the real game.
- All agents interact with the environment using legal actions and receive rewards according to the rules.
- Shuffling is robust: each hand is unique, and random seeds are managed to avoid repeats.

## Inner Workings & Progressive Information Revelation

The Mississippi Stud RL environment is designed to closely follow the rules and flow of the actual game:

- **Shuffling & Dealing:** Each round, the deck is shuffled to ensure unique hands. Five cards are dealt: two to the player (face up), and three community cards (initially hidden).
- **Progressive Information Revelation:**
  - **First Betting Round:** The player sees only their two cards and bets on the 3rd street.
  - **Second Betting Round:** The first community card is revealed. The player can bet on the 4th street.
  - **Third Betting Round:** The second community card is revealed. The player can bet on the 5th street.
  - **Final Reveal:** The last community card is revealed, and the hand is evaluated for payout.
- **Agent Observations:** At each betting round, agents receive only the information available at that stage (player cards + revealed community cards). This enforces realistic decision-making and prevents information leakage.
- **Environment Logic:** The environment tracks bets, payouts, and state transitions, ensuring agents interact with the game as in a real casino setting.

## Getting Started

### 1. Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python environment management and package installation.

#### Install uv

```
pip install uv
```

#### Create and activate environment

```
uv venv .venv
uv pip install -r requirements.txt
```

### 2. Run Simulations & RL Agents

- Run Kisenwether strategy, tabular Q, DQN, dueling DQN, and PPO agents from `src/`
- Example: Run analysis comparing all models:

```
python src/analysis_compare_models.py
```

### 3. Run Tests

```
pytest
```

## Project Structure

- `src/` - Core modules, RL agents, and analysis scripts
- `tests/` - Unit tests for all agents and logic
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## Analysis & Metrics

The analysis script compares:

- House edge, average bet, element of risk
- Training time and sample efficiency
- Unique hands dealt (shuffling check)
- Final performance of each agent

**Recommended:** For robust results, increase the number of simulation trials in `analysis_compare_models.py` (e.g., 10,000+ rounds per agent). This reduces variance and provides more reliable metrics.

**Note:** All agents and strategies are evaluated under progressive information revelation, matching the rules of Mississippi Stud.

## Model Improvement Suggestions

**Tabular Q-Learning**

- Increase exploration rate (epsilon) and decay more slowly for better policy discovery.
- Use more training episodes to allow convergence.
- Refine state representation to include more features (e.g., card ranks, suit patterns).
- Try reward shaping to guide learning toward better strategies.

**DQN**

- Tune network architecture (layers, units, regularization).
- Adjust reward scaling and normalization.
- Use prioritized experience replay for better sample efficiency.
- Experiment with different learning rates and batch sizes.
- Add action masking to prevent illegal bets.

**Dueling DQN**

- Further tune dueling architecture (value/advantage streams).
- Increase training rounds and replay buffer size.
- Use double DQN to reduce overestimation bias.
- Refine state encoding for more granular information.

**PPO**

- Increase training epochs and batch size.
- Tune clipping parameter and entropy bonus for stable learning.
- Normalize observations and rewards.
- Use a larger or deeper policy network.
- Experiment with different learning rates and GAE lambda.

**General RL Tips**

- Increase training rounds for all models.
- Use more granular state features (e.g., encode card combinations, betting history).
- Visualize learning curves to diagnose instability or overfitting.
- Compare with the optimal (Kisenwether) strategy for benchmarking.
- Implement early stopping or model selection based on validation EV.

These improvements can help reduce house edge, improve sample efficiency, and yield more realistic betting behavior.

## Best Practices & Recommendations

- For reliable results, use a large number of trials (e.g., 5000+ hands per agent) in analysis scripts.
- The environment and agents are designed to match the real casino game, with progressive information and legal actions enforced.
- All RL agents are compatible with the OpenSpiel-style API for easy experimentation and extension.

## License

MIT
