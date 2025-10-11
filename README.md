# CS7643-4644-Project
Learning Optimal Strategies in Stud Poker Variants using Deep Reinforcement Learning

## Overview

This project implements a Deep Reinforcement Learning (DRL) model for learning optimal strategies in Stud Poker variants. The implementation is built from scratch using **PyTorch** for the deep learning components and **OpenSpiel** for the game environment API.

## Features

- **Custom DQN Implementation**: Deep Q-Network agent implemented from scratch in PyTorch
- **OpenSpiel Integration**: Uses OpenSpiel API for poker game environments
- **Experience Replay**: Standard and prioritized experience replay buffers
- **Flexible Architecture**: Configurable neural network architectures
- **Training & Evaluation**: Complete training and evaluation pipelines
- **TensorBoard Logging**: Real-time training visualization

## Project Structure

```
CS7643-4644-Project/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── dqn_agent.py          # DQN agent implementation
│   ├── networks/
│   │   ├── __init__.py
│   │   └── networks.py           # Neural network architectures
│   ├── environment/
│   │   ├── __init__.py
│   │   └── poker_env.py          # OpenSpiel environment wrapper
│   └── utils/
│       ├── __init__.py
│       └── replay_buffer.py      # Experience replay buffers
├── config/
│   └── dqn_config.yaml           # Training configuration
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/WaterQualityInspector/CS7643-4644-Project.git
cd CS7643-4644-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Training

Train a DQN agent on Kuhn Poker:

```bash
python train.py --game kuhn_poker --num_episodes 10000 --log_interval 100
```

Train with custom hyperparameters:

```bash
python train.py \
  --game kuhn_poker \
  --num_episodes 20000 \
  --lr 0.0001 \
  --gamma 0.99 \
  --epsilon_start 1.0 \
  --epsilon_end 0.01 \
  --epsilon_decay 0.995 \
  --batch_size 64 \
  --buffer_size 10000 \
  --hidden_dims 256 256 \
  --use_cuda
```

### Evaluation

Evaluate a trained agent:

```bash
python evaluate.py \
  --checkpoint checkpoints/agent_final.pth \
  --game kuhn_poker \
  --num_episodes 100
```

Render episodes during evaluation:

```bash
python evaluate.py \
  --checkpoint checkpoints/agent_final.pth \
  --game kuhn_poker \
  --num_episodes 10 \
  --render \
  --render_episodes 5
```

### Monitoring Training

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

## Algorithm Details

### Deep Q-Network (DQN)

The implementation includes:

- **Q-Network**: Deep neural network that approximates the Q-function
- **Target Network**: Separate network for stable training
- **Experience Replay**: Stores and samples past experiences
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation
- **Legal Action Masking**: Ensures only valid poker actions are selected

### Network Architecture

Default architecture:
- Input layer: State dimension (varies by game)
- Hidden layers: 2 × 256 units with ReLU activation
- Output layer: Action dimension (varies by game)

### Training Process

1. Initialize Q-network and target network
2. For each episode:
   - Reset environment
   - For each step:
     - Select action using epsilon-greedy policy
     - Execute action and observe reward
     - Store transition in replay buffer
     - Sample batch from replay buffer
     - Compute TD target using target network
     - Update Q-network via gradient descent
     - Periodically update target network
3. Save trained model

## Supported Games

The implementation supports various poker variants available in OpenSpiel:

- **kuhn_poker**: Simple 3-card poker variant
- **leduc_poker**: Medium complexity poker variant
- Other OpenSpiel poker games

To use a different game:
```bash
python train.py --game leduc_poker
```

## Configuration

Training parameters can be configured via:

1. Command-line arguments (see `python train.py --help`)
2. YAML configuration file (`config/dqn_config.yaml`)

## Results

Training results and checkpoints are saved in:
- `logs/`: TensorBoard logs
- `checkpoints/`: Model checkpoints

## Educational Purpose

This project is designed for educational purposes to demonstrate:

- Deep Reinforcement Learning implementation from scratch
- Integration with game environments (OpenSpiel)
- PyTorch neural network design
- RL training and evaluation pipelines
- Best practices in ML project organization

## Contributing

This is an educational project. Contributions and improvements are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [OpenSpiel: A Framework for Reinforcement Learning in Games](https://github.com/google-deepmind/open_spiel)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

## Contact

For questions or issues, please open an issue on GitHub.
