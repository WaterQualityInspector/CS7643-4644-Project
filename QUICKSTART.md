# Quick Start Guide

This guide will help you get started with the Stud Poker Deep Reinforcement Learning project.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/WaterQualityInspector/CS7643-4644-Project.git
cd CS7643-4644-Project
```

### Step 2: Install Dependencies

#### Option A: Install with pip
```bash
pip install -r requirements.txt
```

#### Option B: Install in development mode
```bash
pip install -e .
```

### Step 3: Verify Installation

Run the test suite to verify everything is working:
```bash
python tests/test_basic.py
```

## Running a Simple Example

The easiest way to get started is to run the example script:

```bash
python example.py
```

This will:
- Train a DQN agent on Kuhn Poker for 500 episodes
- Evaluate the trained agent over 100 episodes
- Display training progress and final results

## Training Your First Agent

### Basic Training

Train a DQN agent with default settings:

```bash
python train.py --num_episodes 10000
```

### Custom Training

Train with custom hyperparameters:

```bash
python train.py \
    --game kuhn_poker \
    --num_episodes 20000 \
    --lr 0.0001 \
    --gamma 0.99 \
    --hidden_dims 256 256 \
    --batch_size 64 \
    --log_interval 100 \
    --save_interval 1000
```

### Monitor Training Progress

Open a new terminal and run TensorBoard:

```bash
tensorboard --logdir logs
```

Then open your browser to `http://localhost:6006` to see real-time training metrics.

## Evaluating a Trained Agent

After training, evaluate your agent:

```bash
python evaluate.py \
    --checkpoint checkpoints/agent_final.pth \
    --num_episodes 100
```

To see some episodes rendered:

```bash
python evaluate.py \
    --checkpoint checkpoints/agent_final.pth \
    --num_episodes 10 \
    --render \
    --render_episodes 5
```

## Understanding the Components

### 1. DQN Agent (`src/agents/dqn_agent.py`)

The core reinforcement learning agent that learns to play poker:
- Uses experience replay for stable learning
- Implements epsilon-greedy exploration
- Maintains a target network for stable Q-value updates

### 2. Neural Networks (`src/networks/networks.py`)

Three types of networks are available:
- **DQN**: Q-value approximation network
- **PolicyNetwork**: For policy-based methods
- **ValueNetwork**: For actor-critic methods

### 3. Environment Wrapper (`src/environment/poker_env.py`)

Wraps OpenSpiel poker games with a clean interface:
- Handles chance nodes automatically
- Provides legal action masking
- Returns standard (state, reward, done, info) tuples

### 4. Replay Buffer (`src/utils/replay_buffer.py`)

Two types of replay buffers:
- **ReplayBuffer**: Standard experience replay
- **PrioritizedReplayBuffer**: Prioritized experience replay

## Customizing Your Training

### Modifying Hyperparameters

Edit `config/dqn_config.yaml` or pass command-line arguments:

```yaml
# Training Configuration
learning_rate: 0.0001
gamma: 0.99
epsilon_start: 1.0
epsilon_end: 0.01
epsilon_decay: 0.995
buffer_size: 10000
batch_size: 64
```

### Changing the Game

Try different poker variants:

```bash
# Kuhn Poker (simple, 3 cards)
python train.py --game kuhn_poker

# Leduc Poker (more complex)
python train.py --game leduc_poker
```

### Adjusting Network Architecture

Modify the hidden layer dimensions:

```bash
# Larger network
python train.py --hidden_dims 512 512 256

# Deeper network
python train.py --hidden_dims 256 256 256 256
```

## Tips for Better Training

1. **Start Small**: Begin with fewer episodes to verify everything works
2. **Monitor Progress**: Use TensorBoard to track training metrics
3. **Save Often**: Use smaller `--save_interval` for important runs
4. **GPU Training**: Add `--use_cuda` flag if you have a GPU
5. **Experiment**: Try different learning rates, network sizes, and exploration rates

## Troubleshooting

### OpenSpiel Not Found

If you see "OpenSpiel not installed":

```bash
pip install open_spiel
```

Or build from source:
```bash
git clone https://github.com/google-deepmind/open_spiel.git
cd open_spiel
./install.sh
pip install -e .
```

### CUDA Out of Memory

If you run out of GPU memory:
- Reduce batch size: `--batch_size 32`
- Use smaller network: `--hidden_dims 128 128`
- Train on CPU by removing `--use_cuda` flag

### Poor Performance

If the agent isn't learning well:
- Increase training episodes: `--num_episodes 50000`
- Adjust learning rate: `--lr 0.0005`
- Increase replay buffer: `--buffer_size 50000`
- Adjust exploration: `--epsilon_decay 0.999`

## Next Steps

1. Read the full [README.md](README.md) for comprehensive documentation
2. Explore the source code in `src/` directory
3. Implement your own poker variant
4. Try different RL algorithms (PPO, A3C, etc.)
5. Experiment with different neural architectures

## Getting Help

- Check the [tests](tests/) for usage examples
- Review the example script: `example.py`
- Open an issue on GitHub for bugs or questions

Happy learning! 🎰🤖
