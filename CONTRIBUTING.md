# Contributing to CS7643-4644-Project

Thank you for your interest in contributing to this educational deep reinforcement learning project! This document provides guidelines for contributing.

## Project Goals

This project is designed for **educational purposes** to demonstrate:
- Deep Reinforcement Learning from scratch
- PyTorch implementation of RL algorithms
- Integration with OpenSpiel game environments
- Best practices in ML project organization

## How to Contribute

### 1. Code Contributions

We welcome contributions that:
- Implement additional RL algorithms (PPO, A3C, Rainbow DQN, etc.)
- Add new poker variants or game environments
- Improve existing implementations
- Add visualization tools
- Enhance documentation

### 2. Documentation

Help improve our documentation by:
- Adding code examples
- Creating tutorials
- Fixing typos or clarifying explanations
- Adding docstrings to undocumented functions

### 3. Bug Reports

If you find a bug:
1. Check if it's already reported in [Issues](https://github.com/WaterQualityInspector/CS7643-4644-Project/issues)
2. Create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (OS, Python version, etc.)

### 4. Feature Requests

For new features:
1. Open an issue describing the feature
2. Explain why it would be valuable
3. Discuss implementation approach
4. Wait for approval before starting work

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/CS7643-4644-Project.git
cd CS7643-4644-Project
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Install Development Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Make Your Changes

Follow these guidelines:
- Write clean, readable code
- Add docstrings to new functions/classes
- Follow existing code style
- Add tests for new features
- Update documentation as needed

### 5. Run Tests

Before submitting:

```bash
python tests/test_basic.py
```

### 6. Commit Your Changes

```bash
git add .
git commit -m "Brief description of changes"
```

Use clear, descriptive commit messages:
- ✓ Good: "Add PPO algorithm implementation"
- ✗ Bad: "Update files"

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference any related issues
- Explain what changed and why

## Code Style Guidelines

### Python Style

- Follow PEP 8 style guide
- Use 4 spaces for indentation
- Maximum line length: 88 characters
- Use meaningful variable names

### Documentation

- Add docstrings to all public functions/classes
- Use Google-style docstrings format:

```python
def function_name(arg1, arg2):
    """
    Brief description of function.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    """
    pass
```

### Comments

- Use comments to explain *why*, not *what*
- Keep comments up-to-date with code changes
- Avoid obvious comments

## Adding New Algorithms

To add a new RL algorithm:

1. Create a new file in `src/agents/`:
   ```
   src/agents/your_algorithm_agent.py
   ```

2. Implement the agent class:
   ```python
   class YourAlgorithmAgent:
       def __init__(self, ...):
           """Initialize agent."""
           pass
       
       def select_action(self, state, ...):
           """Select action."""
           pass
       
       def train(self):
           """Update agent."""
           pass
       
       def save(self, filepath):
           """Save agent."""
           pass
       
       def load(self, filepath):
           """Load agent."""
           pass
   ```

3. Add to `src/agents/__init__.py`

4. Create tests in `tests/test_your_algorithm.py`

5. Add example script or update existing ones

6. Update documentation

## Project Structure

When adding new features, maintain the project structure:

```
CS7643-4644-Project/
├── src/
│   ├── agents/          # RL agent implementations
│   ├── networks/        # Neural network architectures
│   ├── environment/     # Game environment wrappers
│   └── utils/          # Utility functions
├── tests/              # Test files
├── config/             # Configuration files
├── train.py            # Training script
├── evaluate.py         # Evaluation script
└── example.py          # Simple examples
```

## Testing

### Writing Tests

- Add tests for new features
- Follow existing test patterns
- Test edge cases
- Ensure tests pass before submitting

### Running Tests

```bash
# Run all tests
python tests/test_basic.py

# Test specific module
python -c "from tests.test_basic import test_networks; test_networks()"
```

## Educational Focus

Remember this is an educational project:
- Prioritize code clarity over optimization
- Add comments explaining RL concepts
- Include references to papers/resources
- Make it easy for learners to understand

## Questions?

- Open an issue for questions
- Check existing documentation
- Review code examples in `example.py`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in the README and release notes.

Thank you for helping make this project better for learners! 🎓
