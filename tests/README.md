# Tests

This directory contains tests for the DRL framework.

## Running Tests

To run the basic tests:

```bash
python tests/test_basic.py
```

## Test Coverage

- **test_basic.py**: Tests for core components
  - Neural network architectures (DQN, Policy, Value networks)
  - Replay buffers (standard and prioritized)
  - DQN agent functionality
  - Environment wrapper (requires OpenSpiel)

## Requirements

To run tests, you need:
- PyTorch
- NumPy
- OpenSpiel (optional, for environment tests)

Install with:
```bash
pip install -r requirements.txt
```
