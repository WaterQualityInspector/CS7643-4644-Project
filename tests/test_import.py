#!/usr/bin/env python3
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test the import
try:
    from hand_eval import card_str_to_tuple, eval_5card
    print("✓ Import successful!")
    
    # Test a simple function call
    result = card_str_to_tuple("AS")
    print(f"✓ card_str_to_tuple('AS') = {result}")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
except Exception as e:
    print(f"✗ Test failed: {e}")
