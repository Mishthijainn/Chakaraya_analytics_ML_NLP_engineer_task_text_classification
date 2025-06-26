#!/usr/bin/env python3
"""
Simple training script entry point for text classification pipeline.
This script provides a convenient way to run training from the root directory.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train_model import main

if __name__ == "__main__":
    main()
