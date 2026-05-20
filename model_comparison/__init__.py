"""
Machine Learning Comparison Module.
Provides baseline comparison experiments for KAMEL evaluation.
"""

import sys
import os

# Add algorithms path
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

from .ml_experiment import MLComparisonExperiment
from .algorithms import ALGORITHM_REGISTRY, get_algorithm

__all__ = ['MLComparisonExperiment', 'ALGORITHM_REGISTRY', 'get_algorithm']