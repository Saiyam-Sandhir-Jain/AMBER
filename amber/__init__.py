"""
AMBER: Attention-based Multi-head Bidirectional Enhanced Representations

This package provides context-aware word embeddings by combining TF-IDF weighting
with multi-head self-attention mechanisms on top of static word embeddings.

Main Components:
- AMBERModel: The core contextual embedding model
- MetricsCalculator: Evaluation metrics including Context Sensitivity Score (CSS)
- AMBERComparator: Model comparison and evaluation utilities
"""

from .model import AMBERModel
from .metrics import MetricsCalculator, ContextSensitivityScore
from .comparator import AMBERComparator
from .utils import load_default_word2vec, preprocess_corpus

__version__ = "0.1.0"
__author__ = "Saiyam Jain, Swaroop Bhowmik, Kushagra Srivastava, Hitesh Kaushik"
__email__ = "swaroopbhowmik7@gmail.com"

__all__ = [
    "AMBERModel",
    "MetricsCalculator",
    "ContextSensitivityScore", 
    "AMBERComparator",
    "load_default_word2vec",
    "preprocess_corpus"
]