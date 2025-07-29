"""
Test suite for AMBER package

This module contains comprehensive tests for all AMBER components:
- Core model functionality
- Metrics and evaluation
- Utility functions
- Integration tests
"""

# Test configuration
import pytest
import warnings

# Filter out common warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Test utilities that can be shared across test modules
import numpy as np
from unittest.mock import Mock


def create_mock_w2v_model(vocab_size=1000, vector_size=300):
    """
    Create a mock Word2Vec model for testing
    
    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary
    vector_size : int
        Dimension of word vectors
        
    Returns:
    --------
    Mock
        Mock Word2Vec model
    """
    mock_model = Mock()
    
    # Create vocabulary
    vocab_words = [f"word_{i}" for i in range(vocab_size)]
    mock_model.key_to_index = {word: i for i, word in enumerate(vocab_words)}
    mock_model.vector_size = vector_size
    
    # Mock vector retrieval
    def get_vector(word):
        if word in mock_model.key_to_index:
            # Use hash for consistent vectors
            np.random.seed(hash(word) % 2**32)
            return np.random.randn(vector_size).astype(np.float32)
        else:
            raise KeyError(f"Word '{word}' not in vocabulary")
    
    mock_model.__getitem__ = get_vector
    return mock_model


def create_test_embeddings(n_contexts=3, vector_size=100, similarity_level="medium"):
    """
    Create test embeddings with controlled similarity
    
    Parameters:
    -----------
    n_contexts : int
        Number of context embeddings to create
    vector_size : int
        Dimension of embeddings
    similarity_level : str
        "high", "medium", or "low" similarity between embeddings
        
    Returns:
    --------
    List[np.ndarray]
        List of test embeddings
    """
    embeddings = []
    
    if similarity_level == "high":
        # Very similar embeddings (small random variations)
        base_vector = np.random.randn(vector_size)
        for i in range(n_contexts):
            noise = 0.1 * np.random.randn(vector_size)
            embeddings.append(base_vector + noise)
    
    elif similarity_level == "medium":
        # Moderately similar embeddings
        base_vector = np.random.randn(vector_size)
        for i in range(n_contexts):
            noise = 0.5 * np.random.randn(vector_size)
            embeddings.append(base_vector + noise)
    
    elif similarity_level == "low":
        # Very different embeddings (orthogonal)
        for i in range(n_contexts):
            vector = np.zeros(vector_size)
            if i < vector_size:
                vector[i] = 1.0
            else:
                vector = np.random.randn(vector_size)
            embeddings.append(vector)
    
    return embeddings


# Common test data
SAMPLE_CORPUS = [
    "The quick brown fox jumps over the lazy dog",
    "The dog barks loudly at the cat in the yard",
    "A quick brown cat runs fast through the garden",
    "The fox is a clever animal that hunts at night",
    "He went to the bank to deposit money",
    "The river bank was muddy after the rain",
    "The apple fell from the tree",
    "Apple company released a new iPhone",
    "The bass guitar sounds amazing",
    "He caught a large bass fish"
]

POLYSEMOUS_TEST_CASES = [
    {
        'word': 'bank',
        'contexts': [
            {'sentence': 'He went to the bank to deposit money', 'type': 'Financial'},
            {'sentence': 'The river bank was muddy after rain', 'type': 'Geographic'}
        ]
    },
    {
        'word': 'apple',
        'contexts': [
            {'sentence': 'The apple fell from the tree', 'type': 'Fruit'},
            {'sentence': 'Apple company released new iPhone', 'type': 'Technology'}
        ]
    },
    {
        'word': 'bass',
        'contexts': [
            {'sentence': 'The bass guitar sounds amazing', 'type': 'Music'},
            {'sentence': 'He caught large bass fish', 'type': 'Animal'}
        ]
    }
]

# Export test utilities
__all__ = [
    'create_mock_w2v_model',
    'create_test_embeddings', 
    'SAMPLE_CORPUS',
    'POLYSEMOUS_TEST_CASES'
]