"""
Utility functions for AMBER package

This module provides helper functions for loading models, preprocessing data,
and other common tasks.
"""

import gensim.downloader as api
from gensim.models import KeyedVectors
from typing import List, Union, Optional, Dict, Any
import numpy as np
import re
from sklearn.datasets import fetch_20newsgroups


def load_default_word2vec() -> KeyedVectors:
    """
    Load the default Word2Vec model (Google News 300d)
    
    Returns:
    --------
    KeyedVectors
        Pre-trained Word2Vec model
    """
    print("Loading Google News Word2Vec model (300d)...")
    try:
        model = api.load('word2vec-google-news-300')
        print("✓ Word2Vec model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to load Word2Vec model: {e}")
        raise


def preprocess_corpus(
    corpus: Union[List[str], List[List[str]]],
    lowercase: bool = True,
    remove_punctuation: bool = True,
    min_word_length: int = 2,
    max_corpus_size: Optional[int] = None
) -> List[List[str]]:
    """
    Preprocess text corpus for AMBER model
    
    Parameters:
    -----------
    corpus : List[str] or List[List[str]]
        Input corpus (sentences or pre-tokenized documents)
    lowercase : bool, default=True
        Convert to lowercase
    remove_punctuation : bool, default=True
        Remove punctuation marks
    min_word_length : int, default=2
        Minimum word length to keep
    max_corpus_size : int, optional
        Maximum number of documents to process
        
    Returns:
    --------
    List[List[str]]
        Preprocessed corpus as list of token lists
    """
    processed = []
    
    # Limit corpus size if specified
    if max_corpus_size:
        corpus = corpus[:max_corpus_size]
    
    for doc in corpus:
        if isinstance(doc, list):
            # Already tokenized
            tokens = doc.copy()
        else:
            # String document - tokenize
            tokens = str(doc).split()
        
        # Apply preprocessing
        if lowercase:
            tokens = [token.lower() for token in tokens]
        
        if remove_punctuation:
            # Remove punctuation and non-alphabetic characters
            tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
        
        # Filter by length and remove empty strings
        tokens = [token for token in tokens if len(token) >= min_word_length]
        
        if tokens:  # Only add non-empty documents
            processed.append(tokens)
    
    return processed


def load_sample_corpus(corpus_type: str = "simple") -> List[str]:
    """
    Load sample corpus for testing and demonstration
    
    Parameters:
    -----------
    corpus_type : str, default="simple"
        Type of corpus to load: "simple", "newsgroups", or "polysemy"
        
    Returns:
    --------
    List[str]
        Sample corpus as list of sentences
    """
    if corpus_type == "simple":
        return [
            "The quick brown fox jumps over the lazy dog",
            "The dog barks loudly at the cat",
            "A quick brown cat runs fast",
            "The fox is a clever animal",
            "Dogs are loyal pets and make great companions"
        ]
    
    elif corpus_type == "polysemy":
        return [
            "He went to the bank to deposit money and withdraw cash",
            "The river bank was muddy after the heavy rain",
            "The apple fell from the tree in the orchard",
            "Apple company released a new iPhone model yesterday",
            "The bass guitar sounds amazing in the concert",
            "He caught a large bass fish in the lake",
            "The bat flew through the dark night sky",
            "The baseball bat broke during the game",
            "Python is a popular programming language for data science",
            "The python snake slithered through the grass",
            "The computer mouse stopped working on my desk",
            "The small mouse ran across the kitchen floor",
            "The rock band played loud music at the concert",
            "A heavy rock fell from the mountain cliff",
            "Spring season brings beautiful flowers and warm weather",
            "The metal spring bounced back to its original position"
        ]
    
    elif corpus_type == "newsgroups":
        try:
            newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
            # Return first 1000 documents for efficiency
            return newsgroups.data[:1000]
        except Exception as e:
            print(f"Warning: Could not load newsgroups dataset: {e}")
            return load_sample_corpus("polysemy")
    
    else:
        raise ValueError(f"Unknown corpus type: {corpus_type}")


def create_test_cases() -> List[Dict[str, Any]]:
    """
    Create standard test cases for word sense disambiguation evaluation
    
    Returns:
    --------
    List[Dict]
        List of test cases with polysemous words and contexts
    """
    return [
        {
            'word': 'bank',
            'contexts': [
                {
                    'sentence': 'He went to the bank to deposit money',
                    'doc_idx': 0,
                    'type': 'Financial Context'
                },
                {
                    'sentence': 'The river bank was muddy after rain',
                    'doc_idx': 1, 
                    'type': 'Geographic Context'
                }
            ]
        },
        {
            'word': 'apple',
            'contexts': [
                {
                    'sentence': 'The apple fell from the tree',
                    'doc_idx': 2,
                    'type': 'Fruit Context'
                },
                {
                    'sentence': 'Apple company released new iPhone',
                    'doc_idx': 3,
                    'type': 'Technology Context'
                }
            ]
        },
        {
            'word': 'bass',
            'contexts': [
                {
                    'sentence': 'The bass guitar sounds amazing in concert',
                    'doc_idx': 4,
                    'type': 'Music Context'
                },
                {
                    'sentence': 'He caught large bass fish in the lake',
                    'doc_idx': 5,
                    'type': 'Animal Context'
                }
            ]
        },
        {
            'word': 'python',
            'contexts': [
                {
                    'sentence': 'Python is a programming language',
                    'doc_idx': 6,
                    'type': 'Technology Context'
                },
                {
                    'sentence': 'The python snake slithered through grass',
                    'doc_idx': 7,
                    'type': 'Animal Context'
                }
            ]
        },
        {
            'word': 'mouse',
            'contexts': [
                {
                    'sentence': 'Computer mouse stopped working on desk',
                    'doc_idx': 8,
                    'type': 'Technology Context'
                },
                {
                    'sentence': 'The mouse ran across kitchen floor',
                    'doc_idx': 9,
                    'type': 'Animal Context'
                }
            ]
        },
        {
            'word': 'bat',
            'contexts': [
                {
                    'sentence': 'The bat flew through dark night sky',
                    'doc_idx': 10,
                    'type': 'Animal Context'
                },
                {
                    'sentence': 'Baseball bat broke during the game',
                    'doc_idx': 11,
                    'type': 'Sports Context'
                }
            ]
        }
    ]


def validate_inputs(
    corpus: Union[List[str], List[List[str]]],
    w2v_model: Optional[KeyedVectors] = None,
    vector_size: int = 300
) -> Dict[str, Any]:
    """
    Validate inputs for AMBER model initialization
    
    Parameters:
    -----------
    corpus : List[str] or List[List[str]]
        Input corpus
    w2v_model : KeyedVectors, optional
        Word embedding model
    vector_size : int
        Expected vector dimensionality
        
    Returns:
    --------
    Dict[str, Any]
        Validation results and processed inputs
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'corpus_stats': {},
        'model_stats': {}
    }
    
    # Validate corpus
    if not corpus:
        validation_results['valid'] = False
        validation_results['errors'].append("Corpus cannot be empty")
    else:
        # Calculate corpus statistics
        if isinstance(corpus[0], list):
            # Pre-tokenized
            total_tokens = sum(len(doc) for doc in corpus)
            avg_doc_length = total_tokens / len(corpus)
            vocab_size = len(set(token for doc in corpus for token in doc))
        else:
            # String corpus
            total_tokens = sum(len(str(doc).split()) for doc in corpus)
            avg_doc_length = total_tokens / len(corpus)
            vocab_size = len(set(token for doc in corpus for token in str(doc).split()))
        
        validation_results['corpus_stats'] = {
            'num_documents': len(corpus),
            'total_tokens': total_tokens,
            'avg_doc_length': avg_doc_length,
            'vocabulary_size': vocab_size
        }
        
        if len(corpus) < 10:
            validation_results['warnings'].append(
                f"Small corpus size ({len(corpus)} documents). Consider using more data for better TF-IDF statistics."
            )
    
    # Validate word embedding model
    if w2v_model is not None:
        try:
            model_vocab_size = len(w2v_model.key_to_index)
            model_vector_size = w2v_model.vector_size
            
            validation_results['model_stats'] = {
                'vocabulary_size': model_vocab_size,
                'vector_size': model_vector_size
            }
            
            if model_vector_size != vector_size:
                validation_results['warnings'].append(
                    f"Model vector size ({model_vector_size}) differs from specified size ({vector_size}). Using model size."
                )
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Invalid word embedding model: {e}")
    
    return validation_results


def export_embeddings(
    amber_model,
    words_and_contexts: List[Dict[str, Any]],
    method: str = "multi_head",
    output_format: str = "dict"
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Export contextual embeddings for external use
    
    Parameters:
    -----------
    amber_model : AMBERModel
        Trained AMBER model
    words_and_contexts : List[Dict]
        Word-context pairs to export
    method : str, default="multi_head"
        Embedding method to use
    output_format : str, default="dict"
        Output format: "dict", "array", or "dataframe"
        
    Returns:
    --------
    Union[Dict, np.ndarray]
        Exported embeddings in specified format
    """
    embeddings = []
    labels = []
    
    for item in words_and_contexts:
        word = item['word']
        sentence = item['sentence']
        doc_idx = item.get('doc_idx', 0)
        
        embedding = amber_model.get_contextual_embedding(
            word, sentence, method, doc_idx
        )
        embeddings.append(embedding)
        labels.append(f"{word}:{sentence[:50]}...")
    
    if output_format == "dict":
        return {label: emb for label, emb in zip(labels, embeddings)}
    elif output_format == "array":
        return np.vstack(embeddings)
    elif output_format == "dataframe":
        try:
            import pandas as pd
            df = pd.DataFrame(embeddings)
            df.index = labels
            return df
        except ImportError:
            print("Warning: pandas not available, returning dictionary format")
            return {label: emb for label, emb in zip(labels, embeddings)}
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def quick_demo(corpus_type: str = "polysemy") -> None:
    """
    Run a quick demonstration of AMBER model
    
    Parameters:
    -----------
    corpus_type : str, default="polysemy"
        Type of corpus to use for demo
    """
    print("="*60)
    print("AMBER MODEL QUICK DEMONSTRATION")
    print("="*60)
    
    # Load sample data
    corpus = load_sample_corpus(corpus_type)
    test_cases = create_test_cases()[:3]  # Use first 3 test cases
    
    # Import required classes
    from .model import AMBERModel
    from .comparator import AMBERComparator
    
    # Initialize model
    print("\nInitializing AMBER model...")
    amber_model = AMBERModel(corpus)
    
    # Initialize comparator
    comparator = AMBERComparator(amber_model)
    
    # Run evaluation
    print("\nRunning disambiguation evaluation...")
    results = comparator.evaluate_disambiguation(test_cases, methods=["multi_head", "positional"])
    
    # Generate report
    print("\nGenerating evaluation report...")
    report = comparator.generate_report(test_cases, methods=["multi_head", "positional"])
    print(report)
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)