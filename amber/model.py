"""
Core AMBER Model Implementation

This module contains the main AMBER (Attention-based Multi-head Bidirectional 
Enhanced Representations) model that provides context-aware word embeddings.
"""

import numpy as np
import warnings
from collections import defaultdict
from typing import List, Dict, Union, Optional, Any
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
import gensim.downloader as api

warnings.filterwarnings('ignore')


class AMBERModel:
    """
    AMBER: Attention-based Multi-head Bidirectional Enhanced Representations
    
    A hybrid word embedding model that combines TF-IDF weighting with multi-head
    self-attention to create context-aware embeddings from static word vectors.
    
    Parameters:
    -----------
    corpus : List[str] or List[List[str]]
        Text corpus for TF-IDF computation. Can be list of sentences or pre-tokenized documents.
    w2v_model : gensim.models.KeyedVectors, optional
        Pre-trained word embedding model. If None, loads Google News Word2Vec by default.
    vector_size : int, default=300
        Dimensionality of word vectors.
    max_corpus_size : int, default=10000
        Maximum number of documents to use from corpus for efficiency.
    tfidf_params : dict, optional
        Parameters for TF-IDF vectorizer.
    """
    
    def __init__(
        self,
        corpus: Union[List[str], List[List[str]]],
        w2v_model: Optional[KeyedVectors] = None,
        vector_size: int = 300,
        max_corpus_size: int = 10000,
        tfidf_params: Optional[Dict[str, Any]] = None
    ):
        # Load default Word2Vec if not provided
        if w2v_model is None:
            print("Loading default Word2Vec model (Google News 300d)...")
            self.w2v_model = api.load('word2vec-google-news-300')
            print("Word2Vec model loaded successfully.")
        else:
            self.w2v_model = w2v_model
            
        self.vector_size = vector_size
        
        # Process corpus
        print("Processing corpus...")
        self.processed_corpus = self._process_corpus(corpus, max_corpus_size)
        print(f"Final corpus size: {len(self.processed_corpus)} documents")
        
        # Setup TF-IDF
        self.tfidf_params = tfidf_params or {
            'lowercase': True,
            'stop_words': 'english',
            'min_df': 1,
            'max_df': 0.95,
            'max_features': 10000
        }
        self._setup_tfidf()
        
        # Cache for efficiency
        self.word_vectors_cache = {}
        self.context_cache = {}
    
    def _process_corpus(
        self, 
        corpus: Union[List[str], List[List[str]]], 
        max_size: int
    ) -> List[List[str]]:
        """Process and tokenize corpus"""
        processed = []
        
        for i, doc in enumerate(corpus[:max_size]):
            if isinstance(doc, list):
                # Already tokenized
                processed_doc = [word.lower() for word in doc if word.isalpha()]
            else:
                # String document - tokenize
                processed_doc = [word.lower() for word in str(doc).split() if word.isalpha()]
            
            if len(processed_doc) >= 1:
                processed.append(processed_doc)
                
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} documents...")
        
        return processed
    
    def _setup_tfidf(self):
        """Setup TF-IDF vectorizer and compute scores"""
        print("Setting up TF-IDF vectorizer...")
        
        # Convert to strings for TF-IDF
        corpus_strings = [" ".join(words) for words in self.processed_corpus]
        
        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_params)
        
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus_strings)
            self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
            print(f"TF-IDF vocabulary size: {len(self.feature_names)}")
            
            # Create efficient lookup for TF-IDF scores
            self.word_tfidf_scores = defaultdict(dict)
            for i, doc in enumerate(self.processed_corpus):
                for word in doc:
                    col_idx = self.tfidf_vectorizer.vocabulary_.get(word)
                    if col_idx is not None:
                        score = self.tfidf_matrix[i, col_idx]
                        if score > 0:
                            self.word_tfidf_scores[word][i] = score
                            
        except Exception as e:
            print(f"Warning: TF-IDF setup failed: {e}")
            print("Falling back to uniform weighting...")
            self.tfidf_matrix = None
            self.word_tfidf_scores = defaultdict(dict)
    
    def get_tfidf_score(self, word: str, doc_idx: int) -> float:
        """Get TF-IDF score for a word in a specific document"""
        return self.word_tfidf_scores.get(word, {}).get(doc_idx, 0.1)
    
    def get_word_vector(
        self, 
        word: str, 
        doc_idx: Optional[int] = None, 
        use_tfidf: bool = True
    ) -> np.ndarray:
        """
        Get word vector with optional TF-IDF weighting
        
        Parameters:
        -----------
        word : str
            Target word
        doc_idx : int, optional
            Document index for TF-IDF weighting
        use_tfidf : bool, default=True
            Whether to apply TF-IDF weighting
            
        Returns:
        --------
        np.ndarray
            Word vector (possibly TF-IDF weighted)
        """
        if word not in self.w2v_model.key_to_index:
            return np.zeros(self.vector_size)
        
        base_vector = self.w2v_model[word]
        
        if use_tfidf and doc_idx is not None and self.tfidf_matrix is not None:
            tfidf_score = max(0.1, self.get_tfidf_score(word, doc_idx))
            return base_vector * tfidf_score
        
        return base_vector
    
    def multi_head_attention(
        self,
        target_word: str,
        sentence: List[str],
        doc_idx: int,
        num_heads: int = 4,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Multi-head attention mechanism for contextual understanding
        
        Parameters:
        -----------
        target_word : str
            Word to contextualize
        sentence : List[str]
            Tokenized sentence containing the word
        doc_idx : int
            Document index for TF-IDF weighting
        num_heads : int, default=4
            Number of attention heads
        temperature : float, default=1.0
            Temperature for attention softmax
            
        Returns:
        --------
        np.ndarray
            Context-aware word embedding
        """
        if target_word not in self.w2v_model.key_to_index:
            return np.zeros(self.vector_size)
        
        target_vector = self.get_word_vector(target_word, doc_idx, use_tfidf=True)
        
        # Get context words
        context_words = [w for w in sentence if w != target_word and w in self.w2v_model.key_to_index]
        
        if not context_words:
            return target_vector
        
        # Multi-head attention
        head_size = max(1, self.vector_size // num_heads)
        attention_outputs = []
        
        for head in range(num_heads):
            start_idx = head * head_size
            end_idx = min(start_idx + head_size, self.vector_size)
            
            # Project vectors for this head
            target_head = target_vector[start_idx:end_idx]
            
            similarities = []
            context_vectors = []
            
            for word in context_words:
                context_vec = self.get_word_vector(word, doc_idx, use_tfidf=True)
                context_head = context_vec[start_idx:end_idx]
                
                # Calculate attention score
                if np.linalg.norm(target_head) > 1e-8 and np.linalg.norm(context_head) > 1e-8:
                    attention_score = np.dot(target_head, context_head) / (
                        np.linalg.norm(target_head) * np.linalg.norm(context_head)
                    )
                    similarities.append(attention_score / temperature)
                    context_vectors.append(context_vec)
                else:
                    similarities.append(0.0)
                    context_vectors.append(context_vec)
            
            # Apply softmax
            if similarities:
                exp_similarities = np.exp(np.array(similarities) - np.max(similarities))
                attention_weights = exp_similarities / (np.sum(exp_similarities) + 1e-8)
                
                # Compute weighted context vector
                weighted_context = np.zeros(self.vector_size)
                for i, context_vec in enumerate(context_vectors):
                    weighted_context += attention_weights[i] * context_vec
                
                attention_outputs.append(weighted_context)
        
        # Combine multi-head outputs with residual connection
        if attention_outputs:
            combined_context = np.mean(attention_outputs, axis=0)
            output = target_vector + 0.5 * combined_context
            
            # Layer normalization
            norm = np.linalg.norm(output)
            if norm > 1e-8:
                output = output / norm
            return output
        
        return target_vector
    
    def positional_attention(
        self,
        target_word: str,
        sentence: List[str],
        doc_idx: int,
        window_size: int = 5,
        position_decay: float = 1.0
    ) -> np.ndarray:
        """
        Position-aware attention mechanism
        
        Parameters:
        -----------
        target_word : str
            Word to contextualize
        sentence : List[str]
            Tokenized sentence
        doc_idx : int
            Document index for TF-IDF weighting
        window_size : int, default=5
            Context window size
        position_decay : float, default=1.0
            Decay rate for positional weighting
            
        Returns:
        --------
        np.ndarray
            Position-aware context embedding
        """
        if target_word not in self.w2v_model.key_to_index:
            return np.zeros(self.vector_size)
        
        target_vector = self.get_word_vector(target_word, doc_idx, use_tfidf=True)
        
        # Find target word positions
        target_positions = [i for i, word in enumerate(sentence) if word == target_word]
        
        if not target_positions:
            return target_vector
        
        target_pos = target_positions[0]  # Use first occurrence
        
        # Get context within window
        start_pos = max(0, target_pos - window_size)
        end_pos = min(len(sentence), target_pos + window_size + 1)
        
        context_info = []
        for i in range(start_pos, end_pos):
            if i != target_pos and sentence[i] in self.w2v_model.key_to_index:
                word = sentence[i]
                context_vec = self.get_word_vector(word, doc_idx, use_tfidf=True)
                distance = abs(i - target_pos)
                position_weight = 1.0 / (distance ** position_decay + 1)
                context_info.append((word, context_vec, position_weight))
        
        if not context_info:
            return target_vector
        
        # Calculate weighted similarities
        similarities = []
        weighted_vectors = []
        
        for word, context_vec, pos_weight in context_info:
            if np.linalg.norm(target_vector) > 1e-8 and np.linalg.norm(context_vec) > 1e-8:
                semantic_sim = 1 - cosine(target_vector, context_vec)
                combined_weight = semantic_sim * pos_weight
                similarities.append(combined_weight)
                weighted_vectors.append(context_vec)
            else:
                similarities.append(0.0)
                weighted_vectors.append(context_vec)
        
        # Apply softmax and combine
        if similarities:
            exp_similarities = np.exp(np.array(similarities) - np.max(similarities))
            attention_weights = exp_similarities / (np.sum(exp_similarities) + 1e-8)
            
            weighted_context = np.zeros(self.vector_size)
            for i, context_vec in enumerate(weighted_vectors):
                weighted_context += attention_weights[i] * context_vec
            
            return target_vector + 0.3 * weighted_context
        
        return target_vector
    
    def get_contextual_embedding(
        self,
        word: str,
        sentence: Union[str, List[str]],
        method: str = "multi_head",
        doc_idx: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Get contextual embedding for a word in a sentence
        
        Parameters:
        -----------
        word : str
            Target word
        sentence : str or List[str]
            Context sentence
        method : str, default="multi_head"
            Attention method: "multi_head", "positional", or "tfidf_only"
        doc_idx : int, optional
            Document index (uses 0 if not provided)
        **kwargs
            Additional parameters for attention methods
            
        Returns:
        --------
        np.ndarray
            Contextual word embedding
        """
        # Process sentence
        if isinstance(sentence, str):
            sentence_tokens = sentence.lower().split()
        else:
            sentence_tokens = [token.lower() for token in sentence]
        
        if doc_idx is None:
            doc_idx = 0
        
        if method == "multi_head":
            return self.multi_head_attention(word, sentence_tokens, doc_idx, **kwargs)
        elif method == "positional":
            return self.positional_attention(word, sentence_tokens, doc_idx, **kwargs)
        elif method == "tfidf_only":
            return self.get_word_vector(word, doc_idx, use_tfidf=True)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'multi_head', 'positional', or 'tfidf_only'")
    
    def batch_contextual_embeddings(
        self,
        words_and_contexts: List[Dict[str, Union[str, List[str]]]],
        method: str = "multi_head",
        **kwargs
    ) -> List[np.ndarray]:
        """
        Get contextual embeddings for multiple word-context pairs
        
        Parameters:
        -----------
        words_and_contexts : List[Dict]
            List of dictionaries with 'word' and 'sentence' keys
        method : str, default="multi_head"
            Attention method to use
        **kwargs
            Additional parameters for attention methods
            
        Returns:
        --------
        List[np.ndarray]
            List of contextual embeddings
        """
        embeddings = []
        for item in words_and_contexts:
            word = item['word']
            sentence = item['sentence']
            doc_idx = item.get('doc_idx', 0)
            
            embedding = self.get_contextual_embedding(
                word, sentence, method, doc_idx, **kwargs
            )
            embeddings.append(embedding)
        
        return embeddings