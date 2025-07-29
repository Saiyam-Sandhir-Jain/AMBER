"""
Test suite for AMBER model functionality
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from amber import AMBERModel
from amber.utils import preprocess_corpus


class TestAMBERModel:
    """Test cases for the main AMBER model"""
    
    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus for testing"""
        return [
            "The quick brown fox jumps over the lazy dog",
            "The dog barks loudly at the cat",
            "A quick brown cat runs fast",
            "The fox is a clever animal",
            "He went to the bank to deposit money",
            "The river bank was muddy after rain",
            "The apple fell from the tree",
            "Apple company released new iPhone"
        ]
    
    @pytest.fixture
    def mock_w2v_model(self):
        """Mock Word2Vec model for testing"""
        mock_model = Mock()
        mock_model.key_to_index = {
            'bank': 0, 'apple': 1, 'fox': 2, 'dog': 3, 'cat': 4,
            'quick': 5, 'brown': 6, 'jumps': 7, 'tree': 8, 'company': 9
        }
        mock_model.vector_size = 100
        
        # Mock word vectors
        def get_vector(word):
            if word in mock_model.key_to_index:
                np.random.seed(hash(word) % 2**32)  # Consistent vectors
                return np.random.randn(100).astype(np.float32)
            else:
                raise KeyError(f"Word '{word}' not in vocabulary")
        
        mock_model.__getitem__ = get_vector
        return mock_model
    
    def test_model_initialization(self, sample_corpus, mock_w2v_model):
        """Test AMBER model initialization"""
        model = AMBERModel(
            corpus=sample_corpus,
            w2v_model=mock_w2v_model,
            vector_size=100,
            max_corpus_size=10
        )
        
        assert model.vector_size == 100
        assert len(model.processed_corpus) <= 10
        assert model.w2v_model == mock_w2v_model
        assert hasattr(model, 'tfidf_vectorizer')
    
    def test_corpus_preprocessing(self, mock_w2v_model):
        """Test corpus preprocessing"""
        raw_corpus = [
            "The Quick Brown Fox!",
            "123 Numbers and symbols @#$",
            "",  # Empty document
            "Valid document with words"
        ]
        
        model = AMBERModel(
            corpus=raw_corpus,
            w2v_model=mock_w2v_model,
            vector_size=100
        )
        
        # Should filter empty documents and clean text
        assert len(model.processed_corpus) >= 2
        for doc in model.processed_corpus:
            assert len(doc) > 0
            for word in doc:
                assert word.islower()
                assert word.isalpha()
    
    def test_get_contextual_embedding_multi_head(self, sample_corpus, mock_w2v_model):
        """Test multi-head attention embedding"""
        model = AMBERModel(sample_corpus, mock_w2v_model, vector_size=100)
        
        embedding = model.get_contextual_embedding(
            word="bank",
            sentence="He went to the bank to deposit money",
            method="multi_head",
            doc_idx=0
        )
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (100,)
        assert not np.allclose(embedding, 0)  # Should not be zero vector
    
    def test_get_contextual_embedding_positional(self, sample_corpus, mock_w2v_model):
        """Test positional attention embedding"""
        model = AMBERModel(sample_corpus, mock_w2v_model, vector_size=100)
        
        embedding = model.get_contextual_embedding(
            word="apple",
            sentence="The apple fell from the tree",
            method="positional",
            doc_idx=0
        )
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (100,)
        assert not np.allclose(embedding, 0)
    
    def test_get_contextual_embedding_tfidf_only(self, sample_corpus, mock_w2v_model):
        """Test TF-IDF only embedding"""
        model = AMBERModel(sample_corpus, mock_w2v_model, vector_size=100)
        
        embedding = model.get_contextual_embedding(
            word="fox",
            sentence="The quick brown fox jumps",
            method="tfidf_only",
            doc_idx=0
        )
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (100,)
    
    def test_unknown_word_handling(self, sample_corpus, mock_w2v_model):
        """Test handling of unknown words"""
        model = AMBERModel(sample_corpus, mock_w2v_model, vector_size=100)
        
        embedding = model.get_contextual_embedding(
            word="unknown_word",
            sentence="This unknown_word should return zero vector",
            method="multi_head",
            doc_idx=0
        )
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (100,)
        assert np.allclose(embedding, 0)  # Should be zero vector for unknown words
    
    def test_batch_contextual_embeddings(self, sample_corpus, mock_w2v_model):
        """Test batch processing of embeddings"""
        model = AMBERModel(sample_corpus, mock_w2v_model, vector_size=100)
        
        batch_data = [
            {"word": "bank", "sentence": "He went to the bank", "doc_idx": 0},
            {"word": "apple", "sentence": "The apple fell down", "doc_idx": 1},
            {"word": "fox", "sentence": "The quick brown fox", "doc_idx": 2}
        ]
        
        embeddings = model.batch_contextual_embeddings(batch_data, method="multi_head")
        
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (100,)
    
    def test_invalid_method(self, sample_corpus, mock_w2v_model):
        """Test invalid embedding method"""
        model = AMBERModel(sample_corpus, mock_w2v_model, vector_size=100)
        
        with pytest.raises(ValueError):
            model.get_contextual_embedding(
                word="bank",
                sentence="Test sentence",
                method="invalid_method",
                doc_idx=0
            )
    
    def test_tfidf_score_calculation(self, sample_corpus, mock_w2v_model):
        """Test TF-IDF score calculation"""
        model = AMBERModel(sample_corpus, mock_w2v_model, vector_size=100)
        
        # Test with word that should exist in corpus
        score = model.get_tfidf_score("bank", 0)
        assert isinstance(score, float)
        assert score >= 0
        
        # Test with word that doesn't exist
        score = model.get_tfidf_score("nonexistent", 0)
        assert score == 0.1  # Should return default score
    
    def test_word_vector_with_tfidf(self, sample_corpus, mock_w2v_model):
        """Test word vector retrieval with TF-IDF weighting"""
        model = AMBERModel(sample_corpus, mock_w2v_model, vector_size=100)
        
        # Test with TF-IDF weighting
        vector_with_tfidf = model.get_word_vector("bank", doc_idx=0, use_tfidf=True)
        assert isinstance(vector_with_tfidf, np.ndarray)
        assert vector_with_tfidf.shape == (100,)
        
        # Test without TF-IDF weighting
        vector_without_tfidf = model.get_word_vector("bank", doc_idx=0, use_tfidf=False)
        assert isinstance(vector_without_tfidf, np.ndarray)
        assert vector_without_tfidf.shape == (100,)
        
        # Vectors should be different when TF-IDF is applied
        # (unless TF-IDF score is exactly 1.0, which is unlikely)
    
    def test_empty_context_handling(self, sample_corpus, mock_w2v_model):
        """Test handling of empty context"""
        model = AMBERModel(sample_corpus, mock_w2v_model, vector_size=100)
        
        # Test with empty sentence
        embedding = model.get_contextual_embedding(
            word="bank",
            sentence="",
            method="multi_head",
            doc_idx=0
        )
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (100,)
    
    def test_attention_parameters(self, sample_corpus, mock_w2v_model):
        """Test different attention parameters"""
        model = AMBERModel(sample_corpus, mock_w2v_model, vector_size=100)
        
        # Test multi-head with different parameters
        embedding1 = model.get_contextual_embedding(
            word="bank",
            sentence="The bank is near the river",
            method="multi_head",
            doc_idx=0,
            num_heads=2,
            temperature=0.5
        )
        
        embedding2 = model.get_contextual_embedding(
            word="bank",
            sentence="The bank is near the river",
            method="multi_head",
            doc_idx=0,
            num_heads=4,
            temperature=1.0
        )
        
        assert embedding1.shape == embedding2.shape
        # Different parameters should produce different embeddings
        assert not np.allclose(embedding1, embedding2, atol=1e-6)
    
    def test_positional_attention_parameters(self, sample_corpus, mock_w2v_model):
        """Test positional attention with different parameters"""
        model = AMBERModel(sample_corpus, mock_w2v_model, vector_size=100)
        
        embedding1 = model.get_contextual_embedding(
            word="bank",
            sentence="The quick brown fox near the bank runs fast",
            method="positional",
            doc_idx=0,
            window_size=3
        )
        
        embedding2 = model.get_contextual_embedding(
            word="bank", 
            sentence="The quick brown fox near the bank runs fast",
            method="positional",
            doc_idx=0,
            window_size=5
        )
        
        assert embedding1.shape == embedding2.shape
        # Different window sizes should produce different results
        assert not np.allclose(embedding1, embedding2, atol=1e-6)


class TestAMBERModelIntegration:
    """Integration tests for AMBER model"""
    
    @patch('amber.utils.load_default_word2vec')
    def test_model_with_default_w2v(self, mock_load_w2v):
        """Test model initialization with default Word2Vec"""
        # Mock the default Word2Vec loading
        mock_w2v = Mock()
        mock_w2v.key_to_index = {'test': 0, 'word': 1}
        mock_w2v.vector_size = 300
        mock_w2v.__getitem__ = lambda word: np.random.randn(300)
        mock_load_w2v.return_value = mock_w2v
        
        corpus = ["test sentence", "another test"]
        model = AMBERModel(corpus)  # Should use default Word2Vec
        
        assert model.vector_size == 300
        mock_load_w2v.assert_called_once()
    
    def test_context_sensitivity_demonstration(self, mock_w2v_model):
        """Test that the model produces different embeddings for different contexts"""
        corpus = [
            "He went to the bank to deposit money",
            "The river bank was muddy after rain",
            "The apple fell from the tree",
            "Apple company released new iPhone"
        ]
        
        model = AMBERModel(corpus, mock_w2v_model, vector_size=100)
        
        # Test 'bank' in different contexts
        bank_financial = model.get_contextual_embedding(
            "bank", "He went to the bank to deposit money", "multi_head", 0
        )
        bank_river = model.get_contextual_embedding(
            "bank", "The river bank was muddy after rain", "multi_head", 1
        )
        
        # Test 'apple' in different contexts
        apple_fruit = model.get_contextual_embedding(
            "apple", "The apple fell from the tree", "multi_head", 2
        )
        apple_company = model.get_contextual_embedding(
            "apple", "Apple company released new iPhone", "multi_head", 3
        )
        
        # Embeddings should be different for different contexts
        assert not np.allclose(bank_financial, bank_river, atol=1e-3)
        assert not np.allclose(apple_fruit, apple_company, atol=1e-3)
    
    def test_model_robustness(self, mock_w2v_model):
        """Test model robustness with edge cases"""
        corpus = ["single word", "a", "normal sentence with multiple words"]
        model = AMBERModel(corpus, mock_w2v_model, vector_size=100)
        
        # Test with very short sentence
        embedding = model.get_contextual_embedding(
            "word", "word", "multi_head", 0
        )
        assert embedding.shape == (100,)
        
        # Test with sentence containing only the target word
        embedding = model.get_contextual_embedding(
            "a", "a", "positional", 1
        )
        assert embedding.shape == (100,)


if __name__ == "__main__":
    pytest.main([__file__])