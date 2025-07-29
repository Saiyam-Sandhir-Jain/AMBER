"""
Test suite for AMBER utility functions
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock
from amber.utils import (
    preprocess_corpus, load_sample_corpus, create_test_cases,
    validate_inputs, export_embeddings
)


class TestPreprocessCorpus:
    """Test cases for corpus preprocessing"""
    
    def test_preprocess_string_corpus(self):
        """Test preprocessing of string corpus"""
        corpus = [
            "The Quick Brown Fox!",
            "Hello, World! 123",
            "UPPERCASE and lowercase",
            "Punctuation... removed???"
        ]
        
        processed = preprocess_corpus(corpus)
        
        assert len(processed) == 4
        for doc in processed:
            assert isinstance(doc, list)
            for word in doc:
                assert word.islower()
                assert word.isalpha()
                assert len(word) >= 2  # Default min_word_length
    
    def test_preprocess_tokenized_corpus(self):
        """Test preprocessing of already tokenized corpus"""
        corpus = [
            ["The", "Quick", "Brown", "Fox"],
            ["Hello", "World", "123"],
            ["UPPERCASE", "and", "lowercase"]
        ]
        
        processed = preprocess_corpus(corpus)
        
        assert len(processed) == 3
        for doc in processed:
            assert isinstance(doc, list)
            for word in doc:
                assert word.islower()
                assert word.isalpha()
    
    def test_preprocess_with_custom_parameters(self):
        """Test preprocessing with custom parameters"""
        corpus = ["The Quick Brown Fox Jumps Over The Lazy Dog"]
        
        processed = preprocess_corpus(
            corpus,
            lowercase=False,
            remove_punctuation=False,
            min_word_length=1
        )
        
        # Should preserve case and allow single characters
        assert any(word.isupper() for doc in processed for word in doc)
        assert any(len(word) == 1 for doc in processed for word in doc if word.isalpha())
    
    def test_preprocess_max_corpus_size(self):
        """Test corpus size limiting"""
        large_corpus = [f"Document number {i}" for i in range(100)]
        
        processed = preprocess_corpus(large_corpus, max_corpus_size=10)
        
        assert len(processed) <= 10
    
    def test_preprocess_empty_documents(self):
        """Test handling of empty documents"""
        corpus = [
            "Valid document",
            "",
            "   ",  # Only whitespace
            "123 !@#",  # Only non-alphabetic
            "Another valid document"
        ]
        
        processed = preprocess_corpus(corpus)
        
        # Should filter out empty/invalid documents
        assert len(processed) == 2  # Only 2 valid documents
        for doc in processed:
            assert len(doc) > 0
    
    def test_preprocess_special_characters(self):
        """Test handling of special characters and numbers"""
        corpus = [
            "Hello123World!@#",
            "café naïve résumé",  # Accented characters
            "email@domain.com",
            "http://www.example.com"
        ]
        
        processed = preprocess_corpus(corpus)
        
        for doc in processed:
            for word in doc:
                # Should only contain basic alphabetic characters
                assert word.isalpha()
                assert word.islower()


class TestLoadSampleCorpus:
    """Test cases for sample corpus loading"""
    
    def test_load_simple_corpus(self):
        """Test loading simple corpus"""
        corpus = load_sample_corpus("simple")
        
        assert isinstance(corpus, list)
        assert len(corpus) > 0
        for doc in corpus:
            assert isinstance(doc, str)
            assert len(doc) > 0
    
    def test_load_polysemy_corpus(self):
        """Test loading polysemy corpus"""
        corpus = load_sample_corpus("polysemy")
        
        assert isinstance(corpus, list)
        assert len(corpus) > 10  # Should have many examples
        
        # Should contain polysemous words
        corpus_text = " ".join(corpus).lower()
        assert "bank" in corpus_text
        assert "apple" in corpus_text
        assert "bass" in corpus_text
    
    @patch('amber.utils.fetch_20newsgroups')
    def test_load_newsgroups_corpus(self, mock_fetch):
        """Test loading newsgroups corpus"""
        # Mock the newsgroups data
        mock_data = Mock()
        mock_data.data = [f"Newsgroup document {i}" for i in range(50)]
        mock_fetch.return_value = mock_data
        
        corpus = load_sample_corpus("newsgroups")
        
        assert isinstance(corpus, list)
        assert len(corpus) <= 1000  # Should be limited
        mock_fetch.assert_called_once()
    
    @patch('amber.utils.fetch_20newsgroups')
    def test_load_newsgroups_fallback(self, mock_fetch):
        """Test newsgroups loading with fallback"""
        # Mock failure
        mock_fetch.side_effect = Exception("Dataset not available")
        
        corpus = load_sample_corpus("newsgroups")
        
        # Should fallback to polysemy corpus
        assert isinstance(corpus, list)
        assert len(corpus) > 0
    
    def test_load_invalid_corpus_type(self):
        """Test loading with invalid corpus type"""
        with pytest.raises(ValueError):
            load_sample_corpus("invalid_type")


class TestCreateTestCases:
    """Test cases for test case creation"""
    
    def test_create_test_cases_structure(self):
        """Test structure of created test cases"""
        test_cases = create_test_cases()
        
        assert isinstance(test_cases, list)
        assert len(test_cases) > 0
        
        for case in test_cases:
            assert 'word' in case
            assert 'contexts' in case
            assert isinstance(case['contexts'], list)
            assert len(case['contexts']) >= 2  # Each word should have multiple contexts
            
            for context in case['contexts']:
                assert 'sentence' in context
                assert 'doc_idx' in context
                assert 'type' in context
    
    def test_test_cases_contain_polysemous_words(self):
        """Test that test cases contain expected polysemous words"""
        test_cases = create_test_cases()
        words = [case['word'] for case in test_cases]
        
        expected_words = ['bank', 'apple', 'bass', 'python', 'mouse', 'bat']
        for word in expected_words:
            assert word in words
    
    def test_test_cases_contexts_are_different(self):
        """Test that contexts for each word are meaningfully different"""
        test_cases = create_test_cases()
        
        for case in test_cases:
            contexts = case['contexts']
            sentences = [ctx['sentence'] for ctx in contexts]
            types = [ctx['type'] for ctx in contexts]
            
            # Sentences should be different
            assert len(set(sentences)) == len(sentences)
            # Context types should be different
            assert len(set(types)) == len(types)


class TestValidateInputs:
    """Test cases for input validation"""
    
    def test_validate_valid_inputs(self):
        """Test validation with valid inputs"""
        corpus = ["Valid sentence one", "Valid sentence two"]
        
        mock_w2v = Mock()
        mock_w2v.key_to_index = {'word': 0, 'sentence': 1}
        mock_w2v.vector_size = 300
        
        result = validate_inputs(corpus, mock_w2v, 300)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert 'corpus_stats' in result
        assert 'model_stats' in result
    
    def test_validate_empty_corpus(self):
        """Test validation with empty corpus"""
        result = validate_inputs([])
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert "empty" in result['errors'][0].lower()
    
    def test_validate_small_corpus_warning(self):
        """Test warning for small corpus"""
        small_corpus = ["One", "Two"]
        result = validate_inputs(small_corpus)
        
        assert result['valid'] is True
        assert len(result['warnings']) > 0
        assert "small corpus" in result['warnings'][0].lower()
    
    def test_validate_vector_size_mismatch(self):
        """Test validation with vector size mismatch"""
        corpus = ["Test sentence"]
        
        mock_w2v = Mock()
        mock_w2v.key_to_index = {'test': 0}
        mock_w2v.vector_size = 100  # Different from specified 300
        
        result = validate_inputs(corpus, mock_w2v, 300)
        
        assert result['valid'] is True  # Still valid, just warning
        assert len(result['warnings']) > 0
        assert "vector size" in result['warnings'][0].lower()
    
    def test_validate_invalid_model(self):
        """Test validation with invalid model"""
        corpus = ["Test sentence"]
        invalid_model = "not_a_model"
        
        result = validate_inputs(corpus, invalid_model, 300)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
    
    def test_validate_tokenized_corpus_stats(self):
        """Test corpus statistics with pre-tokenized corpus"""
        tokenized_corpus = [
            ["this", "is", "tokenized"],
            ["another", "tokenized", "document"]
        ]
        
        result = validate_inputs(tokenized_corpus)
        
        assert result['valid'] is True
        stats = result['corpus_stats']
        assert stats['num_documents'] == 2
        assert stats['total_tokens'] == 6
        assert stats['vocabulary_size'] == 6  # All unique words


class TestExportEmbeddings:
    """Test cases for embedding export functionality"""
    
    @pytest.fixture
    def mock_amber_model(self):
        """Mock AMBER model for testing"""
        model = Mock()
        model.get_contextual_embedding.return_value = np.random.randn(100)
        return model
    
    def test_export_dict_format(self, mock_amber_model):
        """Test exporting embeddings as dictionary"""
        words_and_contexts = [
            {"word": "bank", "sentence": "Financial bank", "doc_idx": 0},
            {"word": "bank", "sentence": "River bank", "doc_idx": 1}
        ]
        
        result = export_embeddings(
            mock_amber_model, words_and_contexts, output_format="dict"
        )
        
        assert isinstance(result, dict)
        assert len(result) == 2
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, np.ndarray)
            assert "bank:" in key
    
    def test_export_array_format(self, mock_amber_model):
        """Test exporting embeddings as numpy array"""
        words_and_contexts = [
            {"word": "apple", "sentence": "Red apple", "doc_idx": 0},
            {"word": "apple", "sentence": "Apple Inc", "doc_idx": 1}
        ]
        
        result = export_embeddings(
            mock_amber_model, words_and_contexts, output_format="array"
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2  # Two embeddings
        assert result.shape[1] == 100  # Embedding dimension
    
    @patch('amber.utils.pd')
    def test_export_dataframe_format(self, mock_pd, mock_amber_model):
        """Test exporting embeddings as pandas DataFrame"""
        # Mock pandas DataFrame
        mock_df = Mock()
        mock_pd.DataFrame.return_value = mock_df
        
        words_and_contexts = [
            {"word": "test", "sentence": "Test sentence", "doc_idx": 0}
        ]
        
        result = export_embeddings(
            mock_amber_model, words_and_contexts, output_format="dataframe"
        )
        
        assert result == mock_df
        mock_pd.DataFrame.assert_called_once()
    
    def test_export_dataframe_fallback(self, mock_amber_model):
        """Test DataFrame export fallback when pandas not available"""
        words_and_contexts = [
            {"word": "test", "sentence": "Test sentence", "doc_idx": 0}
        ]
        
        with patch('amber.utils.pd', side_effect=ImportError):
            result = export_embeddings(
                mock_amber_model, words_and_contexts, output_format="dataframe"
            )
        
        # Should fallback to dict format
        assert isinstance(result, dict)
    
    def test_export_invalid_format(self, mock_amber_model):
        """Test export with invalid format"""
        words_and_contexts = [
            {"word": "test", "sentence": "Test", "doc_idx": 0}
        ]
        
        with pytest.raises(ValueError):
            export_embeddings(
                mock_amber_model, words_and_contexts, output_format="invalid"
            )
    
    def test_export_with_method_parameter(self, mock_amber_model):
        """Test export with different embedding method"""
        words_and_contexts = [
            {"word": "test", "sentence": "Test", "doc_idx": 0}
        ]
        
        export_embeddings(
            mock_amber_model, words_and_contexts, 
            method="positional", output_format="dict"
        )
        
        # Should call with specified method
        mock_amber_model.get_contextual_embedding.assert_called_with(
            "test", "Test", "positional", 0
        )


class TestUtilsIntegration:
    """Integration tests for utility functions"""
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline"""
        # Start with messy corpus
        raw_corpus = [
            "The Quick Brown Fox Jumps Over The Lazy Dog!",
            "123 Numbers and @#$ symbols should be removed",
            "",  # Empty
            "   ",  # Whitespace only
            "HELLO world! How are you today???",
            "a",  # Too short
            "Valid sentence with proper words"
        ]
        
        processed = preprocess_corpus(
            raw_corpus,
            lowercase=True,
            remove_punctuation=True,
            min_word_length=2,
            max_corpus_size=10
        )
        
        # Should clean up properly
        assert len(processed) >= 3  # At least some valid documents
        
        for doc in processed:
            assert len(doc) > 0
            for word in doc:
                assert word.islower()
                assert word.isalpha()
                assert len(word) >= 2
    
    def test_sample_corpus_with_preprocessing(self):
        """Test sample corpus with preprocessing"""
        corpus = load_sample_corpus("polysemy")
        processed = preprocess_corpus(corpus, max_corpus_size=10)
        
        assert len(processed) <= 10
        assert all(isinstance(doc, list) for doc in processed)
        
        # Should contain polysemous words after preprocessing
        all_words = [word for doc in processed for word in doc]
        assert "bank" in all_words
        assert "apple" in all_words
    
    def test_validation_with_sample_data(self):
        """Test validation with sample data"""
        corpus = load_sample_corpus("simple")
        test_cases = create_test_cases()
        
        validation = validate_inputs(corpus)
        
        assert validation['valid'] is True
        assert validation['corpus_stats']['num_documents'] > 0
        
        # Test cases should be valid
        assert len(test_cases) >= 6  # At least 6 polysemous words
        for case in test_cases:
            assert len(case['contexts']) >= 2


if __name__ == "__main__":
    pytest.main([__file__])