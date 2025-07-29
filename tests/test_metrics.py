"""
Test suite for AMBER metrics functionality
"""

import pytest
import numpy as np
from unittest.mock import Mock
from amber.metrics import MetricsCalculator, ContextSensitivityScore


class TestContextSensitivityScore:
    """Test cases for Context Sensitivity Score calculation"""
    
    def test_css_identical_embeddings(self):
        """Test CSS with identical embeddings (should be 0)"""
        vector = np.array([1, 0, 0, 0, 0])
        embeddings = [vector, vector, vector]
        
        css = ContextSensitivityScore.calculate(embeddings)
        assert abs(css) < 1e-6  # Should be essentially 0
    
    def test_css_orthogonal_embeddings(self):
        """Test CSS with orthogonal embeddings (should be high)"""
        embeddings = [
            np.array([1, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0]),
            np.array([0, 0, 1, 0, 0])
        ]
        
        css = ContextSensitivityScore.calculate(embeddings)
        assert css > 0.8  # Should be close to 1 for orthogonal vectors
    
    def test_css_single_embedding(self):
        """Test CSS with single embedding (should be 0)"""
        embeddings = [np.array([1, 2, 3, 4, 5])]
        
        css = ContextSensitivityScore.calculate(embeddings)
        assert css == 0.0
    
    def test_css_empty_embeddings(self):
        """Test CSS with empty embeddings list"""
        embeddings = []
        
        css = ContextSensitivityScore.calculate(embeddings)
        assert css == 0.0
    
    def test_css_zero_norm_embeddings(self):
        """Test CSS with zero-norm embeddings"""
        embeddings = [
            np.zeros(5),
            np.zeros(5),
            np.array([1, 0, 0, 0, 0])
        ]
        
        css = ContextSensitivityScore.calculate(embeddings)
        assert css >= 0  # Should handle gracefully
    
    def test_css_batch_calculation(self):
        """Test batch CSS calculation"""
        word_embeddings = {
            'bank': [
                np.array([1, 0, 0]),
                np.array([0, 1, 0])
            ],
            'apple': [
                np.array([1, 1, 0]),
                np.array([1, 1, 0])
            ]
        }
        
        css_scores = ContextSensitivityScore.calculate_batch(word_embeddings)
        
        assert 'bank' in css_scores
        assert 'apple' in css_scores
        assert css_scores['bank'] > css_scores['apple']  # bank should be more context-sensitive
    
    def test_css_numerical_stability(self):
        """Test CSS numerical stability with very similar vectors"""
        base_vector = np.random.randn(100)
        embeddings = [
            base_vector,
            base_vector + 1e-10 * np.random.randn(100),
            base_vector + 1e-10 * np.random.randn(100)
        ]
        
        css = ContextSensitivityScore.calculate(embeddings)
        assert 0 <= css <= 1  # Should be bounded
        assert css < 0.1  # Should be small for very similar vectors


class TestMetricsCalculator:
    """Test cases for MetricsCalculator class"""
    
    @pytest.fixture
    def metrics_calc(self):
        """Fixture for MetricsCalculator instance"""
        return MetricsCalculator()
    
    def test_cosine_distance_matrix(self, metrics_calc):
        """Test cosine distance matrix calculation"""
        embeddings = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]
        
        distance_matrix = metrics_calc.cosine_distance_matrix(embeddings)
        
        assert distance_matrix.shape == (3, 3)
        assert np.allclose(np.diag(distance_matrix), 0)  # Diagonal should be 0
        assert np.allclose(distance_matrix, distance_matrix.T)  # Should be symmetric
        
        # All distances should be close to 1 for orthogonal vectors
        non_diagonal = distance_matrix[np.triu_indices(3, k=1)]
        assert np.allclose(non_diagonal, 1.0, atol=1e-6)
    
    def test_clustering_purity_perfect(self, metrics_calc):
        """Test clustering purity with perfect separation"""
        embeddings = [
            np.array([1, 0]),    # Cluster 0
            np.array([1.1, 0]),  # Cluster 0
            np.array([0, 1]),    # Cluster 1  
            np.array([0, 1.1])   # Cluster 1
        ]
        true_labels = [0, 0, 1, 1]
        
        purity = metrics_calc.clustering_purity(embeddings, true_labels)
        assert purity > 0.8  # Should be high for well-separated clusters
    
    def test_clustering_purity_mixed(self, metrics_calc):
        """Test clustering purity with mixed clusters"""
        embeddings = [
            np.array([0, 0]),    # Mixed cluster
            np.array([0.1, 0.1]),  # Mixed cluster
            np.array([0, 0.1]),    # Mixed cluster
            np.array([0.1, 0])     # Mixed cluster
        ]
        true_labels = [0, 1, 0, 1]  # Alternating labels
        
        purity = metrics_calc.clustering_purity(embeddings, true_labels)
        assert 0 <= purity <= 1  # Should be bounded
    
    def test_clustering_purity_empty(self, metrics_calc):
        """Test clustering purity with empty inputs"""
        purity = metrics_calc.clustering_purity([], [])
        assert purity == 0.0
    
    def test_silhouette_analysis(self, metrics_calc):
        """Test silhouette score calculation"""
        # Well-separated clusters
        embeddings = [
            np.array([0, 0]),
            np.array([0.1, 0.1]),
            np.array([10, 10]),
            np.array([10.1, 10.1])
        ]
        labels = [0, 0, 1, 1]
        
        score = metrics_calc.silhouette_analysis(embeddings, labels)
        assert -1 <= score <= 1  # Silhouette score is bounded
        assert score > 0  # Should be positive for well-separated clusters
    
    def test_silhouette_analysis_single_cluster(self, metrics_calc):
        """Test silhouette analysis with single cluster"""
        embeddings = [np.array([1, 2]), np.array([3, 4])]
        labels = [0, 0]  # All same label
        
        score = metrics_calc.silhouette_analysis(embeddings, labels)
        assert score == 0.0  # Should be 0 for single cluster
    
    def test_disambiguation_accuracy(self, metrics_calc):
        """Test word sense disambiguation accuracy calculation"""
        # Mock AMBER model
        mock_model = Mock()
        mock_model.get_contextual_embedding.side_effect = [
            np.array([1, 0, 0]),  # Context 1
            np.array([0, 1, 0]),  # Context 2
            np.array([1, 0.1, 0]),  # Context 1 (similar to first)
            np.array([0.1, 1, 0])   # Context 2 (similar to second)
        ]
        
        word_contexts = {
            'bank': [
                {'sentence': 'Financial bank', 'label': 0, 'doc_idx': 0},
                {'sentence': 'River bank', 'label': 1, 'doc_idx': 1}
            ],
            'test': [
                {'sentence': 'Test context 1', 'label': 0, 'doc_idx': 2},
                {'sentence': 'Test context 2', 'label': 1, 'doc_idx': 3}
            ]
        }
        
        accuracies = metrics_calc.disambiguation_accuracy(
            word_contexts, mock_model, method="multi_head"
        )
        
        assert 'bank' in accuracies
        assert 'test' in accuracies
        assert 0 <= accuracies['bank'] <= 1
        assert 0 <= accuracies['test'] <= 1
    
    def test_comprehensive_evaluation(self, metrics_calc):
        """Test comprehensive evaluation across methods"""
        # Mock AMBER model
        mock_model = Mock()
        mock_model.get_contextual_embedding.return_value = np.random.randn(100)
        
        test_cases = [
            {
                'word': 'bank',
                'contexts': [
                    {'sentence': 'Financial bank', 'doc_idx': 0},
                    {'sentence': 'River bank', 'doc_idx': 1}
                ]
            }
        ]
        
        results = metrics_calc.comprehensive_evaluation(
            test_cases, mock_model, methods=["multi_head", "positional"]
        )
        
        assert 'multi_head' in results
        assert 'positional' in results
        
        for method in results:
            assert 'avg_css' in results[method]
            assert 'avg_disambiguation_accuracy' in results[method]
            assert 'css_scores' in results[method]
            assert 'disambiguation_accuracies' in results[method]
    
    def test_compare_with_baseline(self, metrics_calc):
        """Test comparison with baseline embeddings"""
        # Mock AMBER model
        mock_model = Mock()
        mock_model.vector_size = 100
        mock_model.get_contextual_embedding.side_effect = [
            np.array([1, 0, 0, 0, 0]),  # Context 1
            np.array([0, 1, 0, 0, 0])   # Context 2  
        ]
        
        test_cases = [
            {
                'word': 'bank',
                'contexts': [
                    {'sentence': 'Financial bank', 'doc_idx': 0},
                    {'sentence': 'River bank', 'doc_idx': 1}
                ]
            }
        ]
        
        baseline_embeddings = {
            'bank': np.array([0.5, 0.5, 0, 0, 0])  # Static embedding
        }
        
        comparison = metrics_calc.compare_with_baseline(
            test_cases, mock_model, baseline_embeddings, method="multi_head"
        )
        
        assert 'amber' in comparison
        assert 'baseline' in comparison
        assert 'improvement' in comparison
        
        # AMBER should show higher CSS than baseline
        assert comparison['amber']['avg_css'] >= comparison['baseline']['avg_css']
        assert comparison['improvement'] >= 0
    
    def test_metrics_with_edge_cases(self, metrics_calc):
        """Test metrics calculator with edge cases"""
        # Test with single embedding
        single_embedding = [np.array([1, 2, 3])]
        single_labels = [0]
        
        # Should handle gracefully
        css = metrics_calc.css_calculator.calculate(single_embedding)
        assert css == 0.0
        
        # Test clustering purity with single point
        purity = metrics_calc.clustering_purity(single_embedding, single_labels)
        assert purity >= 0  # Should not crash
    
    def test_distance_matrix_with_zero_vectors(self, metrics_calc):
        """Test distance matrix calculation with zero vectors"""
        embeddings = [
            np.zeros(3),
            np.array([1, 0, 0]),
            np.zeros(3)
        ]
        
        distance_matrix = metrics_calc.cosine_distance_matrix(embeddings)
        
        assert distance_matrix.shape == (3, 3)
        assert np.allclose(np.diag(distance_matrix), 0)
        # Should handle zero vectors gracefully without NaN
        assert not np.any(np.isnan(distance_matrix))


class TestMetricsIntegration:
    """Integration tests for metrics functionality"""
    
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline"""
        # Create synthetic data
        embeddings_static = [
            np.array([1, 0, 0]) for _ in range(4)  # Same embeddings (static)
        ]
        
        embeddings_contextual = [
            np.array([1, 0, 0]),    # Context 1
            np.array([0, 1, 0]),    # Context 2
            np.array([0.9, 0.1, 0]), # Context 1 (similar)
            np.array([0.1, 0.9, 0])  # Context 2 (similar)
        ]
        
        labels = [0, 1, 0, 1]  # Two contexts
        
        calc = MetricsCalculator()
        
        # Calculate metrics for both
        css_static = calc.css_calculator.calculate(embeddings_static)
        css_contextual = calc.css_calculator.calculate(embeddings_contextual)
        
        purity_static = calc.clustering_purity(embeddings_static, labels)
        purity_contextual = calc.clustering_purity(embeddings_contextual, labels)
        
        # Contextual should outperform static
        assert css_contextual > css_static
        assert purity_contextual >= purity_static
        
        print(f"Static CSS: {css_static:.4f}, Contextual CSS: {css_contextual:.4f}")
        print(f"Static Purity: {purity_static:.4f}, Contextual Purity: {purity_contextual:.4f}")


if __name__ == "__main__":
    pytest.main([__file__])