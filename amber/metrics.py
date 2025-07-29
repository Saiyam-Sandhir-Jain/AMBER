"""
Evaluation Metrics for AMBER Model

This module provides various metrics for evaluating the performance of 
contextual word embeddings, including the Context Sensitivity Score (CSS).
"""

import numpy as np
from typing import List, Dict, Tuple, Union
from scipy.spatial.distance import cosine
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class ContextSensitivityScore:
    """
    Context Sensitivity Score (CSS) - A novel metric for measuring
    how much a word's embedding varies across different contexts.
    
    CSS = 1 - (average pairwise cosine similarity of same word across contexts)
    
    Higher CSS indicates better context sensitivity.
    Static embeddings have CSS ≈ 0, while contextual embeddings have CSS > 0.
    """
    
    @staticmethod
    def calculate(embeddings: List[np.ndarray]) -> float:
        """
        Calculate CSS for a list of embeddings of the same word in different contexts
        
        Parameters:
        -----------
        embeddings : List[np.ndarray]
            List of embedding vectors for the same word in different contexts
            
        Returns:
        --------
        float
            Context Sensitivity Score (0 to 1, higher is better for disambiguation)
        """
        if len(embeddings) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                vec1, vec2 = embeddings[i], embeddings[j]
                if np.linalg.norm(vec1) > 1e-8 and np.linalg.norm(vec2) > 1e-8:
                    sim = 1 - cosine(vec1, vec2)
                    similarities.append(sim)
        
        if not similarities:
            return 0.0
        
        avg_similarity = np.mean(similarities)
        return 1 - avg_similarity
    
    @staticmethod
    def calculate_batch(word_embeddings_dict: Dict[str, List[np.ndarray]]) -> Dict[str, float]:
        """
        Calculate CSS for multiple words
        
        Parameters:
        -----------
        word_embeddings_dict : Dict[str, List[np.ndarray]]
            Dictionary mapping words to their contextual embeddings
            
        Returns:
        --------
        Dict[str, float]
            CSS scores for each word
        """
        css_scores = {}
        for word, embeddings in word_embeddings_dict.items():
            css_scores[word] = ContextSensitivityScore.calculate(embeddings)
        return css_scores


class MetricsCalculator:
    """
    Comprehensive metrics calculator for evaluating contextual word embeddings
    """
    
    def __init__(self):
        self.css_calculator = ContextSensitivityScore()
    
    def cosine_distance_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Calculate pairwise cosine distances between embeddings
        
        Parameters:
        -----------
        embeddings : List[np.ndarray]
            List of embedding vectors
            
        Returns:
        --------
        np.ndarray
            Symmetric distance matrix
        """
        n = len(embeddings)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if (np.linalg.norm(embeddings[i]) > 1e-8 and 
                    np.linalg.norm(embeddings[j]) > 1e-8):
                    dist = cosine(embeddings[i], embeddings[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def clustering_purity(
        self, 
        embeddings: List[np.ndarray], 
        true_labels: List[int],
        n_clusters: int = None
    ) -> float:
        """
        Calculate clustering purity for contextual embeddings
        
        Parameters:
        -----------
        embeddings : List[np.ndarray]
            List of embedding vectors
        true_labels : List[int]
            True cluster labels
        n_clusters : int, optional
            Number of clusters (defaults to number of unique labels)
            
        Returns:
        --------
        float
            Clustering purity score (0 to 1, higher is better)
        """
        if not embeddings or not true_labels:
            return 0.0
        
        if n_clusters is None:
            n_clusters = len(set(true_labels))
        
        # Stack embeddings
        X = np.vstack(embeddings)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(X)
        
        # Calculate purity
        total_samples = len(true_labels)
        cluster_label_counts = defaultdict(lambda: defaultdict(int))
        
        for pred_label, true_label in zip(predicted_labels, true_labels):
            cluster_label_counts[pred_label][true_label] += 1
        
        purity_sum = 0
        for cluster_id in cluster_label_counts:
            max_count = max(cluster_label_counts[cluster_id].values())
            purity_sum += max_count
        
        return purity_sum / total_samples
    
    def silhouette_analysis(self, embeddings: List[np.ndarray], labels: List[int]) -> float:
        """
        Calculate silhouette score for embeddings
        
        Parameters:
        -----------
        embeddings : List[np.ndarray]
            List of embedding vectors
        labels : List[int]
            Cluster labels
            
        Returns:
        --------
        float
            Silhouette score (-1 to 1, higher is better)
        """
        if len(set(labels)) < 2:
            return 0.0
        
        X = np.vstack(embeddings)
        return silhouette_score(X, labels)
    
    def disambiguation_accuracy(
        self,
        word_contexts: Dict[str, List[Dict[str, Union[str, int]]]],
        model,
        method: str = "multi_head"
    ) -> Dict[str, float]:
        """
        Calculate word sense disambiguation accuracy
        
        Parameters:
        -----------
        word_contexts : Dict[str, List[Dict]]
            Dictionary mapping words to their contexts with labels
            Format: {word: [{'sentence': str, 'label': int, 'doc_idx': int}, ...]}
        model : AMBERModel
            Trained AMBER model
        method : str, default="multi_head"
            Embedding method to use
            
        Returns:
        --------
        Dict[str, float]
            Disambiguation accuracy for each word
        """
        accuracies = {}
        
        for word, contexts in word_contexts.items():
            if len(contexts) < 2:
                accuracies[word] = 0.0
                continue
            
            # Get embeddings
            embeddings = []
            true_labels = []
            
            for context in contexts:
                sentence = context['sentence']
                label = context['label']
                doc_idx = context.get('doc_idx', 0)
                
                embedding = model.get_contextual_embedding(
                    word, sentence, method, doc_idx
                )
                embeddings.append(embedding)
                true_labels.append(label)
            
            # Calculate clustering purity as disambiguation accuracy
            accuracy = self.clustering_purity(embeddings, true_labels)
            accuracies[word] = accuracy
        
        return accuracies
    
    def comprehensive_evaluation(
        self,
        test_cases: List[Dict],
        model,
        methods: List[str] = ["multi_head", "positional", "tfidf_only"]
    ) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation across multiple methods
        
        Parameters:
        -----------
        test_cases : List[Dict]
            Test cases with word disambiguation scenarios
        model : AMBERModel
            Trained AMBER model
        methods : List[str]
            Methods to evaluate
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Comprehensive evaluation results
        """
        results = {method: {} for method in methods}
        
        for method in methods:
            print(f"\n=== Evaluating {method} method ===")
            
            # Collect all embeddings and metrics
            all_css_scores = []
            all_disambiguation_accuracies = []
            
            for test_case in test_cases:
                word = test_case['word']
                contexts = test_case['contexts']
                
                # Get embeddings
                embeddings = []
                labels = []
                
                for i, context_info in enumerate(contexts):
                    sentence = context_info['sentence']
                    doc_idx = context_info.get('doc_idx', i)
                    
                    embedding = model.get_contextual_embedding(
                        word, sentence, method, doc_idx
                    )
                    embeddings.append(embedding)
                    labels.append(i)  # Use context index as label
                
                # Calculate CSS
                css_score = self.css_calculator.calculate(embeddings)
                all_css_scores.append(css_score)
                
                # Calculate disambiguation accuracy
                disambiguation_acc = self.clustering_purity(embeddings, labels)
                all_disambiguation_accuracies.append(disambiguation_acc)
            
            # Store method results
            results[method] = {
                'avg_css': np.mean(all_css_scores),
                'std_css': np.std(all_css_scores),
                'avg_disambiguation_accuracy': np.mean(all_disambiguation_accuracies),
                'std_disambiguation_accuracy': np.std(all_disambiguation_accuracies),
                'css_scores': all_css_scores,
                'disambiguation_accuracies': all_disambiguation_accuracies
            }
            
            print(f"Average CSS: {results[method]['avg_css']:.4f} ± {results[method]['std_css']:.4f}")
            print(f"Average Disambiguation Accuracy: {results[method]['avg_disambiguation_accuracy']:.4f} ± {results[method]['std_disambiguation_accuracy']:.4f}")
        
        return results
    
    def compare_with_baseline(
        self,
        test_cases: List[Dict],
        amber_model,
        baseline_embeddings: Dict[str, np.ndarray],
        method: str = "multi_head"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare AMBER with baseline static embeddings
        
        Parameters:
        -----------
        test_cases : List[Dict]
            Test cases for evaluation
        amber_model : AMBERModel
            AMBER model to evaluate
        baseline_embeddings : Dict[str, np.ndarray]
            Static baseline embeddings for each word
        method : str, default="multi_head"
            AMBER method to use
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Comparison results
        """
        amber_results = []
        baseline_results = []
        
        for test_case in test_cases:
            word = test_case['word']
            contexts = test_case['contexts']
            
            # AMBER embeddings
            amber_embeddings = []
            for context_info in contexts:
                sentence = context_info['sentence']
                doc_idx = context_info.get('doc_idx', 0)
                
                embedding = amber_model.get_contextual_embedding(
                    word, sentence, method, doc_idx
                )
                amber_embeddings.append(embedding)
            
            # Baseline embeddings (same for all contexts)
            if word in baseline_embeddings:
                baseline_embs = [baseline_embeddings[word]] * len(contexts)
            else:
                baseline_embs = [np.zeros(amber_model.vector_size)] * len(contexts)
            
            # Calculate CSS for both
            amber_css = self.css_calculator.calculate(amber_embeddings)
            baseline_css = self.css_calculator.calculate(baseline_embs)
            
            amber_results.append(amber_css)
            baseline_results.append(baseline_css)
        
        return {
            'amber': {
                'avg_css': np.mean(amber_results),
                'std_css': np.std(amber_results),
                'css_scores': amber_results
            },
            'baseline': {
                'avg_css': np.mean(baseline_results),
                'std_css': np.std(baseline_results),
                'css_scores': baseline_results
            },
            'improvement': np.mean(amber_results) - np.mean(baseline_results)
        }