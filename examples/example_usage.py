"""
Example Usage of AMBER Package

This script demonstrates how to use the AMBER package for context-aware
word embeddings with your custom corpus and word embedding models.
"""

import numpy as np
from amber import AMBERModel, AMBERComparator, MetricsCalculator, load_default_word2vec, preprocess_corpus

def main():
    print("="*80)
    print("AMBER PACKAGE DEMONSTRATION")
    print("="*80)
    
    # Example 1: Using AMBER with custom corpus
    print("\n1. BASIC USAGE WITH CUSTOM CORPUS")
    print("-" * 50)
    
    # Define your custom corpus
    custom_corpus = [
        "The quick brown fox jumps over the lazy dog",
        "The dog barks loudly at the cat in the yard",
        "A quick brown cat runs fast through the garden",
        "The fox is a clever animal that hunts at night",
        "Dogs are loyal pets and make great companions",
        "He went to the bank to deposit money and check his balance",
        "The river bank was muddy after the heavy rain yesterday",
        "The apple fell from the tree in the beautiful orchard",
        "Apple company released a new iPhone model with advanced features",
        "The bass guitar sounds amazing in the rock concert",
        "He caught a large bass fish in the clear lake",
        "The bat flew through the dark night sky silently",
        "The baseball bat broke during the intense game",
        "Python is a popular programming language for data science",
        "The python snake slithered through the tall grass"
    ]
    
    # Initialize AMBER model (will load default Word2Vec if none provided)
    print("Initializing AMBER model with custom corpus...")
    amber_model = AMBERModel(
        corpus=custom_corpus,
        w2v_model=None,  # Will use default Google News Word2Vec
        vector_size=300,
        max_corpus_size=1000
    )
    print("✓ AMBER model initialized successfully!")
    
    # Example 2: Getting contextual embeddings
    print("\n2. GETTING CONTEXTUAL EMBEDDINGS")
    print("-" * 50)
    
    # Test word "bass" in different contexts
    word = "bass"
    contexts = [
        "The bass guitar sounds amazing in concert",
        "He caught large bass fish in the lake"
    ]
    
    print(f"Testing word: '{word}'")
    for i, sentence in enumerate(contexts):
        print(f"\nContext {i+1}: '{sentence}'")
        
        # Get different types of embeddings
        methods = ["multi_head", "positional", "tfidf_only"]
        
        for method in methods:
            embedding = amber_model.get_contextual_embedding(
                word=word,
                sentence=sentence,
                method=method,
                doc_idx=i
            )
            print(f"  {method.replace('_', ' ').title()}: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
    
    # Example 3: Batch processing
    print("\n3. BATCH PROCESSING")
    print("-" * 50)
    
    # Process multiple word-context pairs at once
    batch_data = [
        {"word": "apple", "sentence": "The apple fell from the tree", "doc_idx": 0},
        {"word": "apple", "sentence": "Apple company released new iPhone", "doc_idx": 1},
        {"word": "bank", "sentence": "He went to the bank to deposit money", "doc_idx": 2},
        {"word": "bank", "sentence": "The river bank was muddy after rain", "doc_idx": 3}
    ]
    
    batch_embeddings = amber_model.batch_contextual_embeddings(
        batch_data, 
        method="multi_head"
    )
    
    print(f"Processed {len(batch_embeddings)} word-context pairs")
    for i, (data, embedding) in enumerate(zip(batch_data, batch_embeddings)):
        print(f"  {i+1}. '{data['word']}' in '{data['sentence'][:30]}...': {embedding.shape}")
    
    # Example 4: Model comparison and evaluation
    print("\n4. MODEL COMPARISON AND EVALUATION")
    print("-" * 50)
    
    # Define test cases for evaluation
    test_cases = [
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
                    'type': 'Fish Context'
                }
            ]
        },
        {
            'word': 'apple',
            'contexts': [
                {
                    'sentence': 'The apple fell from the tree',
                    'doc_idx': 6,
                    'type': 'Fruit Context'
                },
                {
                    'sentence': 'Apple company released new iPhone',
                    'doc_idx': 7,
                    'type': 'Company Context'
                }
            ]
        }
    ]
    
    # Initialize comparator
    comparator = AMBERComparator(amber_model)
    
    # Run disambiguation evaluation
    print("Running disambiguation evaluation...")
    disambiguation_results = comparator.evaluate_disambiguation(
        test_cases, 
        methods=["multi_head", "positional", "tfidf_only"]
    )
    
    # Example 5: Calculate Context Sensitivity Scores
    print("\n5. CONTEXT SENSITIVITY ANALYSIS")
    print("-" * 50)
    
    for test_case in test_cases:
        word = test_case['word']
        contexts = test_case['contexts']
        
        css_scores = comparator.calculate_context_sensitivity(
            word, contexts, methods=["multi_head", "positional", "tfidf_only"]
        )
        
        print(f"\nContext Sensitivity Scores for '{word}':")
        for method, score in css_scores.items():
            print(f"  {method.replace('_', ' ').title()}: {score:.4f}")
    
    # Example 6: Comprehensive evaluation report
    print("\n6. COMPREHENSIVE EVALUATION REPORT")
    print("-" * 50)
    
    # Generate detailed report
    evaluation_report = comparator.generate_report(
        test_cases, 
        methods=["multi_head", "positional", "tfidf_only"]
    )
    print(evaluation_report)
    
    # Example 7: Using custom Word2Vec model
    print("\n7. USING CUSTOM WORD2VEC MODEL")
    print("-" * 50)
    
    try:
        # Load a different pre-trained model (if available)
        import gensim.downloader as api
        
        print("Available models:")
        available_models = api.info()['models'].keys()
        for model_name in list(available_models)[:5]:  # Show first 5
            print(f"  - {model_name}")
        
        # Example with custom model (commented out to avoid long download)
        # custom_w2v = api.load('glove-wiki-gigaword-300')
        # custom_amber = AMBERModel(corpus=custom_corpus, w2v_model=custom_w2v)
        # print("✓ Custom Word2Vec model loaded successfully!")
        
    except Exception as e:
        print(f"Note: {e}")
    
    # Example 8: Metrics calculation
    print("\n8. DETAILED METRICS CALCULATION")
    print("-" * 50)
    
    metrics_calc = MetricsCalculator()
    
    # Get embeddings for "bass" in different contexts
    bass_embeddings = []
    bass_labels = []
    
    for i, context in enumerate(test_cases[0]['contexts']):
        sentence = context['sentence']
        doc_idx = context.get('doc_idx', i)
        
        embedding = amber_model.get_contextual_embedding(
            "bass", sentence, "multi_head", doc_idx
        )
        bass_embeddings.append(embedding)
        bass_labels.append(i)
    
    # Calculate various metrics
    css_score = metrics_calc.css_calculator.calculate(bass_embeddings)
    clustering_purity = metrics_calc.clustering_purity(bass_embeddings, bass_labels)
    cosine_distances = metrics_calc.cosine_distance_matrix(bass_embeddings)
    
    print(f"Metrics for 'bass' disambiguation:")
    print(f"  Context Sensitivity Score: {css_score:.4f}")
    print(f"  Clustering Purity: {clustering_purity:.4f}")
    print(f"  Cosine Distance Matrix:\n{cosine_distances}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nKey takeaways:")
    print("• AMBER successfully creates context-aware embeddings")
    print("• Multi-head attention performs best for disambiguation")
    print("• Context Sensitivity Score quantifies disambiguation ability")
    print("• The framework is modular and easily extensible")
    print("• Custom corpora and word embedding models are supported")

if __name__ == "__main__":
    main()