"""
Test Demo Script for AMBER Package

This script demonstrates AMBER with the specific examples requested:
- "bass" with first corpus
- "apple" with second corpus  
- Random word with custom corpus
"""

import numpy as np
from amber import AMBERModel, AMBERComparator, ContextSensitivityScore

def test_bass_first_corpus():
    """Test 'bass' with the first corpus (from your first code)"""
    print("="*60)
    print("TEST 1: 'BASS' WITH FIRST CORPUS")
    print("="*60)
    
    # First corpus (from your first code)
    first_corpus = [
        "The quick brown fox jumps over the lazy dog",
        "The dog barks loudly at the cat",
        "A quick brown cat runs fast",
        "The fox is a clever animal",
        "Dogs are loyal pets",
        "He went to the bank to deposit money",
        "The river bank was muddy after the rain",
        "The apple fell from the tree",
        "Apple released a new iPhone model",
        "The bass guitar sounds amazing",
        "He caught a large bass fish",
        "The bat flew through the night sky",
        "The baseball bat broke during the game"
    ]
    
    print(f"Corpus size: {len(first_corpus)} sentences")
    
    # Initialize AMBER model
    print("\nInitializing AMBER model...")
    amber_model = AMBERModel(first_corpus, max_corpus_size=100)
    
    # Test 'bass' in different contexts
    bass_contexts = [
        {"sentence": "The bass guitar sounds amazing", "type": "Music Context"},
        {"sentence": "He caught a large bass fish", "type": "Fish Context"}
    ]
    
    print(f"\nTesting word: 'bass'")
    print("-" * 30)
    
    bass_embeddings = []
    methods = ["multi_head", "positional", "tfidf_only"]
    
    for i, context in enumerate(bass_contexts):
        sentence = context["sentence"]
        context_type = context["type"]
        
        print(f"\n{context_type}: '{sentence}'")
        
        for method in methods:
            embedding = amber_model.get_contextual_embedding(
                word="bass",
                sentence=sentence,
                method=method,
                doc_idx=9 + i  # Use relevant doc indices
            )
            
            # Find most similar words
            comparator = AMBERComparator(amber_model)
            similar_words = comparator.find_most_similar_words(
                embedding, topn=5, exclude_words={"bass"}
            )
            
            print(f"  {method.replace('_', ' ').title()}: {[f'{w}({s:.3f})' for w, s in similar_words]}")
            
            if method == "multi_head":
                bass_embeddings.append(embedding)
    
    # Calculate Context Sensitivity Score
    css_score = ContextSensitivityScore.calculate(bass_embeddings)
    print(f"\nContext Sensitivity Score for 'bass': {css_score:.4f}")
    
    # Compare with static Word2Vec
    if "bass" in amber_model.w2v_model.key_to_index:
        static_vec = amber_model.w2v_model["bass"]
        static_similar = comparator.find_most_similar_words(
            static_vec, topn=5, exclude_words={"bass"}
        )
        print(f"Static Word2Vec: {[f'{w}({s:.3f})' for w, s in static_similar]}")
        
        # CSS for static (should be 0)
        static_css = ContextSensitivityScore.calculate([static_vec, static_vec])
        print(f"Static Word2Vec CSS: {static_css:.4f}")
    
    return amber_model, css_score

def test_apple_second_corpus():
    """Test 'apple' with the second corpus (from your second code)"""
    print("\n" + "="*60)
    print("TEST 2: 'APPLE' WITH SECOND CORPUS")
    print("="*60)
    
    # Second corpus (extended version from your second code)
    second_corpus = [
        "The quick brown fox jumps over the lazy dog in the park",
        "The dog barks loudly at the cat sitting on the fence",
        "A quick brown cat runs fast through the garden maze",
        "The fox is a clever animal that hunts during the night",
        "Dogs are loyal pets and make great family companions",
        "He went to the bank to deposit money and check balance",
        "The river bank was muddy after heavy rain yesterday",
        "Economics and banking systems require careful regulation",
        "The apple fell from the tree in the beautiful orchard",
        "Apple company released a new iPhone model with features",
        "Technology companies like Apple drive innovation forward",
        "Red apples are nutritious and taste delicious when fresh",
        "The bass guitar sounds amazing in the rock concert",
        "He caught a large bass fish in the clear mountain lake",
        "Music instruments produce different sounds and frequencies",
        "Fresh fish from the lake make an excellent dinner"
    ]
    
    print(f"Corpus size: {len(second_corpus)} sentences")
    
    # Initialize AMBER model
    print("\nInitializing AMBER model...")
    amber_model = AMBERModel(second_corpus, max_corpus_size=100)
    
    # Test 'apple' in different contexts
    apple_contexts = [
        {"sentence": "The apple fell from the tree in the beautiful orchard", "type": "Fruit Context"},
        {"sentence": "Apple company released a new iPhone model with features", "type": "Technology Context"}
    ]
    
    print(f"\nTesting word: 'apple'")
    print("-" * 30)
    
    apple_embeddings = []
    methods = ["multi_head", "positional", "tfidf_only"]
    
    for i, context in enumerate(apple_contexts):
        sentence = context["sentence"]
        context_type = context["type"]
        
        print(f"\n{context_type}: '{sentence}'")
        
        for method in methods:
            embedding = amber_model.get_contextual_embedding(
                word="apple",
                sentence=sentence,
                method=method,
                doc_idx=8 + i  # Use relevant doc indices
            )
            
            # Find most similar words
            comparator = AMBERComparator(amber_model)
            similar_words = comparator.find_most_similar_words(
                embedding, topn=5, exclude_words={"apple"}
            )
            
            print(f"  {method.replace('_', ' ').title()}: {[f'{w}({s:.3f})' for w, s in similar_words]}")
            
            if method == "multi_head":
                apple_embeddings.append(embedding)
    
    # Calculate Context Sensitivity Score
    css_score = ContextSensitivityScore.calculate(apple_embeddings)
    print(f"\nContext Sensitivity Score for 'apple': {css_score:.4f}")
    
    # Compare with static Word2Vec
    if "apple" in amber_model.w2v_model.key_to_index:
        static_vec = amber_model.w2v_model["apple"]
        static_similar = comparator.find_most_similar_words(
            static_vec, topn=5, exclude_words={"apple"}
        )
        print(f"Static Word2Vec: {[f'{w}({s:.3f})' for w, s in static_similar]}")
        
        # CSS for static (should be 0)
        static_css = ContextSensitivityScore.calculate([static_vec, static_vec])
        print(f"Static Word2Vec CSS: {static_css:.4f}")
    
    return amber_model, css_score

def test_random_word_custom_corpus():
    """Test random word 'python' with custom corpus"""
    print("\n" + "="*60)
    print("TEST 3: 'PYTHON' WITH CUSTOM CORPUS")
    print("="*60)
    
    # Custom corpus with technology and animal contexts
    custom_corpus = [
        "Programming languages are essential tools for software development",
        "Python is a popular programming language used in data science",
        "Machine learning algorithms can be implemented in Python easily",
        "Web development frameworks like Django are built with Python",
        "Data analysis and visualization are common Python applications",
        "Snakes are fascinating reptiles found in many ecosystems worldwide",
        "The python snake is one of the largest species of snakes",
        "Constrictor snakes like pythons hunt by wrapping around prey",
        "Wildlife photographers often capture images of python snakes",
        "Reptile enthusiasts study the behavior of various snake species",
        "Software engineers write code to solve complex problems",
        "Artificial intelligence research benefits from Python programming",
        "Natural language processing libraries are available in Python",
        "Jungle environments provide habitat for large python snakes",
        "Zoo exhibits often feature python snakes for educational purposes"
    ]
    
    print(f"Corpus size: {len(custom_corpus)} sentences")
    
    # Initialize AMBER model
    print("\nInitializing AMBER model...")
    amber_model = AMBERModel(custom_corpus, max_corpus_size=100)
    
    # Test 'python' in different contexts
    python_contexts = [
        {"sentence": "Python is a popular programming language used in data science", "type": "Technology Context"},
        {"sentence": "The python snake is one of the largest species of snakes", "type": "Animal Context"}
    ]
    
    print(f"\nTesting word: 'python'")
    print("-" * 30)
    
    python_embeddings = []
    methods = ["multi_head", "positional", "tfidf_only"]
    
    for i, context in enumerate(python_contexts):
        sentence = context["sentence"]
        context_type = context["type"]
        
        print(f"\n{context_type}: '{sentence}'")
        
        for method in methods:
            embedding = amber_model.get_contextual_embedding(
                word="python",
                sentence=sentence,
                method=method,
                doc_idx=1 + i * 6  # Use relevant doc indices
            )
            
            # Find most similar words
            comparator = AMBERComparator(amber_model)
            similar_words = comparator.find_most_similar_words(
                embedding, topn=5, exclude_words={"python"}
            )
            
            print(f"  {method.replace('_', ' ').title()}: {[f'{w}({s:.3f})' for w, s in similar_words]}")
            
            if method == "multi_head":
                python_embeddings.append(embedding)
    
    # Calculate Context Sensitivity Score
    css_score = ContextSensitivityScore.calculate(python_embeddings)
    print(f"\nContext Sensitivity Score for 'python': {css_score:.4f}")
    
    # Compare with static Word2Vec
    if "python" in amber_model.w2v_model.key_to_index:
        static_vec = amber_model.w2v_model["python"]
        static_similar = comparator.find_most_similar_words(
            static_vec, topn=5, exclude_words={"python"}
        )
        print(f"Static Word2Vec: {[f'{w}({s:.3f})' for w, s in static_similar]}")
        
        # CSS for static (should be 0)
        static_css = ContextSensitivityScore.calculate([static_vec, static_vec])
        print(f"Static Word2Vec CSS: {static_css:.4f}")
    
    return amber_model, css_score

def comprehensive_comparison():
    """Run comprehensive comparison across all test cases"""
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARISON")
    print("="*60)
    
    # Define all test cases
    all_test_cases = [
        {
            'word': 'bass',
            'contexts': [
                {
                    'sentence': 'The bass guitar sounds amazing',
                    'doc_idx': 9,
                    'type': 'Music Context'
                },
                {
                    'sentence': 'He caught a large bass fish',
                    'doc_idx': 10,
                    'type': 'Fish Context'
                }
            ]
        },
        {
            'word': 'apple',
            'contexts': [
                {
                    'sentence': 'The apple fell from the tree',
                    'doc_idx': 8,
                    'type': 'Fruit Context'
                },
                {
                    'sentence': 'Apple company released new iPhone',
                    'doc_idx': 9,
                    'type': 'Technology Context'
                }
            ]
        },
        {
            'word': 'python',
            'contexts': [
                {
                    'sentence': 'Python is a programming language',
                    'doc_idx': 1,
                    'type': 'Technology Context'
                },
                {
                    'sentence': 'The python snake wrapped around prey',
                    'doc_idx': 7,
                    'type': 'Animal Context'
                }
            ]
        }
    ]
    
    # Use the extended corpus from second test
    extended_corpus = [
        "The quick brown fox jumps over the lazy dog in the park",
        "Python is a popular programming language used in data science",
        "The dog barks loudly at the cat sitting on the fence",
        "A quick brown cat runs fast through the garden maze",
        "The fox is a clever animal that hunts during the night",
        "Dogs are loyal pets and make great family companions",
        "He went to the bank to deposit money and check balance",
        "The python snake wrapped around its prey in the jungle",
        "The apple fell from the tree in the beautiful orchard",
        "Apple company released a new iPhone model with features", 
        "The bass guitar sounds amazing in the rock concert",
        "He caught a large bass fish in the clear mountain lake",
        "Technology companies drive innovation in software development",
        "Wildlife photographers capture images of various snake species",
        "Fresh fruits like apples provide essential vitamins and nutrients",
        "Musical instruments create beautiful sounds for entertainment"
    ]
    
    # Initialize AMBER model with comprehensive corpus
    print("Initializing AMBER model with comprehensive corpus...")
    amber_model = AMBERModel(extended_corpus, max_corpus_size=100)
    
    # Initialize comparator
    comparator = AMBERComparator(amber_model)
    
    # Run evaluation
    print("\nRunning disambiguation evaluation...")
    results = comparator.evaluate_disambiguation(
        all_test_cases, 
        methods=["multi_head", "positional", "tfidf_only"]
    )
    
    # Calculate comprehensive metrics
    print("\nCalculating comprehensive metrics...")
    comparison_df = comparator.comprehensive_comparison(
        all_test_cases,
        methods=["multi_head", "positional", "tfidf_only"]
    )
    
    # Display summary
    print("\nSUMMARY RESULTS:")
    print("-" * 40)
    
    method_summary = comparison_df.groupby('method').agg({
        'css_score': ['mean', 'std'],
        'clustering_purity': ['mean', 'std']
    }).round(4)
    
    print("Method Performance:")
    for method in comparison_df['method'].unique():
        method_data = comparison_df[comparison_df['method'] == method]
        css_mean = method_data['css_score'].mean()
        css_std = method_data['css_score'].std()
        purity_mean = method_data['clustering_purity'].mean()
        purity_std = method_data['clustering_purity'].std()
        
        print(f"  {method.replace('_', ' ').title()}:")
        print(f"    CSS: {css_mean:.4f} ± {css_std:.4f}")
        print(f"    Clustering Purity: {purity_mean:.4f} ± {purity_std:.4f}")
    
    # Generate full report
    print("\n" + "="*60)  
    print("DETAILED EVALUATION REPORT")
    print("="*60)
    
    report = comparator.generate_report(
        all_test_cases,
        methods=["multi_head", "positional", "tfidf_only"]
    )
    print(report)
    
    return comparison_df

def main():
    """Run all tests"""
    print("🎯 AMBER PACKAGE DEMONSTRATION WITH SPECIFIC TEST CASES")
    print("=" * 80)
    
    try:
        # Test 1: Bass with first corpus
        model1, css1 = test_bass_first_corpus()
        
        # Test 2: Apple with second corpus  
        model2, css2 = test_apple_second_corpus()
        
        # Test 3: Python with custom corpus
        model3, css3 = test_random_word_custom_corpus()
        
        # Comprehensive comparison
        comparison_df = comprehensive_comparison()
        
        # Final summary
        print("\n" + "="*80)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\nKey Results:")
        print(f"• 'bass' disambiguation CSS: {css1:.4f}")
        print(f"• 'apple' disambiguation CSS: {css2:.4f}")  
        print(f"• 'python' disambiguation CSS: {css3:.4f}")
        print(f"• Average improvement over static embeddings: {np.mean([css1, css2, css3]):.4f}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()