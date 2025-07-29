API Reference
=============

This page contains the API reference documentation for all public classes and functions in the AMBER package.

Core Model
----------

.. automodule:: amber.model
   :members:
   :undoc-members:
   :show-inheritance:

AMBERModel
~~~~~~~~~~

.. autoclass:: amber.model.AMBERModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Metrics and Evaluation
----------------------

.. automodule:: amber.metrics
   :members:
   :undoc-members:
   :show-inheritance:

ContextSensitivityScore
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: amber.metrics.ContextSensitivityScore
   :members:
   :undoc-members:
   :show-inheritance:

MetricsCalculator
~~~~~~~~~~~~~~~~~

.. autoclass:: amber.metrics.MetricsCalculator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Model Comparison
----------------

.. automodule:: amber.comparator
   :members:
   :undoc-members:
   :show-inheritance:

AMBERComparator
~~~~~~~~~~~~~~~

.. autoclass:: amber.comparator.AMBERComparator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Utility Functions
-----------------

.. automodule:: amber.utils
   :members:
   :undoc-members:
   :show-inheritance:

Function Reference
~~~~~~~~~~~~~~~~~~

.. autofunction:: amber.utils.load_default_word2vec

.. autofunction:: amber.utils.preprocess_corpus

.. autofunction:: amber.utils.load_sample_corpus

.. autofunction:: amber.utils.create_test_cases

.. autofunction:: amber.utils.validate_inputs

.. autofunction:: amber.utils.export_embeddings

.. autofunction:: amber.utils.quick_demo

Package Information
-------------------

.. automodule:: amber
   :members:
   :undoc-members:

Constants and Variables
~~~~~~~~~~~~~~~~~~~~~~~

.. data:: amber.__version__
   
   Current version of the AMBER package.

.. data:: amber.__author__
   
   Authors of the AMBER package.

.. data:: amber.__email__
   
   Contact email for the AMBER package.

Examples
--------

Basic Usage Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from amber import AMBERModel, AMBERComparator
   
   # Initialize with your corpus
   corpus = [
       "The bass guitar sounds amazing",
       "He caught a large bass fish",
       "The apple fell from the tree",
       "Apple company released iPhone"
   ]
   
   # Create AMBER model
   model = AMBERModel(corpus)
   
   # Get contextual embeddings
   embedding1 = model.get_contextual_embedding(
       word="bass",
       sentence="The bass guitar sounds amazing",
       method="multi_head"
   )
   
   embedding2 = model.get_contextual_embedding(
       word="bass",
       sentence="He caught a large bass fish", 
       method="multi_head"
   )
   
   # Compare embeddings
   from amber import ContextSensitivityScore
   css = ContextSensitivityScore.calculate([embedding1, embedding2])
   print(f"Context Sensitivity Score: {css:.4f}")

Advanced Usage Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from amber import AMBERModel, AMBERComparator, MetricsCalculator
   import gensim.downloader as api
   
   # Load custom Word2Vec model
   w2v_model = api.load('word2vec-google-news-300')
   
   # Initialize AMBER with custom parameters
   model = AMBERModel(
       corpus=your_corpus,
       w2v_model=w2v_model,
       tfidf_params={
           'max_features': 10000,
           'min_df': 2,
           'max_df': 0.8
       }
   )
   
   # Batch processing
   batch_data = [
       {"word": "bank", "sentence": "Financial bank", "doc_idx": 0},
       {"word": "bank", "sentence": "River bank", "doc_idx": 1}
   ]
   
   embeddings = model.batch_contextual_embeddings(
       batch_data, 
       method="multi_head"
   )
   
   # Comprehensive evaluation
   comparator = AMBERComparator(model)
   test_cases = [
       {
           'word': 'bank',
           'contexts': [
               {'sentence': 'He went to the bank', 'type': 'Financial'},
               {'sentence': 'The river bank was muddy', 'type': 'Geographic'}
           ]
       }
   ]
   
   # Generate evaluation report
   report = comparator.generate_report(test_cases)
   print(report)
   
   # Calculate detailed metrics
   metrics_calc = MetricsCalculator()
   results = metrics_calc.comprehensive_evaluation(test_cases, model)

Error Handling
--------------

The AMBER package defines several custom exceptions and handles common error cases:

Common Error Cases
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from amber import AMBERModel
   
   try:
       # Empty corpus
       model = AMBERModel([])
   except ValueError as e:
       print(f"Error: {e}")
   
   try:
       # Unknown word
       embedding = model.get_contextual_embedding(
           word="nonexistent_word",
           sentence="This word doesn't exist",
           method="multi_head"
       )
       # Returns zero vector for unknown words
   except Exception as e:
       print(f"Unexpected error: {e}")
   
   try:
       # Invalid method
       embedding = model.get_contextual_embedding(
           word="test",
           sentence="Test sentence",
           method="invalid_method"
       )
   except ValueError as e:
       print(f"Invalid method error: {e}")

Performance Considerations
--------------------------

Memory Usage
~~~~~~~~~~~~

The AMBER model loads the entire Word2Vec model into memory. For large models:

.. code-block:: python

   # Limit corpus size for memory efficiency
   model = AMBERModel(
       corpus=large_corpus,
       max_corpus_size=10000  # Process only first 10k documents
   )

Computational Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use batch processing for multiple embeddings
   batch_data = [
       {"word": word, "sentence": sentence, "doc_idx": i}
       for i, (word, sentence) in enumerate(word_sentence_pairs)
   ]
   
   # More efficient than individual calls
   embeddings = model.batch_contextual_embeddings(batch_data)

Customization Options
---------------------

TF-IDF Customization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   custom_tfidf_params = {
       'max_features': 5000,      # Vocabulary size limit
       'min_df': 2,               # Minimum document frequency
       'max_df': 0.8,             # Maximum document frequency
       'stop_words': 'english',   # Remove stop words
       'ngram_range': (1, 2)      # Include bigrams
   }
   
   model = AMBERModel(corpus, tfidf_params=custom_tfidf_params)

Attention Parameters
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Multi-head attention parameters
   embedding = model.get_contextual_embedding(
       word="target_word",
       sentence="Context sentence",
       method="multi_head",
       num_heads=8,           # More heads for complex contexts
       temperature=0.5        # Lower temperature for sharper attention
   )
   
   # Positional attention parameters
   embedding = model.get_contextual_embedding(
       word="target_word",
       sentence="Context sentence",
       method="positional",
       window_size=7,         # Larger context window
       position_decay=1.5     # Stronger distance decay
   )