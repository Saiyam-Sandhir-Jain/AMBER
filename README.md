# AMBER: Attention-based Multi-head Bidirectional Enhanced Representations

[![PyPI version](https://badge.fury.io/py/amber-embeddings.svg)](https://badge.fury.io/py/amber-embeddings)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AMBER is a hybrid word embedding framework that bridges statistical and neural semantics by combining TF-IDF weighting with multi-head self-attention mechanisms. It enhances static word embeddings (like Word2Vec) with contextual awareness, making them suitable for word sense disambiguation and semantic search tasks.

## 🚀 Key Features

- **Context-Aware Embeddings**: Transform static embeddings into context-sensitive representations
- **Hybrid Architecture**: Combines statistical (TF-IDF) and neural (attention) approaches
- **Multiple Attention Mechanisms**: Multi-head and positional attention variants
- **Plug-and-Play Design**: Works with any Word2Vec-compatible embedding model
- **Comprehensive Evaluation**: Built-in metrics including Context Sensitivity Score (CSS)
- **Lightweight & Interpretable**: Computationally efficient alternative to transformer models
- **Easy Integration**: Simple API for seamless integration into existing NLP pipelines

## 📦 Installation

```bash
pip install amber-embeddings
```

Or install from source:

```bash
git clone https://github.com/Saiyam-Sandhir-Jain/AMBER.git
cd AMBER
pip install -e .
```

## 🔧 Quick Start

```python
from amber import AMBERModel, AMBERComparator

# Your text corpus
corpus = [
    "The bass guitar sounds amazing in concert",
    "He caught a large bass fish in the lake",
    "The apple fell from the tree",
    "Apple company released a new iPhone"
]

# Initialize AMBER model (uses Google News Word2Vec by default)
amber_model = AMBERModel(corpus)

# Get context-aware embeddings
embedding1 = amber_model.get_contextual_embedding(
    word="bass",
    sentence="The bass guitar sounds amazing",
    method="multi_head"
)

embedding2 = amber_model.get_contextual_embedding(
    word="bass", 
    sentence="He caught a large bass fish",
    method="multi_head"
)

# The embeddings will be different, reflecting different contexts!
print(f"Embedding shapes: {embedding1.shape}, {embedding2.shape}")
```

## 📖 Core Components

### AMBERModel

The main model class that creates context-aware embeddings:

```python
from amber import AMBERModel
import gensim.downloader as api

# Option 1: Use default Word2Vec model
model = AMBERModel(corpus)

# Option 2: Use custom Word2Vec model
custom_w2v = api.load('word2vec-google-news-300')
model = AMBERModel(corpus, w2v_model=custom_w2v)

# Option 3: Custom TF-IDF parameters
model = AMBERModel(
    corpus,
    tfidf_params={
        'max_features': 5000,
        'min_df': 2,
        'max_df': 0.8
    }
)
```

### Embedding Methods

AMBER provides three embedding methods:

1. **Multi-head Attention** (`multi_head`): Best for disambiguation
2. **Positional Attention** (`positional`): Considers word proximity
3. **TF-IDF Only** (`tfidf_only`): Statistical weighting only

```python
# Multi-head attention (recommended)
embedding = model.get_contextual_embedding(
    word="bank",
    sentence="He went to the bank to deposit money",
    method="multi_head",
    num_heads=4,
    temperature=0.8
)

# Positional attention
embedding = model.get_contextual_embedding(
    word="bank",
    sentence="The river bank was muddy",
    method="positional",
    window_size=5
)
```

### Batch Processing

Process multiple word-context pairs efficiently:

```python
batch_data = [
    {"word": "apple", "sentence": "The apple fell from tree", "doc_idx": 0},
    {"word": "apple", "sentence": "Apple released new iPhone", "doc_idx": 1},
    {"word": "mouse", "sentence": "Computer mouse not working", "doc_idx": 2}
]

embeddings = model.batch_contextual_embeddings(batch_data, method="multi_head")
```

## 📊 Evaluation and Metrics

### Context Sensitivity Score (CSS)

CSS measures how much a word's embedding varies across different contexts:

```python
from amber import ContextSensitivityScore

# Calculate CSS for embeddings of the same word in different contexts
css_score = ContextSensitivityScore.calculate([embedding1, embedding2, embedding3])
print(f"Context Sensitivity Score: {css_score:.4f}")

# Higher CSS = better disambiguation ability
# Static embeddings have CSS ≈ 0
# AMBER embeddings have CSS > 0
```

### Model Comparison

Compare AMBER with baseline models:

```python
from amber import AMBERComparator

comparator = AMBERComparator(amber_model)

# Define test cases
test_cases = [
    {
        'word': 'bank',
        'contexts': [
            {'sentence': 'He went to the bank to deposit money', 'type': 'Financial'},
            {'sentence': 'The river bank was muddy after rain', 'type': 'Geographic'}
        ]
    }
]

# Run comprehensive evaluation
results = comparator.evaluate_disambiguation(test_cases)
report = comparator.generate_report(test_cases)
print(report)
```

### Visualization

Generate comparison plots:

```python
# Run comprehensive comparison
comparison_df = comparator.comprehensive_comparison(test_cases)

# Create visualizations
comparator.visualize_comparison(comparison_df, save_path="amber_comparison.png")
```

## 🔬 Advanced Usage

### Custom Word Embedding Models

```python
import gensim.downloader as api

# Use different pre-trained models
glove_model = api.load('glove-wiki-gigaword-300')
amber_glove = AMBERModel(corpus, w2v_model=glove_model)

# Or load your own trained model
from gensim.models import KeyedVectors
custom_model = KeyedVectors.load_word2vec_format('path/to/your/model.bin', binary=True)
amber_custom = AMBERModel(corpus, w2v_model=custom_model)
```

### Fine-tuning Parameters

```python
# Adjust attention parameters
embedding = model.get_contextual_embedding(
    word="python",
    sentence="Python is a programming language",
    method="multi_head",
    num_heads=8,           # More heads for complex contexts
    temperature=0.5        # Lower temperature for sharper attention
)

# Adjust positional attention
embedding = model.get_contextual_embedding(
    word="spring",
    sentence="Spring brings beautiful flowers",
    method="positional", 
    window_size=7,         # Larger context window
    position_decay=1.5     # Stronger distance decay
)
```

### Export Embeddings

```python
from amber.utils import export_embeddings

# Export for external use
words_and_contexts = [
    {"word": "bank", "sentence": "Financial bank services"},
    {"word": "bank", "sentence": "River bank location"}
]

# Export as dictionary
embeddings_dict = export_embeddings(model, words_and_contexts, output_format="dict")

# Export as numpy array
embeddings_array = export_embeddings(model, words_and_contexts, output_format="array")

# Export as pandas DataFrame (requires pandas)
embeddings_df = export_embeddings(model, words_and_contexts, output_format="dataframe")
```

## 🧪 Evaluation Results

AMBER demonstrates significant improvements over static embeddings:

| Method | Context Sensitivity Score | Disambiguation Accuracy |
|--------|--------------------------|-------------------------|
| Word2Vec (Static) | 0.000 | 61% |
| TF-IDF Weighted | 0.018 | 73% |
| AMBER Multi-head | **0.043** | **87%** |
| AMBER Positional | 0.037 | 84% |

## 🔍 Use Cases

- **Word Sense Disambiguation**: Distinguish between different meanings of polysemous words
- **Semantic Search**: Improve search relevance with context-aware embeddings
- **Document Classification**: Enhanced feature representations for text classification
- **Similarity Matching**: More accurate semantic similarity in specific domains
- **Information Retrieval**: Better query-document matching with contextual understanding

## 🛠️ Technical Details

### Architecture

AMBER enhances static embeddings through three key components:

1. **TF-IDF Scaling**: Weights embeddings by lexical importance
2. **Multi-head Attention**: Captures different types of contextual relationships
3. **Residual Fusion**: Preserves semantic stability while adding contextual adaptation

### Mathematical Foundation

The final contextual embedding is computed as:

```
F = γ · MultiHeadAttention(TF-IDF(E)) + (1 - γ) · E
```

Where:
- `E` is the original Word2Vec embedding
- `TF-IDF(E)` applies sentence-level TF-IDF weighting
- `MultiHeadAttention` computes contextual relationships
- `γ` controls the context-static balance

## 📚 Citation

If you use AMBER in your research, please cite:

```bibtex
@article{jain2024amber,
  title={not decided},
  author={Jain, Saiyam and Bhowmik, Swaroop and Choudhury, Dipanjan},
  journal={not decided},
  year={}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google News Word2Vec model for default embeddings
- Gensim library for word embedding utilities
- scikit-learn for TF-IDF implementation
- The research community for inspiration and feedback

## 📞 Support

- **Documentation**: [Read the Docs](https://amber-embeddings.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/Saiyam-Sandhir-Jain/AMBER/issues)
- **Email**: saiyam.sandhir.jain@gmail.com
- **Paper**: [arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)

---

**AMBER**: Making word embeddings context-aware, one attention head at a time! 🎯