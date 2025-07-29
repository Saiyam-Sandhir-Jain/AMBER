"""
Setup script for AMBER package
"""

from setuptools import setup, find_packages
import pathlib

# Read the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements(filename):
    """Read requirements from requirements.txt file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

requirements = read_requirements('requirements.txt')

# Extract version from package
def get_version():
    """Extract version from package __init__.py"""
    version_file = here / "amber" / "__init__.py"
    try:
        with open(version_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return "0.1.0"  # fallback version

setup(
    name="amber-embeddings",
    version=get_version(),
    
    # Author and contact information
    author="Saiyam Jain",
    author_email="saiyam.sandhir.jain@gmail.com",
    maintainer="Saiyam Jain",
    maintainer_email="saiyam.sandhir.jain@gmail.com",
    
    # Package description
    description="Attention-based Multi-head Bidirectional Enhanced Representations for contextual word embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # URLs
    url="https://github.com/Saiyam-Sandhir-Jain/AMBER",
    project_urls={
        "Bug Tracker": "https://github.com/Saiyam-Sandhir-Jain/AMBER/issues",
        "Documentation": "https://amber-embeddings.readthedocs.io/",
        "Source Code": "https://github.com/Saiyam-Sandhir-Jain/AMBER",
        "Homepage": "https://github.com/Saiyam-Sandhir-Jain/AMBER",
    },
    
    # Package discovery
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.7",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "isort>=5.0.0",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "test": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "pytest-mock>=3.6.0",
        ],
    },
    
    # Package data
    package_data={
        "amber": ["*.json", "*.yaml", "*.yml"],
    },
    
    # Classifiers for PyPI
    classifiers=[
        # Development status
        "Development Status :: 4 - Beta",
        
        # Intended audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Operating system
        "Operating System :: OS Independent",
        
        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "nlp", "word-embeddings", "attention", "context-aware",
        "word-sense-disambiguation", "semantic-search", "tfidf",
        "word2vec", "machine-learning", "natural-language-processing"
    ],
    
    # Entry points (if any command-line tools)
    entry_points={
        "console_scripts": [
            # Add any command-line scripts here
            # "amber-demo=amber.utils:quick_demo",
        ],
    },
    
    # Zip safety
    zip_safe=False,
)