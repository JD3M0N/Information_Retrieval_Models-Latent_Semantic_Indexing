# Latent Semantic Indexing (LSI) for Information Retrieval

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About
Implementation of a Latent Semantic Indexing (LSI) model for information retrieval systems. The model projects documents and queries into a low-dimensional semantic space using Singular Value Decomposition (SVD) on a TF-IDF matrix, and uses cosine similarity to retrieve the most relevant documents for a query.

**Author**: Josue Rolando Naranjo Sieiro  
**Program**: Computer Science, 3rd year, University of Havana

## Table of Contents
- [Latent Semantic Indexing (LSI) for Information Retrieval](#latent-semantic-indexing-lsi-for-information-retrieval)
  - [About](#about)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Repository Structure](#repository-structure)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [LSI Implementation Details](#lsi-implementation-details)
  - [Internal Validations](#internal-validations)
  - [Implemented Improvements](#implemented-improvements)
  - [Future Enhancements](#future-enhancements)
  - [References](#references)
  - [Contact](#contact)

## Features
- Document vectorization using TF-IDF
- Dimensionality reduction with Truncated SVD
- Cosine similarity for ranking documents
- Evaluation metrics: precision, recall, and F1
- Support for the Cranfield test collection

## Repository Structure
- **models/** - Modules for each IR model, including `Josue_Rolando_Naranjo_Sieiro_C_311.py` which implements LSI
- **main.py** - Main script that processes all models, executes fit and evaluate, and displays results
- **template.py** - Base template for implementing any retrieval model
- **metrics.py** - Functions to calculate precision, recall and F1
- **ranking.py** - Results table generation
- **start.sh** - Bash script to launch main.py with default parameters
- **test_lsi.py** - (Optional) Test script to print results and metrics for specific queries

## Requirements
- Python 3.8+
- ir_datasets
- scikit-learn
- numpy
- scipy
- tabulate
- gensim (only if using the alternative approach with Gensim)

## Installation
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the main script to evaluate all models with the Cranfield corpus:
```bash
bash start.sh
```
or
```bash
python main.py --corpus cranfield --metrics precision recall f1
```

To view top-k results for a specific query, use the test script:
```bash
python test_lsi.py
```

## LSI Implementation Details
- **Initialization**: Configuration of TfidfVectorizer with lowercase=True, stop_words='english' and max_df=0.8
- **Fit Method**:
    - Loads documents and queries from ir_datasets
    - Vectorizes documents using TF-IDF
    - Applies truncated SVD with n_components=100
    - Stores the document matrix projected to latent space
- **Predict Method**:
    - Vectorizes the query
    - Projects it to latent space
    - Calculates cosine similarity between query and documents
    - Returns a list of the most similar document IDs
- **Evaluate Method**:
    - Gets predictions with predict
    - Compares with qrels to calculate all_relevant, all_retrieved, and relevant_retrieved
    - Returns a dictionary with metrics aggregated by query

## Internal Validations
The implementation includes assertions to ensure:
- TF-IDF matrix has as many rows as documents
- Resulting latent matrix has shape (n_docs, n_components)
- Fit is executed before predict

## Implemented Improvements
- **Preprocessing**: TF-IDF configured with lowercase=True, stop_words='english' and max_df=0.8 to filter noise and trivial terms before applying SVD
    - Expected benefit: higher quality of latent space, better precision and consistency in rankings

## Future Enhancements
- Adjust the number of dimensions k (n_components) and measure its effect on precision/recall
- Incorporate lemmatization or stemming using spaCy before vectorization
- Experiment with ngram_range in TfidfVectorizer to capture bigrams
- Test term filtering with additional min_df and max_df parameters
- Evaluate alternative approach using Gensim LsiModel

## References
- Deerwester, S.; Dumais, S. T.; Furnas, G. W.; Landauer, T. K.; Harshman, R. (1990). Indexing by Latent Semantic Analysis. Journal of the American Society for Information Science, 41(6): 391–407.
- Dumais, S. T.; Furnas, G. W.; Landauer, T. K.; Deerwester, S. C.; Harshman, R. (1988). Using latent semantic analysis to improve access to textual information. SIGCHI Conference on Human Factors in Computing Systems.
- Berry, M. W.; Dumais, S. T.; O'Brien, G. W. (1995). Using linear algebra for intelligent information retrieval. SIAM Review, 37(4): 573–595.
- Manning, C. D.; Raghavan, P.; Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.
- Baeza-Yates, R.; Ribeiro-Neto, B. (2011). Modern Information Retrieval: The Concepts and Technology Behind Search (2ª ed.). Addison-Wesley.

## Contact
**Josue Rolando Naranjo Sieiro**  
Telegram: [@HaRcArRyDeMoN](https://t.me/HaRcArRyDeMoN)