# Practical 02 - Word2Vec Exploration & Visualization

## Overview

This practical demonstrates how Word2Vec learns the semantic meaning of words by converting them into dense numerical vectors (embeddings). We train a Word2Vec model using Gensim, explore similar words, test analogies, and visualize embeddings using PCA and t-SNE.

## Objective

- Train a Word2Vec model.
- Understand word embeddings.
- Explore vocabulary.
- Find similar words.
- Test word analogies.
- Visualize embeddings using PCA and t-SNE.
- Save the trained model.

## Technologies

- Python
- Pandas
- NumPy
- Gensim
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Project Structure

```text
practical-02-word2vec-exploration/
├── data/
├── models/
├── notebooks/
├── README.md
├── requirements.txt
└── .gitignore
```

## Workflow

### 1. Import Libraries

Import all required libraries for preprocessing, model training, and visualization.

### 2. Load Text Corpus

Load the dataset containing text documents.

**Why?**
Word2Vec learns relationships from sentences.

### 3. Text Preprocessing

Convert text into lowercase tokens.

Example:

Machine Learning is Amazing

↓

['machine', 'learning', 'is', 'amazing']

### 4. Train Word2Vec

Train the model with parameters such as:

- vector_size = 100
- window = 5
- min_count = 2
- sg = 1 (Skip-Gram)
- epochs = 20

**Why?**
The model learns contextual relationships between words.

### 5. Explore Vocabulary

Display all learned words.

### 6. Most Similar Words

Use:
`model.wv.most_similar("computer")`

This shows semantically related words.

### 7. Word Analogy

Example:
King - Man + Woman ≈ Queen

This demonstrates that Word2Vec captures relationships between words.

### 8. Select 50 Words

Choose words from Sports, Food and Technology.

### 9. PCA Visualization

Reduce embeddings to 2 dimensions for visualization.

### 10. t-SNE Visualization

Generate clearer clusters of similar words.

### 11. Save Model

Save the trained Word2Vec model for future use.

## Expected Output

- Trained Word2Vec model
- Learned vocabulary
- Similar words
- Word analogy results
- PCA plot
- t-SNE plot
- Saved model

## Learning Outcomes

- Understand Word2Vec
- Learn semantic similarity
- Understand word embeddings
- Visualize high-dimensional vectors
- Compare PCA and t-SNE

## Future Improvements

- Use a larger corpus
- Compare CBOW vs Skip-Gram
- Compare Word2Vec with FastText and GloVe

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook
```

## Conclusion

Word2Vec converts words into dense numerical vectors that preserve semantic meaning. Words used in similar contexts are placed close together in vector space. PCA and t-SNE help visualize these relationships, making Word2Vec a fundamental concept in modern NLP and Generative AI.
