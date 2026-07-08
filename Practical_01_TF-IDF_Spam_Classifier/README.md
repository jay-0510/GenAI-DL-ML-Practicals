# Practical 01 - TF-IDF Spam Classifier

## Objective

Build a **TF-IDF + Logistic Regression** based SMS Spam Classifier and
compare its performance with a **Multinomial Naive Bayes** baseline.
Perform experiments using stopword removal and different n-gram ranges.

---

## Dataset

**SMS Spam Collection Dataset (Kaggle)**

Target Classes:

- Ham (0)
- Spam (1)

---

## Project Structure

```text
practical-01-tfidf-spam-classifier/
│
├── data/
│   └── spam.csv
├── notebooks/
│   └── practical-01-tfidf-spam-classifier.ipynb
├── models/
│   ├── logistic_model.pkl
│   └── vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Workflow

1.  Load Dataset
2.  Clean Dataset
3.  Exploratory Data Analysis
4.  Label Encoding
5.  Train/Test Split
6.  TF-IDF Feature Extraction
7.  Train Logistic Regression
8.  Evaluate Model
9.  Compare with Naive Bayes
10. Experiment with Stopword Removal
11. Experiment with n-grams
12. Save Model

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## Experiments

### Experiment 1

- TF-IDF without stopwords

### Experiment 2

- Unigram `(1,1)`
- Bigram `(1,2)`
- Trigram `(1,3)`

Compare performance and document observations.

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run

Open the Jupyter Notebook and execute all cells sequentially.

```bash
jupyter notebook
```

---

## Expected Output

- Trained Logistic Regression model
- Trained TF-IDF Vectorizer
- Model comparison with Naive Bayes
- Experimental observations

---
