# Practical 6 — Sentence Embeddings & Similarity

1. Encodes 20 sentences using **SBERT (`all-MiniLM-L6-v2`)**, computes
   pairwise cosine similarity, and finds the best match for a query.
2. Visualizes the embeddings with **t-SNE**, grouped by topic, and checks
   whether meaningful clusters actually form.

---

## Project structure

```
practical_06_sentence_embeddings/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/
│   └── sentence_embeddings.ipynb   # narrated, end-to-end walkthrough
├── src/
│   ├── __init__.py
│   ├── data.py                      # the 20 labeled example sentences
│   ├── embedder.py                  # loads SBERT, encodes sentences
│   ├── similarity.py                # pairwise cosine similarity + best-match search
│   └── visualize.py                 # t-SNE projection + topic-colored plotting
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # shared fake-SBERT-model fixture
│   ├── test_embedder.py
│   ├── test_similarity.py
│   └── test_visualize.py
├── utils/
│   ├── __init__.py
│   └── config.py                    # model name, random seed, t-SNE settings
└── outputs/                          # generated plot PNGs land here (gitignored)
```

### Design principle

| File | Responsibility |
|---|---|
| `utils/config.py` | Model name, reproducibility seed, t-SNE hyperparameters |
| `src/data.py` | The fixed 20-sentence, 4-topic dataset used throughout |
| `src/embedder.py` | Loads SBERT once (cached) and encodes text into vectors |
| `src/similarity.py` | Pairwise similarity matrix + best-match / top-k search |
| `src/visualize.py` | t-SNE reduction + topic-colored scatter plot, saved as PNG |

Each module does one thing, so each can be tested and reasoned about in
isolation — `similarity.py`'s tests never need a real model (they work
directly on hand-built vectors), and `embedder.py`'s tests never need real
weights (a fake model fixture stands in).

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. No credentials needed
SBERT downloads `all-MiniLM-L6-v2` (~80MB) from Hugging Face the first
time it's used, then caches it locally — no account, no API key. After
that first run, this project works fully offline.

### 3. (Optional) override defaults via environment variables
```bash
export SBERT_MODEL_NAME=all-MiniLM-L6-v2
export RANDOM_SEED=42
export TSNE_PERPLEXITY=5
```

---

## Usage

### Run the notebook (recommended)
```bash
cd notebook
jupyter notebook sentence_embeddings.ipynb
```

### Or run modules directly
```bash
python -m src.embedder   # smoke test: encodes the 20 sentences, prints the shape
```

### Run the tests
```bash
pytest tests/ -v
```
All 10 tests run in a couple of seconds against a mocked SBERT model — no
model download, no GPU, fully deterministic.

---

## Task 1 summary: encoding & similarity search

**Flow:** encode 20 labeled sentences with SBERT → compute an (n, n)
cosine similarity matrix → for a new query, encode it the same way and
find which of the 20 sentences it's most similar to.

| Choice | Why this, not that |
|---|---|
| SBERT, not plain BERT | Plain BERT's token embeddings weren't trained to be directly comparable by cosine similarity — averaging them gives poor similarity rankings. SBERT is fine-tuned specifically so semantic closeness maps to embedding closeness. |
| `all-MiniLM-L6-v2`, not a larger model | For 20 short sentences the accuracy gap vs. a bigger model (e.g. `all-mpnet-base-v2`) is marginal, but the speed/size gap isn't — MiniLM is faster and lighter, and it's the model named in the brief. |
| Cosine similarity, not Euclidean distance | SBERT embeddings encode meaning in *direction*, not magnitude — cosine similarity measures angle only, which is what's semantically meaningful here. |
| Normalize embeddings at encode time | Once vectors are unit-length, cosine similarity is just a dot product — faster, and removes a common source of inconsistent similarity math if normalization were applied unevenly downstream. |

## Task 2 summary: t-SNE visualization

**Flow:** reduce the 384-dim embeddings to 2D with t-SNE → plot, colored
by the 4 topic labels → visually check whether same-topic sentences
cluster together.

| Choice | Why this, not that |
|---|---|
| t-SNE, not PCA | PCA is linear and can flatten genuinely separate clusters if their separation isn't aligned with the top principal components. t-SNE is non-linear and optimizes specifically to preserve local neighborhoods, which suits a "do similar things visually group together?" question. |
| `perplexity=5`, not the commonly-cited 5-50 range | That range assumes hundreds+ of data points; with only 20 sentences, perplexity must stay well below the sample count, and a low value keeps each point's "neighborhood" roughly topic-sized. |
| Fixed random seed | t-SNE is stochastic — a fixed seed means the plot looks the same on every re-run, which matters if you want to discuss "this specific layout" with someone else. |
| 4 topics, 5 sentences each | Balanced, clearly-distinct topics give the visualization known ground-truth groups to check against — an unlabeled or imbalanced dataset would make "did meaningful clusters form?" much harder to judge. |

---

## Troubleshooting

- **`OSError: We couldn't connect to huggingface.co`** — the model
  hasn't been downloaded yet and there's no internet access at that
  moment. Run once with a working internet connection; after that,
  the model is cached and no further downloads are needed.
- **t-SNE raises a perplexity error** — perplexity must be strictly less
  than the number of samples. If you shrink the dataset below ~10
  sentences, lower `TSNE_PERPLEXITY` accordingly.
