"""
utils/config.py
=================

WHAT THIS FILE HOLDS
----------------------
Settings for the embedding model, t-SNE reproducibility, and where the
output plot gets saved.

WORTH NOTING: NO AWS, NO SECRETS, NO .env FILE HERE
-------------------------------------------------------
Unlike Practicals 3-5 (all Bedrock-based, needing AWS credentials kept
out of git via python-dotenv), this practical runs entirely LOCALLY.
SBERT downloads its public model weights from Hugging Face once (cached
afterward by the sentence-transformers library itself) and needs no API
key. That means there's nothing sensitive to hide in a .env file this
time — plain environment variables (or just editing the defaults below)
are enough, so this file skips python-dotenv entirely rather than adding
a dependency with nothing to protect.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    # all-MiniLM-L6-v2: a small (~80MB) SBERT model producing 384-dim
    # embeddings. Chosen over a larger model (e.g. all-mpnet-base-v2,
    # ~420MB, 768-dim, slightly higher accuracy) because for 20 short
    # sentences the accuracy gap is marginal, but the speed/size gap
    # isn't — MiniLM encodes near-instantly, which matters for a
    # practical meant to run quickly end-to-end. It's also the specific
    # model named in the practical brief.
    model_name: str = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")

    # Fixed seed so re-running the notebook reproduces the same t-SNE
    # layout every time. t-SNE is stochastic by default — without a fixed
    # seed, the plot (and any conclusions drawn from "these points are
    # near each other") would look different on every run, making it
    # hard to discuss a specific result with someone else.
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

    # t-SNE's perplexity is roughly "how many neighbors" each point
    # balances against, and MUST be smaller than the number of samples —
    # scikit-learn raises an error otherwise. With only 20 sentences
    # (4 topics x 5 each), 5 keeps each point's "neighborhood" roughly
    # topic-sized; the commonly-cited perplexity range of 5-50 assumes
    # datasets with hundreds+ of points, so that guidance doesn't
    # directly apply at this scale.
    tsne_perplexity: int = int(os.getenv("TSNE_PERPLEXITY", "5"))

    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")


config = EmbeddingConfig()
