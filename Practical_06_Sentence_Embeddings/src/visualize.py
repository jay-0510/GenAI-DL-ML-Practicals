"""
src/visualize.py
===================

WHAT THIS FILE DOES
--------------------
Reduces high-dimensional sentence embeddings down to 2D via t-SNE, and
plots them colored by topic — so we can visually check whether sentences
about the same topic actually end up near each other in embedding space.

WHY t-SNE INSTEAD OF, SAY, PCA?
------------------------------------
PCA is a LINEAR projection — it preserves global variance well, but can
flatten genuinely separate clusters on top of each other if their
separation isn't aligned with the top principal components. t-SNE is
non-linear and specifically optimizes to keep points that were CLOSE in
the original high-dimensional space close in the 2D layout, which tends
to reveal cluster structure more clearly for exactly this "do
semantically similar things visually group together?" question — at the
cost that 2D distances no longer mean anything on an absolute scale
(only relative closeness/separation is interpretable, not exact
distances between clusters).
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils.config import config


def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    """
    Projects the (n, embedding_dim) embedding matrix down to (n, 2)
    coordinates for plotting.

    perplexity and random_state come from utils.config so every run (and
    every person running this notebook) gets the same, reproducible
    layout — see config.py's docstring for why perplexity specifically
    needs to be kept small for a 20-sentence dataset.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=config.tsne_perplexity,
        random_state=config.random_seed,
        init="pca",
    )
    return tsne.fit_transform(embeddings)


def plot_embeddings(
    coords_2d: np.ndarray,
    topics: List[str],
    sentences: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter-plots 2D coordinates, colored by topic, with a legend.

    Args:
        coords_2d: output of reduce_to_2d() — shape (n, 2).
        topics: topic label per point, in the SAME order as coords_2d's rows.
        sentences: if given, annotates each point with its first few
            words — optional, since with 20 points full labels can get
            cluttered; useful when eyeballing exactly which sentence
            ended up where.
        save_path: if given, saves the figure as a PNG there in addition
            to returning it — this is what turns the plot into a
            downloadable artifact instead of something that only exists
            inside the notebook session.

    Returns:
        The matplotlib Figure, so the notebook can display it inline via
        plt.show() without needing to re-read the file it just saved.
    """
    unique_topics = sorted(set(topics))
    color_map = plt.colormaps.get_cmap("tab10")
    topic_to_color = {topic: color_map(i / max(len(unique_topics) - 1, 1)) for i, topic in enumerate(unique_topics)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for topic in unique_topics:
        idxs = [i for i, t in enumerate(topics) if t == topic]
        ax.scatter(
            coords_2d[idxs, 0],
            coords_2d[idxs, 1],
            label=topic,
            color=topic_to_color[topic],
            s=80,
            alpha=0.85,
        )

    if sentences:
        for i, sentence in enumerate(sentences):
            short_label = " ".join(sentence.split()[:4]) + "..."
            ax.annotate(short_label, (coords_2d[i, 0], coords_2d[i, 1]), fontsize=7, alpha=0.7)

    ax.set_title("Sentence Embeddings (t-SNE projection), grouped by topic")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(title="Topic")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig
