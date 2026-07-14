"""
tests/test_visualize.py
==========================
Unit tests for src/visualize.py.

What we verify:
    1. reduce_to_2d returns an (n, 2) array for n input embeddings.
    2. plot_embeddings returns a matplotlib Figure and, when given
       save_path, actually writes a PNG file to disk.

We deliberately DON'T assert anything about pixel content — that's not
meaningfully testable and isn't the point. We're checking the function
does what it's supposed to structurally (right shapes, file gets
written), not judging the plot's aesthetics.
"""

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so tests don't need a display

import numpy as np

from src.visualize import plot_embeddings, reduce_to_2d


def test_reduce_to_2d_returns_correct_shape():
    # t-SNE needs n_samples > perplexity (config default is 5), so use
    # more than 5 points here.
    embeddings = np.random.RandomState(0).rand(8, 16)
    coords = reduce_to_2d(embeddings)
    assert coords.shape == (8, 2)


def test_plot_embeddings_returns_figure_and_saves_png(tmp_path):
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])
    topics = ["A", "A", "B", "B"]
    save_path = str(tmp_path / "plot.png")

    fig = plot_embeddings(coords, topics, save_path=save_path)

    assert fig is not None
    assert os.path.exists(save_path)


def test_plot_embeddings_works_without_save_path():
    # save_path is optional -- should still return a usable Figure.
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    topics = ["A", "B"]
    fig = plot_embeddings(coords, topics)
    assert fig is not None
