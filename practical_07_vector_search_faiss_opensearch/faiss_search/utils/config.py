"""
faiss_search/utils/config.py
===============================

WHAT THIS FILE HOLDS
----------------------
Settings for the local FAISS half of the practical: which SBERT model to
embed with, and default search behavior (how many results to return).

No AWS involved here at all -- this whole sub-project runs on your
machine with no cloud account, which is the entire point of "start
local" before "move to AWS" in the next sub-project.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FaissConfig:
    # Same model as any other SBERT-based practical in this repo -- keeps
    # embedding behavior (and dimensionality) consistent and comparable.
    model_name: str = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")

    # How many nearest neighbors to return by default when searching.
    default_top_k: int = int(os.getenv("FAISS_TOP_K", "5"))


config = FaissConfig()
