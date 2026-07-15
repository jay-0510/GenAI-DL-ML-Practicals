"""
src/data.py
=============

WHAT THIS FILE HOLDS
----------------------
The fixed set of 20 example sentences used throughout this practical,
each tagged with a topic label.

WHY 20 SENTENCES ACROSS 4 TOPICS (5 EACH)?
----------------------------------------------
The practical asks two different things of this dataset:

  1. "Find the best match for a query" needs sentences similar enough
     WITHIN a topic that a query can meaningfully prefer one over
     another, but different enough ACROSS topics that a mismatch is
     obviously wrong — otherwise every similarity score looks about the
     same and "best match" stops being a meaningful question.

  2. "Do meaningful clusters form?" (via t-SNE) needs known GROUND-TRUTH
     groups — the topic labels — to check the visual clustering against.
     Without labels, there'd be no way to judge whether a t-SNE layout
     is actually meaningful or just visually plausible.

Topics were picked to be obviously distinct in everyday meaning (Sports,
Technology, Food & Cooking, Travel & Places), so that if clusters DON'T
form clearly, that's informative about the embedding model or method —
not just a symptom of picking topics that were too similar to separate
in the first place.
"""

from typing import List, TypedDict


class LabeledSentence(TypedDict):
    sentence: str
    topic: str


SENTENCE_DATA: List[LabeledSentence] = [
    # --- Sports ---
    {"sentence": "The striker scored a stunning goal in the final minute.", "topic": "Sports"},
    {"sentence": "She trains every morning to prepare for the marathon.", "topic": "Sports"},
    {"sentence": "The home team won the championship after a tense overtime.", "topic": "Sports"},
    {"sentence": "His serve was clocked at over 200 kilometers per hour.", "topic": "Sports"},
    {"sentence": "The coach called a timeout to reset the team's strategy.", "topic": "Sports"},

    # --- Technology ---
    {"sentence": "The new smartphone features a faster processor and better camera.", "topic": "Technology"},
    {"sentence": "Developers are increasingly using AI to write and review code.", "topic": "Technology"},
    {"sentence": "The startup raised funding to build its cloud infrastructure.", "topic": "Technology"},
    {"sentence": "A software update fixed the security vulnerability overnight.", "topic": "Technology"},
    {"sentence": "The company unveiled a robot capable of folding laundry.", "topic": "Technology"},

    # --- Food & Cooking ---
    {"sentence": "She simmered the tomato sauce for hours to deepen the flavor.", "topic": "Food & Cooking"},
    {"sentence": "The bakery's sourdough loaf sells out within an hour of opening.", "topic": "Food & Cooking"},
    {"sentence": "He seasoned the steak generously before searing it in the pan.", "topic": "Food & Cooking"},
    {"sentence": "The recipe calls for fresh basil, garlic, and olive oil.", "topic": "Food & Cooking"},
    {"sentence": "Street food vendors grilled skewers over an open flame.", "topic": "Food & Cooking"},

    # --- Travel & Places ---
    {"sentence": "The train wound through the mountains toward the coastal town.", "topic": "Travel & Places"},
    {"sentence": "Tourists gathered at sunset to watch the view from the old fort.", "topic": "Travel & Places"},
    {"sentence": "She backpacked across three countries in a single summer.", "topic": "Travel & Places"},
    {"sentence": "The airline announced a new direct route to the island.", "topic": "Travel & Places"},
    {"sentence": "The old quarter's narrow streets are best explored on foot.", "topic": "Travel & Places"},
]


def get_sentences() -> List[str]:
    """Returns just the sentence text, in the fixed order defined above —
    used anywhere we just need the raw strings to encode (embedder.py)."""
    return [item["sentence"] for item in SENTENCE_DATA]


def get_topics() -> List[str]:
    """Returns just the topic labels, in the SAME order as
    get_sentences(). Kept as a parallel list rather than looking topics
    up by sentence text, because both t-SNE coordinates and this list
    need to line up by POSITION (index i's coordinate belongs to index
    i's topic) when plotting — a dict keyed by sentence text would work
    too, but position-based lookups are simpler here and match how numpy
    arrays are indexed anyway."""
    return [item["topic"] for item in SENTENCE_DATA]
