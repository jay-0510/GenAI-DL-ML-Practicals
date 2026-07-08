"""
dataloader.py
-------------
Handles everything related to loading and preprocessing the text corpus.
Import this in your notebook or any script — keeps the notebook clean.
"""

import pandas as pd
from gensim.utils import simple_preprocess


# words we want to visualise — 50 diverse words across 5 topic groups
VISUAL_WORDS = [
    # sports
    'football', 'cricket', 'tennis', 'goal', 'bat',
    'ball', 'coach', 'player', 'stadium', 'match',
    # food
    'pizza', 'burger', 'rice', 'apple', 'banana',
    'coffee', 'tea', 'bread', 'milk', 'cake',
    # tech
    'computer', 'python', 'java', 'ai', 'data',
    'network', 'cloud', 'server', 'robot', 'algorithm',
    # lifestyle
    'music', 'movie', 'book', 'school', 'teacher',
    'student', 'doctor', 'hospital', 'travel', 'car',
    # devices
    'phone', 'camera', 'internet', 'software', 'hardware',
    'keyboard', 'mouse', 'monitor', 'science', 'math',
]

# colour map — one colour per topic group for scatter plots
GROUP_COLORS = {
    'sports':    ('#4C9BE8', ['football','cricket','tennis','goal','bat','ball','coach','player','stadium','match']),
    'food':      ('#E87B4C', ['pizza','burger','rice','apple','banana','coffee','tea','bread','milk','cake']),
    'tech':      ('#7B4CE8', ['computer','python','java','ai','data','network','cloud','server','robot','algorithm']),
    'lifestyle': ('#4CE87B', ['music','movie','book','school','teacher','student','doctor','hospital','travel','car']),
    'devices':   ('#E84C7B', ['phone','camera','internet','software','hardware','keyboard','mouse','monitor','science','math']),
}


def load_corpus(csv_path: str) -> pd.DataFrame:
    """
    Loads the raw CSV corpus into a DataFrame.

    Args:
        csv_path: path to text_corpus.csv

    Returns:
        DataFrame with the raw text column
    """
    df = pd.read_csv(csv_path)          # reads the CSV — expects a 'text' column in first position
    print(f"Loaded {len(df)} sentences from {csv_path}")
    return df


def preprocess(df: pd.DataFrame) -> list:
    """
    Converts raw text rows into tokenised word lists for Word2Vec.

    gensim's simple_preprocess:
      - lowercases everything
      - strips punctuation and numbers
      - returns a list of tokens per sentence

    Args:
        df: DataFrame returned by load_corpus()

    Returns:
        List of token lists  e.g. [['football','match',...], ['pizza','burger',...]]
    """
    sentences = df.iloc[:, 0].astype(str).apply(simple_preprocess).tolist()
    total_tokens = sum(len(s) for s in sentences)
    print(f"Preprocessed → {len(sentences)} sentences | {total_tokens} total tokens")
    return sentences


def get_word_color(word: str) -> str:
    """
    Returns the hex colour for a word based on its topic group.
    Falls back to grey if the word isn't in any group.

    Args:
        word: any string

    Returns:
        hex colour string
    """
    for color, words in GROUP_COLORS.values():       # loops through each topic group
        if word in words:
            return color
    return '#AAAAAA'                                  # grey for unknown words


def filter_vocab_words(model_wv, word_list: list) -> tuple:
    """
    Removes any words from the target list that the model never saw
    during training (they won't have a vector).

    Args:
        model_wv:  the trained model's .wv KeyedVectors object
        word_list: list of words we want to plot

    Returns:
        (filtered_words, missing_words) — both as lists
    """
    present = [w for w in word_list if w in model_wv]      # only keep words with vectors
    missing = [w for w in word_list if w not in model_wv]  # track what got dropped
    if missing:
        print(f"Words not in vocab (skipped): {missing}")
    print(f"Visualising {len(present)} / {len(word_list)} words")
    return present, missing
