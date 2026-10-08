"""
Clean raw YouTube comments prior to corpus construction.

This script:
1. Calculates comment word counts.
2. Identifies English-language comments.
3. Retains emoji-only comments.
4. Removes text comments containing fewer than five words.
5. Exports the cleaned comment dataset.

This cleaning stage is applied separately to each candidate's
YouTube comment data before the datasets are combined.
"""

from pathlib import Path
import re

import pandas as pd
from langdetect import detect, DetectorFactory


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path("data/raw/Biden2024_comments.csv")
OUTPUT_FILE = Path("data/clean/Biden2024_comments_clean.csv")

MIN_WORDS = 5

# Make langdetect results reproducible.
DetectorFactory.seed = 42


# ---------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols and pictographs
    "\U0001F680-\U0001F6FF"  # Transport and map symbols
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002700-\U000027BF"   # Dingbats
    "\U000024C2-\U0001F251"   # Enclosed characters
    "]+",
    flags=re.UNICODE,
)


def count_words(text):
    """Return the number of whitespace-separated words in a comment."""

    if pd.isna(text):
        return 0

    return len(str(text).split())


def is_english_or_emoji(text):
    """Identify English-language or emoji-only comments."""

    if pd.isna(text):
        return False

    text = str(text)

    # Remove non-word symbols before language detection.
    text_for_detection = re.sub(
        r"[^\w\s,.!?]",
        "",
        text
    ).strip()

    # Retain comments containing no detectable words,
    # such as emoji-only comments.
    if not text_for_detection:
        return True

    try:
        return detect(text_for_detection) == "en"

    except Exception:
        return False


def meets_length_requirement(text, word_count):
    """
    Retain comments with at least five words or comments
    containing no textual content, such as emoji-only comments.
    """

    if pd.isna(text):
        return False

    has_text = bool(re.search(r"\w", str(text)))

    if has_text and word_count < MIN_WORDS:
        return False

    return True


# ---------------------------------------------------------------------
# Cleaning pipeline
# ---------------------------------------------------------------------

def clean_comments(df):
    """Apply language and minimum-length cleaning rules."""

    if "text" not in df.columns:
        raise ValueError("Input data must contain a 'text' column.")

    df = df.copy()

    df["word_count"] = df["text"].apply(count_words)

    df["is_english"] = df["text"].apply(
        is_english_or_emoji
    )

    df["text_confirmation"] = df.apply(
        lambda row: meets_length_requirement(
            row["text"],
            row["word_count"]
        ),
        axis=1
    )

    return df[
        df["is_english"]
        & df["text_confirmation"]
    ].copy()


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():

    print(f"Loading: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    n_original = len(df)

    cleaned = clean_comments(df)

    n_removed = n_original - len(cleaned)

    print(f"Original comments: {n_original:,}")
    print(f"Comments retained: {len(cleaned):,}")
    print(f"Comments removed:  {n_removed:,}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cleaned.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        index=False
    )

    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
