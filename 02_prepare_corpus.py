"""
Prepare the 2024 YouTube campaign corpus for analysis.

This script:

1. Loads raw comments and video metadata for Biden, Harris, and Trump.
2. Cleans comment text and detects English-language comments.
3. Applies the study's comment eligibility rules.
4. Restricts observations to the 2024 campaign period.
5. Merges comments with video metadata.
6. Removes exact duplicate comments.
7. Creates unique comment IDs and analysis variables.
8. Saves the final EAP-ready research corpus.

The resulting dataset forms the population from which annotation
samples are drawn and to which the final classifiers are applied.
"""

from pathlib import Path
import re

import pandas as pd

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

OUTPUT_FILE = (
    PROCESSED_DIR
    / "EAP_2024_annotation_ready.csv"
)

CANDIDATES = [
    "Biden",
    "Harris",
    "Trump",
]

CAMPAIGN_START = pd.Timestamp(
    "2024-03-08",
    tz="UTC"
)

CAMPAIGN_END = pd.Timestamp(
    "2024-11-05 23:59:59",
    tz="UTC"
)

PARTY_MAP = {
    "Biden": "Democrat",
    "Harris": "Democrat",
    "Trump": "Republican",
}

# langdetect contains a stochastic component.
DetectorFactory.seed = 42


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def remove_unnamed_columns(df):
    """Remove index columns introduced by previous CSV exports."""

    return df.loc[
        :,
        ~df.columns.str.startswith("Unnamed")
    ].copy()


# ---------------------------------------------------------------------
# Comment loading
# ---------------------------------------------------------------------

def load_comments(candidate):
    """Load raw comments for one candidate channel."""

    path = (
        RAW_DIR
        / f"{candidate}2024_comments.csv"
    )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )

    df = remove_unnamed_columns(df)

    required = [
        "video_id",
        "author",
        "text",
        "published_at",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{candidate} comments missing: "
            f"{missing}"
        )

    df["channel_candidate"] = candidate

    df["comment_date"] = pd.to_datetime(
        df["published_at"],
        errors="coerce",
        utc=True
    )

    return df


# ---------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------

def load_videos(candidate):
    """Load raw video metadata for one candidate channel."""

    path = (
        RAW_DIR
        / f"{candidate}2024_channel.csv"
    )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )

    df = remove_unnamed_columns(df)

    if "video_id" not in df.columns:
        raise ValueError(
            f"{candidate} video metadata "
            "does not contain video_id."
        )

    df["candidate_video"] = candidate

    if "video_published" in df.columns:

        df["video_published"] = (
            pd.to_datetime(
                df["video_published"],
                errors="coerce",
                utc=True
            )
        )

    return df


# ---------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------

def word_count(text):
    """Count whitespace-separated tokens in a comment."""

    if pd.isna(text):
        return 0

    return len(
        str(text).split()
    )


def detect_english(text):
    """
    Identify English-language comments.

    Emoji-only comments are retained because language detection
    cannot meaningfully classify them.
    """

    if pd.isna(text):
        return False

    text = str(text).strip()

    if not text:
        return False

    # Remove symbols that do not contribute to language detection.
    cleaned = re.sub(
        r"[^\w\s,.!?]",
        "",
        text
    ).strip()

    # Retain comments containing only emoji/symbols.
    if not cleaned:
        return True

    try:

        return (
            detect(cleaned) == "en"
        )

    except LangDetectException:

        return False


def text_is_eligible(text):
    """
    Apply the original minimum-text eligibility rule.

    Textual comments shorter than five words are excluded.
    Emoji-only comments remain eligible.
    """

    if pd.isna(text):
        return False

    text = str(text).strip()

    if not text:
        return False

    words = word_count(text)

    contains_text = bool(
        re.search(
            r"\w",
            text,
            flags=re.UNICODE
        )
    )

    if contains_text and words < 5:
        return False

    return True


def clean_comments(df):
    """Apply text and language eligibility rules."""

    df = df.copy()

    df["word_count"] = (
        df["text"]
        .apply(word_count)
    )

    df["is_english"] = (
        df["text"]
        .apply(detect_english)
    )

    df["text_confirmation"] = (
        df["text"]
        .apply(text_is_eligible)
    )

    return df


# ---------------------------------------------------------------------
# Combine candidate data
# ---------------------------------------------------------------------

def build_comment_corpus():
    """Load and combine comments from all candidate channels."""

    frames = []

    for candidate in CANDIDATES:

        print(
            f"Loading {candidate} comments..."
        )

        df = load_comments(
            candidate
        )

        df = clean_comments(
            df
        )

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True
    )


def build_video_corpus():
    """Load and combine video metadata."""

    frames = []

    for candidate in CANDIDATES:

        print(
            f"Loading {candidate} videos..."
        )

        frames.append(
            load_videos(candidate)
        )

    videos = pd.concat(
        frames,
        ignore_index=True
    )

    if videos["video_id"].duplicated().any():

        duplicates = int(
            videos["video_id"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"{duplicates} duplicate video IDs "
            "found in metadata."
        )

    return videos


# ---------------------------------------------------------------------
# Corpus eligibility
# ---------------------------------------------------------------------

def apply_eligibility_rules(df):
    """Apply language, text, and campaign-period restrictions."""

    initial_n = len(df)

    # Valid timestamps

    df = df[
        df["comment_date"].notna()
    ].copy()

    # English-language comments

    df = df[
        df["is_english"] == True
    ].copy()

    # Minimum textual content

    df = df[
        df["text_confirmation"] == True
    ].copy()

    # Campaign period

    df = df[
        df["comment_date"].between(
            CAMPAIGN_START,
            CAMPAIGN_END
        )
    ].copy()

    print("\n" + "=" * 70)
    print("ELIGIBILITY")
    print("=" * 70)

    print(
        f"Before eligibility: "
        f"{initial_n:,}"
    )

    print(
        f"After eligibility:  "
        f"{len(df):,}"
    )

    print(
        f"Removed:            "
        f"{initial_n - len(df):,}"
    )

    return df


# ---------------------------------------------------------------------
# Merge metadata
# ---------------------------------------------------------------------

def merge_video_metadata(
    comments,
    videos
):
    """Merge comments with their source-video metadata."""

    # Candidate is already recorded on the comment dataset.
    # Avoid duplicating it during the metadata merge.

    metadata_columns = [
        column
        for column in videos.columns
        if column not in {
            "channel_candidate"
        }
    ]

    merged = comments.merge(
        videos[metadata_columns],
        on="video_id",
        how="left",
        validate="many_to_one"
    )

    missing_metadata = int(
        merged["candidate_video"]
        .isna()
        .sum()
    )

    if missing_metadata:

        print(
            f"WARNING: {missing_metadata:,} comments "
            "have no matching video metadata."
        )

    mismatches = (
        merged["candidate_video"].notna()
        &
        (
            merged["channel_candidate"]
            != merged["candidate_video"]
        )
    )

    if mismatches.any():

        raise ValueError(
            "Candidate mismatch detected between "
            "comments and video metadata."
        )

    return merged


# ---------------------------------------------------------------------
# Duplicate removal
# ---------------------------------------------------------------------

def remove_duplicates(df):
    """
    Remove exact duplicate comments.

    Duplicate identity is defined using video, author,
    timestamp, and comment text.
    """

    duplicate_key = [
        "video_id",
        "author",
        "comment_date",
        "text",
    ]

    duplicates = df.duplicated(
        subset=duplicate_key,
        keep="first"
    )

    n_duplicates = int(
        duplicates.sum()
    )

    df = df[
        ~duplicates
    ].copy()

    print("\n" + "=" * 70)
    print("DUPLICATE REMOVAL")
    print("=" * 70)

    print(
        f"Exact duplicates removed: "
        f"{n_duplicates:,}"
    )

    print(
        f"Rows remaining:           "
        f"{len(df):,}"
    )

    return df


# ---------------------------------------------------------------------
# Analysis variables
# ---------------------------------------------------------------------

def create_analysis_variables(df):
    """Create identifiers and variables used in later analysis."""

    df = df.copy()

    # Stable ordering before assigning study IDs.

    df = df.sort_values(
        [
            "comment_date",
            "video_id",
            "author",
        ],
        kind="stable"
    ).reset_index(drop=True)

    # -------------------------------------------------------------
    # Study ID
    # -------------------------------------------------------------

    df["eap_comment_id"] = [
        f"EAP2024_{number:07d}"
        for number in range(
            1,
            len(df) + 1
        )
    ]

    # -------------------------------------------------------------
    # Political variables
    # -------------------------------------------------------------

    df["party"] = (
        df["channel_candidate"]
        .map(PARTY_MAP)
    )

    if df["party"].isna().any():
        raise ValueError(
            "Candidate could not be mapped to party."
        )

    # -------------------------------------------------------------
    # Temporal variables
    # -------------------------------------------------------------

    df["comment_day"] = (
        df["comment_date"]
        .dt.floor("D")
    )

    df["comment_month"] = (
        df["comment_date"]
        .dt.strftime("%Y-%m")
    )

    iso_calendar = (
        df["comment_date"]
        .dt.isocalendar()
    )

    df["comment_week"] = (
        iso_calendar["year"]
        .astype(str)
        + "-W"
        + iso_calendar["week"]
        .astype(str)
        .str.zfill(2)
    )

    df["week_start"] = (
        df["comment_day"]
        - pd.to_timedelta(
            df["comment_day"]
            .dt.weekday,
            unit="D"
        )
    )

    df["campaign_day"] = (
        df["comment_day"]
        - CAMPAIGN_START
    ).dt.days

    # -------------------------------------------------------------
    # Text variables
    # -------------------------------------------------------------

    df["word_count"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    df["character_count"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    # Retained as a diagnostic variable rather than an
    # additional exclusion criterion.

    df["very_short_comment"] = (
        df["word_count"] <= 2
    )

    return df


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_corpus(df):
    """Run final integrity checks on the prepared corpus."""

    print("\n" + "=" * 70)
    print("FINAL CORPUS VALIDATION")
    print("=" * 70)

    if df["eap_comment_id"].duplicated().any():
        raise ValueError(
            "Duplicate EAP comment IDs detected."
        )

    if df["comment_date"].isna().any():
        raise ValueError(
            "Missing comment dates detected."
        )

    if df["text"].isna().any():
        raise ValueError(
            "Missing comment text detected."
        )

    if not df["comment_date"].between(
        CAMPAIGN_START,
        CAMPAIGN_END
    ).all():

        raise ValueError(
            "Comments outside campaign period detected."
        )

    if not df["is_english"].all():
        raise ValueError(
            "Non-English comments detected."
        )

    duplicate_key = [
        "video_id",
        "author",
        "comment_date",
        "text",
    ]

    if df.duplicated(
        subset=duplicate_key
    ).any():

        raise ValueError(
            "Exact duplicate comments remain."
        )

    print(
        f"Comments: "
        f"{len(df):,}"
    )

    print(
        f"Videos:   "
        f"{df['video_id'].nunique():,}"
    )

    print("\nCandidate counts:")

    print(
        df["channel_candidate"]
        .value_counts()
    )

    print("\nParty counts:")

    print(
        df["party"]
        .value_counts()
    )

    print(
        "\nDate range: "
        f"{df['comment_date'].min()} "
        f"to "
        f"{df['comment_date'].max()}"
    )

    print(
        "\nPASS: Final corpus validated."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Construct the final research corpus."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("2024 YOUTUBE CORPUS PREPARATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load raw data
    # -------------------------------------------------------------

    comments = build_comment_corpus()

    videos = build_video_corpus()

    print(
        f"\nRaw comments: "
        f"{len(comments):,}"
    )

    print(
        f"Video metadata rows: "
        f"{len(videos):,}"
    )

    # -------------------------------------------------------------
    # Eligibility
    # -------------------------------------------------------------

    comments = apply_eligibility_rules(
        comments
    )

    # -------------------------------------------------------------
    # Merge metadata
    # -------------------------------------------------------------

    df = merge_video_metadata(
        comments,
        videos
    )

    # -------------------------------------------------------------
    # Remove exact duplicates
    # -------------------------------------------------------------

    df = remove_duplicates(
        df
    )

    # -------------------------------------------------------------
    # Construct analysis variables
    # -------------------------------------------------------------

    df = create_analysis_variables(
        df
    )

    # -------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------

    validate_corpus(
        df
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        index=False
    )

    print(
        f"\nSaved final corpus to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
