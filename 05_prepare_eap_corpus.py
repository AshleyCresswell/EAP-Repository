"""
Prepare the master YouTube corpus for EAP analysis.

This script:
1. Loads the combined comment and video dataset.
2. Standardises text, dates, and candidate labels.
3. Applies the campaign-period eligibility criteria.
4. Removes exact duplicate comments.
5. Creates unique comment identifiers.
6. Constructs candidate, party, and temporal variables.
7. Exports the final EAP corpus.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path(
    "data/processed/comments_2024_analysis.csv"
)

OUTPUT_FILE = Path(
    "data/processed/EAP_2024_annotation_ready.csv"
)

CAMPAIGN_START = pd.Timestamp(
    "2024-03-08",
    tz="UTC"
)

CAMPAIGN_END = pd.Timestamp(
    "2024-11-05 23:59:59",
    tz="UTC"
)


# ---------------------------------------------------------------------
# Data standardisation
# ---------------------------------------------------------------------

def standardise_text(df):
    """Standardise comment text and remove empty comments."""

    df = df.copy()

    df["text"] = (
        df["text"]
        .astype("string")
        .str.strip()
    )

    df.loc[
        df["text"] == "",
        "text"
    ] = pd.NA

    return df


def standardise_dates(df):
    """Convert relevant date variables to datetime format."""

    df = df.copy()

    date_columns = [
        "comment_date",
        "updated_at",
        "video_published",
    ]

    for column in date_columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
                utc=True
            )

    return df


def standardise_candidates(df):
    """Standardise candidate labels."""

    df = df.copy()

    candidate_map = {
        "Donald Trump": "Trump",
        "Joe Biden": "Biden",
        "Kamala Harris": "Harris",
    }

    df["channel_candidate"] = (
        df["channel_candidate"]
        .astype("string")
        .str.strip()
        .replace(candidate_map)
    )

    return df


# ---------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------

def apply_eligibility_rules(df):
    """Apply the eligibility criteria for the EAP corpus."""

    df = df.copy()

    eligible_date = df["comment_date"].between(
        CAMPAIGN_START,
        CAMPAIGN_END,
        inclusive="both"
    )

    eligible_english = (
        df["is_english"] == True
    )

    eligible_text = (
        df["text"].notna()
    )

    eligible = (
        eligible_date
        & eligible_english
        & eligible_text
    )

    print("\nEligibility checks")
    print(
        f"Campaign period: "
        f"{eligible_date.sum():,} / {len(df):,}"
    )
    print(
        f"English language: "
        f"{eligible_english.sum():,} / {len(df):,}"
    )
    print(
        f"Text present: "
        f"{eligible_text.sum():,} / {len(df):,}"
    )
    print(
        f"Overall eligible: "
        f"{eligible.sum():,} / {len(df):,}"
    )

    return df[eligible].copy()


# ---------------------------------------------------------------------
# Duplicate removal
# ---------------------------------------------------------------------

def remove_duplicates(df):
    """Remove exact duplicate comment records."""

    duplicate_columns = [
        "video_id",
        "author",
        "comment_date",
        "text",
    ]

    duplicate_rows = df.duplicated(
        subset=duplicate_columns,
        keep="first"
    )

    n_duplicates = duplicate_rows.sum()

    print(
        f"\nExact duplicate comments: "
        f"{n_duplicates:,}"
    )

    return df[
        ~duplicate_rows
    ].copy()


# ---------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------

def create_comment_ids(df):
    """Create unique internal identifiers for EAP comments."""

    df = df.reset_index(
        drop=True
    )

    df["eap_comment_id"] = [
        f"EAP2024_{i:07d}"
        for i in range(1, len(df) + 1)
    ]

    return df


# ---------------------------------------------------------------------
# Political variables
# ---------------------------------------------------------------------

def create_political_variables(df):
    """Create party identifiers from channel candidate."""

    df = df.copy()

    party_map = {
        "Trump": "Republican",
        "Biden": "Democrat",
        "Harris": "Democrat",
    }

    df["party"] = (
        df["channel_candidate"]
        .map(party_map)
    )

    return df


# ---------------------------------------------------------------------
# Temporal variables
# ---------------------------------------------------------------------

def create_temporal_variables(df):
    """Construct temporal variables for campaign analysis."""

    df = df.copy()

    # Calendar day
    df["comment_day"] = (
        df["comment_date"]
        .dt.floor("D")
    )

    # Month
    df["comment_month"] = (
        df["comment_date"]
        .dt.strftime("%Y-%m")
    )

    # ISO week
    iso = (
        df["comment_date"]
        .dt.isocalendar()
    )

    df["comment_week"] = (
        iso["year"].astype(str)
        + "-W"
        + iso["week"]
        .astype(str)
        .str.zfill(2)
    )

    # Monday corresponding to each comment week
    df["week_start"] = (
        df["comment_date"]
        .dt.floor("D")
        - pd.to_timedelta(
            df["comment_date"].dt.weekday,
            unit="D"
        )
    )

    # Days elapsed since start of campaign
    df["campaign_day"] = (
        df["comment_day"]
        - CAMPAIGN_START
    ).dt.days

    return df


# ---------------------------------------------------------------------
# Text diagnostics
# ---------------------------------------------------------------------

def create_text_variables(df):
    """Create descriptive comment-length variables."""

    df = df.copy()

    # Recalculate to ensure consistency.
    df["word_count"] = (
        df["text"]
        .fillna("")
        .str.split()
        .str.len()
    )

    df["character_count"] = (
        df["text"]
        .fillna("")
        .str.len()
    )

    df["very_short_comment"] = (
        df["word_count"] <= 2
    )

    return df


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_corpus(df):
    """Run final integrity checks on the EAP corpus."""

    if not df["eap_comment_id"].is_unique:
        raise ValueError(
            "Duplicate EAP comment IDs detected."
        )

    if df["text"].isna().any():
        raise ValueError(
            "Missing comment text detected."
        )

    if df["comment_date"].isna().any():
        raise ValueError(
            "Missing comment dates detected."
        )

    expected_candidates = {
        "Biden",
        "Harris",
        "Trump"
    }

    observed_candidates = set(
        df["channel_candidate"]
        .dropna()
        .unique()
    )

    if observed_candidates != expected_candidates:
        raise ValueError(
            "Unexpected candidate labels detected."
        )


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

    print(
        f"Master corpus rows: "
        f"{len(df):,}"
    )

    # Standardise data.
    df = standardise_text(df)
    df = standardise_dates(df)
    df = standardise_candidates(df)

    # Apply final corpus eligibility criteria.
    df = apply_eligibility_rules(df)

    # Remove exact duplicates.
    df = remove_duplicates(df)

    # Construct analytical variables.
    df = create_comment_ids(df)
    df = create_political_variables(df)
    df = create_temporal_variables(df)
    df = create_text_variables(df)

    # Validate final corpus.
    validate_corpus(df)

    # Save.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        index=False
    )

    # Summary.
    print("\nFinal EAP corpus")
    print(f"Comments: {len(df):,}")

    print("\nComments by candidate:")
    print(
        df["channel_candidate"]
        .value_counts()
    )

    print("\nComments by party:")
    print(
        df["party"]
        .value_counts()
    )

    print(
        f"\nDate range: "
        f"{df['comment_date'].min()} to "
        f"{df['comment_date'].max()}"
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
