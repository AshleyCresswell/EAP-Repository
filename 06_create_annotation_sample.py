"""
Create the stratified EAP annotation sample.

This script:
1. Loads the prepared EAP corpus.
2. Applies annotation-sample eligibility criteria.
3. Allocates approximately equal samples across candidates.
4. Stratifies sampling across campaign weeks within candidates.
5. Produces a reproducible 2,000-comment annotation sample.
6. Validates and exports the final sample.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path(
    "data/processed/EAP_2024_annotation_ready.csv"
)

OUTPUT_FILE = Path(
    "data/annotation/EAP_2024_training_sample_2000.csv"
)

RANDOM_SEED = 42

SAMPLE_SIZES = {
    "Trump": 667,
    "Harris": 667,
    "Biden": 666,
}


# ---------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------

def get_eligible_comments(df):
    """Select comments eligible for the annotation sample."""

    eligible = df[
        (df["is_english"] == True)
        & (df["very_short_comment"] == False)
        & (df["text"].notna())
    ].copy()

    return eligible


# ---------------------------------------------------------------------
# Weekly stratified sampling
# ---------------------------------------------------------------------

def stratified_week_sample(
    candidate_df,
    n_sample,
    random_seed=42
):
    """
    Sample comments proportionally across campaign weeks.

    Weekly sample sizes reflect each candidate's temporal
    distribution while preserving the requested total sample size.
    """

    candidate_df = candidate_df.copy()

    week_counts = (
        candidate_df["comment_week"]
        .value_counts()
        .sort_index()
    )

    total_available = len(candidate_df)

    if n_sample > total_available:
        raise ValueError(
            f"Requested {n_sample} comments but only "
            f"{total_available} are available."
        )

    # Calculate proportional allocation by week.
    allocation = (
        week_counts
        / total_available
        * n_sample
    )

    # Start with the integer component.
    allocation_floor = (
        np.floor(allocation)
        .astype(int)
    )

    # Allocate remaining observations to weeks
    # with the largest fractional remainder.
    remainder = (
        n_sample
        - allocation_floor.sum()
    )

    if remainder > 0:

        fractional_remainder = (
            allocation
            - allocation_floor
        )

        extra_weeks = (
            fractional_remainder
            .sort_values(ascending=False)
            .head(remainder)
            .index
        )

        allocation_floor.loc[
            extra_weeks
        ] += 1

    # Draw observations from each week.
    samples = []

    for week, week_n in allocation_floor.items():

        if week_n == 0:
            continue

        week_df = candidate_df[
            candidate_df["comment_week"] == week
        ]

        week_sample = week_df.sample(
            n=week_n,
            random_state=random_seed
        )

        samples.append(
            week_sample
        )

    candidate_sample = pd.concat(
        samples,
        ignore_index=True
    )

    # Safety correction in case the proportional
    # allocation does not produce the exact target.
    if len(candidate_sample) < n_sample:

        n_missing = (
            n_sample
            - len(candidate_sample)
        )

        remaining = candidate_df[
            ~candidate_df["eap_comment_id"].isin(
                candidate_sample["eap_comment_id"]
            )
        ]

        additional = remaining.sample(
            n=n_missing,
            random_state=random_seed
        )

        candidate_sample = pd.concat(
            [
                candidate_sample,
                additional
            ],
            ignore_index=True
        )

    return candidate_sample


# ---------------------------------------------------------------------
# Annotation sample
# ---------------------------------------------------------------------

def create_annotation_sample(df):
    """Create the candidate- and week-stratified annotation sample."""

    samples = []

    for candidate, n_sample in SAMPLE_SIZES.items():

        candidate_df = df[
            df["channel_candidate"] == candidate
        ].copy()

        print(
            f"Sampling {n_sample:,} comments "
            f"for {candidate}"
        )

        sample = stratified_week_sample(
            candidate_df,
            n_sample,
            RANDOM_SEED
        )

        samples.append(sample)

    annotation_sample = pd.concat(
        samples,
        ignore_index=True
    )

    # Shuffle candidate groups before annotation.
    annotation_sample = (
        annotation_sample
        .sample(
            frac=1,
            random_state=RANDOM_SEED
        )
        .reset_index(drop=True)
    )

    return annotation_sample


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_sample(df):
    """Validate the final annotation sample."""

    expected_total = sum(
        SAMPLE_SIZES.values()
    )

    if len(df) != expected_total:
        raise ValueError(
            f"Expected {expected_total:,} comments, "
            f"but found {len(df):,}."
        )

    if not df["eap_comment_id"].is_unique:
        raise ValueError(
            "Duplicate EAP comment IDs detected."
        )

    for candidate, expected_n in SAMPLE_SIZES.items():

        observed_n = (
            df["channel_candidate"]
            .eq(candidate)
            .sum()
        )

        if observed_n != expected_n:
            raise ValueError(
                f"{candidate}: expected {expected_n}, "
                f"found {observed_n}."
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
        f"Corpus comments: "
        f"{len(df):,}"
    )

    # Select annotation-eligible comments.
    eligible = get_eligible_comments(df)

    print(
        f"Annotation-eligible comments: "
        f"{len(eligible):,}"
    )

    print("\nEligible comments by candidate:")
    print(
        eligible["channel_candidate"]
        .value_counts()
    )

    # Create sample.
    annotation_sample = (
        create_annotation_sample(
            eligible
        )
    )

    # Validate.
    validate_sample(
        annotation_sample
    )

    # Save.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    annotation_sample.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        index=False
    )

    # Summary.
    print("\nAnnotation sample complete")
    print(
        f"Sample size: "
        f"{len(annotation_sample):,}"
    )

    print("\nCandidate distribution:")
    print(
        annotation_sample[
            "channel_candidate"
        ].value_counts()
    )

    print("\nWeeks represented:")
    print(
        annotation_sample
        .groupby(
            "channel_candidate"
        )["comment_week"]
        .nunique()
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
