"""
Prepare the final annotated EAP dataset for BERT modelling.

This script:
1. Combines the original annotation sample with enrichment batches.
2. Records the annotation source of each observation.
3. Removes duplicate comment IDs.
4. Validates Social Distance, Aversion, and Othering labels.
5. Constructs descriptive multi-label EAP variables.
6. Checks duplicate-text label consistency.
7. Exports the final annotated training master.

The frozen train/validation/test splits used for final BERT modelling
are retained separately rather than regenerated here.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ANNOTATION_DIR = Path("data/annotation")
OUTPUT_DIR = Path("data/model")

ANNOTATION_FILES = {
    "original_sample":
        ANNOTATION_DIR
        / "EAP_2024_training_sample_2000_annotated_v2.csv",

    "social_distance_1":
        ANNOTATION_DIR
        / "EAP_2024_social_distance_candidates_500_annotated_v2.csv",

    "social_distance_2":
        ANNOTATION_DIR
        / "EAP_2024_social_distance_candidates_remaining_annotated_v2.csv",

    "social_distance_3":
        ANNOTATION_DIR
        / "EAP_2024_social_distance_final_enrichment_annotated_v2.csv",

    "social_distance_4":
        ANNOTATION_DIR
        / "EAP_2024_social_distance_batch4_annotated_v2.csv",

    "othering":
        ANNOTATION_DIR
        / "EAP_2024_othering_candidates_500_annotated_v2.csv",
}

OUTPUT_FILE = (
    OUTPUT_DIR
    / "EAP_2024_BERT_training_master.csv"
)

LABEL_COLUMNS = [
    "social_distance",
    "aversion",
    "othering",
]

REQUIRED_COLUMNS = [
    "eap_comment_id",
    "text",
    *LABEL_COLUMNS,
]


# ---------------------------------------------------------------------
# Load annotated batches
# ---------------------------------------------------------------------

def load_annotations(files):
    """Load and combine all final EAP annotation batches."""

    frames = []

    for source, file in files.items():

        if not file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {file}"
            )

        df = pd.read_csv(
            file,
            encoding="utf-8-sig",
            low_memory=False
        )

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{file.name} is missing columns: {missing}"
            )

        df["annotation_source"] = source

        print(
            f"{source:<22} "
            f"{len(df):>6,} rows"
        )

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )


# ---------------------------------------------------------------------
# Comment IDs
# ---------------------------------------------------------------------

def clean_comment_ids(df):
    """Standardise comment IDs and remove duplicate observations."""

    df = df.copy()

    df["eap_comment_id"] = (
        df["eap_comment_id"]
        .astype("string")
        .str.strip()
    )

    if df["eap_comment_id"].isna().any():
        raise ValueError(
            "Missing EAP comment IDs detected."
        )

    duplicate_ids = df[
        df["eap_comment_id"].duplicated(
            keep=False
        )
    ]

    print(
        f"\nRows with duplicate comment IDs: "
        f"{len(duplicate_ids):,}"
    )

    df = (
        df
        .drop_duplicates(
            subset="eap_comment_id",
            keep="first"
        )
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------

def clean_labels(df):
    """Validate and standardise the three binary EAP labels."""

    df = df.copy()

    for label in LABEL_COLUMNS:

        df[label] = pd.to_numeric(
            df[label],
            errors="coerce"
        )

        missing = df[label].isna().sum()

        if missing:
            raise ValueError(
                f"{label} contains "
                f"{missing:,} missing labels."
            )

        invalid = set(
            df[label].unique()
        ) - {0, 1}

        if invalid:
            raise ValueError(
                f"{label} contains invalid "
                f"values: {invalid}"
            )

        df[label] = (
            df[label]
            .astype(int)
        )

    return df


# ---------------------------------------------------------------------
# Duplicate-text validation
# ---------------------------------------------------------------------

def validate_duplicate_text(df):
    """
    Check that identical normalised comments do not have
    conflicting EAP labels.
    """

    df = df.copy()

    df["_text_group"] = (
        df["text"]
        .astype("string")
        .str.lower()
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )

    repeated = df[
        df["_text_group"].duplicated(
            keep=False
        )
    ]

    print(
        f"\nRows in repeated text groups: "
        f"{len(repeated):,}"
    )

    for label in LABEL_COLUMNS:

        conflicting = (
            df.groupby("_text_group")[label]
            .nunique()
        )

        conflicting = conflicting[
            conflicting > 1
        ]

        print(
            f"Conflicting {label} text groups: "
            f"{len(conflicting):,}"
        )

        if len(conflicting) > 0:
            raise ValueError(
                f"Identical normalised comments "
                f"have conflicting {label} labels."
            )

    return df.drop(
        columns="_text_group"
    )


# ---------------------------------------------------------------------
# Derived EAP variables
# ---------------------------------------------------------------------

def create_eap_variables(df):
    """Create descriptive multi-label EAP indicators."""

    df = df.copy()

    df["any_eap"] = (
        df[LABEL_COLUMNS]
        .max(axis=1)
        .astype(int)
    )

    df["n_eap_dimensions"] = (
        df[LABEL_COLUMNS]
        .sum(axis=1)
        .astype(int)
    )

    return df


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def print_label_summary(df):
    """Print the final EAP label distribution."""

    print("\n" + "=" * 60)
    print("FINAL EAP TRAINING DATA")
    print("=" * 60)

    print(
        f"\nTotal annotated comments: "
        f"{len(df):,}"
    )

    for label in LABEL_COLUMNS:

        positive = int(
            df[label].sum()
        )

        negative = (
            len(df) - positive
        )

        prevalence = (
            positive
            / len(df)
            * 100
        )

        print(f"\n{label}")
        print(f"  Negative: {negative:,}")
        print(f"  Positive: {positive:,}")
        print(
            f"  Positive rate: "
            f"{prevalence:.2f}%"
        )

    print("\nAny EAP:")
    print(
        df["any_eap"]
        .value_counts()
        .sort_index()
    )

    print("\nNumber of EAP dimensions:")
    print(
        df["n_eap_dimensions"]
        .value_counts()
        .sort_index()
    )


# ---------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------

def validate_master(df):
    """Run final integrity checks."""

    if not df["eap_comment_id"].is_unique:
        raise ValueError(
            "Duplicate comment IDs remain."
        )

    if df["text"].isna().any():
        raise ValueError(
            "Missing comment text detected."
        )

    for label in LABEL_COLUMNS:

        if not set(
            df[label].unique()
        ).issubset({0, 1}):

            raise ValueError(
                f"{label} is not binary."
            )


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():

    print("=" * 60)
    print("LOADING EAP ANNOTATIONS")
    print("=" * 60)

    df = load_annotations(
        ANNOTATION_FILES
    )

    print(
        f"\nRows before duplicate removal: "
        f"{len(df):,}"
    )

    df = clean_comment_ids(df)

    print(
        f"Rows after duplicate removal: "
        f"{len(df):,}"
    )

    df = clean_labels(df)

    df = validate_duplicate_text(df)

    df = create_eap_variables(df)

    validate_master(df)

    print_label_summary(df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        index=False
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
