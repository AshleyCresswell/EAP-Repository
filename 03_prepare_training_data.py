"""
Prepare manually annotated data for EAP and incivility modelling.

This script:

1. Draws the main 2,000-comment EAP annotation sample.
2. Combines completed EAP annotation and enrichment files.
3. Validates the final EAP annotation dataset.
4. Validates the final EAP train/validation/test splits.
5. Validates the final APE/incivility train/validation/test splits.

Manual annotation and targeted enrichment occurred between the
sampling and modelling stages. Completed annotation files are therefore
treated as research inputs rather than generated automatically here.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CORPUS_FILE = Path(
    "data/processed/EAP_2024_annotation_ready.csv"
)

ANNOTATION_DIR = Path("data/annotation")
MODEL_DIR = Path("data/model")

SAMPLE_FILE = (
    ANNOTATION_DIR
    / "EAP_2024_training_sample_2000.csv"
)

MASTER_FILE = (
    ANNOTATION_DIR
    / "EAP_2024_BERT_training_master.csv"
)

SEED = 42
SAMPLE_SIZE = 2000

ID_COL = "eap_comment_id"
TEXT_COL = "text"

EAP_LABELS = [
    "social_distance",
    "aversion",
    "othering",
]

CANDIDATE_SAMPLE = {
    "Trump": 667,
    "Harris": 667,
    "Biden": 666,
}


# Completed manually annotated files used in the final
# EAP training dataset.

EAP_ANNOTATION_FILES = [
    ANNOTATION_DIR
    / "EAP_2024_training_sample_2000_annotated_v2.csv",

    ANNOTATION_DIR
    / "EAP_2024_social_distance_candidates_500_annotated_v2.csv",

    ANNOTATION_DIR
    / "EAP_2024_social_distance_candidates_remaining_annotated_v2.csv",

    ANNOTATION_DIR
    / "EAP_2024_social_distance_final_enrichment_annotated_v2.csv",

    ANNOTATION_DIR
    / "EAP_2024_othering_candidates_500_annotated_v2.csv",
]


# Final frozen modelling splits.

EAP_SPLITS = {
    "train":
        MODEL_DIR
        / "EAP_2024_BERT_train_v2.csv",

    "validation":
        MODEL_DIR
        / "EAP_2024_BERT_validation_v2.csv",

    "test":
        MODEL_DIR
        / "EAP_2024_BERT_test_v2.csv",
}


APE_SPLITS = {
    "train":
        MODEL_DIR
        / "APE_V1_train.csv",

    "validation":
        MODEL_DIR
        / "APE_V1_validation.csv",

    "test":
        MODEL_DIR
        / "APE_V1_test.csv",
}


# ---------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------

def load_corpus():
    """Load the final eligible comment corpus."""

    df = pd.read_csv(
        CORPUS_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    required = [
        ID_COL,
        TEXT_COL,
        "channel_candidate",
        "comment_week",
        "is_english",
        "very_short_comment",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Corpus missing columns: {missing}"
        )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            "Duplicate comment IDs detected."
        )

    return df


# ---------------------------------------------------------------------
# Stratified annotation sample
# ---------------------------------------------------------------------

def allocate_weekly_sample(
    candidate_df,
    target_n
):
    """
    Allocate a candidate's sample proportionally across
    campaign weeks using the largest-remainder method.
    """

    week_counts = (
        candidate_df["comment_week"]
        .value_counts()
        .sort_index()
    )

    proportions = (
        week_counts
        / week_counts.sum()
    )

    exact = (
        proportions
        * target_n
    )

    allocation = (
        np.floor(exact)
        .astype(int)
    )

    remainder = (
        target_n
        - allocation.sum()
    )

    if remainder > 0:

        fractional = (
            exact - allocation
        ).sort_values(
            ascending=False
        )

        for week in fractional.index[
            :remainder
        ]:
            allocation.loc[week] += 1

    # A week cannot contribute more comments
    # than actually exist in that week.

    allocation = pd.concat(
        [
            allocation,
            week_counts,
        ],
        axis=1
    )

    allocation.columns = [
        "requested",
        "available",
    ]

    allocation["sample_n"] = (
        allocation[
            [
                "requested",
                "available",
            ]
        ]
        .min(axis=1)
        .astype(int)
    )

    return allocation


def sample_candidate(
    candidate_df,
    target_n
):
    """Draw one candidate's week-stratified sample."""

    allocation = allocate_weekly_sample(
        candidate_df,
        target_n
    )

    samples = []

    for week, row in allocation.iterrows():

        n = int(
            row["sample_n"]
        )

        if n == 0:
            continue

        week_df = candidate_df[
            candidate_df["comment_week"]
            == week
        ]

        samples.append(
            week_df.sample(
                n=n,
                random_state=SEED
            )
        )

    sampled = pd.concat(
        samples,
        ignore_index=False
    )

    # Safety fill if a small week could not supply
    # its proportional allocation.

    shortfall = (
        target_n
        - len(sampled)
    )

    if shortfall > 0:

        remaining = candidate_df[
            ~candidate_df[ID_COL]
            .isin(sampled[ID_COL])
        ]

        fill = remaining.sample(
            n=shortfall,
            random_state=SEED
        )

        sampled = pd.concat(
            [
                sampled,
                fill,
            ]
        )

    return sampled


def create_annotation_sample(df):
    """
    Draw the main 2,000-comment annotation sample.

    Sampling balances candidates approximately equally while
    retaining each candidate's temporal distribution across
    campaign weeks.
    """

    eligible = df[
        (df["is_english"] == True)
        & (df["very_short_comment"] == False)
        & df[TEXT_COL].notna()
    ].copy()

    samples = []

    for candidate, target_n in (
        CANDIDATE_SAMPLE.items()
    ):

        candidate_df = eligible[
            eligible["channel_candidate"]
            == candidate
        ].copy()

        sample = sample_candidate(
            candidate_df,
            target_n
        )

        samples.append(sample)

    sample_df = pd.concat(
        samples,
        ignore_index=True
    )

    sample_df = sample_df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    if len(sample_df) != SAMPLE_SIZE:
        raise ValueError(
            f"Expected {SAMPLE_SIZE:,} comments, "
            f"found {len(sample_df):,}."
        )

    if sample_df[ID_COL].duplicated().any():
        raise ValueError(
            "Duplicate IDs found in annotation sample."
        )

    print("\n" + "=" * 70)
    print("MAIN EAP ANNOTATION SAMPLE")
    print("=" * 70)

    print(
        f"Sample size: {len(sample_df):,}"
    )

    print("\nCandidate counts:")

    print(
        sample_df[
            "channel_candidate"
        ].value_counts()
    )

    return sample_df


# ---------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------

def load_annotation_file(path):
    """Load and standardise one completed annotation file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Annotation file not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )

    df = df.loc[
        :,
        ~df.columns.str.startswith(
            "Unnamed"
        )
    ].copy()

    required = [
        ID_COL,
        TEXT_COL,
        *EAP_LABELS,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} missing columns: "
            f"{missing}"
        )

    df[ID_COL] = (
        df[ID_COL]
        .astype(str)
        .str.strip()
    )

    df[TEXT_COL] = (
        df[TEXT_COL]
        .astype(str)
        .str.strip()
    )

    for label in EAP_LABELS:

        df[label] = pd.to_numeric(
            df[label],
            errors="coerce"
        )

    return df


# ---------------------------------------------------------------------
# EAP annotation master
# ---------------------------------------------------------------------

def build_eap_master():
    """
    Combine the completed manually annotated EAP datasets.

    Targeted enrichment samples are included because Social Distance
    and Othering were comparatively rare in the initial sample.
    """

    frames = []

    for path in EAP_ANNOTATION_FILES:

        print(
            f"Loading {path.name}"
        )

        frames.append(
            load_annotation_file(path)
        )

    master = pd.concat(
        frames,
        ignore_index=True
    )

    # -------------------------------------------------------------
    # Missing labels
    # -------------------------------------------------------------

    missing_labels = (
        master[EAP_LABELS]
        .isna()
        .sum()
    )

    if missing_labels.any():

        raise ValueError(
            "Missing EAP labels detected:\n"
            f"{missing_labels}"
        )

    # -------------------------------------------------------------
    # Binary labels
    # -------------------------------------------------------------

    for label in EAP_LABELS:

        master[label] = (
            master[label]
            .astype(int)
        )

        if not set(
            master[label].unique()
        ).issubset({0, 1}):

            raise ValueError(
                f"{label} contains "
                "non-binary values."
            )

    # -------------------------------------------------------------
    # Duplicate comment IDs
    # -------------------------------------------------------------

    duplicated_ids = (
        master[ID_COL]
        .duplicated(
            keep=False
        )
    )

    if duplicated_ids.any():

        duplicates = master[
            duplicated_ids
        ].sort_values(
            ID_COL
        )

        # Duplicate IDs are acceptable only if all labels agree.
        conflict = (
            duplicates
            .groupby(ID_COL)[EAP_LABELS]
            .nunique()
            .gt(1)
            .any(axis=1)
        )

        if conflict.any():

            bad_ids = (
                conflict[
                    conflict
                ].index.tolist()
            )

            raise ValueError(
                "Conflicting labels found for "
                f"duplicate IDs: {bad_ids[:10]}"
            )

        master = (
            master
            .drop_duplicates(
                subset=ID_COL,
                keep="first"
            )
            .reset_index(drop=True)
        )

    # -------------------------------------------------------------
    # Composite annotation variables
    # -------------------------------------------------------------

    master["any_eap"] = (
        master[EAP_LABELS]
        .max(axis=1)
        .astype(int)
    )

    master[
        "n_eap_dimensions"
    ] = (
        master[EAP_LABELS]
        .sum(axis=1)
        .astype(int)
    )

    return master


# ---------------------------------------------------------------------
# EAP validation
# ---------------------------------------------------------------------

def validate_eap_master(master):
    """Summarise the final manually annotated EAP dataset."""

    print("\n" + "=" * 70)
    print("FINAL EAP ANNOTATION DATA")
    print("=" * 70)

    print(
        f"Annotated comments: "
        f"{len(master):,}"
    )

    print("\nLabel prevalence:")

    for label in EAP_LABELS:

        positive = int(
            master[label].sum()
        )

        prevalence = (
            master[label].mean()
            * 100
        )

        print(
            f"{label:<20} "
            f"{positive:>5,} "
            f"({prevalence:.2f}%)"
        )

    print(
        f"\n{'any_eap':<20} "
        f"{master['any_eap'].sum():>5,} "
        f"({master['any_eap'].mean() * 100:.2f}%)"
    )


# ---------------------------------------------------------------------
# Frozen split validation
# ---------------------------------------------------------------------

def load_split(
    path,
    labels
):
    """Load and validate one frozen modelling split."""

    if not path.exists():
        raise FileNotFoundError(
            f"Split not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )

    required = [
        ID_COL,
        TEXT_COL,
        *labels,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} missing: {missing}"
        )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"Duplicate IDs in {path.name}."
        )

    for label in labels:

        df[label] = pd.to_numeric(
            df[label],
            errors="raise"
        ).astype(int)

        if not set(
            df[label].unique()
        ).issubset({0, 1}):

            raise ValueError(
                f"{label} is not binary "
                f"in {path.name}."
            )

    return df


def validate_split_collection(
    split_paths,
    labels,
    name
):
    """Check train/validation/test separation."""

    splits = {
        split_name:
            load_split(
                path,
                labels
            )

        for split_name, path
        in split_paths.items()
    }

    train_ids = set(
        splits["train"][ID_COL]
    )

    validation_ids = set(
        splits["validation"][ID_COL]
    )

    test_ids = set(
        splits["test"][ID_COL]
    )

    if train_ids & validation_ids:
        raise ValueError(
            f"{name}: train/validation "
            "ID leakage detected."
        )

    if train_ids & test_ids:
        raise ValueError(
            f"{name}: train/test "
            "ID leakage detected."
        )

    if validation_ids & test_ids:
        raise ValueError(
            f"{name}: validation/test "
            "ID leakage detected."
        )

    print("\n" + "=" * 70)
    print(f"{name} MODELLING SPLITS")
    print("=" * 70)

    for split_name, df in (
        splits.items()
    ):

        print(
            f"{split_name:<12} "
            f"{len(df):>5,}"
        )

    print(
        "\nPASS: No ID leakage "
        "between splits."
    )

    return splits


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Prepare and validate annotation/training data."""

    ANNOTATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # 1. Main EAP annotation sample
    # -------------------------------------------------------------

    corpus = load_corpus()

    annotation_sample = (
        create_annotation_sample(
            corpus
        )
    )

    annotation_sample.to_csv(
        SAMPLE_FILE,
        encoding="utf-8-sig",
        index=False
    )

    print(
        f"\nAnnotation sample saved to:\n"
        f"{SAMPLE_FILE}"
    )

    # -------------------------------------------------------------
    # 2. Combine completed EAP annotations
    # -------------------------------------------------------------

    eap_master = (
        build_eap_master()
    )

    validate_eap_master(
        eap_master
    )

    eap_master.to_csv(
        MASTER_FILE,
        encoding="utf-8-sig",
        index=False
    )

    print(
        f"\nEAP annotation master saved to:\n"
        f"{MASTER_FILE}"
    )

    # -------------------------------------------------------------
    # 3. Validate final EAP modelling splits
    # -------------------------------------------------------------

    validate_split_collection(
        EAP_SPLITS,
        EAP_LABELS,
        "EAP"
    )

    # -------------------------------------------------------------
    # 4. Validate final incivility modelling splits
    # -------------------------------------------------------------

    validate_split_collection(
        APE_SPLITS,
        ["ape"],
        "INCIVILITY / APE"
    )

    print("\n" + "=" * 70)
    print("TRAINING-DATA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
