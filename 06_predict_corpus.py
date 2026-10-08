"""
Apply the final BERT classifiers to the full 2024 campaign corpus.

This script applies four frozen binary classifiers:

EAP dimensions
--------------
1. Aversion
2. Othering
3. Social Distance

Benchmark measure
-----------------
4. Incivility (APE)

For each classifier, the script:

1. Loads the frozen model from 04_train_models.py.
2. Loads its validation-selected decision threshold.
3. Generates probabilities for every comment in the corpus.
4. Converts probabilities into binary predictions.

The three EAP predictions are then combined into:
- any_eap
- n_eap_dimensions

No model training or threshold selection occurs here.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CORPUS_FILE = Path(
    "data/processed/EAP_2024_annotation_ready.csv"
)

MODEL_DIR = Path("models")

OUTPUT_DIR = Path(
    "results/full_corpus_predictions"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "EAP_2024_full_predictions.csv"
)

ID_COL = "eap_comment_id"
TEXT_COL = "text"

EAP_LABELS = [
    "aversion",
    "othering",
    "social_distance",
]

MAX_LENGTH = 128
BATCH_SIZE = 64


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

MODELS = {
    "aversion": {
        "model_dir":
            MODEL_DIR
            / "eap"
            / "aversion"
            / "model",

        "threshold_file":
            MODEL_DIR
            / "eap"
            / "aversion"
            / "threshold.json",
    },

    "othering": {
        "model_dir":
            MODEL_DIR
            / "eap"
            / "othering"
            / "model",

        "threshold_file":
            MODEL_DIR
            / "eap"
            / "othering"
            / "threshold.json",
    },

    "social_distance": {
        "model_dir":
            MODEL_DIR
            / "eap"
            / "social_distance"
            / "model",

        "threshold_file":
            MODEL_DIR
            / "eap"
            / "social_distance"
            / "threshold.json",
    },

    "incivility": {
        "model_dir":
            MODEL_DIR
            / "incivility"
            / "model",

        "threshold_file":
            MODEL_DIR
            / "incivility"
            / "threshold.json",
    },
}


# ---------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------

def load_corpus():
    """Load and validate the final research corpus."""

    df = pd.read_csv(
        CORPUS_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    required = [
        ID_COL,
        TEXT_COL,
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

    if df[TEXT_COL].isna().any():
        raise ValueError(
            "Missing comment text detected."
        )

    print(
        f"Corpus loaded: {len(df):,} comments"
    )

    return df


# ---------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------

def load_threshold(path):
    """Load a frozen validation-selected decision threshold."""

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if "threshold" not in data:
        raise ValueError(
            f"No threshold found in {path}."
        )

    return float(
        data["threshold"]
    )


# ---------------------------------------------------------------------
# Inference dataset
# ---------------------------------------------------------------------

def prepare_dataset(
    texts,
    tokenizer
):
    """Tokenise the full corpus for model inference."""

    dataset = Dataset.from_dict(
        {
            TEXT_COL:
                texts.astype(str).tolist()
        }
    )

    def tokenize(batch):

        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    dataset = dataset.map(
        tokenize,
        batched=True
    )

    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
        ]
    )

    return dataset


# ---------------------------------------------------------------------
# Apply one classifier
# ---------------------------------------------------------------------

def predict_model(
    df,
    name,
    model_dir,
    threshold_file
):
    """Apply one frozen classifier to the full corpus."""

    print("\n" + "=" * 70)
    print(f"FULL-CORPUS INFERENCE: {name.upper()}")
    print("=" * 70)

    threshold = load_threshold(
        threshold_file
    )

    print(
        f"Frozen threshold: {threshold:.6f}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_dir
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_dir
        )
    )

    dataset = prepare_dataset(
        df[TEXT_COL],
        tokenizer
    )

    inference_args = TrainingArguments(
        output_dir=str(
            OUTPUT_DIR
            / "temporary"
            / name
        ),
        per_device_eval_batch_size=(
            BATCH_SIZE
        ),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=inference_args,
        processing_class=tokenizer
    )

    output = trainer.predict(
        dataset
    )

    probabilities = torch.softmax(
        torch.tensor(
            output.predictions
        ),
        dim=1
    )[:, 1].numpy()

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print(
        f"Positive predictions: "
        f"{predictions.sum():,}"
    )

    print(
        f"Prevalence: "
        f"{predictions.mean() * 100:.3f}%"
    )

    return (
        probabilities,
        predictions
    )


# ---------------------------------------------------------------------
# EAP variables
# ---------------------------------------------------------------------

def create_eap_variables(df):
    """Combine the three EAP dimensions into composite measures."""

    dimension_columns = [
        "aversion",
        "othering",
        "social_distance",
    ]

    df["any_eap"] = (
        df[dimension_columns]
        .max(axis=1)
        .astype(int)
    )

    df["n_eap_dimensions"] = (
        df[dimension_columns]
        .sum(axis=1)
        .astype(int)
    )

    return df


# ---------------------------------------------------------------------
# Overlap variables
# ---------------------------------------------------------------------

def create_overlap_variables(df):
    """Construct the EAP × incivility comparison categories."""

    conditions = [
        (
            (df["any_eap"] == 0)
            & (df["incivility"] == 0)
        ),

        (
            (df["any_eap"] == 0)
            & (df["incivility"] == 1)
        ),

        (
            (df["any_eap"] == 1)
            & (df["incivility"] == 0)
        ),

        (
            (df["any_eap"] == 1)
            & (df["incivility"] == 1)
        ),
    ]

    categories = [
        "Neither",
        "Incivility only",
        "EAP only",
        "Both",
    ]

    df["eap_incivility_group"] = (
        np.select(
            conditions,
            categories,
            default="Unknown"
        )
    )

    return df


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_predictions(df):
    """Run integrity checks on full-corpus predictions."""

    print("\n" + "=" * 70)
    print("FULL-CORPUS PREDICTION QC")
    print("=" * 70)

    binary_columns = [
        "aversion",
        "othering",
        "social_distance",
        "incivility",
        "any_eap",
    ]

    probability_columns = [
        "aversion_prob",
        "othering_prob",
        "social_distance_prob",
        "incivility_prob",
    ]

    for column in binary_columns:

        if not set(
            df[column].unique()
        ).issubset({0, 1}):

            raise ValueError(
                f"{column} contains "
                "non-binary predictions."
            )

    for column in probability_columns:

        if not df[column].between(
            0,
            1
        ).all():

            raise ValueError(
                f"{column} contains probabilities "
                "outside [0, 1]."
            )

    if not df[
        "n_eap_dimensions"
    ].between(
        0,
        3
    ).all():

        raise ValueError(
            "Invalid EAP dimension count."
        )

    print(
        f"Comments: "
        f"{len(df):,}"
    )

    print("\nPrediction prevalence:")

    for column in [
        "aversion",
        "othering",
        "social_distance",
        "any_eap",
        "incivility",
    ]:

        positive = int(
            df[column].sum()
        )

        prevalence = (
            df[column].mean()
            * 100
        )

        print(
            f"{column:<20} "
            f"{positive:>8,} "
            f"({prevalence:.3f}%)"
        )

    print(
        "\nPASS: Full-corpus predictions validated."
    )


# ---------------------------------------------------------------------
# Overlap summary
# ---------------------------------------------------------------------

def print_overlap_summary(df):
    """Summarise the relationship between EAP and incivility."""

    print("\n" + "=" * 70)
    print("EAP × INCIVILITY")
    print("=" * 70)

    counts = (
        df["eap_incivility_group"]
        .value_counts()
        .reindex(
            [
                "Neither",
                "Incivility only",
                "EAP only",
                "Both",
            ]
        )
    )

    for category, count in (
        counts.items()
    ):

        percentage = (
            count
            / len(df)
            * 100
        )

        print(
            f"{category:<20} "
            f"{count:>8,} "
            f"({percentage:.3f}%)"
        )

    both = int(
        (
            (df["any_eap"] == 1)
            & (df["incivility"] == 1)
        ).sum()
    )

    eap_total = int(
        df["any_eap"].sum()
    )

    incivility_total = int(
        df["incivility"].sum()
    )

    if incivility_total:

        print(
            "\nP(EAP | Incivility): "
            f"{both / incivility_total * 100:.2f}%"
        )

    if eap_total:

        print(
            "P(Incivility | EAP): "
            f"{both / eap_total * 100:.2f}%"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Apply all four frozen models to the research corpus."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("FULL-CORPUS BERT INFERENCE")
    print("=" * 70)

    df = load_corpus()

    # -------------------------------------------------------------
    # EAP dimensions
    # -------------------------------------------------------------

    for label in EAP_LABELS:

        probabilities, predictions = (
            predict_model(
                df=df,
                name=label,
                model_dir=(
                    MODELS[label][
                        "model_dir"
                    ]
                ),
                threshold_file=(
                    MODELS[label][
                        "threshold_file"
                    ]
                ),
            )
        )

        df[
            f"{label}_prob"
        ] = probabilities

        df[label] = predictions

    # -------------------------------------------------------------
    # Incivility
    # -------------------------------------------------------------

    probabilities, predictions = (
        predict_model(
            df=df,
            name="incivility",
            model_dir=(
                MODELS[
                    "incivility"
                ]["model_dir"]
            ),
            threshold_file=(
                MODELS[
                    "incivility"
                ]["threshold_file"]
            ),
        )
    )

    df[
        "incivility_prob"
    ] = probabilities

    df[
        "incivility"
    ] = predictions

    # -------------------------------------------------------------
    # Composite EAP measure
    # -------------------------------------------------------------

    df = create_eap_variables(
        df
    )

    # -------------------------------------------------------------
    # EAP × incivility comparison
    # -------------------------------------------------------------

    df = create_overlap_variables(
        df
    )

    # -------------------------------------------------------------
    # QC
    # -------------------------------------------------------------

    validate_predictions(
        df
    )

    print_overlap_summary(
        df
    )

    # -------------------------------------------------------------
    # Save final predicted corpus
    # -------------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        index=False
    )

    print("\n" + "=" * 70)
    print("FULL-CORPUS INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"Saved predicted corpus to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
