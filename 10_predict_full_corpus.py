"""
Apply the final EAP BERT classifiers to the full 2024 comment corpus.

The frozen Aversion, Othering, and Social Distance models are applied
to every eligible comment. Validation-selected thresholds are used
unchanged to create binary predictions.

Outputs:
- predicted probability for each EAP dimension
- binary prediction for each EAP dimension
- any_eap_pred
- n_eap_dimensions_pred

No training, threshold optimisation, or model selection occurs here.
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
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/processed")
MODEL_ROOT = Path("models/eap")
OUTPUT_DIR = Path("results/full_corpus_predictions")

INPUT_FILE = (
    DATA_DIR
    / "EAP_2024_annotation_ready.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "EAP_2024_full_BERT_predictions.csv"
)

ID_COL = "eap_comment_id"
TEXT_COL = "text"

LABELS = [
    "aversion",
    "othering",
    "social_distance",
]

MAX_LENGTH = 128
BATCH_SIZE = 64


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def print_device():
    """Report the available computation device."""

    print("=" * 70)
    print("DEVICE")
    print("=" * 70)

    if torch.cuda.is_available():
        print(
            "CUDA GPU available:",
            torch.cuda.get_device_name(0)
        )

    elif torch.backends.mps.is_available():
        print(
            "Apple MPS accelerator available."
        )

    else:
        print(
            "No GPU/MPS accelerator detected."
        )
        print(
            "Inference will use CPU."
        )


# ---------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------

def load_corpus():
    """Load and validate the final EAP corpus."""

    df = pd.read_csv(
        INPUT_FILE,
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
            "Duplicate EAP comment IDs detected."
        )

    if df[TEXT_COL].isna().any():
        raise ValueError(
            "Missing comment text detected."
        )

    df[TEXT_COL] = (
        df[TEXT_COL]
        .astype(str)
    )

    print("\n" + "=" * 70)
    print("FULL EAP CORPUS")
    print("=" * 70)

    print(
        f"Comments loaded: {len(df):,}"
    )

    return df


# ---------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------

def load_threshold(label):
    """Load the frozen validation-selected threshold."""

    threshold_file = (
        MODEL_ROOT
        / label
        / "threshold.json"
    )

    if not threshold_file.exists():
        raise FileNotFoundError(
            f"Threshold file not found: "
            f"{threshold_file}"
        )

    with open(
        threshold_file,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if "threshold" not in data:
        raise ValueError(
            f"No threshold stored for {label}."
        )

    return float(
        data["threshold"]
    )


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def create_dataset(
    df,
    tokenizer
):
    """Tokenise the full comment corpus."""

    dataset = Dataset.from_pandas(
        df[
            [
                ID_COL,
                TEXT_COL,
            ]
        ],
        preserve_index=False
    )

    def tokenize(batch):

        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            max_length=MAX_LENGTH
        )

    return dataset.map(
        tokenize,
        batched=True
    )


# ---------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------

def softmax(logits):
    """Convert model logits to probabilities."""

    logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    probabilities = np.exp(
        logits
    )

    return (
        probabilities
        / probabilities.sum(
            axis=1,
            keepdims=True
        )
    )


# ---------------------------------------------------------------------
# Predict one EAP dimension
# ---------------------------------------------------------------------

def predict_dimension(
    df,
    label
):
    """Apply one frozen EAP classifier to the full corpus."""

    print("\n" + "=" * 70)
    print(
        f"FULL-CORPUS INFERENCE: "
        f"{label.upper()}"
    )
    print("=" * 70)

    model_dir = (
        MODEL_ROOT
        / label
        / "model"
    )

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model not found: {model_dir}"
        )

    # -------------------------------------------------------------
    # Frozen threshold
    # -------------------------------------------------------------

    threshold = load_threshold(
        label
    )

    print(
        f"Threshold: {threshold:.6f}"
    )

    # -------------------------------------------------------------
    # Load frozen model
    # -------------------------------------------------------------

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            str(model_dir)
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            str(model_dir)
        )
    )

    # -------------------------------------------------------------
    # Tokenise corpus
    # -------------------------------------------------------------

    dataset = create_dataset(
        df,
        tokenizer
    )

    data_collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer
        )
    )

    # -------------------------------------------------------------
    # Prediction only
    # -------------------------------------------------------------

    inference_dir = (
        OUTPUT_DIR
        / "_trainer"
        / label
    )

    inference_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    inference_args = (
        TrainingArguments(
            output_dir=str(
                inference_dir
            ),

            per_device_eval_batch_size=(
                BATCH_SIZE
            ),

            report_to="none",
        )
    )

    trainer = Trainer(
        model=model,
        args=inference_args,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    output = trainer.predict(
        dataset
    )

    logits = (
        output.predictions
    )

    probabilities = (
        softmax(logits)[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    positive = int(
        predictions.sum()
    )

    prevalence = (
        positive
        / len(predictions)
        * 100
    )

    print(
        f"Positive predictions: "
        f"{positive:,}"
    )

    print(
        f"Prevalence: "
        f"{prevalence:.3f}%"
    )

    return (
        probabilities,
        predictions
    )


# ---------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------

def run_qc(df):
    """Run final checks on EAP predictions."""

    prediction_columns = [
        f"{label}_pred"
        for label in LABELS
    ]

    probability_columns = [
        f"{label}_prob"
        for label in LABELS
    ]

    print("\n" + "=" * 70)
    print("FULL-CORPUS PREDICTION QC")
    print("=" * 70)

    # Binary predictions

    for column in prediction_columns:

        if df[column].isna().any():
            raise ValueError(
                f"Missing values detected in {column}."
            )

        if not set(
            df[column].unique()
        ).issubset({0, 1}):

            raise ValueError(
                f"{column} is not binary."
            )

    # Probabilities

    for column in probability_columns:

        if df[column].isna().any():
            raise ValueError(
                f"Missing probabilities in {column}."
            )

        if not df[column].between(
            0,
            1
        ).all():

            raise ValueError(
                f"Invalid probabilities in {column}."
            )

    # Composite variables

    if not df[
        "n_eap_dimensions_pred"
    ].between(0, 3).all():

        raise ValueError(
            "Invalid EAP dimension count."
        )

    print(
        "PASS: Prediction variables validated."
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_summary(df):
    """Print final corpus prevalence estimates."""

    print("\n" + "=" * 70)
    print("EAP PREVALENCE")
    print("=" * 70)

    for label in LABELS:

        column = (
            f"{label}_pred"
        )

        positive = int(
            df[column].sum()
        )

        prevalence = (
            positive
            / len(df)
            * 100
        )

        print(
            f"{label:<20} "
            f"{positive:>8,} "
            f"({prevalence:.3f}%)"
        )

    any_eap = int(
        df["any_eap_pred"].sum()
    )

    print(
        f"\n{'any_eap':<20} "
        f"{any_eap:>8,} "
        f"({any_eap / len(df) * 100:.3f}%)"
    )

    print(
        "\nNumber of EAP dimensions:"
    )

    print(
        df[
            "n_eap_dimensions_pred"
        ]
        .value_counts()
        .sort_index()
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print_device()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_corpus()

    # -------------------------------------------------------------
    # Apply each frozen classifier
    # -------------------------------------------------------------

    for label in LABELS:

        probabilities, predictions = (
            predict_dimension(
                df,
                label
            )
        )

        df[
            f"{label}_prob"
        ] = probabilities

        df[
            f"{label}_pred"
        ] = predictions

    # -------------------------------------------------------------
    # Composite EAP measures
    # -------------------------------------------------------------

    prediction_columns = [
        f"{label}_pred"
        for label in LABELS
    ]

    df["any_eap_pred"] = (
        df[prediction_columns]
        .max(axis=1)
        .astype(int)
    )

    df[
        "n_eap_dimensions_pred"
    ] = (
        df[prediction_columns]
        .sum(axis=1)
        .astype(int)
    )

    # -------------------------------------------------------------
    # Quality control
    # -------------------------------------------------------------

    run_qc(df)

    print_summary(df)

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        index=False
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
