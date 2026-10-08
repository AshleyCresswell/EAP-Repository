"""
Evaluate the final EAP BERT classifiers on the held-out test set.

This script performs final evaluation only.

No:
- model training
- checkpoint selection
- threshold optimisation
- hyperparameter tuning

The models and validation-selected thresholds produced by
08_train_bert_models.py are applied unchanged to the held-out
test set.

Models evaluated:
1. Aversion
2. Othering
3. Social Distance
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

DATA_DIR = Path("data/model")
MODEL_ROOT = Path("models/eap")
RESULTS_DIR = Path("results/model_evaluation")

TEST_FILE = DATA_DIR / "EAP_2024_BERT_test_v2.csv"

ID_COL = "eap_comment_id"
TEXT_COL = "text"

LABELS = [
    "aversion",
    "othering",
    "social_distance",
]

MAX_LENGTH = 128
EVAL_BATCH_SIZE = 32


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
            "Evaluation will use CPU."
        )


# ---------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------

def load_test_data():
    """Load and validate the frozen held-out test dataset."""

    df = pd.read_csv(
        TEST_FILE,
        encoding="utf-8-sig"
    )

    required = [
        ID_COL,
        TEXT_COL,
        *LABELS,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Test data missing columns: {missing}"
        )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            "Duplicate comment IDs found in test set."
        )

    if df[TEXT_COL].isna().any():
        raise ValueError(
            "Missing test comment text detected."
        )

    df[TEXT_COL] = (
        df[TEXT_COL]
        .astype(str)
    )

    for label in LABELS:

        df[label] = pd.to_numeric(
            df[label],
            errors="raise"
        ).astype(int)

        if not set(
            df[label].unique()
        ).issubset({0, 1}):

            raise ValueError(
                f"{label} contains non-binary labels."
            )

    print("\n" + "=" * 70)
    print("HELD-OUT TEST DATA")
    print("=" * 70)

    print(
        f"Test observations: {len(df):,}"
    )

    return df


# ---------------------------------------------------------------------
# Load threshold
# ---------------------------------------------------------------------

def load_threshold(label):
    """Load the validation-selected threshold."""

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

        threshold_data = json.load(
            file
        )

    if "threshold" not in threshold_data:
        raise ValueError(
            f"No threshold stored for {label}."
        )

    return float(
        threshold_data["threshold"]
    )


# ---------------------------------------------------------------------
# Tokenise test data
# ---------------------------------------------------------------------

def create_test_dataset(
    df,
    tokenizer,
    label
):
    """Tokenise the held-out test comments."""

    dataset = Dataset.from_pandas(
        df[
            [
                ID_COL,
                TEXT_COL,
                label,
            ]
        ],
        preserve_index=False
    )

    dataset = dataset.rename_column(
        label,
        "labels"
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
    """Convert model logits to class probabilities."""

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
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    labels,
    probabilities,
    threshold
):
    """Calculate final held-out classification metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "threshold":
            threshold,

        "accuracy":
            accuracy_score(
                labels,
                predictions
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                labels,
                predictions
            ),

        "precision":
            precision_score(
                labels,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                labels,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                labels,
                predictions,
                zero_division=0
            ),

        "pr_auc":
            average_precision_score(
                labels,
                probabilities
            ),

        "roc_auc":
            roc_auc_score(
                labels,
                probabilities
            ),
    }

    return predictions, metrics


# ---------------------------------------------------------------------
# Evaluate one classifier
# ---------------------------------------------------------------------

def evaluate_classifier(
    label,
    test_df
):

    print("\n" + "=" * 70)
    print(
        f"HELD-OUT TEST: "
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
        f"Validation-selected threshold: "
        f"{threshold:.6f}"
    )

    # -------------------------------------------------------------
    # Model and tokenizer
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
    # Test dataset
    # -------------------------------------------------------------

    test_dataset = (
        create_test_dataset(
            test_df,
            tokenizer,
            label
        )
    )

    data_collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer
        )
    )

    # -------------------------------------------------------------
    # Prediction only
    # -------------------------------------------------------------

    evaluation_dir = (
        RESULTS_DIR
        / label
        / "trainer_output"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation_args = (
        TrainingArguments(
            output_dir=str(
                evaluation_dir
            ),

            per_device_eval_batch_size=(
                EVAL_BATCH_SIZE
            ),

            report_to="none",
        )
    )

    trainer = Trainer(
        model=model,
        args=evaluation_args,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    output = trainer.predict(
        test_dataset
    )

    logits = (
        output.predictions
    )

    true_labels = (
        output.label_ids
    )

    probabilities = (
        softmax(logits)[:, 1]
    )

    # -------------------------------------------------------------
    # Final metrics
    # -------------------------------------------------------------

    predictions, metrics = (
        calculate_metrics(
            true_labels,
            probabilities,
            threshold
        )
    )

    print("\nFINAL TEST METRICS")

    for metric, value in (
        metrics.items()
    ):
        print(
            f"{metric:<20} "
            f"{value:.4f}"
        )

    # -------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = (
        matrix.ravel()
    )

    print("\nCONFUSION MATRIX")
    print(matrix)

    print("\nBreakdown:")
    print(
        f"True negatives:  {tn}"
    )
    print(
        f"False positives: {fp}"
    )
    print(
        f"False negatives: {fn}"
    )
    print(
        f"True positives:  {tp}"
    )

    # -------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------

    print(
        "\nCLASSIFICATION REPORT"
    )

    print(
        classification_report(
            true_labels,
            predictions,
            target_names=[
                f"No {label}",
                label,
            ],
            digits=3,
            zero_division=0
        )
    )

    # -------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------

    result_dir = (
        RESULTS_DIR
        / label
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df = pd.DataFrame(
        {
            ID_COL:
                test_df[ID_COL].values,

            "true_label":
                true_labels,

            "probability":
                probabilities,

            "prediction":
                predictions,
        }
    )

    prediction_df.to_csv(
        result_dir
        / "test_predictions.csv",
        encoding="utf-8-sig",
        index=False
    )

    # -------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------

    metrics_file = (
        result_dir
        / "test_metrics.json"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print_device()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    test_df = load_test_data()

    all_metrics = {}

    for label in LABELS:

        all_metrics[label] = (
            evaluate_classifier(
                label,
                test_df
            )
        )

    # -------------------------------------------------------------
    # Combined results table
    # -------------------------------------------------------------

    summary = (
        pd.DataFrame(
            all_metrics
        )
        .T
        .reset_index()
        .rename(
            columns={
                "index": "dimension"
            }
        )
    )

    summary.to_csv(
        RESULTS_DIR
        / "EAP_BERT_test_results.csv",
        encoding="utf-8-sig",
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL HELD-OUT RESULTS")
    print("=" * 70)

    print(
        summary[
            [
                "dimension",
                "threshold",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "roc_auc",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to: "
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
