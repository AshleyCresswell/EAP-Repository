"""
Evaluate the final BERT classifiers on held-out test data.

This script evaluates four frozen binary classifiers:

EAP dimensions
--------------
1. Aversion
2. Othering
3. Social Distance

Benchmark measure
-----------------
4. Incivility (APE)

Models and decision thresholds are loaded from 04_train_models.py.
No training, threshold optimisation, or model selection occurs here.

Each classifier is evaluated once against its untouched held-out
test set.
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
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/model")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results/model_evaluation")

EAP_TEST_FILE = (
    DATA_DIR
    / "EAP_2024_BERT_test_v2.csv"
)

APE_TEST_FILE = (
    DATA_DIR
    / "APE_V1_test.csv"
)

TEXT_COL = "text"
ID_COL = "eap_comment_id"

EAP_LABELS = [
    "aversion",
    "othering",
    "social_distance",
]

MAX_LENGTH = 128
BATCH_SIZE = 32


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_test_data(path, label):
    """Load and validate one held-out test dataset."""

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )

    required = [
        ID_COL,
        TEXT_COL,
        label,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} missing columns: {missing}"
        )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"Duplicate IDs detected in {path.name}."
        )

    if df[TEXT_COL].isna().any():
        raise ValueError(
            f"Missing text detected in {path.name}."
        )

    df[label] = pd.to_numeric(
        df[label],
        errors="raise"
    ).astype(int)

    if not set(
        df[label].unique()
    ).issubset({0, 1}):

        raise ValueError(
            f"{label} contains non-binary values."
        )

    return df


# ---------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------

def load_threshold(path):
    """Load the frozen validation-selected threshold."""

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        threshold_data = json.load(file)

    if "threshold" not in threshold_data:
        raise ValueError(
            f"No threshold found in {path}."
        )

    return float(
        threshold_data["threshold"]
    )


# ---------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------

def prepare_dataset(
    df,
    label,
    tokenizer
):
    """Tokenise held-out comments for BERT evaluation."""

    dataset = Dataset.from_pandas(
        df[
            [
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
            "labels",
        ]
    )

    return dataset


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    labels,
    probabilities,
    predictions
):
    """Calculate held-out binary classification metrics."""

    return {
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


# ---------------------------------------------------------------------
# Evaluate one classifier
# ---------------------------------------------------------------------

def evaluate_model(
    name,
    label,
    test_file,
    model_dir,
    threshold_file
):
    """Evaluate one frozen classifier on held-out test data."""

    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)

    # -------------------------------------------------------------
    # Load test data
    # -------------------------------------------------------------

    test_df = load_test_data(
        test_file,
        label
    )

    print(
        f"Test rows: "
        f"{len(test_df):,}"
    )

    print(
        f"Positive labels: "
        f"{test_df[label].sum():,}"
    )

    # -------------------------------------------------------------
    # Load frozen threshold
    # -------------------------------------------------------------

    threshold = load_threshold(
        threshold_file
    )

    print(
        f"Frozen threshold: "
        f"{threshold:.6f}"
    )

    # -------------------------------------------------------------
    # Load frozen model
    # -------------------------------------------------------------

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

    test_dataset = prepare_dataset(
        test_df,
        label,
        tokenizer
    )

    # Trainer is used only for inference.
    # No optimisation occurs.

    evaluation_args = TrainingArguments(
        output_dir=str(
            RESULTS_DIR
            / "temporary"
            / label
        ),
        per_device_eval_batch_size=(
            BATCH_SIZE
        ),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=evaluation_args,
        processing_class=tokenizer
    )

    output = trainer.predict(
        test_dataset
    )

    # -------------------------------------------------------------
    # Probabilities and predictions
    # -------------------------------------------------------------

    probabilities = torch.softmax(
        torch.tensor(
            output.predictions
        ),
        dim=1
    )[:, 1].numpy()

    labels = (
        test_df[label]
        .to_numpy()
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    # -------------------------------------------------------------
    # Performance
    # -------------------------------------------------------------

    metrics = calculate_metrics(
        labels,
        probabilities,
        predictions
    )

    metrics["threshold"] = (
        threshold
    )

    metrics["test_n"] = (
        len(test_df)
    )

    metrics["positive_n"] = int(
        labels.sum()
    )

    print("\nHeld-out test performance:")

    for metric in [
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "balanced_accuracy",
        "accuracy",
    ]:

        print(
            f"{metric:<20} "
            f"{metrics[metric]:.3f}"
        )

    # -------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------

    matrix = confusion_matrix(
        labels,
        predictions
    )

    print("\nConfusion matrix:")
    print(matrix)

    print("\nClassification report:")

    print(
        classification_report(
            labels,
            predictions,
            digits=3,
            zero_division=0
        )
    )

    # -------------------------------------------------------------
    # Save model-specific results
    # -------------------------------------------------------------

    output_dir = (
        RESULTS_DIR
        / label
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df = pd.DataFrame(
        {
            ID_COL:
                test_df[
                    ID_COL
                ].to_numpy(),

            "true_label":
                labels,

            "probability":
                probabilities,

            "prediction":
                predictions,
        }
    )

    prediction_df.to_csv(
        output_dir
        / "test_predictions.csv",
        encoding="utf-8-sig",
        index=False
    )

    with open(
        output_dir
        / "test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    return {
        "model": name,
        **metrics,
    }


# ---------------------------------------------------------------------
# EAP models
# ---------------------------------------------------------------------

def evaluate_eap_models():
    """Evaluate the three EAP classifiers."""

    results = []

    for label in EAP_LABELS:

        result = evaluate_model(
            name=label.replace(
                "_",
                " "
            ).title(),

            label=label,

            test_file=(
                EAP_TEST_FILE
            ),

            model_dir=(
                MODEL_DIR
                / "eap"
                / label
                / "model"
            ),

            threshold_file=(
                MODEL_DIR
                / "eap"
                / label
                / "threshold.json"
            ),
        )

        results.append(result)

    return results


# ---------------------------------------------------------------------
# Incivility model
# ---------------------------------------------------------------------

def evaluate_incivility_model():
    """Evaluate the held-out incivility classifier."""

    result = evaluate_model(
        name="Incivility / APE",

        label="ape",

        test_file=(
            APE_TEST_FILE
        ),

        model_dir=(
            MODEL_DIR
            / "incivility"
            / "model"
        ),

        threshold_file=(
            MODEL_DIR
            / "incivility"
            / "threshold.json"
        ),
    )

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Evaluate all four frozen classifiers."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("HELD-OUT MODEL EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # EAP
    # -------------------------------------------------------------

    results = (
        evaluate_eap_models()
    )

    # -------------------------------------------------------------
    # Incivility
    # -------------------------------------------------------------

    results.append(
        evaluate_incivility_model()
    )

    # -------------------------------------------------------------
    # Combined results table
    # -------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    column_order = [
        "model",
        "test_n",
        "positive_n",
        "threshold",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "balanced_accuracy",
        "accuracy",
    ]

    results_df = results_df[
        column_order
    ]

    results_df.to_csv(
        RESULTS_DIR
        / "model_test_results.csv",
        encoding="utf-8-sig",
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL HELD-OUT RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "f1",
                "pr_auc",
                "roc_auc",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
        f"\n{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
