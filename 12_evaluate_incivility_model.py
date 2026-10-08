"""
Evaluate the final Antagonistic Political Expression (APE) classifier.

The frozen model and validation-selected threshold produced by
11_train_incivility_model.py are applied once to the held-out test set.

No training, threshold optimisation, checkpoint selection, or other
model changes occur during this stage.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

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
    BertForSequenceClassification,
    BertTokenizerFast,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/model")
MODEL_DIR = Path("models/incivility")
RESULTS_DIR = Path("results/model_evaluation/incivility")

TEST_FILE = DATA_DIR / "APE_V1_test.csv"

MODEL_PATH = MODEL_DIR / "model"
THRESHOLD_FILE = MODEL_DIR / "threshold.json"

ID_COL = "eap_comment_id"
TEXT_COL = "text"
TARGET = "ape"

MAX_LENGTH = 128
BATCH_SIZE = 16


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def get_device():
    """Select the available computation device."""

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    return device


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class APEDataset(Dataset):
    """Tokenised comments with binary APE labels."""

    def __init__(
        self,
        texts,
        labels,
        tokenizer
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    self.labels[index],
                    dtype=torch.long
                ),
        }


# ---------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------

def load_test_data():
    """Load and validate the untouched APE test set."""

    df = pd.read_csv(
        TEST_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    required = [
        ID_COL,
        TEXT_COL,
        TARGET,
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
            "Duplicate IDs found in test data."
        )

    if df[TEXT_COL].isna().any():
        raise ValueError(
            "Missing comment text in test data."
        )

    df[TEXT_COL] = (
        df[TEXT_COL]
        .astype(str)
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="raise"
    ).astype(int)

    if not set(
        df[TARGET].unique()
    ).issubset({0, 1}):

        raise ValueError(
            "APE labels must be binary."
        )

    print("=" * 70)
    print("APE HELD-OUT TEST SET")
    print("=" * 70)

    print(
        f"Test observations: {len(df):,}"
    )

    print(
        f"APE prevalence: "
        f"{df[TARGET].mean() * 100:.2f}%"
    )

    return df


# ---------------------------------------------------------------------
# Frozen threshold
# ---------------------------------------------------------------------

def load_threshold():
    """Load the validation-selected APE threshold."""

    with open(
        THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        threshold_data = json.load(file)

    if "threshold" not in threshold_data:
        raise ValueError(
            "Saved threshold file does not "
            "contain 'threshold'."
        )

    threshold = float(
        threshold_data["threshold"]
    )

    print(
        f"Frozen threshold: "
        f"{threshold:.6f}"
    )

    return threshold


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def predict(
    model,
    loader,
    device
):
    """Generate probabilities for the held-out test set."""

    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for batch in loader:

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probabilities = (
                torch.softmax(
                    outputs.logits,
                    dim=1
                )[:, 1]
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    return (
        np.asarray(all_labels),
        np.asarray(all_probabilities),
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true,
    y_prob,
    threshold
):
    """Calculate final held-out performance metrics."""

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    metrics = {
        "threshold":
            threshold,

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                y_prob
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_prob
            ),
    }

    return y_pred, metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    device = get_device()

    test_df = load_test_data()

    threshold = load_threshold()

    # -------------------------------------------------------------
    # Load frozen model
    # -------------------------------------------------------------

    tokenizer = (
        BertTokenizerFast
        .from_pretrained(
            MODEL_PATH
        )
    )

    model = (
        BertForSequenceClassification
        .from_pretrained(
            MODEL_PATH
        )
    )

    model.to(device)

    # -------------------------------------------------------------
    # Test dataset
    # -------------------------------------------------------------

    test_dataset = APEDataset(
        test_df[TEXT_COL].tolist(),
        test_df[TARGET].tolist(),
        tokenizer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -------------------------------------------------------------
    # Frozen-model predictions
    # -------------------------------------------------------------

    y_true, y_prob = predict(
        model,
        test_loader,
        device
    )

    y_pred, metrics = (
        calculate_metrics(
            y_true,
            y_prob,
            threshold
        )
    )

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL APE HELD-OUT RESULTS")
    print("=" * 70)

    for metric, value in metrics.items():

        print(
            f"{metric:<20} "
            f"{value:.4f}"
        )

    print("\nCONFUSION MATRIX")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    print(matrix)

    print("\nCLASSIFICATION REPORT")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "No APE",
                "APE",
            ],
            digits=3,
            zero_division=0
        )
    )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions = pd.DataFrame(
        {
            ID_COL:
                test_df[ID_COL].values,

            "ape_true":
                y_true,

            "ape_prob":
                y_prob,

            "ape_pred":
                y_pred,
        }
    )

    predictions.to_csv(
        RESULTS_DIR
        / "test_predictions.csv",
        encoding="utf-8-sig",
        index=False
    )

    with open(
        RESULTS_DIR
        / "test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"\nResults saved to: "
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
