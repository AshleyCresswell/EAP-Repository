"""
Train the Antagonistic Political Expression (APE) classifier.

APE is used as the incivility benchmark against the three-dimensional
Expressive Affective Polarisation (EAP) measure.

This script:
1. Trains bert-base-uncased on the APE training set.
2. Evaluates performance on the validation set.
3. Selects the probability threshold that maximises validation F1.
4. Saves the frozen model and threshold for held-out evaluation.

The held-out test set is not accessed here.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/model")
OUTPUT_DIR = Path("models/incivility")

TRAIN_FILE = DATA_DIR / "APE_V1_train.csv"
VAL_FILE = DATA_DIR / "APE_V1_validation.csv"

MODEL_NAME = "bert-base-uncased"

ID_COL = "eap_comment_id"
TEXT_COL = "text"
TARGET = "ape"

SEED = 42

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_FRACTION = 0.10


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed):
    """Set random seeds used during model training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def get_device():
    """Select the available computation device."""

    if torch.cuda.is_available():

        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)

    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):

        device = torch.device("mps")
        name = "Apple MPS"

    else:

        device = torch.device("cpu")
        name = "CPU"

    print("=" * 70)
    print("DEVICE")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Device name: {name}")

    return device


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_data():
    """Load and validate APE training and validation data."""

    train_df = pd.read_csv(
        TRAIN_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    val_df = pd.read_csv(
        VAL_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    required = [
        ID_COL,
        TEXT_COL,
        TARGET,
    ]

    for name, df in [
        ("Training", train_df),
        ("Validation", val_df),
    ]:

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} data missing columns: {missing}"
            )

        if df[ID_COL].duplicated().any():
            raise ValueError(
                f"Duplicate IDs found in {name} data."
            )

        if df[TEXT_COL].isna().any():
            raise ValueError(
                f"Missing text found in {name} data."
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
                f"{name} APE labels are not binary."
            )

    print("\n" + "=" * 70)
    print("DATA")
    print("=" * 70)

    print(f"Training:   {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")

    print(
        "\nTraining APE prevalence: "
        f"{train_df[TARGET].mean() * 100:.2f}%"
    )

    print(
        "Validation APE prevalence: "
        f"{val_df[TARGET].mean() * 100:.2f}%"
    )

    return train_df, val_df


# ---------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------

class APEDataset(Dataset):
    """Tokenised YouTube comments with binary APE labels."""

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
                encoding[
                    "input_ids"
                ].squeeze(0),

            "attention_mask":
                encoding[
                    "attention_mask"
                ].squeeze(0),

            "labels":
                torch.tensor(
                    self.labels[index],
                    dtype=torch.long
                ),
        }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device
):
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0

    for batch in loader:

        optimizer.zero_grad()

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
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += (
            loss.item()
        )

    return (
        total_loss
        / len(loader)
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(
    model,
    loader,
    device
):
    """Return validation loss, labels, and positive probabilities."""

    model.eval()

    total_loss = 0.0

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
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += (
                outputs.loss.item()
            )

            probabilities = (
                torch.softmax(
                    outputs.logits,
                    dim=1
                )[:, 1]
            )

            all_labels.extend(
                labels
                .cpu()
                .numpy()
            )

            all_probabilities.extend(
                probabilities
                .cpu()
                .numpy()
            )

    return (
        total_loss / len(loader),
        np.asarray(all_labels),
        np.asarray(all_probabilities),
    )


# ---------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------

def select_threshold(
    y_true,
    y_prob
):
    """
    Select the validation threshold that maximises
    positive-class F1.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_prob
        )
    )

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-12
        )
    )

    best_index = int(
        np.argmax(f1)
    )

    return {
        "threshold":
            float(
                thresholds[
                    best_index
                ]
            ),

        "validation_precision":
            float(
                precision[
                    best_index
                ]
            ),

        "validation_recall":
            float(
                recall[
                    best_index
                ]
            ),

        "validation_f1":
            float(
                f1[
                    best_index
                ]
            ),

        "selection_rule":
            "Maximum validation F1",
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    set_seed(SEED)

    device = get_device()

    train_df, val_df = (
        load_data()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # Tokenizer and datasets
    # -------------------------------------------------------------

    tokenizer = (
        BertTokenizerFast
        .from_pretrained(
            MODEL_NAME
        )
    )

    train_dataset = APEDataset(
        train_df[TEXT_COL].tolist(),
        train_df[TARGET].tolist(),
        tokenizer
    )

    val_dataset = APEDataset(
        val_df[TEXT_COL].tolist(),
        val_df[TARGET].tolist(),
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = (
        BertForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2
        )
    )

    model.to(device)

    # -------------------------------------------------------------
    # Optimizer and scheduler
    # -------------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    total_steps = (
        len(train_loader)
        * EPOCHS
    )

    warmup_steps = int(
        total_steps
        * WARMUP_FRACTION
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    )

    # -------------------------------------------------------------
    # Train
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING APE CLASSIFIER")
    print("=" * 70)

    training_history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device
        )

        (
            val_loss,
            val_labels,
            val_probabilities,
        ) = evaluate(
            model,
            val_loader,
            device
        )

        # 0.50 is used only as an epoch-level
        # diagnostic, not as the final threshold.

        val_predictions = (
            val_probabilities >= 0.50
        ).astype(int)

        (
            precision,
            recall,
            f1,
            _
        ) = (
            precision_recall_fscore_support(
                val_labels,
                val_predictions,
                average="binary",
                zero_division=0
            )
        )

        accuracy = accuracy_score(
            val_labels,
            val_predictions
        )

        roc_auc = roc_auc_score(
            val_labels,
            val_probabilities
        )

        pr_auc = (
            average_precision_score(
                val_labels,
                val_probabilities
            )
        )

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "accuracy_050": accuracy,
                "precision_050": precision,
                "recall_050": recall,
                "f1_050": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train loss:      "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation F1 "
            f"@ 0.50: {f1:.4f}"
        )

        print(
            f"Validation ROC-AUC: "
            f"{roc_auc:.4f}"
        )

        print(
            f"Validation PR-AUC:  "
            f"{pr_auc:.4f}"
        )

    # -------------------------------------------------------------
    # Final validation probabilities
    # -------------------------------------------------------------

    (
        final_val_loss,
        val_labels,
        val_probabilities,
    ) = evaluate(
        model,
        val_loader,
        device
    )

    validation_output = pd.DataFrame(
        {
            ID_COL:
                val_df[
                    ID_COL
                ].values,

            "ape_true":
                val_labels,

            "ape_prob":
                val_probabilities,
        }
    )

    validation_output.to_csv(
        OUTPUT_DIR
        / "validation_probabilities.csv",
        encoding="utf-8-sig",
        index=False
    )

    # -------------------------------------------------------------
    # Validation threshold selection
    # -------------------------------------------------------------

    threshold_results = (
        select_threshold(
            val_labels,
            val_probabilities
        )
    )

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold: "
        f"{threshold_results['threshold']:.6f}"
    )

    print(
        f"Precision: "
        f"{threshold_results['validation_precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{threshold_results['validation_recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{threshold_results['validation_f1']:.4f}"
    )

    # -------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------

    model_dir = (
        OUTPUT_DIR
        / "model"
    )

    model.save_pretrained(
        model_dir
    )

    tokenizer.save_pretrained(
        model_dir
    )

    # -------------------------------------------------------------
    # Save threshold
    # -------------------------------------------------------------

    with open(
        OUTPUT_DIR
        / "threshold.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            threshold_results,
            file,
            indent=4
        )

    # -------------------------------------------------------------
    # Save training history
    # -------------------------------------------------------------

    pd.DataFrame(
        training_history
    ).to_csv(
        OUTPUT_DIR
        / "training_history.csv",
        encoding="utf-8-sig",
        index=False
    )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    metadata = {
        "construct":
            "Antagonistic Political Expression",

        "construct_version":
            "APE v1",

        "base_model":
            MODEL_NAME,

        "target":
            TARGET,

        "seed":
            SEED,

        "max_length":
            MAX_LENGTH,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "warmup_fraction":
            WARMUP_FRACTION,

        "class_weights":
            False,

        "threshold_selection":
            "Maximum validation F1",

        "test_status":
            "UNTOUCHED",
    }

    with open(
        OUTPUT_DIR
        / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print("APE TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Model saved to: "
        f"{model_dir}"
    )

    print(
        "Held-out test set: "
        "NOT ACCESSED"
    )


if __name__ == "__main__":
    main()
