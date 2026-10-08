"""
Train the three Expressive Affective Polarisation BERT classifiers.

Models are trained independently for:
1. Aversion
2. Othering
3. Social Distance

Each classifier uses:
- bert-base-uncased
- class-weighted cross-entropy loss
- three training epochs
- PR-AUC for checkpoint selection
- validation-set threshold optimisation

The held-out test set is not accessed in this script.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
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
OUTPUT_ROOT = Path("models/eap")

TRAIN_FILE = DATA_DIR / "EAP_2024_BERT_train_v2.csv"
VAL_FILE = DATA_DIR / "EAP_2024_BERT_validation_v2.csv"

MODEL_NAME = "bert-base-uncased"

TEXT_COL = "text"

LABELS = [
    "aversion",
    "othering",
    "social_distance",
]

RANDOM_SEED = 42

MAX_LENGTH = 128
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01


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
# Data
# ---------------------------------------------------------------------

def load_data():
    """Load the frozen training and validation datasets."""

    train_df = pd.read_csv(
        TRAIN_FILE,
        encoding="utf-8-sig"
    )

    val_df = pd.read_csv(
        VAL_FILE,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("DATA LOADED")
    print("=" * 60)

    print(f"Training rows:   {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")

    return train_df, val_df


def prepare_labels(df, label):
    """Validate text and binary labels."""

    df = df.copy()

    required = [
        "eap_comment_id",
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
            f"Missing required columns: {missing}"
        )

    if df[TEXT_COL].isna().any():
        raise ValueError(
            "Missing comment text detected."
        )

    df[TEXT_COL] = (
        df[TEXT_COL]
        .astype(str)
    )

    df[label] = pd.to_numeric(
        df[label],
        errors="raise"
    ).astype(int)

    if not set(
        df[label].unique()
    ).issubset({0, 1}):

        raise ValueError(
            f"{label} must contain only 0/1."
        )

    return df


# ---------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------

def calculate_class_weights(labels):
    """Calculate inverse-frequency weight for the positive class."""

    negative = int(
        (labels == 0).sum()
    )

    positive = int(
        (labels == 1).sum()
    )

    if positive == 0:
        raise ValueError(
            "Training data contains no positive observations."
        )

    positive_weight = (
        negative / positive
    )

    return torch.tensor(
        [1.0, positive_weight],
        dtype=torch.float32
    )


# ---------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------

def create_dataset(df, tokenizer, label):
    """Tokenise comments for Hugging Face Trainer."""

    dataset = Dataset.from_pandas(
        df[
            [
                "eap_comment_id",
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
# Weighted Trainer
# ---------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """Trainer using class-weighted cross-entropy loss."""

    def __init__(
        self,
        *args,
        class_weights=None,
        **kwargs
    ):

        super().__init__(
            *args,
            **kwargs
        )

        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):

        labels = inputs.pop("labels")

        outputs = model(
            **inputs
        )

        logits = outputs.logits

        weights = self.class_weights.to(
            logits.device
        )

        loss_function = nn.CrossEntropyLoss(
            weight=weights
        )

        loss = loss_function(
            logits,
            labels
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def softmax(logits):
    """Convert logits to class probabilities."""

    logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    probabilities = np.exp(logits)

    return (
        probabilities
        / probabilities.sum(
            axis=1,
            keepdims=True
        )
    )


def compute_metrics(eval_pred):
    """Calculate validation metrics at the default threshold."""

    logits, labels = eval_pred

    probabilities = softmax(logits)

    positive_probability = (
        probabilities[:, 1]
    )

    predictions = (
        positive_probability >= 0.5
    ).astype(int)

    metrics = {
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
                positive_probability
            ),
    }

    if len(np.unique(labels)) == 2:

        metrics["roc_auc"] = (
            roc_auc_score(
                labels,
                positive_probability
            )
        )

    return metrics


# ---------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------

def optimise_threshold(labels, probabilities):
    """
    Select the validation probability threshold
    that maximises positive-class F1.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            labels,
            probabilities
        )
    )

    # precision and recall contain one additional value
    # beyond the threshold array.
    precision = precision[:-1]
    recall = recall[:-1]

    denominator = (
        precision + recall
    )

    f1_scores = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0
    )

    best_index = int(
        np.argmax(f1_scores)
    )

    return {
        "threshold":
            float(
                thresholds[best_index]
            ),

        "validation_f1":
            float(
                f1_scores[best_index]
            ),

        "validation_precision":
            float(
                precision[best_index]
            ),

        "validation_recall":
            float(
                recall[best_index]
            ),
    }


# ---------------------------------------------------------------------
# Train one EAP dimension
# ---------------------------------------------------------------------

def train_classifier(
    label,
    train_df,
    val_df
):

    print("\n" + "=" * 70)
    print(f"TRAINING: {label.upper()}")
    print("=" * 70)

    train = prepare_labels(
        train_df,
        label
    )

    validation = prepare_labels(
        val_df,
        label
    )

    # -------------------------------------------------------------
    # Distribution
    # -------------------------------------------------------------

    negative = int(
        (train[label] == 0).sum()
    )

    positive = int(
        (train[label] == 1).sum()
    )

    print(
        f"\nTraining negatives: "
        f"{negative:,}"
    )

    print(
        f"Training positives: "
        f"{positive:,}"
    )

    print(
        f"Positive prevalence: "
        f"{positive / len(train) * 100:.2f}%"
    )

    # -------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    train_dataset = create_dataset(
        train,
        tokenizer,
        label
    )

    val_dataset = create_dataset(
        validation,
        tokenizer,
        label
    )

    data_collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer
        )
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2
        )
    )

    class_weights = (
        calculate_class_weights(
            train[label]
        )
    )

    print(
        "\nClass weights:",
        class_weights.tolist()
    )

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    model_dir = (
        OUTPUT_ROOT
        / label
    )

    checkpoint_dir = (
        model_dir
        / "checkpoints"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # Training arguments
    # -------------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(
            checkpoint_dir
        ),

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        num_train_epochs=NUM_EPOCHS,

        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="pr_auc",
        greater_is_better=True,

        save_total_limit=1,

        report_to="none",

        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
    )

    trainer = WeightedTrainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=val_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics,

        class_weights=class_weights,
    )

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    trainer.train()

    print(
        "\nBest checkpoint:",
        trainer.state.best_model_checkpoint
    )

    print(
        "Best validation PR-AUC:",
        trainer.state.best_metric
    )

    # -------------------------------------------------------------
    # Validation probabilities
    # -------------------------------------------------------------

    prediction_output = (
        trainer.predict(
            val_dataset
        )
    )

    logits = (
        prediction_output.predictions
    )

    true_labels = (
        prediction_output.label_ids
    )

    probabilities = softmax(
        logits
    )[:, 1]

    # -------------------------------------------------------------
    # Optimise threshold
    # -------------------------------------------------------------

    threshold_results = (
        optimise_threshold(
            true_labels,
            probabilities
        )
    )

    print("\nOptimised threshold")

    for key, value in (
        threshold_results.items()
    ):
        print(
            f"{key}: {value:.6f}"
        )

    # -------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------

    final_model_dir = (
        model_dir
        / "model"
    )

    trainer.save_model(
        str(final_model_dir)
    )

    tokenizer.save_pretrained(
        str(final_model_dir)
    )

    # -------------------------------------------------------------
    # Save threshold
    # -------------------------------------------------------------

    threshold_file = (
        model_dir
        / "threshold.json"
    )

    with open(
        threshold_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            threshold_results,
            file,
            indent=4
        )

    # -------------------------------------------------------------
    # Save validation predictions
    # -------------------------------------------------------------

    validation_output = (
        validation.copy()
    )

    validation_output[
        f"{label}_probability"
    ] = probabilities

    validation_output[
        f"{label}_predicted"
    ] = (
        probabilities
        >= threshold_results[
            "threshold"
        ]
    ).astype(int)

    validation_output.to_csv(
        model_dir
        / "validation_predictions.csv",

        encoding="utf-8-sig",
        index=False
    )

    print(
        f"\nModel saved to: "
        f"{final_model_dir}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    set_seed(
        RANDOM_SEED
    )

    train_df, val_df = (
        load_data()
    )

    for label in LABELS:

        # Reset the seed before each independent
        # classifier training run.
        set_seed(
            RANDOM_SEED
        )

        train_classifier(
            label,
            train_df,
            val_df
        )

    print("\n" + "=" * 70)
    print("ALL EAP MODELS TRAINED")
    print("=" * 70)


if __name__ == "__main__":
    main()
