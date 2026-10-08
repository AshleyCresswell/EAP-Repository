"""
Train the final BERT classifiers used in Paper 3.

Four independent binary classifiers are trained:

EAP dimensions
--------------
1. Aversion
2. Othering
3. Social Distance

Benchmark measure
-----------------
4. Incivility (APE)

Training uses only the training and validation datasets.
The held-out test sets are not accessed in this script.

Validation data are used for model selection and/or decision-threshold
selection. Final held-out evaluation is performed separately in
05_evaluate_models.py.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data/model")
MODEL_DIR = Path("models")

EAP_TRAIN_FILE = (
    DATA_DIR
    / "EAP_2024_BERT_train_v2.csv"
)

EAP_VALIDATION_FILE = (
    DATA_DIR
    / "EAP_2024_BERT_validation_v2.csv"
)

APE_TRAIN_FILE = (
    DATA_DIR
    / "APE_V1_train.csv"
)

APE_VALIDATION_FILE = (
    DATA_DIR
    / "APE_V1_validation.csv"
)

BASE_MODEL = "bert-base-uncased"

TEXT_COL = "text"

EAP_LABELS = [
    "aversion",
    "othering",
    "social_distance",
]

SEED = 42
MAX_LENGTH = 128

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 3

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_random_seeds():
    """Set random seeds used during model training."""

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_data(path, label):
    """Load one modelling dataset and validate required columns."""

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )

    required = [
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

    df = df[
        df[TEXT_COL].notna()
        & df[label].notna()
    ].copy()

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
# Tokenisation
# ---------------------------------------------------------------------

def prepare_dataset(
    df,
    label,
    tokenizer
):
    """Tokenise comments for BERT classification."""

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
# Class-weighted trainer
# ---------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """
    Trainer using class-weighted cross-entropy.

    Used for EAP dimensions where positive examples are
    substantially less common than negative examples.
    """

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

        self.class_weights = (
            class_weights
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.pop(
            "labels"
        )

        outputs = model(
            **inputs
        )

        logits = outputs.logits

        weights = (
            self.class_weights
            .to(logits.device)
        )

        loss_function = (
            torch.nn.CrossEntropyLoss(
                weight=weights
            )
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
# Validation metric
# ---------------------------------------------------------------------

def compute_pr_auc(eval_pred):
    """Calculate positive-class PR-AUC on validation data."""

    logits, labels = eval_pred

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1
    )[:, 1].numpy()

    return {
        "pr_auc":
            average_precision_score(
                labels,
                probabilities
            )
    }


# ---------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------

def select_f1_threshold(
    labels,
    probabilities
):
    """
    Select the validation threshold that maximises
    positive-class F1.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            labels,
            probabilities
        )
    )

    # precision and recall contain one more element
    # than the threshold array.

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = (
        precision + recall
    )

    f1_scores = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(
            denominator
        ),
        where=denominator != 0
    )

    best_index = int(
        np.argmax(f1_scores)
    )

    return {
        "threshold":
            float(
                thresholds[
                    best_index
                ]
            ),

        "validation_f1":
            float(
                f1_scores[
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
    }


# ---------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------

def calculate_class_weights(labels):
    """Calculate balanced binary class weights."""

    classes = np.array(
        [0, 1]
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    return torch.tensor(
        weights,
        dtype=torch.float
    )


# ---------------------------------------------------------------------
# Train one EAP classifier
# ---------------------------------------------------------------------

def train_eap_model(label):
    """Train one EAP dimension classifier."""

    print("\n" + "=" * 70)
    print(f"EAP MODEL: {label.upper()}")
    print("=" * 70)

    train_df = load_data(
        EAP_TRAIN_FILE,
        label
    )

    validation_df = load_data(
        EAP_VALIDATION_FILE,
        label
    )

    print(
        f"Training rows:   {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(validation_df):,}"
    )

    print(
        f"Training positives: "
        f"{train_df[label].sum():,}"
    )

    output_dir = (
        MODEL_DIR
        / "eap"
        / label
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    final_model_dir = (
        output_dir
        / "model"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL
        )
    )

    train_dataset = prepare_dataset(
        train_df,
        label,
        tokenizer
    )

    validation_dataset = (
        prepare_dataset(
            validation_df,
            label,
            tokenizer
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BASE_MODEL,
            num_labels=2
        )
    )

    class_weights = (
        calculate_class_weights(
            train_df[label].to_numpy()
        )
    )

    print(
        "Class weights: "
        f"{class_weights.tolist()}"
    )

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
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        save_total_limit=1,
        seed=SEED,
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_pr_auc,
        class_weights=class_weights
    )

    trainer.train()

    # -------------------------------------------------------------
    # Validation probabilities
    # -------------------------------------------------------------

    prediction_output = (
        trainer.predict(
            validation_dataset
        )
    )

    probabilities = torch.softmax(
        torch.tensor(
            prediction_output.predictions
        ),
        dim=1
    )[:, 1].numpy()

    threshold_results = (
        select_f1_threshold(
            validation_df[
                label
            ].to_numpy(),
            probabilities
        )
    )

    print(
        "\nSelected threshold: "
        f"{threshold_results['threshold']:.6f}"
    )

    print(
        "Validation F1: "
        f"{threshold_results['validation_f1']:.3f}"
    )

    # -------------------------------------------------------------
    # Save frozen model
    # -------------------------------------------------------------

    trainer.save_model(
        final_model_dir
    )

    tokenizer.save_pretrained(
        final_model_dir
    )

    with open(
        output_dir
        / "threshold.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            threshold_results,
            file,
            indent=4
        )

    validation_output = pd.DataFrame(
        {
            "probability":
                probabilities,

            "label":
                validation_df[
                    label
                ].to_numpy(),
        }
    )

    if "eap_comment_id" in (
        validation_df.columns
    ):

        validation_output.insert(
            0,
            "eap_comment_id",
            validation_df[
                "eap_comment_id"
            ].to_numpy()
        )

    validation_output.to_csv(
        output_dir
        / "validation_predictions.csv",
        encoding="utf-8-sig",
        index=False
    )


# ---------------------------------------------------------------------
# Train incivility classifier
# ---------------------------------------------------------------------

def train_incivility_model():
    """
    Train the APE/incivility classifier.

    Unlike the EAP dimensions, the APE labels are comparatively
    balanced, so class-weighted loss is not applied.
    """

    label = "ape"

    print("\n" + "=" * 70)
    print("INCIVILITY / APE MODEL")
    print("=" * 70)

    train_df = load_data(
        APE_TRAIN_FILE,
        label
    )

    validation_df = load_data(
        APE_VALIDATION_FILE,
        label
    )

    print(
        f"Training rows:   {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(validation_df):,}"
    )

    print(
        f"Training positives: "
        f"{train_df[label].sum():,}"
    )

    output_dir = (
        MODEL_DIR
        / "incivility"
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    final_model_dir = (
        output_dir
        / "model"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL
        )
    )

    train_dataset = prepare_dataset(
        train_df,
        label,
        tokenizer
    )

    validation_dataset = (
        prepare_dataset(
            validation_df,
            label,
            tokenizer
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BASE_MODEL,
            num_labels=2
        )
    )

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
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        save_total_limit=1,
        seed=SEED,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_pr_auc
    )

    trainer.train()

    # -------------------------------------------------------------
    # Validation threshold
    # -------------------------------------------------------------

    prediction_output = (
        trainer.predict(
            validation_dataset
        )
    )

    probabilities = torch.softmax(
        torch.tensor(
            prediction_output.predictions
        ),
        dim=1
    )[:, 1].numpy()

    threshold_results = (
        select_f1_threshold(
            validation_df[
                label
            ].to_numpy(),
            probabilities
        )
    )

    print(
        "\nSelected threshold: "
        f"{threshold_results['threshold']:.6f}"
    )

    print(
        "Validation F1: "
        f"{threshold_results['validation_f1']:.3f}"
    )

    # -------------------------------------------------------------
    # Save frozen model
    # -------------------------------------------------------------

    trainer.save_model(
        final_model_dir
    )

    tokenizer.save_pretrained(
        final_model_dir
    )

    with open(
        output_dir
        / "threshold.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            threshold_results,
            file,
            indent=4
        )

    validation_output = pd.DataFrame(
        {
            "probability":
                probabilities,

            "label":
                validation_df[
                    label
                ].to_numpy(),
        }
    )

    if "eap_comment_id" in (
        validation_df.columns
    ):

        validation_output.insert(
            0,
            "eap_comment_id",
            validation_df[
                "eap_comment_id"
            ].to_numpy()
        )

    validation_output.to_csv(
        output_dir
        / "validation_predictions.csv",
        encoding="utf-8-sig",
        index=False
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Train and freeze all four classifiers."""

    set_random_seeds()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("BERT MODEL TRAINING")
    print("=" * 70)

    # -------------------------------------------------------------
    # EAP classifiers
    # -------------------------------------------------------------

    for label in EAP_LABELS:

        train_eap_model(
            label
        )

    # -------------------------------------------------------------
    # Incivility benchmark
    # -------------------------------------------------------------

    train_incivility_model()

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(
        "\nFour frozen classifiers created:"
    )

    print(
        "  1. Aversion\n"
        "  2. Othering\n"
        "  3. Social Distance\n"
        "  4. Incivility / APE"
    )


if __name__ == "__main__":
    main()
