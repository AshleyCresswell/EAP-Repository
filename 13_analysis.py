"""
Final analysis pipeline for Paper 3.

This script combines the validated EAP and incivility measures to:

1. Apply the frozen APE/incivility classifier to the EAP corpus.
2. Construct EAP × incivility overlap measures.
3. Estimate campaign-level temporal trajectories.
4. Construct seven major campaign-event windows.
5. Estimate event-specific pre/post changes.
6. Test whether EAP and incivility respond differently across events.
7. Test post-event dissipation relative to pre-event baselines.

Model development and held-out evaluation occur in earlier scripts.
No classifiers or decision thresholds are modified here.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import statsmodels.formula.api as smf

from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

EAP_FILE = Path(
    "results/full_corpus_predictions/"
    "EAP_2024_full_BERT_predictions.csv"
)

APE_MODEL_DIR = Path(
    "models/incivility/model"
)

APE_THRESHOLD_FILE = Path(
    "models/incivility/threshold.json"
)

OUTPUT_DIR = Path(
    "results/analysis"
)

FINAL_DATA_FILE = (
    OUTPUT_DIR
    / "EAP_2024_analysis.csv"
)


# ---------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------

ID_COL = "eap_comment_id"
TEXT_COL = "text"

EAP_DIMENSIONS = [
    "aversion_pred",
    "othering_pred",
    "social_distance_pred",
]

EAP_COL = "any_eap_pred"
INCIVILITY_COL = "ape_pred"

MAX_LENGTH = 128
BATCH_SIZE = 64

CAMPAIGN_START = pd.Timestamp(
    "2024-03-08"
)

CAMPAIGN_END = pd.Timestamp(
    "2024-11-05"
)

TRANSITION_DATE = pd.Timestamp(
    "2024-07-21"
)


# ---------------------------------------------------------------------
# Campaign events
# ---------------------------------------------------------------------

EVENTS = [
    {
        "event_key": "trump_conviction",
        "event_name": "Trump criminal conviction",
        "event_type": "Political/legal shock",
        "event_start": "2024-05-30",
        "event_end": "2024-05-30",
    },
    {
        "event_key": "debate_1",
        "event_name": "Biden-Trump presidential debate",
        "event_type": "Electoral confrontation",
        "event_start": "2024-06-27",
        "event_end": "2024-06-27",
    },
    {
        "event_key": "assassination_attempt",
        "event_name": "Trump assassination attempt",
        "event_type": "External political shock",
        "event_start": "2024-07-13",
        "event_end": "2024-07-13",
    },
    {
        "event_key": "rnc",
        "event_name": "Republican National Convention",
        "event_type": "Party mobilisation",
        "event_start": "2024-07-15",
        "event_end": "2024-07-18",
    },
    {
        "event_key": "biden_withdrawal",
        "event_name": "Biden withdrawal / Harris transition",
        "event_type": "Campaign transition shock",
        "event_start": "2024-07-21",
        "event_end": "2024-07-21",
    },
    {
        "event_key": "dnc",
        "event_name": "Democratic National Convention",
        "event_type": "Party mobilisation",
        "event_start": "2024-08-19",
        "event_end": "2024-08-22",
    },
    {
        "event_key": "debate_2",
        "event_name": "Harris-Trump presidential debate",
        "event_type": "Electoral confrontation",
        "event_start": "2024-09-10",
        "event_end": "2024-09-10",
    },
]


# ---------------------------------------------------------------------
# APE inference dataset
# ---------------------------------------------------------------------

class InferenceDataset(Dataset):
    """Tokenised comments for full-corpus APE inference."""

    def __init__(
        self,
        texts,
        tokenizer
    ):
        self.texts = texts
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
        }


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def get_device():
    """Select available inference device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------------------
# Load EAP corpus
# ---------------------------------------------------------------------

def load_eap_corpus():
    """Load the full corpus containing frozen EAP predictions."""

    df = pd.read_csv(
        EAP_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    required = [
        ID_COL,
        TEXT_COL,
        "channel_candidate",
        *EAP_DIMENSIONS,
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

    if df[ID_COL].duplicated().any():
        raise ValueError(
            "Duplicate comment IDs detected."
        )

    # Reconstruct composite EAP measure from the
    # three independently classified dimensions.

    df[EAP_COL] = (
        df[EAP_DIMENSIONS]
        .max(axis=1)
        .astype(int)
    )

    df["n_eap_dimensions_pred"] = (
        df[EAP_DIMENSIONS]
        .sum(axis=1)
        .astype(int)
    )

    # Standardise date.

    date_source = (
        "comment_day"
        if "comment_day" in df.columns
        else "comment_date"
    )

    df["comment_day"] = (
        pd.to_datetime(
            df[date_source],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if df["comment_day"].isna().any():
        raise ValueError(
            "Missing comment dates detected."
        )

    return df


# ---------------------------------------------------------------------
# Apply frozen APE model
# ---------------------------------------------------------------------

def add_incivility_predictions(df):
    """Apply the frozen APE classifier to the full EAP corpus."""

    print("\n" + "=" * 70)
    print("FULL-CORPUS INCIVILITY INFERENCE")
    print("=" * 70)

    with open(
        APE_THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        threshold = float(
            json.load(file)["threshold"]
        )

    print(
        f"Frozen APE threshold: "
        f"{threshold:.6f}"
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    tokenizer = (
        BertTokenizerFast
        .from_pretrained(
            APE_MODEL_DIR
        )
    )

    model = (
        BertForSequenceClassification
        .from_pretrained(
            APE_MODEL_DIR
        )
    )

    model.to(device)
    model.eval()

    dataset = InferenceDataset(
        df[TEXT_COL]
        .astype(str)
        .tolist(),
        tokenizer
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    probabilities = []

    with torch.no_grad():

        for batch in loader:

            outputs = model(
                input_ids=(
                    batch["input_ids"]
                    .to(device)
                ),
                attention_mask=(
                    batch["attention_mask"]
                    .to(device)
                )
            )

            batch_probabilities = (
                torch.softmax(
                    outputs.logits,
                    dim=1
                )[:, 1]
                .cpu()
                .numpy()
            )

            probabilities.extend(
                batch_probabilities
            )

    probabilities = np.asarray(
        probabilities
    )

    df["ape_prob"] = probabilities

    df[INCIVILITY_COL] = (
        probabilities >= threshold
    ).astype(int)

    print(
        f"APE positives: "
        f"{df[INCIVILITY_COL].sum():,}"
    )

    print(
        f"APE prevalence: "
        f"{df[INCIVILITY_COL].mean() * 100:.3f}%"
    )

    return df


# ---------------------------------------------------------------------
# EAP × incivility comparison
# ---------------------------------------------------------------------

def create_overlap_variables(df):
    """Construct EAP × incivility overlap categories."""

    conditions = [
        (
            (df[INCIVILITY_COL] == 0)
            & (df[EAP_COL] == 0)
        ),
        (
            (df[INCIVILITY_COL] == 1)
            & (df[EAP_COL] == 0)
        ),
        (
            (df[INCIVILITY_COL] == 0)
            & (df[EAP_COL] == 1)
        ),
        (
            (df[INCIVILITY_COL] == 1)
            & (df[EAP_COL] == 1)
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

    print("\n" + "=" * 70)
    print("EAP × INCIVILITY OVERLAP")
    print("=" * 70)

    overlap = (
        df["eap_incivility_group"]
        .value_counts()
        .reindex(categories)
    )

    overlap_pct = (
        overlap
        / len(df)
        * 100
    )

    overlap_table = pd.DataFrame(
        {
            "n": overlap,
            "percent": overlap_pct,
        }
    )

    print(overlap_table)

    both = int(
        (
            (df[EAP_COL] == 1)
            & (df[INCIVILITY_COL] == 1)
        ).sum()
    )

    eap_total = int(
        df[EAP_COL].sum()
    )

    incivility_total = int(
        df[INCIVILITY_COL].sum()
    )

    print(
        "\nP(EAP | Incivility): "
        f"{both / incivility_total * 100:.2f}%"
    )

    print(
        "P(Incivility | EAP): "
        f"{both / eap_total * 100:.2f}%"
    )

    overlap_table.to_csv(
        OUTPUT_DIR
        / "eap_incivility_overlap.csv",
        encoding="utf-8-sig"
    )

    return df


# ---------------------------------------------------------------------
# Campaign trajectory
# ---------------------------------------------------------------------

def campaign_trajectory(df):
    """Estimate weekly EAP and incivility campaign trends."""

    print("\n" + "=" * 70)
    print("CAMPAIGN TRAJECTORY")
    print("=" * 70)

    campaign = df[
        df["comment_day"].between(
            CAMPAIGN_START,
            CAMPAIGN_END
        )
    ].copy()

    # Candidate periods used in the final trajectory analysis.
    #
    # Trump: entire campaign
    # Biden: through 21 July
    # Harris: from 21 July

    trajectory = campaign[
        (campaign["channel_candidate"] == "Trump")
        |
        (
            (campaign["channel_candidate"] == "Biden")
            & (
                campaign["comment_day"]
                <= TRANSITION_DATE
            )
        )
        |
        (
            (campaign["channel_candidate"] == "Harris")
            & (
                campaign["comment_day"]
                >= TRANSITION_DATE
            )
        )
    ].copy()

    trajectory["week_start"] = (
        trajectory["comment_day"]
        - pd.to_timedelta(
            trajectory[
                "comment_day"
            ].dt.weekday,
            unit="D"
        )
    )

    weekly = (
        trajectory
        .groupby(
            [
                "channel_candidate",
                "week_start",
            ],
            as_index=False
        )
        .agg(
            comments=(
                ID_COL,
                "size"
            ),
            eap_pct=(
                EAP_COL,
                "mean"
            ),
            incivility_pct=(
                INCIVILITY_COL,
                "mean"
            ),
        )
    )

    weekly["eap_pct"] *= 100
    weekly["incivility_pct"] *= 100

    weekly["campaign_day"] = (
        weekly["week_start"]
        - CAMPAIGN_START
    ).dt.days

    weekly.to_csv(
        OUTPUT_DIR
        / "weekly_campaign_trajectory.csv",
        encoding="utf-8-sig",
        index=False
    )

    eap_model = smf.ols(
        "eap_pct ~ campaign_day "
        "+ C(channel_candidate)",
        data=weekly
    ).fit(
        cov_type="HC3"
    )

    incivility_model = smf.ols(
        "incivility_pct ~ campaign_day "
        "+ C(channel_candidate)",
        data=weekly
    ).fit(
        cov_type="HC3"
    )

    trajectory_results = pd.DataFrame(
        {
            "measure": [
                "EAP",
                "Incivility",
            ],
            "campaign_day_b": [
                eap_model.params[
                    "campaign_day"
                ],
                incivility_model.params[
                    "campaign_day"
                ],
            ],
            "se": [
                eap_model.bse[
                    "campaign_day"
                ],
                incivility_model.bse[
                    "campaign_day"
                ],
            ],
            "p": [
                eap_model.pvalues[
                    "campaign_day"
                ],
                incivility_model.pvalues[
                    "campaign_day"
                ],
            ],
        }
    )

    trajectory_results.to_csv(
        OUTPUT_DIR
        / "campaign_trend_models.csv",
        encoding="utf-8-sig",
        index=False
    )

    print(trajectory_results)

    return weekly


# ---------------------------------------------------------------------
# Event variables
# ---------------------------------------------------------------------

def add_event_variables(df):
    """Create ±7-day variables for the seven campaign events."""

    print("\n" + "=" * 70)
    print("CREATING EVENT VARIABLES")
    print("=" * 70)

    for event in EVENTS:

        key = event["event_key"]

        start = pd.Timestamp(
            event["event_start"]
        )

        end = pd.Timestamp(
            event["event_end"]
        )

        pre_start = (
            start
            - pd.Timedelta(days=7)
        )

        pre_end = (
            start
            - pd.Timedelta(days=1)
        )

        post_start = (
            end
            + pd.Timedelta(days=1)
        )

        post_end = (
            end
            + pd.Timedelta(days=7)
        )

        df[f"{key}_pre7"] = (
            df["comment_day"]
            .between(
                pre_start,
                pre_end
            )
            .astype(int)
        )

        df[f"{key}_during"] = (
            df["comment_day"]
            .between(
                start,
                end
            )
            .astype(int)
        )

        df[f"{key}_post7"] = (
            df["comment_day"]
            .between(
                post_start,
                post_end
            )
            .astype(int)
        )

        # Signed distance from event boundary.

        distance = np.where(
            df["comment_day"] < start,
            (
                df["comment_day"]
                - start
            ).dt.days,
            np.where(
                df["comment_day"] > end,
                (
                    df["comment_day"]
                    - end
                ).dt.days,
                0
            )
        )

        df[f"{key}_distance"] = (
            distance
        )

    return df


# ---------------------------------------------------------------------
# Candidate-day × event dataset
# ---------------------------------------------------------------------

def build_event_dataset(df):
    """
    Construct candidate-day × event observations for
    the seven-day pre/post event analysis.
    """

    rows = []

    for event in EVENTS:

        key = event["event_key"]

        temp = df[
            (
                df[f"{key}_pre7"] == 1
            )
            |
            (
                df[f"{key}_post7"] == 1
            )
        ].copy()

        temp["post_event"] = (
            temp[f"{key}_post7"]
            .astype(int)
        )

        temp["event_distance"] = (
            temp[f"{key}_distance"]
        )

        daily = (
            temp
            .groupby(
                [
                    "channel_candidate",
                    "comment_day",
                    "post_event",
                    "event_distance",
                ],
                as_index=False
            )
            .agg(
                comments=(
                    ID_COL,
                    "size"
                ),
                eap_pct=(
                    EAP_COL,
                    "mean"
                ),
                incivility_pct=(
                    INCIVILITY_COL,
                    "mean"
                ),
            )
        )

        daily["eap_pct"] *= 100
        daily["incivility_pct"] *= 100

        daily["event_key"] = key
        daily["event_name"] = (
            event["event_name"]
        )

        daily["event_type"] = (
            event["event_type"]
        )

        rows.append(daily)

    event_daily = pd.concat(
        rows,
        ignore_index=True
    )

    event_daily.to_csv(
        OUTPUT_DIR
        / "event_candidate_day_data.csv",
        encoding="utf-8-sig",
        index=False
    )

    return event_daily


# ---------------------------------------------------------------------
# RQ2: event heterogeneity
# ---------------------------------------------------------------------

def event_heterogeneity(event_daily):
    """Estimate event-specific pre/post changes."""

    print("\n" + "=" * 70)
    print("RQ2 — EVENT HETEROGENEITY")
    print("=" * 70)

    reference = "trump_conviction"

    eap_formula = (
        "eap_pct ~ "
        "post_event * "
        f"C(event_key, Treatment(reference='{reference}')) "
        "+ C(channel_candidate)"
    )

    incivility_formula = (
        "incivility_pct ~ "
        "post_event * "
        f"C(event_key, Treatment(reference='{reference}')) "
        "+ C(channel_candidate)"
    )

    eap_model = smf.ols(
        eap_formula,
        data=event_daily
    ).fit(
        cov_type="HC3"
    )

    incivility_model = smf.ols(
        incivility_formula,
        data=event_daily
    ).fit(
        cov_type="HC3"
    )

    # Joint Wald test:
    # post-event change does not vary across events.

    def interaction_test(model):

        terms = [
            term
            for term
            in model.params.index
            if (
                "post_event:" in term
                and "event_key" in term
            )
        ]

        restriction = np.zeros(
            (
                len(terms),
                len(model.params)
            )
        )

        for row, term in enumerate(terms):

            restriction[
                row,
                model.params.index.get_loc(
                    term
                )
            ] = 1

        return model.wald_test(
            restriction,
            scalar=True
        )

    eap_wald = interaction_test(
        eap_model
    )

    incivility_wald = interaction_test(
        incivility_model
    )

    results = pd.DataFrame(
        {
            "measure": [
                "EAP",
                "Incivility",
            ],
            "wald": [
                float(
                    eap_wald.statistic
                ),
                float(
                    incivility_wald.statistic
                ),
            ],
            "p": [
                float(
                    eap_wald.pvalue
                ),
                float(
                    incivility_wald.pvalue
                ),
            ],
        }
    )

    results.to_csv(
        OUTPUT_DIR
        / "event_heterogeneity_tests.csv",
        encoding="utf-8-sig",
        index=False
    )

    print(results)

    return eap_model, incivility_model


# ---------------------------------------------------------------------
# Direct EAP vs incivility comparison
# ---------------------------------------------------------------------

def compare_measures(event_daily):
    """
    Test whether the difference between EAP and incivility
    event responses varies across campaign events.
    """

    print("\n" + "=" * 70)
    print("EAP vs INCIVILITY EVENT RESPONSES")
    print("=" * 70)

    long_df = event_daily.melt(
        id_vars=[
            "channel_candidate",
            "comment_day",
            "event_key",
            "event_name",
            "post_event",
        ],
        value_vars=[
            "eap_pct",
            "incivility_pct",
        ],
        var_name="measure",
        value_name="outcome_pct"
    )

    long_df["measure"] = (
        long_df["measure"]
        .replace(
            {
                "eap_pct": "EAP",
                "incivility_pct":
                    "Incivility",
            }
        )
    )

    # EAP and incivility observations from the same
    # candidate-day × event are paired.

    long_df["pair_id"] = (
        long_df[
            [
                "channel_candidate",
                "comment_day",
                "event_key",
            ]
        ]
        .astype(str)
        .agg("_".join, axis=1)
    )

    reference = "trump_conviction"

    formula = (
        "outcome_pct ~ "
        "post_event * "
        f"C(event_key, Treatment(reference='{reference}')) "
        "* C(measure, Treatment(reference='EAP')) "
        "+ C(channel_candidate)"
    )

    model = smf.ols(
        formula,
        data=long_df
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                long_df["pair_id"]
        }
    )

    three_way_terms = [
        term
        for term
        in model.params.index
        if (
            "post_event:" in term
            and "event_key" in term
            and "measure" in term
        )
    ]

    restriction = np.zeros(
        (
            len(three_way_terms),
            len(model.params)
        )
    )

    for row, term in enumerate(
        three_way_terms
    ):

        restriction[
            row,
            model.params.index.get_loc(
                term
            )
        ] = 1

    wald = model.wald_test(
        restriction,
        scalar=True
    )

    results = pd.DataFrame(
        {
            "test": [
                "EAP × Incivility "
                "event-response heterogeneity"
            ],
            "wald": [
                float(
                    wald.statistic
                )
            ],
            "p": [
                float(
                    wald.pvalue
                )
            ],
            "clusters": [
                long_df[
                    "pair_id"
                ].nunique()
            ],
        }
    )

    results.to_csv(
        OUTPUT_DIR
        / "eap_incivility_event_comparison.csv",
        encoding="utf-8-sig",
        index=False
    )

    print(results)

    return model


# ---------------------------------------------------------------------
# RQ2b: post-event dissipation
# ---------------------------------------------------------------------

def post_event_dissipation(df):
    """
    Test whether event-related departures from the pre-event
    baseline diminish across post-event days +1 to +7.
    """

    print("\n" + "=" * 70)
    print("RQ2b — POST-EVENT DISSIPATION")
    print("=" * 70)

    rows = []

    for event in EVENTS:

        start = pd.Timestamp(
            event["event_start"]
        )

        end = pd.Timestamp(
            event["event_end"]
        )

        pre = df[
            df["comment_day"].between(
                start - pd.Timedelta(days=7),
                start - pd.Timedelta(days=1)
            )
        ]

        pre_baseline = (
            pre
            .groupby(
                "channel_candidate",
                as_index=False
            )
            .agg(
                pre_eap=(
                    EAP_COL,
                    "mean"
                ),
                pre_incivility=(
                    INCIVILITY_COL,
                    "mean"
                ),
            )
        )

        pre_baseline[
            [
                "pre_eap",
                "pre_incivility",
            ]
        ] *= 100

        post = df[
            df["comment_day"].between(
                end + pd.Timedelta(days=1),
                end + pd.Timedelta(days=7)
            )
        ].copy()

        post["event_distance"] = (
            post["comment_day"]
            - end
        ).dt.days

        post_daily = (
            post
            .groupby(
                [
                    "channel_candidate",
                    "comment_day",
                    "event_distance",
                ],
                as_index=False
            )
            .agg(
                eap_pct=(
                    EAP_COL,
                    "mean"
                ),
                incivility_pct=(
                    INCIVILITY_COL,
                    "mean"
                ),
            )
        )

        post_daily[
            [
                "eap_pct",
                "incivility_pct",
            ]
        ] *= 100

        post_daily = post_daily.merge(
            pre_baseline,
            on="channel_candidate",
            how="inner",
            validate="many_to_one"
        )

        post_daily[
            "eap_abs_deviation"
        ] = (
            post_daily["eap_pct"]
            - post_daily["pre_eap"]
        ).abs()

        post_daily[
            "incivility_abs_deviation"
        ] = (
            post_daily["incivility_pct"]
            - post_daily[
                "pre_incivility"
            ]
        ).abs()

        post_daily["event_key"] = (
            event["event_key"]
        )

        rows.append(
            post_daily
        )

    decay = pd.concat(
        rows,
        ignore_index=True
    )

    reference = "trump_conviction"

    eap_model = smf.ols(
        (
            "eap_abs_deviation ~ "
            "event_distance + "
            f"C(event_key, Treatment(reference='{reference}')) "
            "+ C(channel_candidate)"
        ),
        data=decay
    ).fit(
        cov_type="HC3"
    )

    incivility_model = smf.ols(
        (
            "incivility_abs_deviation ~ "
            "event_distance + "
            f"C(event_key, Treatment(reference='{reference}')) "
            "+ C(channel_candidate)"
        ),
        data=decay
    ).fit(
        cov_type="HC3"
    )

    results = pd.DataFrame(
        {
            "measure": [
                "EAP",
                "Incivility",
            ],
            "distance_b": [
                eap_model.params[
                    "event_distance"
                ],
                incivility_model.params[
                    "event_distance"
                ],
            ],
            "se": [
                eap_model.bse[
                    "event_distance"
                ],
                incivility_model.bse[
                    "event_distance"
                ],
            ],
            "p": [
                eap_model.pvalues[
                    "event_distance"
                ],
                incivility_model.pvalues[
                    "event_distance"
                ],
            ],
        }
    )

    decay.to_csv(
        OUTPUT_DIR
        / "post_event_dissipation_data.csv",
        encoding="utf-8-sig",
        index=False
    )

    results.to_csv(
        OUTPUT_DIR
        / "post_event_dissipation_models.csv",
        encoding="utf-8-sig",
        index=False
    )

    print(results)

    return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # Full-corpus measures
    # -------------------------------------------------------------

    df = load_eap_corpus()

    print(
        f"Full corpus: {len(df):,}"
    )

    df = add_incivility_predictions(
        df
    )

    df = create_overlap_variables(
        df
    )

    # -------------------------------------------------------------
    # Campaign trajectory
    # -------------------------------------------------------------

    campaign_trajectory(
        df
    )

    # -------------------------------------------------------------
    # Campaign events
    # -------------------------------------------------------------

    df = add_event_variables(
        df
    )

    event_daily = (
        build_event_dataset(
            df
        )
    )

    event_heterogeneity(
        event_daily
    )

    compare_measures(
        event_daily
    )

    post_event_dissipation(
        df
    )

    # -------------------------------------------------------------
    # Save final analysis dataset
    # -------------------------------------------------------------

    df.to_csv(
        FINAL_DATA_FILE,
        encoding="utf-8-sig",
        index=False
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Final analysis dataset:\n"
        f"{FINAL_DATA_FILE}"
    )


if __name__ == "__main__":
    main()
