"""
Final temporal analysis for Paper 3.

Using the full corpus predictions produced by 06_predict_corpus.py,
this script examines:

1. Campaign trajectories in EAP and incivility.
2. Changes around seven major campaign events.
3. Whether event responses differ between EAP and incivility.
4. Whether event-related changes dissipate over time.

Analyses are conducted on aggregated candidate-week or
candidate-day observations rather than individual comments.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path(
    "results/full_corpus_predictions/"
    "EAP_2024_full_predictions.csv"
)

OUTPUT_DIR = Path(
    "results/analysis"
)

EAP_COL = "any_eap"
INCIVILITY_COL = "incivility"

CAMPAIGN_START = pd.Timestamp(
    "2024-03-08"
)

CAMPAIGN_END = pd.Timestamp(
    "2024-11-05"
)

TRANSITION_DATE = pd.Timestamp(
    "2024-07-21"
)

EVENT_WINDOW = 7

CANDIDATE_REFERENCE = "Trump"
EVENT_REFERENCE = "trump_conviction"


# ---------------------------------------------------------------------
# Major campaign events
# ---------------------------------------------------------------------

EVENTS = [
    {
        "event_key": "trump_conviction",
        "event_name": "Trump criminal conviction",
        "event_start": "2024-05-30",
        "event_end": "2024-05-30",
    },
    {
        "event_key": "debate_1",
        "event_name": "Biden-Trump presidential debate",
        "event_start": "2024-06-27",
        "event_end": "2024-06-27",
    },
    {
        "event_key": "assassination_attempt",
        "event_name": "Trump assassination attempt",
        "event_start": "2024-07-13",
        "event_end": "2024-07-13",
    },
    {
        "event_key": "rnc",
        "event_name": "Republican National Convention",
        "event_start": "2024-07-15",
        "event_end": "2024-07-18",
    },
    {
        "event_key": "biden_withdrawal",
        "event_name": "Biden withdrawal / Harris transition",
        "event_start": "2024-07-21",
        "event_end": "2024-07-21",
    },
    {
        "event_key": "dnc",
        "event_name": "Democratic National Convention",
        "event_start": "2024-08-19",
        "event_end": "2024-08-22",
    },
    {
        "event_key": "debate_2",
        "event_name": "Harris-Trump presidential debate",
        "event_start": "2024-09-10",
        "event_end": "2024-09-10",
    },
]


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_data():
    """Load the final predicted corpus."""

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    required = [
        "eap_comment_id",
        "comment_date",
        "channel_candidate",
        EAP_COL,
        INCIVILITY_COL,
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

    df["comment_day"] = (
        pd.to_datetime(
            df["comment_date"],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
        .dt.floor("D")
    )

    df[EAP_COL] = pd.to_numeric(
        df[EAP_COL],
        errors="raise"
    )

    df[INCIVILITY_COL] = pd.to_numeric(
        df[INCIVILITY_COL],
        errors="raise"
    )

    df = df.dropna(
        subset=[
            "comment_day",
            "channel_candidate",
        ]
    ).copy()

    df = df[
        df["comment_day"].between(
            CAMPAIGN_START,
            CAMPAIGN_END
        )
    ].copy()

    return df


# =====================================================================
# 1. CAMPAIGN TRAJECTORY
# =====================================================================

def build_weekly_trajectory(df):
    """
    Construct candidate-week observations.

    Candidate periods reflect the Democratic candidate transition:
    Trump is retained throughout, Biden through 21 July, and Harris
    from 21 July onward.
    """

    trajectory = df[
        (df["channel_candidate"] == "Trump")
        |
        (
            (df["channel_candidate"] == "Biden")
            & (df["comment_day"] <= TRANSITION_DATE)
        )
        |
        (
            (df["channel_candidate"] == "Harris")
            & (df["comment_day"] >= TRANSITION_DATE)
        )
    ].copy()

    trajectory["week_start"] = (
        trajectory["comment_day"]
        - pd.to_timedelta(
            trajectory["comment_day"].dt.weekday,
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
            n_comments=(
                "eap_comment_id",
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

    weekly["channel_candidate"] = (
        weekly["channel_candidate"]
        .astype("object")
    )

    return weekly


def analyse_campaign_trajectory(df):
    """Estimate overall campaign trends using HAC standard errors."""

    print("\n" + "=" * 70)
    print("CAMPAIGN TRAJECTORY")
    print("=" * 70)

    weekly = build_weekly_trajectory(
        df
    )

    results = []

    for measure, outcome in [
        ("EAP", "eap_pct"),
        ("Incivility", "incivility_pct"),
    ]:

        model = smf.ols(
            (
                f"{outcome} ~ campaign_day "
                f"+ C(channel_candidate, "
                f"Treatment(reference='{CANDIDATE_REFERENCE}'))"
            ),
            data=weekly
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": 2
            }
        )

        ci = model.conf_int().loc[
            "campaign_day"
        ]

        results.append(
            {
                "measure": measure,
                "daily_slope":
                    model.params["campaign_day"],
                "se":
                    model.bse["campaign_day"],
                "ci_lower":
                    ci.iloc[0],
                "ci_upper":
                    ci.iloc[1],
                "p":
                    model.pvalues["campaign_day"],
                "change_per_30_days":
                    model.params["campaign_day"] * 30,
                "n_candidate_weeks":
                    int(model.nobs),
            }
        )

    results = pd.DataFrame(
        results
    )

    weekly.to_csv(
        OUTPUT_DIR
        / "weekly_campaign_trajectory.csv",
        encoding="utf-8-sig",
        index=False
    )

    results.to_csv(
        OUTPUT_DIR
        / "campaign_trajectory_results.csv",
        encoding="utf-8-sig",
        index=False
    )

    print(results.to_string(index=False))

    return weekly


# =====================================================================
# 2. EVENT ANALYSIS
# =====================================================================

def build_event_data(df):
    """
    Construct candidate-day × event observations.

    For each event:
    - pre-event = days -7 to -1
    - event period is excluded
    - post-event = days +1 to +7
    """

    event_frames = []

    for event in EVENTS:

        start = pd.Timestamp(
            event["event_start"]
        )

        end = pd.Timestamp(
            event["event_end"]
        )

        # ---------------------------------------------------------
        # Pre-event
        # ---------------------------------------------------------

        pre = df[
            df["comment_day"].between(
                start - pd.Timedelta(
                    days=EVENT_WINDOW
                ),
                start - pd.Timedelta(days=1)
            )
        ].copy()

        pre["post_event"] = 0

        pre["event_distance"] = (
            pre["comment_day"]
            - start
        ).dt.days

        # ---------------------------------------------------------
        # Post-event
        # ---------------------------------------------------------

        post = df[
            df["comment_day"].between(
                end + pd.Timedelta(days=1),
                end + pd.Timedelta(
                    days=EVENT_WINDOW
                )
            )
        ].copy()

        post["post_event"] = 1

        post["event_distance"] = (
            post["comment_day"]
            - end
        ).dt.days

        # Event-period comments are deliberately excluded.

        temp = pd.concat(
            [
                pre,
                post,
            ],
            ignore_index=True
        )

        temp["event_key"] = (
            event["event_key"]
        )

        temp["event_name"] = (
            event["event_name"]
        )

        daily = (
            temp
            .groupby(
                [
                    "event_key",
                    "event_name",
                    "channel_candidate",
                    "comment_day",
                    "post_event",
                    "event_distance",
                ],
                as_index=False
            )
            .agg(
                n_comments=(
                    "eap_comment_id",
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

        event_frames.append(
            daily
        )

    event_daily = pd.concat(
        event_frames,
        ignore_index=True
    )

    event_daily[
        "channel_candidate"
    ] = (
        event_daily[
            "channel_candidate"
        ].astype("object")
    )

    event_daily["event_key"] = (
        event_daily["event_key"]
        .astype("object")
    )

    event_daily.to_csv(
        OUTPUT_DIR
        / "event_candidate_day_data.csv",
        encoding="utf-8-sig",
        index=False
    )

    return event_daily


# ---------------------------------------------------------------------
# Event-specific pre/post estimates
# ---------------------------------------------------------------------

def extract_event_effects(
    model,
    measure
):
    """Recover the estimated pre/post effect for each event."""

    effects = []

    parameters = list(
        model.params.index
    )

    base_effect = (
        model.params["post_event"]
    )

    covariance = model.cov_params()

    for event in EVENTS:

        key = event["event_key"]

        if key == EVENT_REFERENCE:

            estimate = base_effect

            variance = covariance.loc[
                "post_event",
                "post_event"
            ]

        else:

            interaction = next(
                (
                    term
                    for term in parameters
                    if (
                        "post_event:" in term
                        and f"[T.{key}]" in term
                    )
                ),
                None
            )

            if interaction is None:
                raise ValueError(
                    f"Could not locate interaction "
                    f"for {key}."
                )

            estimate = (
                base_effect
                + model.params[interaction]
            )

            variance = (
                covariance.loc[
                    "post_event",
                    "post_event"
                ]
                + covariance.loc[
                    interaction,
                    interaction
                ]
                + 2
                * covariance.loc[
                    "post_event",
                    interaction
                ]
            )

        se = np.sqrt(
            variance
        )

        z = estimate / se

        # Normal approximation for HC3 coefficient tests.
        p = np.math.erfc(
            abs(z) / np.sqrt(2)
        )

        effects.append(
            {
                "measure": measure,
                "event_key": key,
                "event_name":
                    event["event_name"],
                "post_change_pp":
                    estimate,
                "se": se,
                "p": p,
            }
        )

    return effects


# ---------------------------------------------------------------------
# Joint heterogeneity test
# ---------------------------------------------------------------------

def event_heterogeneity_test(
    model,
    measure
):
    """
    Test whether the pre/post change differs across events.

    H0: all post_event × event interaction terms equal zero.
    """

    interaction_terms = [
        term
        for term in model.params.index
        if (
            "post_event:" in term
            and "event_key" in term
        )
    ]

    restriction = np.zeros(
        (
            len(interaction_terms),
            len(model.params)
        )
    )

    for row, term in enumerate(
        interaction_terms
    ):

        restriction[
            row,
            model.params.index.get_loc(
                term
            )
        ] = 1

    test = model.wald_test(
        restriction,
        scalar=True
    )

    return {
        "measure": measure,
        "wald":
            float(test.statistic),
        "df":
            len(interaction_terms),
        "p":
            float(test.pvalue),
    }


def analyse_events(event_daily):
    """Estimate event-contingent changes in EAP and incivility."""

    print("\n" + "=" * 70)
    print("MAJOR CAMPAIGN EVENTS")
    print("=" * 70)

    models = {}
    effects = []
    heterogeneity = []

    for measure, outcome in [
        ("EAP", "eap_pct"),
        ("Incivility", "incivility_pct"),
    ]:

        formula = (
            f"{outcome} ~ "
            f"post_event * "
            f"C(event_key, "
            f"Treatment(reference='{EVENT_REFERENCE}')) "
            f"+ C(channel_candidate, "
            f"Treatment(reference='{CANDIDATE_REFERENCE}'))"
        )

        model = smf.ols(
            formula,
            data=event_daily
        ).fit(
            cov_type="HC3"
        )

        models[measure] = model

        effects.extend(
            extract_event_effects(
                model,
                measure
            )
        )

        heterogeneity.append(
            event_heterogeneity_test(
                model,
                measure
            )
        )

    effects = pd.DataFrame(
        effects
    )

    heterogeneity = pd.DataFrame(
        heterogeneity
    )

    effects.to_csv(
        OUTPUT_DIR
        / "event_specific_changes.csv",
        encoding="utf-8-sig",
        index=False
    )

    heterogeneity.to_csv(
        OUTPUT_DIR
        / "event_heterogeneity_tests.csv",
        encoding="utf-8-sig",
        index=False
    )

    print("\nEvent-specific changes:")
    print(
        effects.to_string(
            index=False
        )
    )

    print("\nJoint heterogeneity tests:")
    print(
        heterogeneity.to_string(
            index=False
        )
    )

    return models


# =====================================================================
# 3. DIRECT EAP vs INCIVILITY COMPARISON
# =====================================================================

def compare_eap_incivility(
    event_daily
):
    """
    Test whether EAP and incivility respond differently
    across campaign events.

    The three-way interaction tests:
    post-event × event × measure.
    """

    print("\n" + "=" * 70)
    print("EAP vs INCIVILITY EVENT RESPONSES")
    print("=" * 70)

    long_df = event_daily.melt(
        id_vars=[
            "event_key",
            "event_name",
            "channel_candidate",
            "comment_day",
            "post_event",
            "event_distance",
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
        .astype("object")
    )

    # EAP and incivility from the same candidate-day/event
    # are paired observations.

    long_df["cluster_id"] = (
        long_df[
            [
                "event_key",
                "channel_candidate",
                "comment_day",
            ]
        ]
        .astype(str)
        .agg("_".join, axis=1)
    )

    formula = (
        "outcome_pct ~ "
        "post_event * "
        f"C(event_key, Treatment(reference='{EVENT_REFERENCE}')) "
        "* C(measure, Treatment(reference='EAP')) "
        f"+ C(channel_candidate, "
        f"Treatment(reference='{CANDIDATE_REFERENCE}'))"
    )

    model = smf.ols(
        formula,
        data=long_df
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                long_df["cluster_id"]
        }
    )

    three_way_terms = [
        term
        for term in model.params.index
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

    test = model.wald_test(
        restriction,
        scalar=True
    )

    results = pd.DataFrame(
        [
            {
                "test":
                    "Event × post-event × measure",
                "wald":
                    float(test.statistic),
                "df":
                    len(three_way_terms),
                "p":
                    float(test.pvalue),
                "clusters":
                    long_df[
                        "cluster_id"
                    ].nunique(),
            }
        ]
    )

    results.to_csv(
        OUTPUT_DIR
        / "eap_incivility_comparison.csv",
        encoding="utf-8-sig",
        index=False
    )

    print(results.to_string(index=False))

    return model


# =====================================================================
# 4. POST-EVENT DISSIPATION
# =====================================================================

def build_dissipation_data(df):
    """
    Calculate absolute deviation from each candidate's
    pre-event baseline for post-event days +1 to +7.
    """

    frames = []

    for event in EVENTS:

        start = pd.Timestamp(
            event["event_start"]
        )

        end = pd.Timestamp(
            event["event_end"]
        )

        # ---------------------------------------------------------
        # Candidate-specific pre-event baseline
        # ---------------------------------------------------------

        pre = df[
            df["comment_day"].between(
                start - pd.Timedelta(
                    days=EVENT_WINDOW
                ),
                start - pd.Timedelta(days=1)
            )
        ].copy()

        baseline = (
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

        baseline[
            [
                "pre_eap",
                "pre_incivility",
            ]
        ] *= 100

        # ---------------------------------------------------------
        # Post-event candidate-days
        # ---------------------------------------------------------

        post = df[
            df["comment_day"].between(
                end + pd.Timedelta(days=1),
                end + pd.Timedelta(
                    days=EVENT_WINDOW
                )
            )
        ].copy()

        post["event_distance"] = (
            post["comment_day"]
            - end
        ).dt.days

        daily = (
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

        daily[
            [
                "eap_pct",
                "incivility_pct",
            ]
        ] *= 100

        daily = daily.merge(
            baseline,
            on="channel_candidate",
            how="inner",
            validate="many_to_one"
        )

        daily[
            "eap_abs_deviation"
        ] = (
            daily["eap_pct"]
            - daily["pre_eap"]
        ).abs()

        daily[
            "incivility_abs_deviation"
        ] = (
            daily["incivility_pct"]
            - daily["pre_incivility"]
        ).abs()

        daily["event_key"] = (
            event["event_key"]
        )

        daily["event_name"] = (
            event["event_name"]
        )

        frames.append(
            daily
        )

    return pd.concat(
        frames,
        ignore_index=True
    )


def analyse_dissipation(df):
    """
    Test whether deviations from pre-event levels diminish
    as temporal distance from the event increases.
    """

    print("\n" + "=" * 70)
    print("POST-EVENT DISSIPATION")
    print("=" * 70)

    decay = build_dissipation_data(
        df
    )

    results = []

    for measure, outcome in [
        (
            "EAP",
            "eap_abs_deviation"
        ),
        (
            "Incivility",
            "incivility_abs_deviation"
        ),
    ]:

        formula = (
            f"{outcome} ~ event_distance "
            f"+ C(channel_candidate, "
            f"Treatment(reference='{CANDIDATE_REFERENCE}')) "
            f"+ C(event_key, "
            f"Treatment(reference='{EVENT_REFERENCE}'))"
        )

        model = smf.ols(
            formula,
            data=decay
        ).fit(
            cov_type="HC3"
        )

        ci = model.conf_int().loc[
            "event_distance"
        ]

        results.append(
            {
                "measure": measure,
                "distance_b":
                    model.params[
                        "event_distance"
                    ],
                "se":
                    model.bse[
                        "event_distance"
                    ],
                "ci_lower":
                    ci.iloc[0],
                "ci_upper":
                    ci.iloc[1],
                "p":
                    model.pvalues[
                        "event_distance"
                    ],
                "n":
                    int(model.nobs),
            }
        )

    results = pd.DataFrame(
        results
    )

    decay.to_csv(
        OUTPUT_DIR
        / "post_event_dissipation_data.csv",
        encoding="utf-8-sig",
        index=False
    )

    results.to_csv(
        OUTPUT_DIR
        / "post_event_dissipation_results.csv",
        encoding="utf-8-sig",
        index=False
    )

    print(results.to_string(index=False))

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    """Run the final Paper 3 analyses."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("PAPER 3 — FINAL ANALYSIS")
    print("=" * 70)

    df = load_data()

    print(
        f"Comments loaded: {len(df):,}"
    )

    # -------------------------------------------------------------
    # 1. Campaign trajectory
    # -------------------------------------------------------------

    analyse_campaign_trajectory(
        df
    )

    # -------------------------------------------------------------
    # 2. Major campaign events
    # -------------------------------------------------------------

    event_daily = build_event_data(
        df
    )

    analyse_events(
        event_daily
    )

    # -------------------------------------------------------------
    # 3. EAP versus incivility
    # -------------------------------------------------------------

    compare_eap_incivility(
        event_daily
    )

    # -------------------------------------------------------------
    # 4. Post-event dissipation
    # -------------------------------------------------------------

    analyse_dissipation(
        df
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
