"""
Construct the master 2024 YouTube campaign corpus.

This script:
1. Combines cleaned comment datasets for Biden, Harris, and Trump.
2. Standardises comment timestamps and candidate identifiers.
3. Restricts comments to Election Day or earlier.
4. Combines candidate-level video metadata.
5. Checks video IDs, dates, duplicates, and candidate consistency.
6. Merges video metadata onto the comment-level corpus.
7. Exports the master comment, video, and analysis datasets.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path("data")
CLEAN_DIR = DATA_DIR / "clean"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "processed"

ELECTION_DAY_END = pd.Timestamp(
    "2024-11-05 23:59:59",
    tz="UTC"
)

COMMENT_FILES = {
    "Biden": CLEAN_DIR / "Biden2024_comments_clean.csv",
    "Harris": CLEAN_DIR / "Harris2024_comments_clean.csv",
    "Trump": CLEAN_DIR / "Trump2024_comments_clean.csv",
}

VIDEO_FILES = {
    "Biden": RAW_DIR / "Biden2024_channel.csv",
    "Harris": RAW_DIR / "Harris2024_channel.csv",
    "Trump": RAW_DIR / "Trump2024_channel.csv",
}

COMMENTS_OUTPUT = OUTPUT_DIR / "comments_2024_master.csv"
VIDEOS_OUTPUT = OUTPUT_DIR / "videos_2024_master.csv"
ANALYSIS_OUTPUT = OUTPUT_DIR / "comments_2024_analysis.csv"


# ---------------------------------------------------------------------
# Comment data
# ---------------------------------------------------------------------

def load_comment_data(files):
    """Load and combine cleaned candidate comment datasets."""

    datasets = []

    for candidate, file in files.items():

        print(f"Loading comments: {candidate}")

        df = pd.read_csv(
            file,
            encoding="utf-8-sig",
            low_memory=False
        )

        # Remove accidental CSV index columns.
        df = df.drop(
            columns=["Unnamed: 0"],
            errors="ignore"
        )

        # Earlier collection files differ slightly in their
        # timestamp field, so construct one consistent variable.
        if "published_at" in df.columns:
            date_source = "published_at"

        elif "published" in df.columns:
            date_source = "published"

        elif "updated_at" in df.columns:
            date_source = "updated_at"

        else:
            raise ValueError(
                f"No comment timestamp found in {file}"
            )

        df["comment_date"] = pd.to_datetime(
            df[date_source],
            errors="coerce",
            utc=True
        )

        if "updated_at" in df.columns:
            df["updated_at"] = pd.to_datetime(
                df["updated_at"],
                errors="coerce",
                utc=True
            )

        df["channel_candidate"] = candidate
        df["election_year"] = 2024
        df["comment_source_file"] = file.name

        datasets.append(df)

    return pd.concat(
        datasets,
        ignore_index=True,
        sort=False
    )


def apply_election_cutoff(comments):
    """Remove comments posted after Election Day 2024."""

    n_before = len(comments)

    comments = comments[
        comments["comment_date"] <= ELECTION_DAY_END
    ].copy()

    n_removed = n_before - len(comments)

    print("\nElection Day cutoff")
    print(f"Before cutoff: {n_before:,}")
    print(f"Retained:      {len(comments):,}")
    print(f"Removed:       {n_removed:,}")

    return comments


# ---------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------

def load_video_data(files):
    """Load and combine candidate video metadata."""

    datasets = []

    for candidate, file in files.items():

        print(f"Loading video metadata: {candidate}")

        df = pd.read_csv(
            file,
            encoding="utf-8-sig",
            low_memory=False
        )

        df = df.drop(
            columns=["Unnamed: 0"],
            errors="ignore"
        )

        # Standardise metadata variable names.
        df = df.rename(
            columns={
                "id": "video_id",
                "video_length": "length",
                "published": "video_published",
                "published_at": "video_published",
            }
        )

        if "video_id" not in df.columns:
            raise ValueError(
                f"No 'video_id' column found in {file}"
            )

        if "video_published" in df.columns:
            df["video_published"] = pd.to_datetime(
                df["video_published"],
                errors="coerce",
                utc=True
            )

        df["candidate_video"] = candidate
        df["election_year_video"] = 2024
        df["video_source_file"] = file.name

        datasets.append(df)

    videos = pd.concat(
        datasets,
        ignore_index=True,
        sort=False
    )

    # Each video should have one metadata record.
    duplicate_ids = videos["video_id"].duplicated().sum()

    print(f"\nDuplicate video IDs: {duplicate_ids:,}")

    videos = videos.drop_duplicates(
        subset="video_id",
        keep="first"
    )

    return videos


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_data(comments, videos):
    """Run integrity checks before constructing the master corpus."""

    print("\nData validation")

    # Comment dates
    missing_dates = comments["comment_date"].isna().sum()

    print(f"Missing comment dates: {missing_dates:,}")
    print(
        f"Comment date range: "
        f"{comments['comment_date'].min()} to "
        f"{comments['comment_date'].max()}"
    )

    # Potential duplicate comments.
    duplicate_fields = [
        "author",
        "video_id",
        "text"
    ]

    duplicate_fields = [
        col for col in duplicate_fields
        if col in comments.columns
    ]

    potential_duplicates = comments.duplicated(
        subset=duplicate_fields,
        keep=False
    ).sum()

    print(
        f"Potential duplicate comment rows: "
        f"{potential_duplicates:,}"
    )

    # Comment-video matching.
    comment_video_ids = set(
        comments["video_id"]
        .dropna()
        .astype(str)
    )

    metadata_video_ids = set(
        videos["video_id"]
        .dropna()
        .astype(str)
    )

    missing_metadata = (
        comment_video_ids - metadata_video_ids
    )

    videos_without_comments = (
        metadata_video_ids - comment_video_ids
    )

    print(
        f"Comment video IDs without metadata: "
        f"{len(missing_metadata):,}"
    )

    print(
        f"Metadata videos without comments: "
        f"{len(videos_without_comments):,}"
    )

    # Candidate consistency between comment source
    # and video metadata.
    candidate_check = comments[
        ["video_id", "channel_candidate"]
    ].merge(
        videos[
            ["video_id", "candidate_video"]
        ],
        on="video_id",
        how="left"
    )

    mismatches = candidate_check[
        candidate_check["candidate_video"].notna()
        & (
            candidate_check["channel_candidate"]
            != candidate_check["candidate_video"]
        )
    ]

    print(
        f"Candidate/video mismatches: "
        f"{len(mismatches):,}"
    )

    if missing_dates > 0:
        raise ValueError(
            "Missing comment dates detected."
        )

    if missing_metadata:
        raise ValueError(
            "Some comment video IDs have no metadata."
        )

    if len(mismatches) > 0:
        raise ValueError(
            "Candidate/video mismatches detected."
        )


# ---------------------------------------------------------------------
# Master corpus
# ---------------------------------------------------------------------

def build_analysis_corpus(comments, videos):
    """Merge video metadata onto the comment-level corpus."""

    return comments.merge(
        videos,
        on="video_id",
        how="left",
        validate="many_to_one"
    )


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load comments.
    comments = load_comment_data(
        COMMENT_FILES
    )

    comments = apply_election_cutoff(
        comments
    )

    print(
        f"\nComments retained: "
        f"{len(comments):,}"
    )

    print("\nComments by candidate:")
    print(
        comments["channel_candidate"]
        .value_counts()
    )

    # Load video metadata.
    videos = load_video_data(
        VIDEO_FILES
    )

    print(
        f"\nUnique videos: "
        f"{len(videos):,}"
    )

    print("\nVideos by candidate:")
    print(
        videos["candidate_video"]
        .value_counts()
    )

    # Validate before merging.
    validate_data(
        comments,
        videos
    )

    # Construct master analysis corpus.
    analysis = build_analysis_corpus(
        comments,
        videos
    )

    # Save outputs.
    comments.to_csv(
        COMMENTS_OUTPUT,
        encoding="utf-8-sig",
        index=False
    )

    videos.to_csv(
        VIDEOS_OUTPUT,
        encoding="utf-8-sig",
        index=False
    )

    analysis.to_csv(
        ANALYSIS_OUTPUT,
        encoding="utf-8-sig",
        index=False
    )

    print("\nMaster corpus complete")
    print(f"Comments: {len(comments):,}")
    print(f"Videos:   {len(videos):,}")
    print(f"Rows:     {len(analysis):,}")

    print(
        f"Date range: "
        f"{comments['comment_date'].min()} to "
        f"{comments['comment_date'].max()}"
    )

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
