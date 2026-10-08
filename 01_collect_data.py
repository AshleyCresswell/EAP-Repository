"""
Collect YouTube data for the 2024 US presidential election corpus.

For each candidate channel, this script:

1. Retrieves all uploaded videos from the channel.
2. Collects video-level metadata.
3. Collects top-level comments from each video.
4. Saves separate raw video and comment datasets.

Replies are not collected because the study focuses on top-level
comments posted directly beneath campaign videos.

Requires a YouTube Data API key stored in the environment variable:
YOUTUBE_API_KEY
"""

from pathlib import Path
from datetime import timedelta
import os
import re

import pandas as pd

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("data/raw")

API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

# Official campaign YouTube channels used in the study.
# Replace these values only if rebuilding the corpus from different
# source channels.

CHANNELS = {
    "Biden": "CHANNEL_ID_HERE",
    "Harris": "CHANNEL_ID_HERE",
    "Trump": "CHANNEL_ID_HERE",
}

VIDEO_BATCH_SIZE = 50
COMMENT_PAGE_SIZE = 100


# ---------------------------------------------------------------------
# YouTube client
# ---------------------------------------------------------------------

def create_youtube_client():
    """Create an authenticated YouTube Data API client."""

    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY environment variable is not set."
        )

    return build(
        API_SERVICE_NAME,
        API_VERSION,
        developerKey=api_key
    )


# ---------------------------------------------------------------------
# Channel videos
# ---------------------------------------------------------------------

def get_upload_playlist(youtube, channel_id):
    """Return the uploads playlist associated with a channel."""

    response = (
        youtube
        .channels()
        .list(
            part="contentDetails",
            id=channel_id
        )
        .execute()
    )

    items = response.get("items", [])

    if not items:
        raise ValueError(
            f"Channel not found: {channel_id}"
        )

    return (
        items[0]
        ["contentDetails"]
        ["relatedPlaylists"]
        ["uploads"]
    )


def get_video_ids(youtube, playlist_id):
    """Retrieve all video IDs from a channel uploads playlist."""

    video_ids = []
    page_token = None

    while True:

        response = (
            youtube
            .playlistItems()
            .list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=page_token
            )
            .execute()
        )

        video_ids.extend(
            item["contentDetails"]["videoId"]
            for item in response.get("items", [])
        )

        page_token = response.get(
            "nextPageToken"
        )

        if page_token is None:
            break

    return video_ids


# ---------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------

def iso_duration_to_hms(duration):
    """Convert an ISO 8601 YouTube duration to HH:MM:SS."""

    if not duration:
        return None

    match = re.fullmatch(
        r"PT"
        r"(?:(\d+)H)?"
        r"(?:(\d+)M)?"
        r"(?:(\d+)S)?",
        duration
    )

    if match is None:
        return duration

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    return str(
        timedelta(
            hours=hours,
            minutes=minutes,
            seconds=seconds
        )
    )


def get_video_metadata(
    youtube,
    video_ids,
    candidate
):
    """Retrieve metadata for a collection of YouTube videos."""

    records = []

    for start in range(
        0,
        len(video_ids),
        VIDEO_BATCH_SIZE
    ):

        batch = video_ids[
            start:start + VIDEO_BATCH_SIZE
        ]

        response = (
            youtube
            .videos()
            .list(
                part="snippet,contentDetails,statistics",
                id=",".join(batch)
            )
            .execute()
        )

        for item in response.get("items", []):

            snippet = item.get(
                "snippet",
                {}
            )

            statistics = item.get(
                "statistics",
                {}
            )

            content = item.get(
                "contentDetails",
                {}
            )

            records.append(
                {
                    "video_id":
                        item["id"],

                    "channel_candidate":
                        candidate,

                    "title":
                        snippet.get("title"),

                    "description":
                        snippet.get(
                            "description"
                        ),

                    "video_published":
                        snippet.get(
                            "publishedAt"
                        ),

                    "length":
                        iso_duration_to_hms(
                            content.get(
                                "duration"
                            )
                        ),

                    "view_count":
                        statistics.get(
                            "viewCount"
                        ),

                    "like_count":
                        statistics.get(
                            "likeCount"
                        ),

                    "comment_count":
                        statistics.get(
                            "commentCount"
                        ),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------

def get_video_comments(
    youtube,
    video_id,
    candidate
):
    """
    Retrieve all available top-level comments for one video.

    Reply chains are intentionally not collected.
    """

    records = []
    page_token = None

    while True:

        try:

            response = (
                youtube
                .commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=COMMENT_PAGE_SIZE,
                    pageToken=page_token,
                    textFormat="plainText"
                )
                .execute()
            )

        except HttpError as error:

            # Comments may be disabled or unavailable
            # for individual videos.

            print(
                f"Skipping comments for "
                f"{video_id}: {error}"
            )

            break

        for item in response.get(
            "items",
            []
        ):

            comment = (
                item["snippet"]
                ["topLevelComment"]
            )

            snippet = comment["snippet"]

            records.append(
                {
                    "comment_id":
                        comment.get("id"),

                    "video_id":
                        video_id,

                    "channel_candidate":
                        candidate,

                    "author":
                        snippet.get(
                            "authorDisplayName"
                        ),

                    "text":
                        snippet.get(
                            "textDisplay"
                        ),

                    "published_at":
                        snippet.get(
                            "publishedAt"
                        ),

                    "like_count_comment":
                        snippet.get(
                            "likeCount"
                        ),

                    "public":
                        item["snippet"].get(
                            "isPublic"
                        ),
                }
            )

        page_token = response.get(
            "nextPageToken"
        )

        if page_token is None:
            break

    return records


def get_channel_comments(
    youtube,
    video_ids,
    candidate
):
    """Retrieve top-level comments from all videos for one channel."""

    records = []

    total = len(video_ids)

    for number, video_id in enumerate(
        video_ids,
        start=1
    ):

        print(
            f"[{candidate}] "
            f"Video {number:,}/{total:,}: "
            f"{video_id}"
        )

        comments = get_video_comments(
            youtube,
            video_id,
            candidate
        )

        records.extend(comments)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------

def collect_candidate(
    youtube,
    candidate,
    channel_id
):
    """Collect video metadata and comments for one candidate."""

    print("\n" + "=" * 70)
    print(candidate.upper())
    print("=" * 70)

    # -------------------------------------------------------------
    # Videos
    # -------------------------------------------------------------

    playlist_id = get_upload_playlist(
        youtube,
        channel_id
    )

    video_ids = get_video_ids(
        youtube,
        playlist_id
    )

    print(
        f"Videos found: {len(video_ids):,}"
    )

    video_df = get_video_metadata(
        youtube,
        video_ids,
        candidate
    )

    # -------------------------------------------------------------
    # Comments
    # -------------------------------------------------------------

    comment_df = get_channel_comments(
        youtube,
        video_ids,
        candidate
    )

    print(
        f"Comments collected: "
        f"{len(comment_df):,}"
    )

    # -------------------------------------------------------------
    # Save raw candidate data
    # -------------------------------------------------------------

    video_file = (
        OUTPUT_DIR
        / f"{candidate}2024_channel.csv"
    )

    comment_file = (
        OUTPUT_DIR
        / f"{candidate}2024_comments.csv"
    )

    video_df.to_csv(
        video_file,
        encoding="utf-8-sig",
        index=False
    )

    comment_df.to_csv(
        comment_file,
        encoding="utf-8-sig",
        index=False
    )

    print(
        f"Saved video metadata: "
        f"{video_file}"
    )

    print(
        f"Saved comments: "
        f"{comment_file}"
    )

    return video_df, comment_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """Collect the raw YouTube campaign data."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    youtube = create_youtube_client()

    print("=" * 70)
    print("2024 YOUTUBE CAMPAIGN DATA COLLECTION")
    print("=" * 70)

    for candidate, channel_id in CHANNELS.items():

        if channel_id == "CHANNEL_ID_HERE":

            raise ValueError(
                f"Channel ID has not been set "
                f"for {candidate}."
            )

        collect_candidate(
            youtube,
            candidate,
            channel_id
        )

    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
