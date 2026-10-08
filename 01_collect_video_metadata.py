"""
Collect YouTube video metadata for a specified channel and year.

This script uses the YouTube Data API v3 to:
1. Identify a channel's uploads playlist.
2. Retrieve all video IDs from that playlist.
3. Collect metadata and engagement statistics.
4. Retain videos published during the target year.
5. Export the resulting dataset as a CSV file.

The script forms the first stage of the data collection pipeline used
to construct the YouTube political communication corpus.
"""

import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from googleapiclient.discovery import build


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CHANNEL_ID = "UCAql2DyGU2un1Ei2nMYsqOA"
TARGET_YEAR = 2024

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / f"youtube_videos_{TARGET_YEAR}.csv"


# ---------------------------------------------------------------------
# YouTube API
# ---------------------------------------------------------------------

def create_youtube_client():
    """Create an authenticated YouTube Data API client."""

    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise ValueError(
            "YOUTUBE_API_KEY environment variable is not set."
        )

    return build("youtube", "v3", developerKey=api_key)


def get_upload_playlist(youtube, channel_id):
    """Return the uploads playlist ID for a YouTube channel."""

    response = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    if not response["items"]:
        raise ValueError(f"Channel not found: {channel_id}")

    return response["items"][0]["contentDetails"][
        "relatedPlaylists"
    ]["uploads"]


def get_video_ids(youtube, playlist_id):
    """Retrieve all video IDs from a channel's uploads playlist."""

    video_ids = []
    page_token = None

    while True:
        response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token
        ).execute()

        video_ids.extend(
            item["contentDetails"]["videoId"]
            for item in response["items"]
        )

        page_token = response.get("nextPageToken")

        if not page_token:
            break

    return video_ids


# ---------------------------------------------------------------------
# Metadata collection
# ---------------------------------------------------------------------

def iso_duration_to_hms(duration):
    """Convert an ISO-8601 YouTube duration to HH:MM:SS."""

    hours = re.search(r"(\d+)H", duration)
    minutes = re.search(r"(\d+)M", duration)
    seconds = re.search(r"(\d+)S", duration)

    h = int(hours.group(1)) if hours else 0
    m = int(minutes.group(1)) if minutes else 0
    s = int(seconds.group(1)) if seconds else 0

    return f"{h:02d}:{m:02d}:{s:02d}"


def get_video_metadata(youtube, video_ids, target_year):
    """Collect metadata for videos published in the target year."""

    records = []

    # YouTube's videos endpoint accepts a maximum of 50 IDs per request.
    for start in range(0, len(video_ids), 50):

        response = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_ids[start:start + 50]
        ).execute()

        for video in response["items"]:

            published = video["snippet"]["publishedAt"]

            published_dt = datetime.fromisoformat(
                published.replace("Z", "+00:00")
            )

            if published_dt.year != target_year:
                continue

            statistics = video.get("statistics", {})

            records.append({
                "video_id": video["id"],
                "title": video["snippet"]["title"],
                "description": video["snippet"]["description"],
                "published_at": published,
                "video_length": iso_duration_to_hms(
                    video["contentDetails"].get("duration", "PT0S")
                ),
                "view_count": int(statistics.get("viewCount", 0)),
                "like_count": int(statistics.get("likeCount", 0)),
                "comment_count": int(statistics.get("commentCount", 0)),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():

    youtube = create_youtube_client()

    print("Retrieving channel uploads playlist...")
    playlist_id = get_upload_playlist(youtube, CHANNEL_ID)

    print("Retrieving video IDs...")
    video_ids = get_video_ids(youtube, playlist_id)
    print(f"Found {len(video_ids):,} videos")

    print(f"Collecting metadata for {TARGET_YEAR}...")
    videos = get_video_metadata(
        youtube,
        video_ids,
        TARGET_YEAR
    )

    print(f"Videos retained: {len(videos):,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        index=False
    )

    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
