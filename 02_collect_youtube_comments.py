"""
Collect top-level YouTube comments from a list of videos.

This script uses the YouTube Data API v3 to:
1. Retrieve top-level comments for each specified video.
2. Handle API pagination to collect all available comments.
3. Combine comments across videos into a single dataset.
4. Export the resulting comment corpus as a CSV file.

Replies to comments are not collected.
"""

import os
from pathlib import Path

import pandas as pd
from googleapiclient.discovery import build


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path("data/youtube_videos.csv")
OUTPUT_FILE = Path("data/youtube_comments.csv")


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


def get_comments(youtube, video_id):
    """Retrieve all available top-level comments for a YouTube video."""

    comments = []
    page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=page_token,
            textFormat="plainText"
        )

        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "author": comment["authorDisplayName"],
                "published_at": comment["publishedAt"],
                "like_count": comment["likeCount"],
                "text": comment["textOriginal"],
                "video_id": comment["videoId"],
                "public": item["snippet"]["isPublic"]
            })

        page_token = response.get("nextPageToken")

        if not page_token:
            break

    return comments


# ---------------------------------------------------------------------
# Comment collection
# ---------------------------------------------------------------------

def collect_comments(youtube, video_ids):
    """Collect comments across a list of YouTube videos."""

    all_comments = []

    for number, video_id in enumerate(video_ids, start=1):

        print(
            f"Collecting video {number}/{len(video_ids)}: "
            f"{video_id}"
        )

        comments = get_comments(youtube, video_id)
        all_comments.extend(comments)

    return pd.DataFrame(all_comments)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():

    youtube = create_youtube_client()

    videos = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    if "video_id" not in videos.columns:
        raise ValueError("Input file must contain a 'video_id' column.")

    video_ids = videos["video_id"].dropna().unique().tolist()

    print(f"Videos to process: {len(video_ids):,}")

    comments = collect_comments(
        youtube,
        video_ids
    )

    print(f"Comments collected: {len(comments):,}")
    print(f"Videos represented: {comments['video_id'].nunique():,}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comments.to_csv(
        OUTPUT_FILE,
        encoding="utf-8-sig",
        index=False
    )

    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
