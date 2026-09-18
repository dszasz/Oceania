# Gets Artist & Title from Apple Music playlist and Youtube URL

import pandas
import pathlib
# %%
import re

from pathlib import Path

import pandas as pd

data_dir = Path("/Users/dianaszasz/PycharmProjects/Oceania")

txt_file = data_dir/"AM_playlist_import/brunello_stuff.txt"
input_file = data_dir / "AM_playlist_export" / "brunello_stuff.csv"
output_file = data_dir / "AM_playlist_export" / "PL01_brunello_stuff.csv"


# 1---- Convert Playlist txt to csv ---

# Load Apple Music playlist export
df = pd.read_csv(txt_file, sep="\t", encoding="utf-16")

# Rename available Apple Music columns
df = df.rename(columns={
    "Name": "Title",
    "Location": "URL"
})

# Define metadata columns for Oceania
columns = [
    "Title",
    "Artist",
    "Album",
    "Label",
    "Genre",
    "Subgenre",
    "Subgenre 2",
    "BPM",
    "Key",
    "Mood",
    "Light",
    "URL"
]

# Add missing columns as empty values
for col in columns:
    if col not in df.columns:
        df[col] = pd.NA

# Keep only the Oceania metadata
df = df[columns]


# Make individual track identifier basesed on name
def make_track_id(title, artist):
    text = f"{title}_{artist}".lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text[:9]

df.insert(0, "Track_ID", df.apply(lambda row: make_track_id(row["Title"], row["Artist"]), axis=1))

# Optional: save as CSV
df.to_csv(input_file, index=False)
# %%

# 2---- Get Youtube URL ---

from pathlib import Path
import requests

from rapidfuzz.fuzz import ratio

yt_API_key = "AIzaSyDuLVYZcnug3NTKNcHdmXH91rMf22mc24c"

CONFIDENCE_THRESHOLD = 60
MAX_RESULTS = 5

df = pd.read_csv(input_file)

# Youtube search function

def get_youtube_match(title, artist):
    url = "https://www.googleapis.com/youtube/v3/search"

    query = f"{artist} {title}"

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": MAX_RESULTS,
        "key": yt_API_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as e:
        print(f"Request failed: {title} - {artist}: {e}")
        return pd.Series([pd.NA, pd.NA, pd.NA])

    items = data.get("items", [])

    if not items:
        print(f"No YouTube results: {title} - {artist}")
        return pd.Series([pd.NA, pd.NA, pd.NA])

    # Text we expect to find in the YouTube result
    target = f"{artist} {title}".lower()

    # Compare all returned YouTube titles
    matches = []

    for item in items:
        youtube_title = item["snippet"]["title"]
        video_id = item["id"]["videoId"]

        score = ratio(
            target,
            youtube_title.lower()
        )

        matches.append({
            "video_id": video_id,
            "youtube_title": youtube_title,
            "score": score
        })

    # Select result with highest similarity
    best_match = max(
        matches,
        key=lambda x: x["score"]
    )

    score = best_match["score"]

    # Reject uncertain matches
    if score < CONFIDENCE_THRESHOLD:
        print(
            f"Low confidence ({score:.0f}): "
            f"{title} - {artist} -> "
            f"{best_match['youtube_title']}"
        )

        return pd.Series([
            pd.NA,
            best_match["youtube_title"],
            score
        ])

    youtube_url = (
        "https://www.youtube.com/watch?v="
        + best_match["video_id"]
    )

    print(
        f"Match ({score:.0f}): "
        f"{title} - {artist} -> "
        f"{best_match['youtube_title']}"
    )

    return pd.Series([
        youtube_url,
        best_match["youtube_title"],
        score
    ])


# Search all tracks
df[
    [
        "YouTube_URL",
        "YouTube_title",
        "YouTube_match_score"
    ]
] = df.apply(
    lambda row: get_youtube_match(
        row["Title"],
        row["Artist"]
    ),
    axis=1
)

df.to_csv(output_file, index=False)



print("\nFinished.")
print(f"Saved to: {output_file}")
