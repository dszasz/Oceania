import pandas as pd
import numpy as np
from itertools import combinations
import pathlib
from pathlib import Path

# 1. SETTINGS

data_dir = Path("/Users/dianaszasz/PycharmProjects/Oceania")

OUTPUT_DIR = data_dir/"2_connection_mapping/output"
INPUT_FILE = data_dir/"AM_playlist_export/PL01_final.csv"

# Size of entire map
MAP_MIN = 0
MAP_MAX = 100

# Genre reef centers
# Arranged as an approximately equilateral triangle across the 100x100 map
GENRE_CENTERS = {
    "Techno": np.array([25.0, 25.0]),
    "House": np.array([75.0, 25.0]),
    "Electronic": np.array([50.0, 75.0]),
}

# Maximum distance a track can move away from its genre center
REEF_RADIUS = 18

# Distance of Subgenre1 regions from genre center
SUBGENRE1_RADIUS = 10

# Distance of Subgenre2 regions from their Subgenre1 center
SUBGENRE2_RADIUS = 3

# Strength of spatial metadata
MOOD1_WEIGHT = 2.5
MOOD2_WEIGHT = 1.2
LIGHT_WEIGHT = 1.0

# Connection score weights
BPM_CONNECTION_WEIGHT = 0.6
KEY_CONNECTION_WEIGHT = 0.4

# Only keep connections above this value
CONNECTION_THRESHOLD = 0.60


# ============================================================
# 2. IMPORT DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} tracks.")
print(df[["Track_ID", "Title", "Genre", "Subgenre1",
          "Subgenre2", "Mood1", "Mood2",
          "Light", "BPM", "Camelot"]].head())


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    """
    Converts text into a clean lowercase string.
    Missing values become None.
    """

    if pd.isna(value):
        return None

    return str(value).strip().lower()


def circular_offsets(labels, radius):
    """
    Places categories evenly around a circle.

    Example:
    Driving, Deep, Minimal, Acid...
    each receive a different direction around the reef.
    """

    labels = sorted([x for x in labels if x is not None])

    if len(labels) == 0:
        return {}

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False
    )

    return {
        label: np.array([
            radius * np.cos(angle),
            radius * np.sin(angle)
        ])
        for label, angle in zip(labels, angles)
    }


def constrain_to_reef(position, reef_center, reef_radius):
    """
    Prevents mood/light/subgenre offsets from moving a track
    outside its genre reef.
    """

    difference = position - reef_center
    distance = np.linalg.norm(difference)

    if distance > reef_radius:
        difference = difference / distance * reef_radius
        position = reef_center + difference

    return position


# ============================================================
# 4. CLEAN CATEGORICAL COLUMNS
# ============================================================

for column in [
    "Genre",
    "Subgenre1",
    "Subgenre2",
    "Mood1",
    "Mood2",
    "Light"
]:
    df[column + "_clean"] = df[column].apply(clean_text)


# ============================================================
# 5. GENRE CENTERS
# ============================================================

# Convert keys to lowercase because cleaned data are lowercase
genre_centers = {
    genre.lower(): position
    for genre, position in GENRE_CENTERS.items()
}


# ============================================================
# 6. CREATE SUBGENRE1 REGIONS
# ============================================================

subgenre1_centers = {}

for genre in df["Genre_clean"].dropna().unique():

    genre_tracks = df[df["Genre_clean"] == genre]

    subgenres = genre_tracks["Subgenre1_clean"].dropna().unique()

    offsets = circular_offsets(
        subgenres,
        SUBGENRE1_RADIUS
    )

    reef_center = genre_centers[genre]

    for subgenre, offset in offsets.items():

        subgenre1_centers[(genre, subgenre)] = (
            reef_center + offset
        )


# ============================================================
# 7. CREATE SUBGENRE2 LOCAL REGIONS
# ============================================================

subgenre2_centers = {}

for (genre, subgenre1), subgenre1_center in subgenre1_centers.items():

    relevant_tracks = df[
        (df["Genre_clean"] == genre) &
        (df["Subgenre1_clean"] == subgenre1)
    ]

    subgenres2 = (
        relevant_tracks["Subgenre2_clean"]
        .dropna()
        .unique()
    )

    offsets = circular_offsets(
        subgenres2,
        SUBGENRE2_RADIUS
    )

    for subgenre2, offset in offsets.items():

        subgenre2_centers[
            (genre, subgenre1, subgenre2)
        ] = subgenre1_center + offset


# ============================================================
# 8. MOOD DIRECTIONS
# ============================================================

# Same moods point in the same spatial direction.
#
# These are NOT "scientific" coordinates.
# They simply create consistent attraction between
# tracks with similar moods.

MOOD_VECTORS = {

    "happy":
        np.array([1.0, 1.0]),

    "emotional":
        np.array([-1.0, 1.0]),

    "groovy":
        np.array([1.0, 0.0]),

    "bouncy":
        np.array([1.0, -1.0]),

    "sleek":
        np.array([-1.0, 0.0]),

    "hypnotic":
        np.array([-1.0, -1.0]),
}


# Normalize diagonal vectors so all moods have equal strength

MOOD_VECTORS = {
    mood: vector / np.linalg.norm(vector)
    for mood, vector in MOOD_VECTORS.items()
}


# ============================================================
# 9. LIGHTING DIRECTIONS
# ============================================================

LIGHT_VECTORS = {

    "dark":
        np.array([-1.0, 0.0]),

    "bright":
        np.array([1.0, 0.0]),
}


# ============================================================
# 10. CALCULATE X/Y POSITION
# ============================================================

def calculate_xy(row):

    genre = row["Genre_clean"]
    subgenre1 = row["Subgenre1_clean"]
    subgenre2 = row["Subgenre2_clean"]

    mood1 = row["Mood1_clean"]
    mood2 = row["Mood2_clean"]

    light = row["Light_clean"]


    # --------------------------------------------------------
    # Start at genre reef
    # --------------------------------------------------------

    reef_center = genre_centers[genre]

    position = reef_center.copy()


    # --------------------------------------------------------
    # Move to Subgenre1 region
    # --------------------------------------------------------

    key1 = (genre, subgenre1)

    if key1 in subgenre1_centers:
        position = subgenre1_centers[key1].copy()


    # --------------------------------------------------------
    # Move to Subgenre2 local region
    # --------------------------------------------------------

    key2 = (
        genre,
        subgenre1,
        subgenre2
    )

    if key2 in subgenre2_centers:
        position = subgenre2_centers[key2].copy()


    # --------------------------------------------------------
    # Mood1 = strong local influence
    # --------------------------------------------------------

    if mood1 in MOOD_VECTORS:
        position += (
            MOOD1_WEIGHT *
            MOOD_VECTORS[mood1]
        )


    # --------------------------------------------------------
    # Mood2 = weaker local influence
    # --------------------------------------------------------

    if mood2 in MOOD_VECTORS:
        position += (
            MOOD2_WEIGHT *
            MOOD_VECTORS[mood2]
        )


    # --------------------------------------------------------
    # Lighting
    # --------------------------------------------------------

    if light in LIGHT_VECTORS:
        position += (
            LIGHT_WEIGHT *
            LIGHT_VECTORS[light]
        )


    # --------------------------------------------------------
    # Keep track inside its genre reef
    # --------------------------------------------------------

    position = constrain_to_reef(
        position,
        reef_center,
        REEF_RADIUS
    )


    # Keep everything inside 0-100 map
    position = np.clip(
        position,
        MAP_MIN,
        MAP_MAX
    )


    return pd.Series({
        "x": position[0],
        "y": position[1]
    })


df[["x", "y"]] = df.apply(
    calculate_xy,
    axis=1
)


# ============================================================
# 11. BPM → Z AXIS
# ============================================================

# Normalize BPM to 0-100.
#
# Lowest BPM in playlist  -> z = 0
# Highest BPM in playlist -> z = 100

bpm_min = df["BPM"].min()
bpm_max = df["BPM"].max()

df["z"] = (
    (df["BPM"] - bpm_min)
    /
    (bpm_max - bpm_min)
    * 100
)


# ============================================================
# 12. BPM SIMILARITY
# ============================================================

def bpm_similarity(bpm1, bpm2):
    """
    Returns a value between 0 and 1.

    Same BPM:
        1.0

    Difference of 5 BPM:
        0.5

    Difference of >=10 BPM:
        0.0
    """

    if pd.isna(bpm1) or pd.isna(bpm2):
        return 0

    difference = abs(bpm1 - bpm2)

    return max(
        0,
        1 - difference / 10
    )


# ============================================================
# 13. CAMELOT KEY SIMILARITY
# ============================================================

def camelot_similarity(key1, key2):

    if pd.isna(key1) or pd.isna(key2):
        return 0

    key1 = str(key1).strip().upper()
    key2 = str(key2).strip().upper()


    # Same key
    if key1 == key2:
        return 1.0


    try:

        number1 = int(key1[:-1])
        number2 = int(key2[:-1])

        letter1 = key1[-1]
        letter2 = key2[-1]

    except ValueError:

        return 0


    # --------------------------------------------------------
    # Relative major / minor
    #
    # Example:
    # 4A ↔ 4B
    # --------------------------------------------------------

    if (
        number1 == number2
        and letter1 != letter2
    ):
        return 0.8


    # --------------------------------------------------------
    # Adjacent Camelot number
    #
    # Example:
    # 4A ↔ 3A
    # 4A ↔ 5A
    # 12A ↔ 1A
    # --------------------------------------------------------

    number_difference = min(
        abs(number1 - number2),
        12 - abs(number1 - number2)
    )

    if (
        number_difference == 1
        and letter1 == letter2
    ):
        return 0.8


    return 0


# ============================================================
# 14. CALCULATE CONNECTIONS BETWEEN ALL TRACKS
# ============================================================

connections = []


for index_a, index_b in combinations(df.index, 2):

    track_a = df.loc[index_a]
    track_b = df.loc[index_b]


    bpm_score = bpm_similarity(
        track_a["BPM"],
        track_b["BPM"]
    )


    key_score = camelot_similarity(
        track_a["Camelot"],
        track_b["Camelot"]
    )


    connection_score = (
        BPM_CONNECTION_WEIGHT * bpm_score
        +
        KEY_CONNECTION_WEIGHT * key_score
    )


    connections.append({

        "Track_A":
            track_a["Track_ID"],

        "Title_A":
            track_a["Title"],

        "Track_B":
            track_b["Track_ID"],

        "Title_B":
            track_b["Title"],

        "BPM_A":
            track_a["BPM"],

        "BPM_B":
            track_b["BPM"],

        "Camelot_A":
            track_a["Camelot"],

        "Camelot_B":
            track_b["Camelot"],

        "BPM_similarity":
            round(bpm_score, 3),

        "Key_similarity":
            round(key_score, 3),

        "Connection_score":
            round(connection_score, 3)
    })


connections_df = pd.DataFrame(connections)


# ============================================================
# 15. FILTER STRONG CONNECTIONS
# ============================================================

strong_connections = connections_df[
    connections_df["Connection_score"]
    >= CONNECTION_THRESHOLD
].copy()


strong_connections = strong_connections.sort_values(
    "Connection_score",
    ascending=False
)


# ============================================================
# 16. EXPORT TRACK COORDINATES
# ============================================================

coordinate_columns = [

    "Track_ID",
    "Title",
    "Artist",

    "Genre",
    "Subgenre1",
    "Subgenre2",

    "Mood1",
    "Mood2",
    "Light",

    "BPM",
    "Camelot",

    "x",
    "y",
    "z"
]


df[coordinate_columns].to_csv(
    "track_coordinates.csv",
    index=False
)


# ============================================================
# 17. EXPORT CONNECTIONS
# ============================================================

connections_df.to_csv(OUTPUT_DIR/
    "all_connections.csv",
    index=False
)


strong_connections.to_csv(OUTPUT_DIR/
    "connections.csv",
    index=False
)


# ============================================================
# 18. SUMMARY
# ============================================================

print("\n--- MAP SUMMARY ---")

print(
    f"Tracks: {len(df)}"
)

print(
    f"Possible track pairs: {len(connections_df)}"
)

print(
    f"Connections >= {CONNECTION_THRESHOLD}: "
    f"{len(strong_connections)}"
)

print(
    f"BPM range: {bpm_min:.1f}–{bpm_max:.1f}"
)

print("\nGenre centers:")

for genre, center in GENRE_CENTERS.items():

    print(
        f"{genre}: "
        f"x={center[0]:.1f}, "
        f"y={center[1]:.1f}"
    )


print("\nFiles created:")

print("track_coordinates.csv")
print("all_connections.csv")
print("connections.csv")