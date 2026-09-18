from pathlib import Path
import tempfile
import shutil

import numpy as np
import pandas as pd
import librosa
import yt_dlp


# PATHS

data_dir = Path("/Users/dianaszasz/PycharmProjects/Oceania")

INPUT_FILE = data_dir / "AM_playlist_export" / "PL01_brunello_stuff.csv"
OUTPUT_FILE = data_dir / "AM_playlist_export" / "PL01_brunello_stuff_analyzed.csv"


# --------------------------------------------------
# CAMELOT MAP
# --------------------------------------------------

CAMELOT = {
    ("C", "major"): "8B",
    ("G", "major"): "9B",
    ("D", "major"): "10B",
    ("A", "major"): "11B",
    ("E", "major"): "12B",
    ("B", "major"): "1B",
    ("F#", "major"): "2B",
    ("C#", "major"): "3B",
    ("G#", "major"): "4B",
    ("D#", "major"): "5B",
    ("A#", "major"): "6B",
    ("F", "major"): "7B",

    ("A", "minor"): "8A",
    ("E", "minor"): "9A",
    ("B", "minor"): "10A",
    ("F#", "minor"): "11A",
    ("C#", "minor"): "12A",
    ("G#", "minor"): "1A",
    ("D#", "minor"): "2A",
    ("A#", "minor"): "3A",
    ("F", "minor"): "4A",
    ("C", "minor"): "5A",
    ("G", "minor"): "6A",
    ("D", "minor"): "7A",
}


# --------------------------------------------------
# KEY PROFILES
# Krumhansl-Schmuckler profiles
# --------------------------------------------------

MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33,
    4.38, 4.09, 2.52, 5.19,
    2.39, 3.66, 2.29, 2.88
])

MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38,
    2.60, 3.53, 2.54, 4.75,
    3.98, 2.69, 3.34, 3.17
])

NOTE_NAMES = [
    "C", "C#", "D", "D#",
    "E", "F", "F#", "G",
    "G#", "A", "A#", "B"
]


# --------------------------------------------------
# DOWNLOAD AUDIO
# --------------------------------------------------

def download_audio(youtube_url, temp_dir):

    output_template = str(Path(temp_dir) / "audio.%(ext)s")

    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([youtube_url])

    audio_file = Path(temp_dir) / "audio.wav"

    if not audio_file.exists():
        raise FileNotFoundError("Audio file was not created.")

    return audio_file


# --------------------------------------------------
# BPM
# --------------------------------------------------

def estimate_bpm(y, sr):

    tempo, _ = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    tempo = float(np.asarray(tempo).squeeze())

    # Correct common half/double tempo estimates
    while tempo < 70:
        tempo *= 2

    while tempo > 190:
        tempo /= 2

    return round(tempo, 1)


# --------------------------------------------------
# MUSICAL KEY
# --------------------------------------------------

def estimate_key(y, sr):

    # Separate harmonic content from drums/percussion
    harmonic = librosa.effects.harmonic(y)

    # Convert harmonic signal to 12 pitch classes
    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=sr
    )

    # Average pitch-class energy across the song
    chroma_mean = np.mean(chroma, axis=1)

    scores = []

    for shift in range(12):

        major = np.roll(MAJOR_PROFILE, shift)
        minor = np.roll(MINOR_PROFILE, shift)

        major_score = np.corrcoef(
            chroma_mean,
            major
        )[0, 1]

        minor_score = np.corrcoef(
            chroma_mean,
            minor
        )[0, 1]

        scores.append(
            (major_score, shift, "major")
        )

        scores.append(
            (minor_score, shift, "minor")
        )

    _, note_index, mode = max(
        scores,
        key=lambda x: x[0]
    )

    note = NOTE_NAMES[note_index]

    camelot = CAMELOT.get(
        (note, mode),
        pd.NA
    )

    return note, mode, camelot


# --------------------------------------------------
# ANALYZE ONE TRACK
# --------------------------------------------------

def analyze_track(youtube_url):

    if pd.isna(youtube_url):
        return pd.Series([
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA
        ])

    temp_dir = tempfile.mkdtemp()

    try:
        print(f"\nDownloading: {youtube_url}")

        audio_file = download_audio(
            youtube_url,
            temp_dir
        )

        print("Analyzing audio...")

        y, sr = librosa.load(
            audio_file,
            sr=22050,
            mono=True
        )

        bpm = estimate_bpm(y, sr)

        note, mode, camelot = estimate_key(
            y,
            sr
        )

        musical_key = f"{note} {mode}"

        print(
            f"BPM: {bpm} | "
            f"Key: {musical_key} | "
            f"Camelot: {camelot}"
        )

        return pd.Series([
            bpm,
            musical_key,
            camelot,
            "YouTube_audio_analysis"
        ])

    except Exception as e:

        print(f"Analysis failed: {e}")

        return pd.Series([
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA
        ])

    finally:

        # Delete downloaded audio + temporary directory
        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# --------------------------------------------------
# LOAD CSV
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)


# --------------------------------------------------
# ANALYZE ALL TRACKS
# --------------------------------------------------

df[
    [
        "BPM",
        "Key",
        "Camelot",
        "Audio_analysis_source"
    ]
] = df["YouTube_URL"].apply(
    analyze_track
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFinished.")
print(f"Saved to: {OUTPUT_FILE}")