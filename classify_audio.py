import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import requests

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # or "WARNING" if you want warnings

# ---------------- Notes -------------------- #
# This script is getting a high correct classification rate but is missing singing and the beast is loose
# also some quiet music (fur alina) and missing music in some mostly vocal tracks, and un normalized tracks

# ---------------- Utilities ---------------- #


def clear_dir(dir_path: Path) -> None:
    """Remove an existing directory and all its contents."""
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)


def get_contents(url: str):
    """Fetch JSON data from a URL, raising for non-200 responses."""
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    return res.json()


def filter_data(data, stream_key: str):
    """Filter items by the 'stream_2' key."""
    return [item for item in data if item.get("stream_2") == stream_key]


def save_new_data(data, filename: Path) -> None:
    """Save data to a JSON file."""
    with filename.open("w") as json_file:
        json.dump(data, json_file, indent=4)


# ---------------- Audio / Classification ---------------- #


def is_singing(vocals: np.ndarray, sr: int, voiced_thresh: float = 0.6) -> bool:
    """Check whether a vocal track contains singing using voiced frame ratio."""
    print("Just vocals detected, checking for singing")

    start_time = 10.0
    end_time = 20.0

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    if end_sample <= start_sample or end_sample > len(vocals):
        # Fallback to using the full signal if the clip range is invalid
        segment = vocals
    else:
        segment = vocals[start_sample:end_sample]

    try:
        _, voiced_flag, _ = librosa.pyin(
            segment,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
    except Exception as e:
        print(f"pyin failed: {e}")
        return False

    voiced_ratio = float(np.mean(voiced_flag.astype(float)))
    print(f"Voiced ratio is: {voiced_ratio:.3f}")

    return voiced_ratio >= voiced_thresh


def rms_energy(y: np.ndarray) -> float:
    """Compute RMS energy of a mono waveform."""
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y**2)))


def run_demucs(input_path: Path, out_dir: Path) -> Path | None:
    """Run demucs to produce two stems (vocals + no_vocals)."""
    try:
        subprocess.run(
            [
                "demucs",
                "--two-stems=vocals",
                "--out",
                str(out_dir),
                str(input_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"demucs timed out on {input_path}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"demucs failed on {input_path}: {e}")
        return None

    return out_dir / "htdemucs" / input_path.stem


def load_stems(
    stems_dir: Path, target_sr: int
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Load vocals and music stems, resample to target_sr if needed."""
    vocals_path = stems_dir / "vocals.wav"
    music_path = stems_dir / "no_vocals.wav"

    if not vocals_path.is_file() or not music_path.is_file():
        print(f"Missing expected stems in {stems_dir}")
        return None

    try:
        vocals, sr = librosa.load(vocals_path, sr=None, mono=True)
        music, sr_music = librosa.load(music_path, sr=None, mono=True)
    except Exception as e:
        print(f"Error loading demucs stems from {stems_dir}: {e}")
        return None

    if sr != sr_music:
        print(
            f"Warning: different sampling rates for stems ({sr} vs {sr_music}), using vocals sr."
        )

    if sr != target_sr:
        vocals = librosa.resample(vocals, orig_sr=sr, target_sr=target_sr)
        music = librosa.resample(music, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return vocals, music, sr


def classify_audio(
    input_path: Path,
    voiced_thresh: float,
    target_sr: int = 16000,
    min_rms: float = 0.015,
) -> str | None:
    """Classify audio into 'vocals + music', 'just vocals', 'just music', or 'silence/other'."""
    input_path = Path(input_path)
    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        return None

    out_dir = Path("./separated")
    clear_dir(out_dir)

    stems_dir = run_demucs(input_path, out_dir)
    if stems_dir is None:
        clear_dir(out_dir)
        return None

    loaded = load_stems(stems_dir, target_sr=target_sr)
    if loaded is None:
        clear_dir(out_dir)
        return None

    vocals, music, sr = loaded

    vocal_rms = rms_energy(vocals)
    music_rms = rms_energy(music)

    if vocal_rms > min_rms and music_rms > min_rms:
        label = "vocals + music"
    elif vocal_rms > min_rms:
        label = "just vocals"
    elif music_rms > min_rms:
        label = "just music"
    else:
        label = "silence/other"

    if label == "just vocals" and is_singing(vocals, sr, voiced_thresh=voiced_thresh):
        print("Predicted singing detected in just vocals.")
        label = "vocals + music"

    clear_dir(out_dir)

    print(f"{label} | vocals: {vocal_rms:.4f} | music: {music_rms:.4f}")
    return label


# ---------------- Main script ---------------- #


def process_single_file(archive: Path, fn: str, voiced_thresh: float) -> None:
    """Process a single local file passed via CLI."""
    data = {"ID": "test", "url": fn, "title": "test", "stream_2": "music"}
    p = archive / data["url"]

    print(f"Processing {data['title']}, file: {data['url']}")
    result = classify_audio(p, voiced_thresh)

    if result is None:
        print("Classification failed")
    else:
        print(f"{data['title']} was {data['stream_2']} and was classified as {result}")


def process_remote_data(
    archive: Path,
    url: str,
    stream_key: str,
    voiced_thresh: float,
    output_path: Path,
) -> None:
    """Fetch remote metadata, classify matching items, and save results."""
    try:
        data = get_contents(url)
    except Exception as e:
        print(f"Failed to fetch data from {url}: {e}")
        return

    filtered_data = filter_data(data, stream_key)
    new_data: dict[str, str] = {}

    for item in filtered_data:
        item_id = item.get("ID")
        fn = item.get("url")
        title = item.get("title", "<unknown title>")

        if not item_id or not fn:
            print(f"Skipping item with missing ID or url: {item}")
            continue

        p = archive / fn
        print(f"Processing {title}, file: {fn}")

        result = classify_audio(p, voiced_thresh)
        if result is None:
            print("Classification failed")
            continue

        print(f"{title} was {item.get('stream_2')} and was classified as {result}")
        new_data[item_id] = result

    save_new_data(new_data, output_path)
    print(f"Results saved to {output_path}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify audio files as vocals/music/silence"
    )
    parser.add_argument(
        "--fn",
        type=str,
        default="",
        help="Single filename (relative to archive) to classify",
    )
    parser.add_argument(
        "--vt",
        type=float,
        default=0.6,
        help="Voiced threshold for singing detection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # location of audio files
    archive = Path("./")

    key = "music"
    url = "https://localhost:8080/content"
    output_filename = Path("classified_audio_data.json")

    if args.fn:
        process_single_file(archive, args.fn, args.vt)
    else:
        process_remote_data(
            archive=archive,
            url=url,
            stream_key=key,
            voiced_thresh=args.vt,
            output_path=output_filename,
        )


if __name__ == "__main__":
    main()
