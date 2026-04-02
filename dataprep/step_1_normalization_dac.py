"""
Audio normalization for the RNNDAC pipeline.

This version targets the 44.1 kHz DAC model:
- audio is converted to 44.1 kHz mono
- optional peak-windowed RMS normalization is applied
- CSV files are copied alongside normalized WAVs
- unlike the old EnCodec pipeline, audio is not trimmed to CSV row count,
  because step 3 resamples annotations onto DAC frame times
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from .auxiliary_functions_dac import DAC_SAMPLE_RATE, find_audio_csv_pairs


def calculate_windowed_rms(audio: np.ndarray, sample_rate: int, window_ms: int = 250) -> np.ndarray:
    """Calculate sliding-window RMS values with 75% overlap."""
    window_samples = int(sample_rate * window_ms / 1000)
    hop_samples = max(1, window_samples // 4)

    if len(audio) < max(1, window_samples):
        return np.array([np.sqrt(np.mean(audio**2))], dtype=np.float64)

    rms_values = []
    for i in range(0, len(audio) - window_samples + 1, hop_samples):
        window = audio[i : i + window_samples]
        rms_values.append(np.sqrt(np.mean(window**2)))
    return np.asarray(rms_values)


def get_peak_windowed_rms(filepath: Path, window_ms: int = 250) -> Optional[float]:
    """Get the peak windowed RMS after loading the file as DAC-ready audio."""
    try:
        audio, sr_original = librosa.load(str(filepath), sr=None, mono=True)
        audio, sr = librosa.load(str(filepath), sr=DAC_SAMPLE_RATE, mono=True)
        if sr != sr_original and len(audio) > 24:
            audio = audio[:-24]
        rms_values = calculate_windowed_rms(audio, sr, window_ms)
        return float(np.max(rms_values))
    except Exception as e:  # pragma: no cover
        print(f"Error processing {filepath}: {e}")
        return None


def process_file(
    input_path: Path,
    output_path: Path,
    target_peak_rms: float,
    window_ms: int = 250,
    apply_rms_normalization: bool = True,
) -> bool:
    """Normalize/resample one file and save it as 44.1 kHz mono WAV."""
    print(f"Processing: {input_path.name}")

    gain_linear = 1.0
    if apply_rms_normalization:
        current_peak_rms = get_peak_windowed_rms(input_path, window_ms)
        if current_peak_rms is None or current_peak_rms == 0:
            print("    ✗ Skipping - could not calculate RMS")
            return False
        gain_linear = target_peak_rms / current_peak_rms
        gain_db = 20 * np.log10(gain_linear)
        print(f"    Current peak RMS: {current_peak_rms:.6f}")
        print(f"    Target peak RMS: {target_peak_rms:.6f}")
        print(f"    Applying gain: {gain_db:.2f} dB")
    else:
        print(f"    RMS normalization disabled - resampling to {DAC_SAMPLE_RATE} Hz only")

    try:
        audio, sr_original = librosa.load(str(input_path), sr=None, mono=True)
        audio, sr = librosa.load(str(input_path), sr=DAC_SAMPLE_RATE, mono=True)
        if sr != sr_original and len(audio) > 24:
            audio = audio[:-24]

        audio_normalized = audio * gain_linear
        sf.write(str(output_path), audio_normalized, sr, subtype="PCM_16")
        print(f"    ✓ Saved: {output_path.name}")
        return True
    except Exception as e:  # pragma: no cover
        print(f"    ✗ Error: {e}")
        return False


def normalize_dataset(
    input_folder: Path,
    output_folder: Path,
    target_rms: float = 0.1,
    window_ms: int = 250,
    apply_rms_normalization: bool = True,
) -> dict:
    """Normalize all audio/CSV pairs in a raw dataset tree."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder {input_folder} does not exist")

    output_folder.mkdir(parents=True, exist_ok=True)
    audio_csv_pairs = find_audio_csv_pairs(input_folder)
    if not audio_csv_pairs:
        print(f"No audio-CSV pairs found in {input_folder}")
        return {"total": 0, "success": 0, "failed": 0}

    print("\n\033[1mDATA SUMMARY:\033[0m\n")
    print(f"Found {len(audio_csv_pairs)} audio-CSV pairs")
    print(f"Target sample rate: {DAC_SAMPLE_RATE} Hz mono")
    if apply_rms_normalization:
        print(f"Target peak windowed RMS: {target_rms}")
        print(f"Window size: {window_ms} ms")
    else:
        print("RMS normalization: DISABLED")
    print(f"Input:  {input_folder}")
    print(f"Output: {output_folder}")

    print("\n\033[1mDATA PROCESSING:\033[0m\n")
    success_count = 0

    for audio_file, csv_file in sorted(audio_csv_pairs):
        rel_path = audio_file.relative_to(input_folder)
        output_audio = output_folder / rel_path.parent / f"{audio_file.stem}.wav"
        output_csv = output_folder / rel_path.parent / f"{audio_file.stem}.csv"
        output_audio.parent.mkdir(parents=True, exist_ok=True)

        if process_file(audio_file, output_audio, target_rms, window_ms, apply_rms_normalization):
            try:
                shutil.copy2(csv_file, output_csv)
                print(f"    ✅ Copied CSV: {csv_file.name} → {output_csv}")
                success_count += 1
            except Exception as e:
                print(f"    ❌ Error copying CSV {csv_file}: {e}")
        print()

    print(f"Successfully processed {success_count}/{len(audio_csv_pairs)} pairs")
    if success_count < len(audio_csv_pairs):
        print(f"Failed to process {len(audio_csv_pairs) - success_count} pairs")

    return {
        "total": len(audio_csv_pairs),
        "success": success_count,
        "failed": len(audio_csv_pairs) - success_count,
    }


def quick_normalize(dataset_dir, target_rms=0.1, window_ms=250, apply_rms_normalization=True):
    input_dir = str(Path(dataset_dir) / "raw")
    output_dir = str(Path(dataset_dir) / "normalized")
    normalize_dataset(
        Path(input_dir),
        Path(output_dir),
        target_rms=target_rms,
        window_ms=window_ms,
        apply_rms_normalization=apply_rms_normalization,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize audio for the RNNDAC pipeline")
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--target-rms", type=float, default=0.1)
    parser.add_argument("--window-ms", type=int, default=250)
    parser.add_argument("--no-rms-normalization", action="store_true")
    args = parser.parse_args()

    normalize_dataset(
        args.input_folder,
        args.output_folder,
        target_rms=args.target_rms,
        window_ms=args.window_ms,
        apply_rms_normalization=not args.no_rms_normalization,
    )
