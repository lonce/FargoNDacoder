"""
Shared utility functions for the RNNDAC dataset processing pipeline.

This is the DAC/44.1 kHz version of the earlier EnCodec-oriented helpers.
The main differences are:
- audio is analyzed at 44.1 kHz mono
- frame counts are based on DAC's hop length (512 samples for the 44.1 kHz model)
- .dac files can be inspected without importing the DAC package by reading the
  saved numpy artifact format directly
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch

DAC_SAMPLE_RATE = 44_100
DAC_HOP_LENGTH = 512
DAC_FRAMES_PER_SECOND = DAC_SAMPLE_RATE / DAC_HOP_LENGTH
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aif"}


def load_parameter_config(config_path: Path) -> Dict:
    """Load and parse ``parameters.json``."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_audio_csv_pairs(raw_dir: Path) -> List[Tuple[Path, Path]]:
    """Find all audio/CSV pairs under ``raw_dir`` recursively."""
    raw_dir = Path(raw_dir)
    audio_files: List[Path] = []
    for ext in SUPPORTED_AUDIO_EXTS:
        audio_files.extend(raw_dir.rglob(f"*{ext}"))

    pairs: List[Tuple[Path, Path]] = []
    for audio_file in sorted(audio_files):
        csv_file = audio_file.with_suffix(".csv")
        if csv_file.exists():
            pairs.append((audio_file, csv_file))
        else:
            print(f"⚠️  Warning: No CSV found for {audio_file.name}")
    return pairs


def find_audio_files(input_dir: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """Find all audio files recursively."""
    input_dir = Path(input_dir)
    if extensions is None:
        extensions = ["wav", "WAV", "mp3", "MP3", "flac", "FLAC", "m4a", "M4A", "ogg", "OGG", "aiff", "AIFF", "aif", "AIF"]

    audio_files: List[Path] = []
    for ext in extensions:
        audio_files.extend(input_dir.rglob(f"*.{ext}"))
    return sorted(audio_files)


def load_audio_dac_mono(audio_path: Path, sample_rate: int = DAC_SAMPLE_RATE) -> Tuple[np.ndarray, int, int]:
    """
    Load audio as mono at the DAC sample rate.

    Returns
    -------
    audio:
        Resampled mono waveform.
    sr:
        Target sample rate (normally 44100).
    sr_original:
        Original file sample rate reported by librosa.
    """
    audio_path_str = str(audio_path)
    audio_orig, sr_original = librosa.load(audio_path_str, sr=None, mono=True)
    audio, sr = librosa.load(audio_path_str, sr=sample_rate, mono=True)

    # Mirror the slight end-trim behavior from the earlier pipeline when resampling.
    if sr != sr_original and len(audio) > 0:
        trim = min(24, len(audio))
        audio = audio[:-trim] if trim < len(audio) else audio[:0]

    return audio, sr, sr_original


def get_audio_info(audio_path: Path) -> Optional[Dict]:
    """
    Get DAC-oriented audio metadata.

    Frame counts are based on ceil(num_samples / 512), matching DAC preprocess
    behavior for the 44.1 kHz model.
    """
    try:
        audio, sr, _ = load_audio_dac_mono(audio_path, sample_rate=DAC_SAMPLE_RATE)
        duration = len(audio) / sr if sr > 0 else 0.0
        expected_dac_frames = math.ceil(len(audio) / DAC_HOP_LENGTH)
        return {
            "duration": duration,
            "samplerate": sr,
            "frames": len(audio),
            "channels": 1,
            "expected_dac_frames": expected_dac_frames,
            "dac_hop_length": DAC_HOP_LENGTH,
            "dac_fps": DAC_FRAMES_PER_SECOND,
        }
    except Exception as e:  # pragma: no cover - defensive utility
        print(f"❌ Error reading {audio_path}: {e}")
        return None


def get_parameter_names(config: Dict) -> List[str]:
    """Extract parameter names in config order."""
    return [param_info["name"] for param_info in config.values()]


def get_parameter_unit(config: Dict, param_name: str) -> Optional[str]:
    """Get the unit string for a named parameter."""
    for param_info in config.values():
        if param_info["name"] == param_name:
            return param_info.get("unit")
    return None


def expected_dac_frames_from_samples(num_samples: int, hop_length: int = DAC_HOP_LENGTH) -> int:
    """Convert audio sample count to DAC frame count using ceiling division."""
    return math.ceil(num_samples / hop_length)


def audio_samples_from_dac_frames(num_frames: int, hop_length: int = DAC_HOP_LENGTH) -> int:
    """Convert DAC frame count to the corresponding nominal sample count."""
    return int(num_frames * hop_length)


def resample_parameter_series(values: np.ndarray, target_length: int, kind: str = "linear") -> np.ndarray:
    """
    Resample a 1D parameter trajectory to ``target_length``.

    Parameters
    ----------
    values:
        Source sequence of length T.
    target_length:
        Desired number of samples.
    kind:
        "linear" for continuous parameters, "nearest" for categorical indices.
    """
    values = np.asarray(values)

    if target_length <= 0:
        return np.zeros((0,), dtype=values.dtype)
    if len(values) == 0:
        return np.zeros((target_length,), dtype=np.float32)
    if len(values) == target_length:
        return values.copy()
    if len(values) == 1:
        return np.repeat(values, target_length)

    src_x = np.linspace(0.0, 1.0, num=len(values), endpoint=True)
    dst_x = np.linspace(0.0, 1.0, num=target_length, endpoint=True)

    if kind == "nearest":
        src_idx = np.arange(len(values), dtype=np.float64)
        mapped = np.interp(dst_x, src_x, src_idx)
        mapped = np.clip(np.round(mapped).astype(int), 0, len(values) - 1)
        return values[mapped]

    return np.interp(dst_x, src_x, values).astype(np.float32)


def load_dac_artifact(dac_path: Path) -> Dict:
    """
    Load a ``.dac`` file directly from disk.

    The official DAC implementation stores a numpy ``artifacts`` dictionary with
    ``codes`` and ``metadata`` fields in the file.
    """
    dac_path = Path(dac_path)
    artifacts = np.load(dac_path, allow_pickle=True)[()]
    if not isinstance(artifacts, dict) or "codes" not in artifacts:
        raise ValueError(f"Invalid DAC file format: {dac_path}")
    return artifacts


def load_dac_codes(dac_path: Path) -> torch.Tensor:
    """Load discrete DAC codes as a torch tensor of shape [B, N, T]."""
    artifacts = load_dac_artifact(dac_path)
    codes = torch.from_numpy(artifacts["codes"].astype(np.int64))
    if codes.ndim == 2:
        codes = codes.unsqueeze(0)
    return codes


def infer_frames_from_dac(dac_path: Path) -> int:
    """Infer the number of codec frames T from a ``.dac`` file."""
    codes = load_dac_codes(dac_path)
    return int(codes.shape[-1])


def get_dac_metadata(dac_path: Path) -> Dict:
    """Return metadata saved inside a ``.dac`` file."""
    artifacts = load_dac_artifact(dac_path)
    metadata = dict(artifacts.get("metadata", {}))
    metadata["n_codebooks"] = int(load_dac_codes(dac_path).shape[1])
    metadata["frames"] = int(load_dac_codes(dac_path).shape[-1])
    return metadata
