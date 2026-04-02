"""
Step 2: DAC encoding for dataset pipeline.

This step is intentionally thin: it uses DAC's own file-oriented compression API
instead of reimplementing chunking, normalization, or quantization behavior.

Defaults:
- no chunking by default      -> win_duration=None
- no normalization by default -> normalize_db=None
- use DAC default n_q=9       -> n_quantizers=9

The user may still override any of these.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
from tqdm import tqdm

from .auxiliary_functions_dac import find_audio_files


def _load_dac_model(model_type: str = "44khz", device: str = "cpu"):
    """Load the DAC model from the installed descript-audio-codec package."""
    try:
        from dac import DAC
        from dac.utils import download
    except ImportError as e:
        raise ImportError(
            "Could not import DAC. Make sure descript-audio-codec is installed "
            "in the current environment."
        ) from e

    model_key = model_type.lower().replace("_", "").replace("-", "")
    if model_key not in {"44khz", "44k", "44100"}:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. This pipeline currently supports only the 44.1 kHz DAC model."
        )

    model_path = download(model_type="44khz")
    model = DAC.load(model_path)
    model.to(device)
    model.eval()
    return model


def _expected_out_path(in_dir: Path, out_dir: Path, audio_path: Path) -> Path:
    """Preserve relative structure and save with .dac suffix."""
    rel = audio_path.relative_to(in_dir)
    return (out_dir / rel).with_suffix(".dac")


def _encode_file(
    audio_path: Path,
    out_path: Path,
    model,
    *,
    overwrite: bool = False,
    win_duration: Optional[float] = None,
    normalize_db: Optional[float] = None,
    n_quantizers: Optional[int] = 9,
) -> tuple[bool, str]:
    """Encode one audio file using DAC's native compress API."""
    try:
        if out_path.exists() and not overwrite:
            return True, "exists"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            model.compress(
                audio_path_or_signal=str(audio_path),
                win_duration=win_duration,
                normalize_db=normalize_db,
                n_quantizers=n_quantizers,
            ).save(out_path)

        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def encode_dataset(
    input_folder: Path,
    output_folder: Path,
    model_type: str = "44khz",
    device: str = "cpu",
    overwrite: bool = False,
    win_duration: Optional[float] = None,
    normalize_db: Optional[float] = None,
    n_quantizers: Optional[int] = 9,
) -> Dict:
    """
    Encode all audio files in a dataset directory to DAC .dac files.

    Args:
        input_folder: Directory containing normalized audio files.
        output_folder: Directory where .dac files will be written.
        model_type: DAC model selector. Currently only '44khz' is supported.
        device: Torch device string, e.g. 'cpu' or 'cuda'.
        overwrite: Whether to overwrite existing .dac files.
        win_duration: Optional DAC chunk/window duration in seconds.
            None means let DAC encode the full file without forced chunking.
        normalize_db: Optional DAC pre-encode loudness normalization target.
            None means do not normalize in step 2.
        n_quantizers: Number of DAC quantizers/codebooks to use.
            Default is 9, matching DAC's default for the pretrained 44 kHz model.

    Returns:
        Dictionary with processing stats.
    """
    print("\n\033[1mDATA PROCESSING:\033[0m\n")

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files(input_folder)
    if not audio_files:
        print(f"No audio files found in {input_folder}")
        return {"total": 0, "success": 0, "skipped": 0, "failed": 0, "errors": []}

    print(f"Found {len(audio_files)} audio files under {input_folder}")
    print(f"Using device: {device}")
    print(f"DAC model: {model_type}")
    print(f"win_duration: {win_duration}")
    print(f"normalize_db: {normalize_db}")
    print(f"n_quantizers: {n_quantizers}")

    model = _load_dac_model(model_type=model_type, device=device)

    ok, skipped, failed = 0, 0, 0
    errors = []

    for audio_file in tqdm(audio_files, desc="Encoding", unit="file"):
        out_path = _expected_out_path(input_folder, output_folder, audio_file)
        success, msg = _encode_file(
            audio_file,
            out_path,
            model,
            overwrite=overwrite,
            win_duration=win_duration,
            normalize_db=normalize_db,
            n_quantizers=n_quantizers,
        )

        if success:
            if msg == "exists":
                skipped += 1
            else:
                ok += 1
        else:
            failed += 1
            errors.append((audio_file, msg))

    if errors:
        print(f"\nShowing first {min(10, len(errors))} errors:")
        for audio_path, msg in errors[:10]:
            print(f"  ERR {audio_path.name}: {msg}")

    return {
        "total": len(audio_files),
        "success": ok,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
    }


def quick_encode(
    dataset_path,
    model_type: str = "44khz",
    device: str = "cpu",
    overwrite: bool = False,
    win_duration: Optional[float] = None,
    normalize_db: Optional[float] = None,
    n_quantizers: Optional[int] = 9,
):
    """Notebook-friendly wrapper for DAC encoding."""
    dataset_path = Path(dataset_path)
    input_dir = dataset_path / "normalized"
    output_dir = dataset_path / "tokens"

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Normalized folder not found: {input_dir}\n"
            f"Please run step_1_normalization_dac first."
        )

    return encode_dataset(
        input_folder=input_dir,
        output_folder=output_dir,
        model_type=model_type,
        device=device,
        overwrite=overwrite,
        win_duration=win_duration,
        normalize_db=normalize_db,
        n_quantizers=n_quantizers,
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode a dataset to DAC .dac files")
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--model-type", type=str, default="44khz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--win-duration",
        type=float,
        default=None,
        help="Optional DAC chunk/window duration in seconds. Default: None (no forced chunking).",
    )
    parser.add_argument(
        "--normalize-db",
        type=float,
        default=None,
        help="Optional DAC normalization target in dB. Default: None (no step-2 normalization).",
    )
    parser.add_argument(
        "--n-quantizers",
        type=int,
        default=9,
        help="Number of DAC quantizers/codebooks to use. Default: 9.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    encode_dataset(
        args.input_folder,
        args.output_folder,
        model_type=args.model_type,
        device=args.device,
        overwrite=args.overwrite,
        win_duration=args.win_duration,
        normalize_db=args.normalize_db,
        n_quantizers=args.n_quantizers,
    )
