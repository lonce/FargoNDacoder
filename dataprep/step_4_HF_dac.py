"""
Step 4: Create a HuggingFace-style dataset directory for RNNDAC.

This DAC version mirrors the old EnCodec step but points to `.dac` files and
requires `.cond.npy` sidecars. It also saves a central conditioning config.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from datasets import Dataset, DatasetDict
    import datasets
    datasets.disable_progress_bar()
except ImportError:  # pragma: no cover - optional runtime dependency
    Dataset = None
    DatasetDict = None


from .auxiliary_functions_dac import DAC_FRAMES_PER_SECOND


def _require_datasets():
    if Dataset is None or DatasetDict is None:
        raise ImportError("This step requires the 'datasets' package. Install it with: pip install datasets")


def expand_parameters_config(parameters_path: Path) -> Dict:
    """Expand categorical parameters to one-hot feature names."""
    with open(parameters_path, "r", encoding="utf-8") as f:
        original_config = json.load(f)

    expanded_config: Dict[str, Dict] = {}
    feature_list: List[str] = []

    for _, param_info in original_config.items():
        param_name = param_info["name"]
        param_type = param_info.get("type", "continuous")

        if param_type == "continuous":
            expanded_config[param_name] = {
                "type": "continuous",
                "min": param_info.get("min", 0.0),
                "max": param_info.get("max", 1.0),
                "unit": param_info.get("unit", ""),
                "description": f"Continuous parameter: {param_name}",
            }
            feature_list.append(param_name)
        elif param_type == "class":
            classes = param_info.get("classes", [])
            num_classes = len(classes) if classes else param_info.get("num_classes", 2)
            if classes:
                for class_name in classes:
                    feature_name = f"{param_name}_{class_name}"
                    expanded_config[feature_name] = {
                        "type": "binary",
                        "source_parameter": param_name,
                        "class_name": class_name,
                        "unit": "probability",
                        "description": f"One-hot encoding for {param_name} class '{class_name}'",
                    }
                    feature_list.append(feature_name)
            else:
                for i in range(num_classes):
                    feature_name = f"{param_name}_{i}"
                    expanded_config[feature_name] = {
                        "type": "binary",
                        "source_parameter": param_name,
                        "class_index": i,
                        "unit": "probability",
                        "description": f"One-hot encoding for {param_name} class {i}",
                    }
                    feature_list.append(feature_name)

    return {
        "schema_version": 1,
        "fps": DAC_FRAMES_PER_SECOND,
        "feature_names": feature_list,
        "features": expanded_config,
        "num_features": len(feature_list),
    }


def detect_split_structure(tokens_dir: Path) -> Optional[List[str]]:
    """Detect train/validation/test-like folder structure under tokens_dir."""
    tokens_dir = Path(tokens_dir)
    if not (tokens_dir / "train").is_dir():
        return None
    all_splits = [d.name for d in tokens_dir.iterdir() if d.is_dir()]
    return ["train"] + sorted([s for s in all_splits if s != "train"])


def collect_token_files(tokens_dir: Path, suffix: str = ".dac", recursive: bool = True) -> List[Path]:
    """Collect all DAC token files."""
    pattern = f"*{suffix}"
    paths = tokens_dir.rglob(pattern) if recursive else tokens_dir.glob(pattern)
    return sorted([p for p in paths if p.is_file()], key=lambda p: p.as_posix().lower())


def verify_sidecar_files(dac_path: Path) -> Dict[str, bool]:
    """Check required sidecar files for one DAC token file."""
    cond_npy_path = dac_path.with_suffix(".cond.npy")
    return {"npy_exists": cond_npy_path.exists(), "both_exist": cond_npy_path.exists()}


def materialize_files(src_dac: Path, dst_dac: Path, materialize_mode: str = "link"):
    """Link or copy a .dac file and its .cond.npy sidecar into the HF dataset dir."""
    dst_dac.parent.mkdir(parents=True, exist_ok=True)
    files_to_materialize = [
        (src_dac, dst_dac),
        (src_dac.with_suffix(".cond.npy"), dst_dac.with_suffix(".cond.npy")),
        (src_dac.with_suffix(".cond.json"), dst_dac.with_suffix(".cond.json")),
    ]

    for src_file, dst_file in files_to_materialize:
        if not src_file.exists():
            continue
        if materialize_mode == "none":
            continue
        if dst_file.exists() or dst_file.is_symlink():
            dst_file.unlink()
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if materialize_mode == "link":
            try:
                target = os.path.relpath(src_file.resolve(), start=dst_file.parent.resolve())
                os.symlink(target, dst_file)
            except OSError:
                shutil.copy2(src_file, dst_file)
        elif materialize_mode == "copy":
            shutil.copy2(src_file, dst_file)
        else:
            raise ValueError(f"Unknown materialize mode: {materialize_mode}")


def verify_dataset_files(df: pd.DataFrame, output_dir: Path) -> int:
    """Verify that all dataset-referenced DAC files exist."""
    missing_files = []
    for _, row in df.iterrows():
        audio_path = output_dir / row["audio"]
        if not audio_path.exists():
            missing_files.append(str(audio_path))
    if missing_files:
        print(f"⚠️ Found {len(missing_files)} missing files:")
        for missing in missing_files[:10]:
            print(f"   • {missing}")
        if len(missing_files) > 10:
            print(f"   ... and {len(missing_files) - 10} more")
    else:
        print("✅ All DAC files verified successfully")
    return len(missing_files)


def cleanup_dataset_duplicates(output_dir: Path, tokens_subdir: str = "tokens"):
    """Remove and recreate the token subtree inside the HF dataset dir."""
    tokens_dir = output_dir / tokens_subdir
    if tokens_dir.exists():
        shutil.rmtree(tokens_dir)
    tokens_dir.mkdir(parents=True, exist_ok=True)


def create_single_split_dataset(
    tokens_split_dir: Path,
    output_dir: Path,
    split_name: str,
    tokens_subdir: str,
    materialize_mode: str,
):
    """Create one HuggingFace split from a DAC token subtree."""
    _require_datasets()
    token_paths = collect_token_files(tokens_split_dir, suffix=".dac", recursive=True)
    if not token_paths:
        print(f"   ⚠️  No .dac files found in '{split_name}' split")
        return Dataset.from_pandas(pd.DataFrame(columns=["audio"])), {"total_files": 0, "valid_files": 0, "missing_sidecars": 0}

    print(f"   📂 {split_name}: {len(token_paths)} .dac files found")
    rows = []
    valid_files = 0
    missing_sidecars = 0

    for dac_path in token_paths:
        sidecar_status = verify_sidecar_files(dac_path)
        if not sidecar_status["both_exist"]:
            missing_sidecars += 1
            continue
        rel_within_split = dac_path.relative_to(tokens_split_dir)
        rel_path_in_dataset = Path(tokens_subdir) / split_name / rel_within_split
        dst_dac_path = output_dir / rel_path_in_dataset
        materialize_files(dac_path, dst_dac_path, materialize_mode)
        rows.append({"audio": str(rel_path_in_dataset)})
        valid_files += 1

    if missing_sidecars > 0:
        print(f"      ⚠️  {missing_sidecars} files missing sidecars")

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["audio"])
    dataset = Dataset.from_pandas(df, preserve_index=False)
    stats = {"total_files": len(token_paths), "valid_files": valid_files, "missing_sidecars": missing_sidecars}
    return dataset, stats


def create_huggingface_dataset(
    tokens_dir: Path,
    output_dir: Path,
    raw_dir: Optional[Path] = None,
    split_name: str = "train",
    tokens_subdir: str = "tokens",
    materialize_mode: str = "link",
    verify_files: bool = True,
) -> Dict:
    """Create the on-disk HF dataset wrapper around the DAC token tree."""
    _require_datasets()
    tokens_dir = Path(tokens_dir)
    output_dir = Path(output_dir)
    raw_dir = Path(raw_dir) if raw_dir else None

    print("\n\033[1mHUGGINGFACE DATASET CREATION:\033[0m\n")
    print(f"📁 Tokens directory: {tokens_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔗 Materialize mode: {materialize_mode}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_dataset_duplicates(output_dir, tokens_subdir)
    detected_splits = detect_split_structure(tokens_dir)

    if detected_splits:
        print(f"\n🎯 Split structure detected: {', '.join(detected_splits)}")
        datasets_dict = {}
        all_stats = {}
        for split in detected_splits:
            split_dir = tokens_dir / split
            dataset, stats = create_single_split_dataset(split_dir, output_dir, split, tokens_subdir, materialize_mode)
            datasets_dict[split] = dataset
            all_stats[split] = stats
        dataset_dict = DatasetDict(datasets_dict)
        total_files = sum(s["total_files"] for s in all_stats.values())
        total_valid = sum(s["valid_files"] for s in all_stats.values())
        total_missing = sum(s["missing_sidecars"] for s in all_stats.values())
    else:
        print(f"\n📊 No split structure detected")
        print(f"📌 Using all data as '{split_name}' split\n")
        dataset, stats = create_single_split_dataset(tokens_dir, output_dir, split_name, tokens_subdir, materialize_mode)
        dataset_dict = DatasetDict({split_name: dataset})
        total_files = stats["total_files"]
        total_valid = stats["valid_files"]
        total_missing = stats["missing_sidecars"]

    dataset_dict.save_to_disk(str(output_dir))

    if raw_dir:
        parameters_path = raw_dir / "parameters.json"
        if parameters_path.exists():
            expanded_params = expand_parameters_config(parameters_path)
            config_output_path = output_dir / "conditioning_config.json"
            with open(config_output_path, "w", encoding="utf-8") as f:
                json.dump(expanded_params, f, indent=2)
            print(f"\n📋 Saved expanded conditioning config to: {config_output_path.name}")
            print(f"   • Features: {', '.join(expanded_params['feature_names'])}")
            print(f"   • Total features: {expanded_params['num_features']}")

    print(f"\n💾 Saved DatasetDict to: {output_dir}")
    print("📈 Dataset summary:")
    for split, dataset in dataset_dict.items():
        print(f"   • {split}: {len(dataset)} samples")

    missing_count = 0
    if verify_files and total_valid > 0:
        print("\n🔍 Verifying dataset files...")
        all_rows = []
        for dataset in dataset_dict.values():
            if len(dataset) > 0:
                all_rows.extend(dataset.to_pandas()["audio"].tolist())
        if all_rows:
            df_verify = pd.DataFrame({"audio": all_rows})
            missing_count = verify_dataset_files(df_verify, output_dir)

    return {
        "splits_detected": detected_splits if detected_splits else [split_name],
        "total_dac_files": total_files,
        "valid_files": total_valid,
        "missing_sidecars": total_missing,
        "missing_files": missing_count,
        "output_dir": str(output_dir),
    }


def quick_create_dataset(dataset_dir: str, split_name: str = "train", materialize_mode: str = "link") -> Dict:
    dataset_path = Path(dataset_dir)
    tokens_dir = dataset_path / "tokens"
    output_dir = dataset_path / "hf_dataset"
    raw_dir = dataset_path / "raw"
    if not tokens_dir.exists():
        raise FileNotFoundError(f"Tokens directory not found: {tokens_dir}")
    return create_huggingface_dataset(
        tokens_dir=tokens_dir,
        output_dir=output_dir,
        raw_dir=raw_dir if raw_dir.exists() else None,
        split_name=split_name,
        tokens_subdir="tokens",
        materialize_mode=materialize_mode,
        verify_files=True,
    )


def quick_load_dataset(dataset_dir: str):
    from datasets import load_from_disk
    dataset_path = Path(dataset_dir)
    hf_dataset_dir = dataset_path / "hf_dataset"
    if not hf_dataset_dir.exists():
        raise FileNotFoundError(f"HuggingFace dataset not found: {hf_dataset_dir}")
    return load_from_disk(str(hf_dataset_dir))
