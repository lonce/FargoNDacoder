"""
Step 3: Create DAC sidecar files (.cond.npy).

Key DAC-oriented behavior:
- reads frame count from .dac files
- resamples raw CSV parameter trajectories to DAC frame count
- continuous parameters use linear interpolation
- class parameters use nearest-neighbor resampling before one-hot encoding

This is the main conceptual difference from the earlier EnCodec pipeline, where
CSV rows were expected to already match codec frame count.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .auxiliary_functions_dac import (
    DAC_FRAMES_PER_SECOND,
    find_audio_csv_pairs,
    infer_frames_from_dac,
    load_parameter_config,
    resample_parameter_series,
)


def normalize_parameter_values(values: np.ndarray, param_info: Dict, param_name: str) -> np.ndarray:
    """Normalize continuous parameters or validate categorical indices."""
    param_type = param_info.get("type", "continuous")

    if param_type == "continuous":
        min_val = param_info.get("min", 0.0)
        max_val = param_info.get("max", 1.0)
        clipped_values = np.clip(values.astype(np.float32), min_val, max_val)
        if max_val != min_val:
            normalized = (clipped_values - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(clipped_values)
        return normalized.astype(np.float32)

    if param_type == "class":
        classes = param_info.get("classes", [])
        if classes:
            class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
            num_classes = len(classes)
            class_indices = []
            warning_count = 0
            for value in values:
                if isinstance(value, str) and value in class_to_idx:
                    class_indices.append(class_to_idx[value])
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    idx = int(value)
                    if 0 <= idx < num_classes:
                        class_indices.append(idx)
                    else:
                        warning_count += 1
                        class_indices.append(0)
                else:
                    warning_count += 1
                    class_indices.append(0)
            if warning_count > 0:
                print(f"   • ⚠️  {warning_count} invalid class values for '{param_name}', defaulted to '{classes[0]}'")
            return np.asarray(class_indices, dtype=int)

        num_classes = param_info.get("num_classes", 2)
        class_values = np.asarray(values, dtype=int)
        return np.clip(class_values, 0, num_classes - 1)

    raise ValueError(f"Unknown parameter type '{param_type}' for parameter '{param_name}'")


def create_one_hot_encoding(class_values: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer class indices to one-hot rows."""
    class_values = np.asarray(class_values, dtype=int)
    one_hot = np.zeros((len(class_values), num_classes), dtype=np.float32)
    for i, class_idx in enumerate(class_values):
        if 0 <= class_idx < num_classes:
            one_hot[i, class_idx] = 1.0
    return one_hot


def create_sidecar_files(dac_path: Path, csv_path: Path, config: Dict, tokens_dir: Path) -> bool:
    """Create ``.cond.npy`` for one ``.dac`` file using CSV→DAC-frame resampling."""
    try:
        t_dac = infer_frames_from_dac(dac_path)
        df = pd.read_csv(csv_path)
        t_csv = len(df)

        print(f"📄 {dac_path.stem} sidecar:")
        print(f"    DAC frames: {t_dac}")
        print(f"    CSV rows:   {t_csv}")
        print(f"    DAC fps:    {DAC_FRAMES_PER_SECOND:.6f}")

        feature_columns = []
        feature_names = []
        feature_metadata = {}

        for _, param_info in config.items():
            param_name = param_info["name"]
            param_type = param_info.get("type", "continuous")

            if param_name not in df.columns:
                print(f"    ⚠️ Parameter '{param_name}' not found in CSV, skipping")
                continue

            values = df[param_name].to_numpy()

            if param_type == "continuous":
                values_resampled = resample_parameter_series(values.astype(np.float32), t_dac, kind="linear")
                normalized_values = normalize_parameter_values(values_resampled, param_info, param_name)
                feature_columns.append(normalized_values.reshape(-1, 1))
                feature_names.append(param_name)
                feature_metadata[param_name] = {
                    "min": float(np.min(normalized_values)) if len(normalized_values) else 0.0,
                    "max": float(np.max(normalized_values)) if len(normalized_values) else 0.0,
                    "mean": float(np.mean(normalized_values)) if len(normalized_values) else 0.0,
                    "std": float(np.std(normalized_values)) if len(normalized_values) else 0.0,
                    "units": param_info.get("unit", ""),
                    "doc_string": f"Normalized {param_name} parameter resampled to DAC frame times",
                }

            elif param_type == "class":
                class_values = normalize_parameter_values(values, param_info, param_name)
                class_values_resampled = resample_parameter_series(class_values, t_dac, kind="nearest").astype(int)
                classes = param_info.get("classes", [])
                num_classes = len(classes) if classes else param_info.get("num_classes", 2)
                one_hot = create_one_hot_encoding(class_values_resampled, num_classes)
                feature_columns.append(one_hot)

                class_names = [f"{param_name}_{c}" for c in classes] if classes else [f"{param_name}_{i}" for i in range(num_classes)]
                feature_names.extend(class_names)
                for i, class_name in enumerate(class_names):
                    class_col = one_hot[:, i]
                    actual_class_name = class_name.replace(f"{param_name}_", "") if "_" in class_name else str(i)
                    feature_metadata[class_name] = {
                        "min": float(np.min(class_col)) if len(class_col) else 0.0,
                        "max": float(np.max(class_col)) if len(class_col) else 0.0,
                        "mean": float(np.mean(class_col)) if len(class_col) else 0.0,
                        "std": float(np.std(class_col)) if len(class_col) else 0.0,
                        "units": "probability",
                        "doc_string": f"One-hot encoding for {param_name} class {actual_class_name} resampled to DAC frame times",
                    }
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")

        conditioning_matrix = np.concatenate(feature_columns, axis=1) if feature_columns else np.zeros((t_dac, 0), dtype=np.float32)
        conditioning_matrix = conditioning_matrix.astype(np.float32)

        cond_npy_path = dac_path.with_suffix(".cond.npy")
        cond_json_path = dac_path.with_suffix(".cond.json")

        metadata = {
            "schema_version": 1,
            "fps": DAC_FRAMES_PER_SECOND,
            "source_rate": "raw_csv_resampled_to_dac_frames",
            "names": feature_names,
            "features": feature_metadata,
            "norm": {
                "min": [feature_metadata[name]["min"] for name in feature_names],
                "max": [feature_metadata[name]["max"] for name in feature_names],
                "mean": [feature_metadata[name]["mean"] for name in feature_names],
                "std": [feature_metadata[name]["std"] for name in feature_names],
            },
        }

        temp_npy = cond_npy_path.with_suffix(".tmp.npy")
        np.save(temp_npy, conditioning_matrix)
        temp_npy.rename(cond_npy_path)

        # Save per-file JSON for debugging / inspection.
        temp_json = cond_json_path.with_suffix(".tmp.json")
        with open(temp_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        temp_json.rename(cond_json_path)

        print(f"    ✅ Sidecar created ({t_dac} frames, {len(feature_names)} features)\n")
        return True
    except Exception as e:
        print(f"\n📄 {dac_path.stem} sidecar:")
        print(f"   • ❌ Error: {e}")
        return False


def create_sidecars_dataset(tokens_dir: Path, raw_dir: Path) -> Dict:
    """Create sidecars for all ``.dac`` files in the tokens directory."""
    print("\n\033[1mSIDECAR CREATION:\033[0m\n")
    tokens_dir = Path(tokens_dir)
    raw_dir = Path(raw_dir)

    config_path = raw_dir / "parameters.json"
    if not config_path.exists():
        raise FileNotFoundError(f"parameters.json not found in {raw_dir}")
    config = load_parameter_config(config_path)

    dac_files = list(tokens_dir.rglob("*.dac"))
    if not dac_files:
        raise ValueError(f"No .dac files found in {tokens_dir}")

    dac_csv_pairs = []
    for dac_path in dac_files:
        rel_path = dac_path.relative_to(tokens_dir)
        csv_path = raw_dir / rel_path.with_suffix(".csv")
        if csv_path.exists():
            dac_csv_pairs.append((dac_path, csv_path))
        else:
            print(f"⚠️ No CSV found for {dac_path.name}, skipping")

    if not dac_csv_pairs:
        raise ValueError("No matching CSV files found for .dac files")

    print(f"📁 Tokens directory: {tokens_dir}")
    print(f"📁 Raw directory:    {raw_dir}")
    print(f"📊 Found {len(dac_csv_pairs)} .dac-CSV pairs")
    print(f"🎛️ Parameters:      {', '.join([info['name'] for info in config.values()])}")

    success_count = 0
    print("\n\033[1mSIDECARS SUMMARY:\033[0m\n")
    for dac_path, csv_path in dac_csv_pairs:
        if create_sidecar_files(dac_path, csv_path, config, tokens_dir):
            success_count += 1

    print(f"✅ Successfully created {success_count}/{len(dac_csv_pairs)} sidecar pairs")
    if success_count < len(dac_csv_pairs):
        print(f"❌ Failed to create {len(dac_csv_pairs) - success_count} sidecar pairs")

    return {
        "total": len(dac_csv_pairs),
        "success": success_count,
        "failed": len(dac_csv_pairs) - success_count,
    }


def validate_sidecars(tokens_dir: Path) -> Dict:
    """Validate that ``.cond.npy`` row count matches DAC frame count."""
    tokens_dir = Path(tokens_dir)
    dac_files = list(tokens_dir.rglob("*.dac"))
    results = {
        "total_dac": len(dac_files),
        "valid_sidecars": 0,
        "missing_sidecars": 0,
        "alignment_errors": 0,
        "issues": [],
    }

    for dac_path in dac_files:
        cond_npy_path = dac_path.with_suffix(".cond.npy")
        if not cond_npy_path.exists():
            results["missing_sidecars"] += 1
            results["issues"].append(f"Missing sidecar .npy file for {dac_path.name}")
            continue

        try:
            t_dac = infer_frames_from_dac(dac_path)
            cond_array = np.load(cond_npy_path)
            if cond_array.shape[0] != t_dac:
                results["alignment_errors"] += 1
                results["issues"].append(f"Frame mismatch in {dac_path.name}: dac={t_dac}, sidecar={cond_array.shape[0]}")
                continue
            results["valid_sidecars"] += 1
        except Exception as e:
            results["alignment_errors"] += 1
            results["issues"].append(f"Error validating {dac_path.name}: {e}")

    return results


def quick_create_sidecars(dataset_dir: str) -> Dict:
    dataset_path = Path(dataset_dir)
    tokens_dir = dataset_path / "tokens"
    raw_dir = dataset_path / "raw"
    return create_sidecars_dataset(tokens_dir, raw_dir)
