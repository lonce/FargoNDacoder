from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .auxiliary_functions_dac import (
    DAC_FRAMES_PER_SECOND,
    DAC_HOP_LENGTH,
    DAC_SAMPLE_RATE,
    find_audio_csv_pairs,
    get_audio_info,
    get_parameter_names,
    get_parameter_unit,
    load_audio_dac_mono,
    load_parameter_config,
)


def summarize_dataset(raw_dir: Path) -> Dict:
    """Summarize the raw dataset for the DAC pipeline."""
    raw_dir = Path(raw_dir)
    config_path = raw_dir / "parameters.json"
    if not config_path.exists():
        raise FileNotFoundError(f"parameters.json not found in {raw_dir}")

    config = load_parameter_config(config_path)
    param_names = [param_info["name"] for param_info in config.values()]
    pairs = find_audio_csv_pairs(raw_dir)
    if not pairs:
        raise ValueError(f"No audio-CSV pairs found in {raw_dir}")

    summary = {
        "total_files": len(pairs),
        "parameters": param_names,
        "files": [],
        "total_audio_duration": 0.0,
        "total_csv_rows": 0,
        "dac_fps": DAC_FRAMES_PER_SECOND,
        "dac_hop_length": DAC_HOP_LENGTH,
    }

    for audio_path, csv_path in pairs:
        audio_info = get_audio_info(audio_path)
        if audio_info is None:
            continue

        df = pd.read_csv(csv_path)
        csv_rows = len(df)
        duration = audio_info["duration"]
        csv_fps = (csv_rows / duration) if duration > 0 else 0.0
        fps_ratio = (csv_fps / DAC_FRAMES_PER_SECOND) if DAC_FRAMES_PER_SECOND > 0 else 0.0

        file_info = {
            "name": audio_path.stem,
            "audio_duration": duration,
            "audio_samplerate": audio_info["samplerate"],
            "audio_samples": audio_info["frames"],
            "expected_dac_frames": audio_info["expected_dac_frames"],
            "csv_rows": csv_rows,
            "csv_fps": csv_fps,
            "fps_ratio_to_dac": fps_ratio,
            "row_difference": csv_rows - audio_info["expected_dac_frames"],
        }
        summary["files"].append(file_info)
        summary["total_audio_duration"] += duration
        summary["total_csv_rows"] += csv_rows

    return summary


def plot_parameter_patterns(raw_dir: Path, file_name: Optional[str] = None):
    """Plot parameter trajectories from a selected raw CSV file."""
    raw_dir = Path(raw_dir)
    config = load_parameter_config(raw_dir / "parameters.json")
    param_names = get_parameter_names(config)

    if file_name is None:
        pairs = find_audio_csv_pairs(raw_dir)
        if not pairs:
            raise ValueError("No audio-CSV pairs found")
        audio_path, csv_path = random.choice(pairs)
        file_name = audio_path.stem
    else:
        csv_path = raw_dir / f"{file_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    n_params = len(param_names)
    cols = min(3, max(1, n_params))
    rows = (n_params + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    fig.suptitle(f"Parameter Trajectories: {file_name}", fontsize=16, fontweight="bold")

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        if param_name in df.columns:
            values = df[param_name].to_numpy()
            ax.plot(values, linewidth=1.5)
            ax.set_title(param_name)
            ax.set_xlabel("CSV Row Index")
            unit = get_parameter_unit(config, param_name)
            ax.set_ylabel(f"Value ({unit})" if unit else "Value")
            ax.grid(True, alpha=0.3)
            ax.axhline(np.mean(values), color="red", linestyle="--", alpha=0.7, label=f"Mean: {np.mean(values):.2f}")
            ax.legend()
        else:
            ax.text(0.5, 0.5, f'Parameter "{param_name}"\nnot found in CSV', ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{param_name} (Missing)")

    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_sample(raw_dir: Path, file_name: Optional[str] = None):
    """
    Plot raw audio and raw CSV trajectories over a shared duration axis.

    Unlike the old EnCodec version, this DAC-oriented plot does not assume that
    the CSV row count already matches codec frame count. Instead, the CSV is
    stretched across the file duration so you can visually inspect whether the
    annotation sampling density looks reasonable before sidecar resampling.
    """
    raw_dir = Path(raw_dir)
    pairs = find_audio_csv_pairs(raw_dir)
    if not pairs:
        raise ValueError("No audio-CSV pairs found")

    if file_name is None:
        audio_path, csv_path = random.choice(pairs)
        file_name = audio_path.stem
        print(f"📊 Randomly selected file: {file_name}")
    else:
        audio_path = None
        csv_path = None
        for ap, cp in pairs:
            if ap.stem == file_name:
                audio_path, csv_path = ap, cp
                break
        if audio_path is None or csv_path is None:
            raise FileNotFoundError(f"File '{file_name}' not found in dataset")

    audio, sr, _ = load_audio_dac_mono(audio_path, sample_rate=DAC_SAMPLE_RATE)
    duration = len(audio) / sr if sr > 0 else 0.0
    df = pd.read_csv(csv_path)
    config = load_parameter_config(raw_dir / "parameters.json")
    param_names = get_parameter_names(config)

    audio_time = np.linspace(0, duration, len(audio), endpoint=False) if len(audio) > 0 else np.array([])
    csv_time = np.linspace(0, duration, len(df), endpoint=False) if len(df) > 0 else np.array([])

    n_params = len(param_names)
    fig, axes = plt.subplots(n_params + 1, 1, figsize=(15, 3 * (n_params + 1)))
    fig.suptitle(
        f"File: {file_name} ({duration:.2f}s, DAC target fps={DAC_FRAMES_PER_SECOND:.3f})",
        fontsize=16,
        fontweight="bold",
    )

    axes[0].plot(audio_time, audio, color="blue", alpha=0.7, linewidth=0.5)
    axes[0].set_title("Audio Waveform (44.1 kHz mono for DAC)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    for i, param_name in enumerate(param_names):
        ax = axes[i + 1]
        if param_name in df.columns:
            values = df[param_name].to_numpy()
            ax.plot(csv_time, values, color="red", linewidth=1.5, alpha=0.8)
            ax.set_title(f"Parameter: {param_name}")
            unit = get_parameter_unit(config, param_name)
            ax.set_ylabel(unit if unit else "Value")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Parameter "{param_name}" not found', ha="center", va="center", transform=ax.transAxes)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()


def analyze_dataset(raw_dir: Path):
    """Print a DAC-oriented raw-data summary."""
    raw_dir = Path(raw_dir)
    print("\033[1mDATASET SUMMARY (DAC / 44.1 kHz):\033[0m\n")
    summary = summarize_dataset(raw_dir)

    print(f"✅ Found {summary['total_files']} audio-CSV pairs")
    print(f"📁 Parameters: {', '.join(summary['parameters'])}")
    print(f"⏱️ Total audio duration: {summary['total_audio_duration']:.1f} seconds")
    print(f"🎛️ DAC sample rate: {DAC_SAMPLE_RATE} Hz")
    print(f"🎛️ DAC hop length: {DAC_HOP_LENGTH} samples")
    print(f"🎛️ DAC frame rate: {DAC_FRAMES_PER_SECOND:.6f} fps")

    print("\n\033[1mFILE DETAILS:\033[0m\n")
    for file_info in summary["files"]:
        row_diff = file_info["row_difference"]
        status = "≈" if abs(row_diff) <= 1 else "•"
        print(
            f"{status} {file_info['name']}: {file_info['audio_duration']:.2f}s, "
            f"CSV rows={file_info['csv_rows']}, target DAC frames={file_info['expected_dac_frames']}, "
            f"CSV fps≈{file_info['csv_fps']:.3f}"
        )

    print(
        "\nℹ️  In the DAC pipeline, CSV rows do not need to match DAC frame count exactly; "
        "step 3 resamples parameter trajectories onto DAC frames."
    )
    return summary


def interactive_file_selector(raw_dir):
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output
    except ImportError:
        print("❌ This function requires ipywidgets. Install with: pip install ipywidgets")
        return None

    print("\n\033[1mFILE PLOTTER:\033[0m\n")
    raw_dir = Path(raw_dir)
    pairs = find_audio_csv_pairs(raw_dir)
    if not pairs:
        print("❌ No audio-CSV pairs found in dataset")
        return None

    file_names = [audio_path.stem for audio_path, _ in pairs]
    file_dropdown = widgets.Dropdown(options=file_names, description="Select File:", style={"description_width": "100px"})
    plot_button = widgets.Button(description="📊 Plot File", button_style="success", tooltip="Click to plot the selected file")
    output = widgets.Output()

    def on_plot_button_clicked(_):
        with output:
            clear_output(wait=True)
            selected_file = file_dropdown.value
            print(f"📈 Plotting file: {selected_file}")
            plot_sample(raw_dir, file_name=selected_file)

    plot_button.on_click(on_plot_button_clicked)
    return widgets.VBox([widgets.HBox([file_dropdown, plot_button]), output])


def quick_analyze(dataset_path, individual_plot_selector=False):
    raw_dir = Path(dataset_path) / "raw"
    analyze_dataset(raw_dir)
    if individual_plot_selector:
        from IPython.display import display
        ui = interactive_file_selector(raw_dir)
        if ui is not None:
            display(ui)
