#!/usr/bin/env python3
"""
sweep_pitch_tracking.py — Hyperparameter grid over:

  Train:  zeroed_latent_proportion × tau
  Eval:   infer_top_k × infer_temperature

Results saved to  dev_notebooks/output/sweep_<timestamp>/
Summary CSV + per-run pitch plots.
"""

import csv, json, os, random, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import dac
import librosa
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.rnndac_dataset import RNNDACLatentDataset, LatentDatasetConfig
from core.rnndac_model import GRUModelConfig, RNNDACModel
from core.inference import infer_streaming_with_lookahead
from utils.io import save_checkpoint

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HF_DATASET_PATH = "/slowdisk/data/DAC/pitchglidesATriangle5octaves/Aglides/hf_dataset"
OUTPUT_BASE = Path(__file__).resolve().parent / "output" / "sweep"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 44100
FRAME_SAMPLES = 512
PARAM_SR = SR / FRAME_SAMPLES  # ~86.13
N_FRAMES = 900

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
TRAIN_GRID = [
    # (zeroed_latent_proportion, training_tau)
    (0.0,  0.4),
    (0.1,  0.4),
    (0.3,  0.4),
    (0.0,  0.7),
    (0.1,  0.7),
    (0.3,  0.7),
]

EVAL_GRID = [
    # (top_k, temperature)
    (5, 0.5),
    (5, 0.9),
    (20, 0.5),
    (20, 0.9),
]

TRAIN_PARAMS = {
    "n_steps": 15001,
    "print_every": 3000,
    "validate_every": 3000,
    "val_batches": 8,
    "checkpoint_every": 15000,
    "cond_injection": "concat",
    "inp_proportion": 5,
    "cond_proportion": 2,
    "tf_schedule": [5000, 5000],
    "batch_size": 16,
    "learning_rate": 1e-3,
    "grad_clip": 1.0,
}

N_Q = 9
CODEBOOK_SIZE = 1024
CODEBOOK_DIM = 8
INPUT_SIZE = N_Q * CODEBOOK_DIM  # 72
CLAMP_VAL = 12

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_datasets():
    train_cfg = LatentDatasetConfig(
        dataset_path=HF_DATASET_PATH,
        sequence_length=128,
        n_q=N_Q,
        clamp_val=CLAMP_VAL,
        files_per_sequence=4,
        add_noise=True,
        noise_weight=0.05,
    )
    val_cfg = LatentDatasetConfig(
        dataset_path=HF_DATASET_PATH,
        sequence_length=128,
        n_q=N_Q,
        clamp_val=CLAMP_VAL,
        files_per_sequence=1,
    )
    ds = RNNDACLatentDataset(train_cfg, split="train", device="cpu")
    val_ds = RNNDACLatentDataset(val_cfg, split="validation", device="cpu")
    loader = DataLoader(ds, batch_size=TRAIN_PARAMS["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_PARAMS["val_batches"], shuffle=False, num_workers=0)
    return loader, val_loader


def make_model(training_tau):
    cfg = GRUModelConfig(
        n_q=N_Q,
        codebook_size=CODEBOOK_SIZE,
        codebook_dim=CODEBOOK_DIM,
        input_size=INPUT_SIZE,
        cond_size=1,
        clamp_val=CLAMP_VAL,
        hidden_size=128,
        num_layers=3,
        cond_injection=TRAIN_PARAMS["cond_injection"],
        inp_proportion=TRAIN_PARAMS["inp_proportion"],
        cond_proportion=TRAIN_PARAMS["cond_proportion"],
        tau=training_tau,
    )
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz")).to(DEVICE)
    dac_model.eval()
    model = RNNDACModel(cfg, dac_model=dac_model).to(DEVICE)
    return model, dac_model

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(model, loader, val_loader, zeroed_prop, training_tau, out_dir, run_name):
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_PARAMS["learning_rate"])
    n_steps = TRAIN_PARAMS["n_steps"]
    tf_on, tf_off = TRAIN_PARAMS["tf_schedule"]
    tf_cycle = tf_on + tf_off

    train_loss_hist = []
    val_loss_hist = []
    train_iter = iter(loader)
    val_iter = iter(val_loader)

    for step in range(n_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        model.train()

        # z e r o e d   l a t e n t s
        if random.random() < zeroed_prop:
            clean_latents = torch.zeros_like(batch["latents"])
        else:
            clean_latents = batch["latents"]

        # teacher-forcing schedule
        if tf_cycle > 0:
            in_tf = (step % tf_cycle) < tf_on
            cascade = "teacher" if in_tf else "free"
        else:
            cascade = "free"

        out = model(
            latents=clean_latents,
            cond=batch["cond"],
            target_codes=batch["targets"],
            cascade_mode=cascade,
        )
        loss = model.compute_loss(out["predicted_logits_per_codebook"], batch["targets"])["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_PARAMS["grad_clip"])
        optimizer.step()

        train_loss_hist.append(loss.item())

        # validation
        if step % TRAIN_PARAMS["validate_every"] == 0 and step > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(TRAIN_PARAMS["val_batches"]):
                    try:
                        vb = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        vb = next(val_iter)
                    vb = {k: v.to(DEVICE) for k, v in vb.items()}
                    vo = model(latents=vb["latents"], cond=vb["cond"], cascade_mode="free")
                    vl = model.compute_loss(vo["predicted_logits_per_codebook"], vb["targets"])["total_loss"].item()
                    val_losses.append(vl)
            avg_val = sum(val_losses) / len(val_losses)
            val_loss_hist.append((step, avg_val))

            recent = train_loss_hist[-TRAIN_PARAMS["print_every"]:]
            avg_train = sum(recent) / len(recent) if recent else loss.item()
            print(f"[{run_name}] Step {step:4d} | Train: {loss.item():.4f} | Avg: {avg_train:.4f} | Val: {avg_val:.4f}")

    # plot loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_loss_hist, label="train", alpha=0.7)
    if val_loss_hist:
        vs, vv = zip(*val_loss_hist)
        ax.plot(vs, vv, "o-", label="val")
    ax.set_xlabel("Step"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{run_name} — training & validation loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(str(out_dir / "loss.png"), dpi=150)
    plt.close(fig)

    return train_loss_hist, val_loss_hist


# ---------------------------------------------------------------------------
# Evaluation: generate + pitch analysis
# ---------------------------------------------------------------------------
def evaluate_pitch(model, dac_model, top_k, temperature, out_dir, run_name):
    cond_seq = torch.linspace(0.05, 0.95, N_FRAMES).unsqueeze(0).unsqueeze(-1).to(DEVICE)

    model.eval()
    with torch.no_grad():
        gen_audio = infer_streaming_with_lookahead(
            rnn_model=model,
            dac_model=dac_model,
            cond_sequence=cond_seq,
            chunk_size=16,
            hop_size=8,
            right_context=4,
            frame_samples=FRAME_SAMPLES,
            tau=temperature,
            top_k=top_k,
        )
    audio_np = gen_audio.squeeze().detach().cpu().numpy()

    # pitch extraction
    f0 = librosa.yin(audio_np, fmin=55, fmax=2000, sr=SR)
    midi = 69 + 12 * np.log2(np.clip(f0, 1e-3, None) / 440.0)
    midi = midi[:len(cond_seq.squeeze())]  # truncate to cond length

    cond_np = cond_seq.squeeze().cpu().numpy()
    mask = (midi >= 33) & (midi <= 93) & np.isfinite(midi)
    valid_frac = mask.sum() / len(mask)

    if mask.sum() > 5:
        cond_norm = cond_np[mask]
        midi_norm = (midi[mask] - 33) / (93 - 33)
        r = np.corrcoef(cond_norm, midi_norm)[0, 1]
        r2 = r ** 2
    else:
        r, r2 = 0.0, 0.0

    cond_midi = cond_np * (93 - 33) + 33

    # plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # top: pitch tracking
    t_ax = np.arange(len(cond_np))
    ax1.plot(t_ax, midi, "b-", label="Extracted MIDI", alpha=0.8)
    ax1.set_ylabel("MIDI pitch", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1b = ax1.twinx()
    ax1b.plot(t_ax, cond_midi, "r-", label="Target MIDI", alpha=0.6)
    ax1b.set_ylabel("Target MIDI", color="r")
    ax1b.tick_params(axis="y", labelcolor="r")
    ax1.set_title(f"{run_name}  |  top_k={top_k}  temp={temperature}  |  R² = {r2:.4f}")
    ax1.grid(True, alpha=0.3)

    # bottom: ideal ramp overlay
    ideal = np.linspace(0.05, 0.95, len(cond_np))
    ax2.plot(t_ax, ideal, "k--", alpha=0.5, label="Ideal ramp")
    ax2.plot(t_ax, (midi - 33) / (93 - 33), "g-", linewidth=1.0, alpha=0.8,
             label=f"Measured (R²={r2:.4f})")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Frame"); ax2.set_ylabel("Normalized pitch")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_title("Pitch vs ideal ramp")

    fig.tight_layout()
    stem = f"pitch_topk{top_k}_temp{temperature}"
    fig.savefig(str(out_dir / f"{stem}.png"), dpi=150)
    plt.close(fig)

    # save audio
    sf.write(str(out_dir / f"{stem}.wav"), audio_np, SR)

    return {
        "pitch_r2": r2,
        "pitch_r": r,
        "valid_frac": valid_frac,
        "top_k": top_k,
        "temperature": temperature,
        "zeroed_prop": None,
        "training_tau": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = OUTPUT_BASE / f"{timestamp}_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    loader, val_loader = make_datasets()

    all_rows = []
    trained_configs = []

    for zeroed_prop, training_tau in TRAIN_GRID:
        run_name = f"zp{zeroed_prop}_tau{training_tau}"
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"TRAINING: zeroed_prop={zeroed_prop}  tau={training_tau}")
        print(f"{'='*60}")

        model, dac_model = make_model(training_tau)
        train_model(model, loader, val_loader, zeroed_prop, training_tau, run_dir, run_name)

        # evaluate all inference configs
        for top_k, temperature in EVAL_GRID:
            print(f"  EVAL: top_k={top_k}  temperature={temperature}   ...", end=" ", flush=True)
            metrics = evaluate_pitch(model, dac_model, top_k, temperature, run_dir, run_name)
            metrics["zeroed_prop"] = zeroed_prop
            metrics["training_tau"] = training_tau
            all_rows.append(metrics)
            print(f"R² = {metrics['pitch_r2']:.4f}")

    # write summary CSV
    csv_path = out_root / "summary.csv"
    fieldnames = [
        "zeroed_prop", "training_tau", "top_k", "temperature",
        "pitch_r", "pitch_r2", "valid_frac",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row[k] for k in fieldnames})

    # write experiment config
    config = {
        "hf_dataset_path": HF_DATASET_PATH,
        "device": DEVICE,
        "train_params": TRAIN_PARAMS,
        "train_grid": TRAIN_GRID,
        "eval_grid": EVAL_GRID,
    }
    with open(out_root / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # print summary
    print(f"\n{'='*60}")
    print(f"Experiment root: {out_root}")
    print(f"{'='*60}")
    print(f"{'zp':>4}  {'tau':>4}  {'k':>3}  {'temp':>4}  {'R²':>6}  {'valid':>5}")
    print("-" * 40)
    for row in sorted(all_rows, key=lambda r: -r["pitch_r2"]):
        print(f"{row['zeroed_prop']:>4.1f}  {row['training_tau']:>4.1f}  "
              f"{row['top_k']:>3d}  {row['temperature']:>4.1f}  "
              f"{row['pitch_r2']:>6.4f}  {row['valid_frac']:>5.2f}")

    print(f"\nAll outputs: {out_root}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
