#!/usr/bin/env python3
"""
run_experiments.py — Systematic comparison of architectural variants
for FargoNDacoder pitch-tracking ability.

Each experiment:
  1. Creates dataset + model (fresh weights)
  2. Trains for N steps
  3. Runs stickiness / exposure-bias diagnostic
  4. Generates audio with a pitch-glide conditioning sweep
  5. Computes pitch track (librosa.yin) and plots it

Output directory:  dev_notebooks/output/experiments/<exp_name>/
Summary:           dev_notebooks/output/experiments/summary.csv
"""

import argparse, csv, json, os, random, sys, time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import dac
import librosa

# matplotlib — must set backend before pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project-level imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.rnndac_dataset import RNNDACLatentDataset, LatentDatasetConfig
from core.rnndac_model import GRUModelConfig, RNNDACModel, RNNDACModelNoCascade
from core.inference import infer_streaming_with_lookahead
from utils.io import save_checkpoint

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
BASE_PARAMS = {
    "n_steps": 6001,
    "print_every": 2000,
    "validate_every": 2000,
    "val_batches": 8,
    "checkpoint_every": 6000,
    "cascade_mode": "free",
    "cond_injection": "concat",
    "inp_proportion": 5,
    "cond_proportion": 2,
    "tf_schedule": [2000, 2000],
    "ss_every": 10,
    "ss_warmup_steps": 0,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "grad_clip": 1.0,
}

EXPERIMENTS = [
    # (name, overrides_dict)
    # name must be a valid directory name

    ("01_baseline", {}),

    ("02_rnn_dropout02", {
        "rnn_dropout": 0.2,
    }),

    ("03_ss_every3", {
        "ss_every": 3,
    }),

    ("04_rnn_dropout02_ss3", {
        "rnn_dropout": 0.2,
        "ss_every": 3,
    }),

    ("05_no_cascade", {
        "no_cascade": True,
    }),

    ("06_no_cascade_drop02_ss3", {
        "no_cascade": True,
        "rnn_dropout": 0.2,
        "ss_every": 3,
    }),

    ("07_nq4", {
        "n_q": 4,
        "input_size": 32,
        "infer_n_q": 4,  # how many codebooks for the dataset (same everywhere)
    }),

    ("08_nq4_drop02_ss3", {
        "n_q": 4,
        "input_size": 32,
        "rnn_dropout": 0.2,
        "ss_every": 3,
    }),

    ("09_larger_gru", {
        "hidden_size": 256,
        "num_layers": 4,
        "rnn_dropout": 0.2,
        "ss_every": 3,
    }),
]

# ---------------------------------------------------------------------------
# Paths (edit these to match your system)
# ---------------------------------------------------------------------------
HF_DATASET_PATH = "/slowdisk/data/DAC/pitchglidesATriangle5octaves/Aglides/hf_dataset"
OUTPUT_BASE = Path(__file__).resolve().parent / "output" / "experiments"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_output_dir(exp_name):
    """Create timestamped output directory for an experiment."""
    d = OUTPUT_BASE / f"{make_timestamp()}_{exp_name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "checkpoints").mkdir(exist_ok=True)
    return d


def make_data_config(n_q, sequence_length=128, clamp_val=12, noise_weight=0.05):
    """Return (train_cfg, val_cfg) for RNNDACLatentDataset."""
    train_cfg = LatentDatasetConfig(
        dataset_path=HF_DATASET_PATH,
        sequence_length=sequence_length,
        n_q=n_q,
        clamp_val=clamp_val,
        files_per_sequence=4,
        add_noise=True,
        noise_weight=noise_weight,
    )
    val_cfg = LatentDatasetConfig(
        dataset_path=HF_DATASET_PATH,
        sequence_length=sequence_length,
        n_q=n_q,
        clamp_val=clamp_val,
        files_per_sequence=1,
    )
    return train_cfg, val_cfg


def make_model(model_cfg, dac_model, no_cascade=False):
    """Create RNNDACModel (or NoCascade variant)."""
    cls = RNNDACModelNoCascade if no_cascade else RNNDACModel
    model = cls(model_cfg, dac_model=dac_model).to(DEVICE)
    return model


def make_model_config(params):
    """Build GRUModelConfig from a merged experiment dict."""
    return GRUModelConfig(
        n_q=params.get("n_q", 9),
        clamp_val=12,
        codebook_size=1024,
        codebook_dim=8,
        input_size=params.get("input_size", 72),
        cond_size=1,
        hidden_size=params.get("hidden_size", 128),
        num_layers=params.get("num_layers", 3),
        cond_injection=params["cond_injection"],
        inp_proportion=params["inp_proportion"],
        cond_proportion=params["cond_proportion"],
        rnn_dropout=params.get("rnn_dropout", 0.0),
    )


def train_model(model, loader, val_loader, params, out_dir, exp_name):
    """
    Training loop matching test_model.ipynb.
    Returns (train_loss_history, val_loss_history).
    """
    optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"])
    grad_clip = params["grad_clip"]
    n_steps = params["n_steps"]
    tf_schedule = params["tf_schedule"]
    cascade_mode = params["cascade_mode"]
    tf_on, tf_off = tf_schedule[0], tf_schedule[1]
    tf_cycle = tf_on + tf_off
    ss_every = params.get("ss_every", 0)
    batch_size = params["batch_size"]

    # --- auto-extend: if val loss still dropping at n_steps, extend to 12001 ---
    step = 0
    auto_n_steps = n_steps
    train_loss_history = []
    val_loss_history = []
    train_iter = iter(loader)
    val_iter = iter(val_loader)

    while step < auto_n_steps:
        # --- fetch batch ---
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        model.train()

        # --- teacher-forcing schedule ---
        if tf_cycle > 0:
            in_tf_phase = (step % tf_cycle) < tf_on
            training_cascade = "teacher" if in_tf_phase else cascade_mode
        else:
            training_cascade = cascade_mode

        # --- scheduled sampling ---
        use_ss = ss_every > 0 and step > 0 and step % ss_every == 0
        if use_ss:
            out = model.forward_autoregressive(
                latents=batch["latents"],
                cond=batch["cond"],
                warmup_steps=params.get("ss_warmup_steps", 0),
            )
        else:
            out = model(
                latents=batch["latents"],
                cond=batch["cond"],
                target_codes=batch["targets"],
                cascade_mode=training_cascade,
            )

        loss_dict = model.compute_loss(
            out["predicted_logits_per_codebook"],
            batch["targets"],
        )
        loss = loss_dict["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        train_loss_history.append(loss.item())

        # --- validation ---
        if step > 0 and step % params["validate_every"] == 0:
            model.eval()
            val_total_losses = []
            with torch.no_grad():
                for _ in range(params["val_batches"]):
                    try:
                        vb = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        vb = next(val_iter)
                    vb = {k: v.to(DEVICE) for k, v in vb.items()}
                    out_val = model(
                        latents=vb["latents"],
                        cond=vb["cond"],
                        target_codes=vb["targets"],
                        cascade_mode=cascade_mode,
                    )
                    vloss = model.compute_loss(
                        out_val["predicted_logits_per_codebook"],
                        vb["targets"],
                    )["total_loss"].item()
                    val_total_losses.append(vloss)
            avg_val = sum(val_total_losses) / len(val_total_losses)
            val_loss_history.append((step, avg_val))

            # --- print ---
            recent = train_loss_history[-params["print_every"]:]
            avg_train = sum(recent) / len(recent) if recent else loss.item()
            ss_tag = " [SS]" if use_ss else ""
            print(f"[{exp_name}] Step {step:4d} | Train: {loss.item():.4f} | "
                  f"Avg: {avg_train:.4f} | Val: {avg_val:.4f}{ss_tag}")

            # --- auto-extend check (once, near original n_steps boundary) ---
            if step >= n_steps - params["validate_every"] and auto_n_steps == n_steps and len(val_loss_history) >= 4:
                recent_vals = [v for _, v in val_loss_history[-4:]]
                # linear fit slope of last 4 validation losses
                xs = list(range(len(recent_vals)))
                sx = sum(xs); sy = sum(recent_vals); sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, recent_vals))
                n = len(recent_vals)
                slope = (n * sxy - sx * sy) / (n * sxx - sx * sx) if (n * sxx - sx * sx) != 0 else 0
                if slope < 0:
                    auto_n_steps = int(n_steps * 2) + 1  # ~12003
                    print(f"[{exp_name}] Val loss still declining (slope={slope:.2e}/val), extending to {auto_n_steps} steps.")

        # --- checkpoint ---
        if step > 0 and step % params["checkpoint_every"] == 0:
            ckpt_path = save_checkpoint(
                output_dir=str(out_dir),
                step=step,
                model=model,
                optimizer=optimizer,
                params=params,
                model_config=model.config,
                extra={
                    "last_train_loss": loss.item(),
                    "train_loss_history_tail": train_loss_history[-10:],
                },
            )
            print(f"[{exp_name}] Saved checkpoint: {ckpt_path}")

        step += 1

    # --- final checkpoint (use actual final step) ---
    final_step = step - 1 if step > 0 else 0
    ckpt_path = save_checkpoint(
        output_dir=str(out_dir),
        step=final_step,
        model=model,
        optimizer=optimizer,
        params=params,
        model_config=model.config,
        extra={
            "last_train_loss": loss.item(),
            "train_loss_history_tail": train_loss_history[-10:],
        },
    )
    print(f"[{exp_name}] Final checkpoint (step {final_step}): {ckpt_path}")

    return train_loss_history, val_loss_history


def plot_loss(train_hist, val_hist, out_dir, exp_name):
    """Save training/validation loss plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_hist, label="train", alpha=0.7)
    if val_hist:
        steps, vals = zip(*val_hist)
        ax.plot(steps, vals, "o-", label="val")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"{exp_name} — training & validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(out_dir / "loss.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostics  (adapted from diagnostic_stickiness.py)
# ---------------------------------------------------------------------------
def run_stickiness_diagnostic(model, val_batch, out_dir, exp_name):
    """
    Run open-loop vs closed-loop MSE + frame-evolution stickiness.
    Returns dict of metrics.
    """
    model.eval()
    B, T = val_batch["latents"].shape[:2]
    n_q = model.config.n_q
    D = model.config.codebook_dim

    # --- teacher cascade (open-loop, best case) ---
    with torch.no_grad():
        out_teacher = model(
            latents=val_batch["latents"],
            cond=val_batch["cond"],
            target_codes=val_batch["targets"],
            cascade_mode="teacher",
        )
        mse_teacher = model.compute_loss(
            out_teacher["predicted_logits_per_codebook"],
            val_batch["targets"],
        )["total_loss"].item()

    # --- free cascade (open-loop, GRU gets GT input) ---
    with torch.no_grad():
        out_free = model(
            latents=val_batch["latents"],
            cond=val_batch["cond"],
            cascade_mode="free",
        )
        mse_free = model.compute_loss(
            out_free["predicted_logits_per_codebook"],
            val_batch["targets"],
        )["total_loss"].item()

        n_q_model = model.config.n_q
        open_preds = torch.stack([
            model.codebook_vectors[q][
                out_free["predicted_logits_per_codebook"][q].argmax(dim=-1)
            ] / model.config.clamp_val
            for q in range(n_q_model)
        ], dim=2)  # [B, T, n_q, D]

    # --- closed-loop (autoregressive) ---
    closed_preds_list = []
    closed_mse_per_step = []
    hidden = None
    latent_t = val_batch["latents"][:, 0, :]

    for t in range(T):
        cond_t = val_batch["cond"][:, t, :]
        target_t = val_batch["targets"][:, t, :]
        normed = torch.clamp(latent_t, -1.0, 1.0)
        with torch.no_grad():
            out = model.forward_step(
                latent_t=normed,
                cond_t=cond_t,
                hidden=hidden,
                cascade_mode="free",
            )
        hidden = out["hidden"]
        preds_q = [p.squeeze(1) for p in out["predicted_vectors_per_codebook"]]
        preds_t = torch.stack(preds_q, dim=1)
        closed_preds_list.append(preds_t)
        loss_step = model.compute_loss(
            out["predicted_logits_per_codebook"],
            target_t.unsqueeze(1),
        )
        closed_mse_per_step.append(loss_step["total_loss"].item())
        latent_t = preds_t.reshape(B, n_q * D)

    closed_preds = torch.stack(closed_preds_list, dim=1)
    mse_closed = float(np.mean(closed_mse_per_step))
    ratio = mse_closed / max(mse_free, 1e-10)

    # --- stickiness (consecutive frame L2) ---
    open_diffs = (open_preds[:, 1:] - open_preds[:, :-1]).norm(dim=-1)
    closed_diffs = (closed_preds[:, 1:] - closed_preds[:, :-1]).norm(dim=-1)
    open_mean_diff = float(open_diffs.mean(dim=(0, 2)).mean().item())
    closed_mean_diff = float(closed_diffs.mean(dim=(0, 2)).mean().item())
    stickiness_ratio = closed_mean_diff / max(open_mean_diff, 1e-10)

    # --- per-step MSE for table ---
    with torch.no_grad():
        target_vecs = model._teacher_vectors_from_codes(val_batch["targets"])
        target_vecs = target_vecs / model.config.clamp_val
    mse_open_per_step = (open_preds - target_vecs).square().mean(dim=(0, 2, 3)).cpu().numpy()

    # --- print summary ---
    print(f"\n  [{exp_name}] Diagnostic:")
    print(f"    Open-loop teacher: {mse_teacher:.4f}  free: {mse_free:.4f}")
    print(f"    Closed-loop:       {mse_closed:.4f}  ratio: {ratio:.2f}x")
    print(f"    Stickiness:        open={open_mean_diff:.4f}  closed={closed_mean_diff:.4f}  "
          f"ratio={stickiness_ratio:.3f}")
    if stickiness_ratio < 0.1:
        print("    *** STICKY ***")

    # --- plot per-step MSE ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    steps = np.arange(min(T, 50))
    ax = axes[0]
    ax.plot(steps, mse_open_per_step[:len(steps)], "b-o", ms=2, label="Open-loop MSE")
    ax.plot(steps, np.array(closed_mse_per_step[:len(steps)]), "r-o", ms=2, label="Closed-loop MSE")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("MSE")
    ax.set_title(f"{exp_name} — per-timestep MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    open_diff_arr = open_diffs.mean(dim=(0, 2)).cpu().numpy()
    closed_diff_arr = closed_diffs.mean(dim=(0, 2)).cpu().numpy()
    ax.plot(steps, open_diff_arr[:len(steps)], "b-o", ms=2, label="Open-loop L2 diff")
    ax.plot(steps, closed_diff_arr[:len(steps)], "r-o", ms=2, label="Closed-loop L2 diff")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Consecutive L2")
    ax.set_title(f"{exp_name} — frame-evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(str(out_dir / "diagnostic.png"), dpi=150)
    plt.close(fig)

    return {
        "mse_open_teacher": mse_teacher,
        "mse_open_free": mse_free,
        "mse_closed": mse_closed,
        "ratio_closed_open": ratio,
        "stickiness_open": open_mean_diff,
        "stickiness_closed": closed_mean_diff,
        "stickiness_ratio": stickiness_ratio,
    }


# ---------------------------------------------------------------------------
# Audio generation + pitch analysis
# ---------------------------------------------------------------------------
def generate_and_analyze(model, dac_model, out_dir, exp_name,
                         n_frames=900, chunk_size=16, hop_size=8,
                         right_context=4):
    """
    Generate audio with a pitch-glide conditioning sweep (0.05 -> 0.95).
    Compute pitch track using librosa.yin and save pitch plot.
    """
    model.eval()

    # --- build conditioning sweep ---
    cond_seq = torch.linspace(0.05, 0.95, n_frames).unsqueeze(0).unsqueeze(-1).to(DEVICE)

    # --- generate ---
    with torch.no_grad():
        gen_audio = infer_streaming_with_lookahead(
            rnn_model=model,
            dac_model=dac_model,
            cond_sequence=cond_seq,
            chunk_size=chunk_size,
            hop_size=hop_size,
            right_context=right_context,
            frame_samples=512,
        )
    gen_audio_np = gen_audio.squeeze().detach().cpu().numpy()
    param_sr = 44100 / 512  # DAC frame rate: ~86.13 Hz

    # --- compute pitch track ---
    f0 = librosa.yin(gen_audio_np, fmin=55, fmax=2000, sr=44100)
    mp = librosa.hz_to_midi(f0)
    norm_mp = (mp - 33) / (93 - 33)  # normalize to [0, 1] approx (MIDI 33..93)

    # --- plot audio + pitch ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # audio waveform
    t_audio = np.arange(len(gen_audio_np)) / 44100
    axes[0].plot(t_audio, gen_audio_np, linewidth=0.5, alpha=0.7)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"{exp_name} — generated audio")
    axes[0].grid(True, alpha=0.3)

    # pitch track
    t_pitch = np.arange(len(norm_mp)) / param_sr
    axes[1].plot(t_pitch, norm_mp, "g-", linewidth=1.0)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].axhline(0.05, color="gray", linestyle="--", alpha=0.3)
    axes[1].axhline(0.95, color="gray", linestyle="--", alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Normalized pitch")
    axes[1].set_title(f"{exp_name} — pitch track (librosa.yin)")
    axes[1].grid(True, alpha=0.3)

    # pitch vs ideal ramp
    ideal = np.linspace(0.05, 0.95, len(norm_mp))
    valid = np.isfinite(norm_mp)
    if valid.sum() > 10:
        r2 = np.corrcoef(ideal[valid], norm_mp[valid])[0, 1] ** 2
    else:
        r2 = 0.0
    axes[2].plot(t_pitch, ideal, "k--", alpha=0.5, label="Ideal ramp")
    axes[2].plot(t_pitch, norm_mp, "g-", linewidth=1.0, alpha=0.8, label=f"Measured (R²={r2:.3f})")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Normalized pitch")
    axes[2].set_title(f"{exp_name} — pitch vs ideal")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.savefig(str(out_dir / "pitch_track.png"), dpi=150)
    plt.close(fig)

    # --- also save raw audio ---
    import soundfile as sf
    sf.write(str(out_dir / "generated.wav"), gen_audio_np, 44100)

    # --- compute pitch-tracking quality ---
    pitch_range = float(np.nanmax(norm_mp) - np.nanmin(norm_mp)) if valid.sum() > 0 else 0.0
    print(f"  [{exp_name}] Pitch: R²={r2:.3f}  range={pitch_range:.2f}  "
          f"valid_frames={valid.sum()}/{len(norm_mp)}")

    return {
        "pitch_r2": r2,
        "pitch_range": pitch_range,
        "pitch_valid_ratio": float(valid.sum()) / len(norm_mp),
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run_experiment(exp_cfg, dac_model):
    """Run a single experiment.  exp_cfg is (name, overrides_dict)."""
    exp_name, overrides = exp_cfg
    exp_dir = make_output_dir(exp_name)
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  Output:     {exp_dir}")
    print(f"{'='*70}")

    # --- merge params ---
    params = deepcopy(BASE_PARAMS)
    params.update(overrides)
    n_q = params.get("n_q", 9)

    # --- dataset ---
    train_cfg, val_cfg = make_data_config(n_q)
    ds = RNNDACLatentDataset(train_cfg, split="train", device="cpu")
    val_ds = RNNDACLatentDataset(val_cfg, split="validation", device="cpu")
    loader = DataLoader(ds, batch_size=params["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False, num_workers=0)

    # --- model ---
    model_cfg = make_model_config(params)
    no_cascade = params.get("no_cascade", False)
    model = make_model(model_cfg, dac_model, no_cascade=no_cascade)
    print(f"  Model: {type(model).__name__} | n_q={n_q} | "
          f"hidden={model_cfg.hidden_size}x{model_cfg.num_layers} | "
          f"rnn_drop={model_cfg.rnn_dropout} | ss_every={params['ss_every']}")

    # --- save config ---
    torch.save({
        "params": params,
        "model_cfg": model_cfg.to_dict(),
        "train_cfg": train_cfg,
        "val_cfg": val_cfg,
    }, str(exp_dir / "config.pt"))

    # --- train ---
    train_hist, val_hist = train_model(model, loader, val_loader, params, exp_dir, exp_name)
    plot_loss(train_hist, val_hist, exp_dir, exp_name)
    torch.save(train_hist, str(exp_dir / "train_loss.pt"))
    if val_hist:
        torch.save(val_hist, str(exp_dir / "val_loss.pt"))

    # --- diagnostic (use a fixed batch of the validation set) ---
    val_batch = next(iter(val_loader))
    val_batch = {k: v.to(DEVICE) for k, v in val_batch.items()}
    diag_metrics = run_stickiness_diagnostic(model, val_batch, exp_dir, exp_name)

    # --- generate audio + pitch analysis ---
    pitch_metrics = generate_and_analyze(model, dac_model, exp_dir, exp_name)

    # --- cleanup ---
    del model
    torch.cuda.empty_cache()

    # --- combine metrics ---
    metrics = {
        "exp_name": exp_name,
        "n_q": n_q,
        "hidden_size": model_cfg.hidden_size,
        "num_layers": model_cfg.num_layers,
        "rnn_dropout": model_cfg.rnn_dropout,
        "ss_every": params["ss_every"],
        "no_cascade": no_cascade,
        "n_steps": params["n_steps"],
        "final_train_loss": float(train_hist[-1]) if train_hist else -1,
    }
    metrics.update(diag_metrics)
    metrics.update(pitch_metrics)

    # --- save per-experiment metrics ---
    with open(str(exp_dir / "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"  [{exp_name}] Done.  Metrics saved.")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated list of experiment names to run "
                             "(default: all)")
    parser.add_argument("--n_steps", type=int, default=6001,
                        help="Override training steps for all experiments")
    parser.add_argument("--list", action="store_true",
                        help="Just list experiments and exit")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name, _ in EXPERIMENTS:
            print(f"  {name}")
        return

    # --- run selected experiments ---
    if args.experiments is not None:
        selected_names = [s.strip() for s in args.experiments.split(",")]
        selected = [(n, o) for n, o in EXPERIMENTS if n in selected_names]
        missing = set(selected_names) - {n for n, _ in selected}
        if missing:
            print(f"Warning: unknown experiments: {missing}")
    else:
        selected = EXPERIMENTS

    print(f"Selected {len(selected)} experiments")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # --- load DAC model once (shared across experiments) ---
    print("Loading DAC model...")
    dac_model_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(dac_model_path).to(DEVICE)
    dac_model.eval()
    print("DAC model loaded.")

    all_metrics = []
    start_time = time.time()

    for i, (name, _) in enumerate(selected):
        # find the full config in EXPERIMENTS
        exp_cfg = [(n, deepcopy(o)) for n, o in EXPERIMENTS if n == name][0]
        # update n_steps if overridden (use deep copy to preserve original)
        if args.n_steps != 6001:
            exp_cfg[1]["n_steps"] = args.n_steps

        try:
            metrics = run_experiment(exp_cfg, dac_model)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\n  *** EXPERIMENT '{name}' FAILED: {e} ***\n")
            import traceback
            traceback.print_exc()
            # save partial metrics with failure flag
            all_metrics.append({
                "exp_name": name,
                "error": str(e),
                "final_train_loss": -1,
                "mse_closed": -1,
                "pitch_r2": -1,
            })

        elapsed = time.time() - start_time
        remaining = (elapsed / (i + 1)) * (len(selected) - i - 1)
        print(f"\n  --- Progress: {i+1}/{len(selected)} | "
              f"Elapsed: {elapsed/60:.1f}m | "
              f"ETA: {remaining/60:.1f}m ---\n")

    # --- summary CSV ---
    csv_path = OUTPUT_BASE / "summary.csv"
    if all_metrics:
        fieldnames = list(all_metrics[0].keys())
        with open(str(csv_path), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\nSummary saved to {csv_path}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
