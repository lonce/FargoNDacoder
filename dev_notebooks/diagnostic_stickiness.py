"""
diagnostic_stickiness.py
========================
Three diagnostics + normalization check for the GRU exposure bias hypothesis.

Usage:
    Paste into a notebook cell, or run:
        %run diagnostic_stickiness.py
    (requires model, dac_model, model_cfg, data_cfg, val_ds, val_loader, DEVICE
     to be defined in the notebook namespace.)

Output:
    - Prints normalization config check
    - Prints comparison table: open-loop MSE vs closed-loop MSE per timestep
    - Prints frame-evolution statistics (mean consecutive L2 diff)
    - Saves three figures to out_dir/:
        * frame_evolution.png
        * input_distribution.png
        * per_step_mse.png
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 70)
print("DIAGNOSTIC: Stickiness / GRU Exposure Bias")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Normalization consistency check
# ---------------------------------------------------------------------------
print("\n--- 1. Normalization Check ---")
print(f"  data_cfg.clamp_val          = {data_cfg.clamp_val}")
print(f"  validation_cfg.clamp_val    = {validation_cfg.clamp_val}")
dcv = getattr(model.config, "clamp_val", "NOT SET (default 15)")
print(f"  model.config.clamp_val      = {dcv}")
train_divisor = data_cfg.clamp_val
infer_divisor = model.config.clamp_val
if train_divisor != infer_divisor:
    print(f"  *** MISMATCH: train normalizes by {train_divisor}, "
          f"inference by {infer_divisor} (ratio={infer_divisor/train_divisor:.2f}) ***")
else:
    print(f"  OK: both use {train_divisor}")

# ---------------------------------------------------------------------------
# 2. Sample a validation batch
# ---------------------------------------------------------------------------
print("\n--- 2. Loading validation sample ---")
val_batch = next(iter(val_loader))
val_batch = {k: v.to(DEVICE) for k, v in val_batch.items()}
B = val_batch["latents"].shape[0]
T = val_batch["latents"].shape[1]

print(f"  Batch shape: latents {tuple(val_batch['latents'].shape)}, "
      f"targets {tuple(val_batch['targets'].shape)}, "
      f"cond {tuple(val_batch['cond'].shape)}")

# Quick statistics on training-time GRU inputs
with torch.no_grad():
    gt_normed = val_batch["latents"]  # already normalized by dataset to [-1, 1]
    print(f"  Training-time GRU input stats: "
          f"mean={gt_normed.mean().item():.4f}  "
          f"std={gt_normed.std().item():.4f}  "
          f"min={gt_normed.min().item():.4f}  "
          f"max={gt_normed.max().item():.4f}")

# ---------------------------------------------------------------------------
# 3. Open-loop vs closed-loop MSE diagnostic
# ---------------------------------------------------------------------------
print("\n--- 3. Open-loop vs Closed-loop MSE ---")

model.eval()
n_q = model.config.n_q
codebook_dim = model.config.codebook_dim

# 3a. Open-loop (GT input, teacher cascade): GT grub input + GT cascade input.
#     This is the easiest case — heads just copy teacher vectors.
with torch.no_grad():
    out_open_teacher = model(
        latents=val_batch["latents"],
        cond=val_batch["cond"],
        target_codes=val_batch["targets"],
        cascade_mode="teacher",
    )
    mse_open_teacher = model.compute_loss(
        out_open_teacher["predicted_logits_per_codebook"],
        val_batch["targets"],
    )["total_loss"].item()

print(f"  Open-loop (GT input + teacher cascade): total MSE = {mse_open_teacher:.6f}")

# 3b. Open-loop (GT input, free cascade): GT GRU input but self-predicted cascade.
#     This isolates the GRU input effect from the cascade effect.
with torch.no_grad():
    out_open_free = model(
        latents=val_batch["latents"],
        cond=val_batch["cond"],
        cascade_mode="free",
    )
    mse_open_free = model.compute_loss(
        out_open_free["predicted_logits_per_codebook"],
        val_batch["targets"],
    )["total_loss"].item()

    # Logits → argmax vectors for downstream vector analysis
    n_q_model = model.config.n_q
    open_preds = torch.stack([
        model.codebook_vectors[q][
            out_open_free["predicted_logits_per_codebook"][q].argmax(dim=-1)
        ] / model.config.clamp_val
        for q in range(n_q_model)
    ], dim=2)  # [B, T, n_q, D]

print(f"  Open-loop (GT input + free cascade):    total MSE = {mse_open_free:.6f}")

# 3b. Closed-loop: autoregressive, feed own predictions as GRU input
closed_preds_list = []
closed_mse_per_step = []
hidden = None
latent_t = val_batch["latents"][:, 0, :]  # first frame from GT as seed

for t in range(T):
    cond_t = val_batch["cond"][:, t, :]
    target_t = val_batch["targets"][:, t, :]  # [B, n_q]

    # Clamp to expected range (head outputs are in normalized space)
    normed = torch.clamp(latent_t, -1.0, 1.0)

    out = model.forward_step(
        latent_t=normed,
        cond_t=cond_t,
        hidden=hidden,
        cascade_mode="free",
    )
    hidden = out["hidden"]

    # Get predictions for this step
    preds_q = [p.squeeze(1) for p in out["predicted_vectors_per_codebook"]]
    preds_t = torch.stack(preds_q, dim=1)  # [B, n_q, D]
    closed_preds_list.append(preds_t)

    # Compute per-step cross-entropy
    target_flat = target_t.unsqueeze(1)  # [B, 1, n_q]
    loss_step = model.compute_loss(
        out["predicted_logits_per_codebook"],
        target_flat,
    )
    closed_mse_per_step.append(loss_step["total_loss"].item())

    # Next input = predicted vectors (closed loop)
    latent_t = preds_t.reshape(B, n_q * codebook_dim)

closed_preds = torch.stack(closed_preds_list, dim=1)  # [B, T, n_q, D]
mse_closed_overall = np.mean(closed_mse_per_step)

# 3d. Compute normalized target vectors (for comparison)
with torch.no_grad():
    target_vecs = model._teacher_vectors_from_codes(val_batch["targets"])  # [B, T, n_q, D]
    target_vecs = target_vecs / model.config.clamp_val  # normalize to match head output

# per-step MSE on normalized targets
mse_open_per_step = (open_preds - target_vecs).square().mean(dim=(0, 2, 3)).cpu().numpy()  # [T]

print(f"\n  Closed-loop (self-predicted input + free cascade): overall MSE = {mse_closed_overall:.6f}")
print(f"  Ratio closed/open-free: {mse_closed_overall / max(mse_open_free, 1e-10):.2f}x")
print(f"\n  Per-step MSE comparison (first 20 steps):")
print(f"  {'step':>5} {'open-free':>12} {'closed':>10} {'ratio':>10}")
print(f"  {'-----':>5} {'------------':>12} {'----------':>10} {'----------':>10}")
for t_ in range(min(20, T)):
    r = closed_mse_per_step[t_] / max(mse_open_per_step[t_], 1e-10)
    print(f"  {t_:>5} {mse_open_per_step[t_]:>10.6f} {closed_mse_per_step[t_]:>10.6f} {r:>10.2f}")

# ---------------------------------------------------------------------------
# 4. Frame-evolution (stickiness) diagnostic
# ---------------------------------------------------------------------------
print("\n--- 4. Frame-evolution (stickiness) ---")

# Compute consecutive L2 distances between predicted frames
# open_preds: [B, T, n_q, D]
open_diffs = (open_preds[:, 1:] - open_preds[:, :-1]).norm(dim=-1)    # [B, T-1, n_q]
closed_diffs = (closed_preds[:, 1:] - closed_preds[:, :-1]).norm(dim=-1)

open_mean_diff = open_diffs.mean(dim=(0, 2)).cpu().numpy()  # [T-1]
closed_mean_diff = closed_diffs.mean(dim=(0, 2)).cpu().numpy()

# Show first 20 steps
print(f"  Consecutive frame L2 distance (averaged over B and n_q):")
print(f"  {'step':>5} {'open-loop':>12} {'closed-loop':>12}")
print(f"  {'-----':>5} {'------------':>12} {'------------':>12}")
for t_ in range(min(20, T - 1)):
    print(f"  {t_:>5} {open_mean_diff[t_]:>12.6f} {closed_mean_diff[t_]:>12.6f}")

print(f"\n  Mean consecutive L2 (over all steps):")
print(f"    open-loop:   {open_mean_diff.mean():.6f}")
print(f"    closed-loop: {closed_mean_diff.mean():.6f}")
stickiness_ratio = closed_mean_diff.mean() / max(open_mean_diff.mean(), 1e-10)
print(f"    ratio:       {stickiness_ratio:.2f}x")
if stickiness_ratio < 0.5:
    print(f"    *** CLOSED-LOOP IS STICKY ({(1 - stickiness_ratio)*100:.0f}% less frame-to-frame variation) ***")
elif stickiness_ratio < 0.8:
    print(f"    ** Closed-loop shows some stickiness ({(1 - stickiness_ratio)*100:.0f}% less variation) **")
else:
    print(f"    OK: closed-loop frame variation is comparable to open-loop")

# 4b. Stickiness vs conditioning parameter
print(f"\n  Stickiness per codebook level (mean consecutive L2):")
for q in range(n_q):
    o = open_diffs[:, :, q].mean().item()
    c = closed_diffs[:, :, q].mean().item()
    print(f"    CB{q}: open={o:.6f}  closed={c:.6f}  ratio={c/max(o,1e-10):.2f}")

# ---------------------------------------------------------------------------
# 5. GRU input distribution: training vs inference
# ---------------------------------------------------------------------------
print("\n--- 5. GRU Input Distribution ---")

# Training-time GRU inputs: from the validation batch (already normalized by dataset)
train_inputs = val_batch["latents"].flatten().cpu().numpy()

# Inference-time GRU inputs: what the model feeds itself during closed-loop
# These are the self-predicted latents, clamped to [-1, 1]
# Collect them during inference
infer_inputs_list = []
latent_t = val_batch["latents"][:, 0, :]
hidden = None

with torch.no_grad():
    for t in range(T):
        # Record what the GRU sees (clamped to expected range)
        normed = torch.clamp(latent_t, -1.0, 1.0)
        infer_inputs_list.append(normed.cpu())

        cond_t = val_batch["cond"][:, t, :]
        out = model.forward_step(
            latent_t=normed,
            cond_t=cond_t,
            hidden=hidden,
            cascade_mode="free",
        )
        hidden = out["hidden"]
        preds_q = [p.squeeze(1) for p in out["predicted_vectors_per_codebook"]]
        preds_t = torch.stack(preds_q, dim=1)
        latent_t = preds_t.reshape(B, n_q * codebook_dim)

infer_inputs = torch.stack(infer_inputs_list, dim=1).flatten().cpu().numpy()

print(f"  Train GRU input:  mean={train_inputs.mean():.4f}  std={train_inputs.std():.4f}  "
      f"min={train_inputs.min():.4f}  max={train_inputs.max():.4f}")
print(f"  Infer GRU input:  mean={infer_inputs.mean():.4f}  std={infer_inputs.std():.4f}  "
      f"min={infer_inputs.min():.4f}  max={infer_inputs.max():.4f}")
print(f"  Distribution shift (infer/train std ratio): {infer_inputs.std()/max(train_inputs.std(), 1e-10):.4f}")

# ---------------------------------------------------------------------------
# 6. PLOTS
# ---------------------------------------------------------------------------
print("\n--- 6. Saving figures ---")
out_dir = Path.cwd() / "output" if "out_dir" not in dir() else Path(out_dir)
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 6a. Frame-evolution
ax = axes[0]
steps = np.arange(min(T - 1, 100))
ax.plot(steps, open_mean_diff[:len(steps)], "b-o", markersize=2, label="Open-loop (GT input)")
ax.plot(steps, closed_mean_diff[:len(steps)], "r-o", markersize=2, label="Closed-loop (self-predicted)")
ax.set_xlabel("Timestep")
ax.set_ylabel("Consecutive L2 distance")
ax.set_title("Frame-evolution: consecutive predicted-vector L2 norm")
ax.legend()
ax.grid(True, alpha=0.3)

# 6b. Per-step MSE
ax = axes[1]
steps = np.arange(min(T, 100))
ax.plot(steps, mse_open_per_step[:len(steps)], "b-o", markersize=2, label="Open-loop MSE")
ax.plot(steps, np.array(closed_mse_per_step[:len(steps)]), "r-o", markersize=2, label="Closed-loop MSE")
ax.set_xlabel("Timestep")
ax.set_ylabel("MSE")
ax.set_title("Per-timestep prediction MSE")
ax.legend()
ax.grid(True, alpha=0.3)

# 6c. GRU input distribution
ax = axes[2]
ax.hist(train_inputs, bins=80, alpha=0.5, label=f"Train n={len(train_inputs)}", density=True)
ax.hist(infer_inputs, bins=80, alpha=0.5, label=f"Infer n={len(infer_inputs)}", density=True)
ax.set_xlabel("Normalized latent value")
ax.set_ylabel("Density")
ax.set_title("GRU input distribution: training vs inference")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
figpath = str(out_dir / "diagnostic_stickiness.png")
plt.savefig(figpath, dpi=150)
print(f"  Saved: {figpath}")
plt.close()

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
