from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


CodePool = List[List[Tuple[torch.Tensor, torch.Tensor]]]
"""
pool[b][q] -> (vectors, indices)
    vectors: [N_bq, 8]    unique 8D codebook vectors at this cond bin
    indices: [N_bq]        corresponding codebook indices
"""


def build_code_pool(
    dataset,
    n_bins: int = 100,
    cond_index: int = 0,
    n_q_for_pool: int = 1,
) -> CodePool:
    """
    One-pass pool builder: iterate dataset, bin cond values,
    record observed 8D DAC codebook vectors per codebook level.

    The pool stores the *actual 8D vectors* from the dataset's latents
    (already projected to DAC latent space), so Euclidean distance
    comparisons are meaningful.

    Args:
        dataset: RNNDACLatentDataset (training split).
        n_bins: number of bins for cond values (assumed in [0, 1]).
        cond_index: which column of cond to bin by.
        n_q_for_pool: number of codebook levels to record (default 1 = CB0 only).

    Returns:
        CodePool: pool[b][q] = (vectors, indices)
    """
    D = 8  # codebook_dim
    pool: CodePool = [[(torch.empty(0, D), torch.empty(0, dtype=torch.long))
                        for _ in range(n_q_for_pool)] for _ in range(n_bins)]

    for i in range(len(dataset)):
        sample = dataset[i]
        cond = sample["cond"]           # [T, p]
        latents = sample["latents"]     # [T, n_q * D]
        targets = sample["targets"]     # [T, n_q]

        T = cond.shape[0]
        for t in range(T):
            bin_idx = int(cond[t, cond_index].item() * n_bins)
            bin_idx = min(max(bin_idx, 0), n_bins - 1)
            for q in range(n_q_for_pool):
                vec = latents[t, q * D : (q + 1) * D]  # [D]
                idx = int(targets[t, q].item())
                cur_vecs, cur_idxs = pool[bin_idx][q]
                if cur_vecs.shape[0] == 0 or not (vec == cur_vecs).all(dim=1).any():
                    pool[bin_idx][q] = (
                        torch.cat([cur_vecs, vec.unsqueeze(0)]),
                        torch.cat([cur_idxs, torch.tensor([idx])]),
                    )

    return pool


def pool_snap(
    pool: CodePool,
    expected_vec: torch.Tensor,
    cond_values: torch.Tensor,
    n_bins: int,
    widen: int = 1,
    q: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Snap expected 8D vectors to nearest pool vectors in Euclidean space.

    Args:
        pool: from build_code_pool.
        expected_vec: [B, 8] soft expected vector from logits @ codebook.
        cond_values: [B] float tensor of cond values in [0, 1].
        n_bins: number of bins used when building pool.
        widen: adjacent bins to include on each side.
        q: codebook level to snap.

    Returns:
        (snapped_vecs, snapped_idxs): each [B, 8] and [B].
    """
    B = expected_vec.shape[0]
    device = expected_vec.device

    snapped_vecs = torch.zeros_like(expected_vec)
    snapped_idxs = torch.zeros(B, dtype=torch.long, device=device)

    for bi in range(B):
        cv = cond_values[bi].item()
        bin_idx = int(cv * n_bins)
        bin_idx = min(max(bin_idx, 0), n_bins - 1)
        lo = max(0, bin_idx - widen)
        hi = min(n_bins - 1, bin_idx + widen)

        all_vecs = []
        all_idxs = []
        for b in range(lo, hi + 1):
            vecs, idxs = pool[b][q]
            if vecs.shape[0] > 0:
                all_vecs.append(vecs)
                all_idxs.append(idxs)

        if not all_vecs:
            # No pool vectors available — return the expected vector unchanged
            snapped_vecs[bi] = expected_vec[bi]
            snapped_idxs[bi] = -1
            continue

        pool_vecs = torch.cat(all_vecs, dim=0).to(device)  # [M, 8]
        pool_idxs = torch.cat(all_idxs, dim=0).to(device)   # [M]

        # Euclidean distance
        diff = expected_vec[bi:bi+1] - pool_vecs
        dist = diff.square().sum(dim=1)  # [M]
        nn = dist.argmin()

        snapped_vecs[bi] = pool_vecs[nn]
        snapped_idxs[bi] = pool_idxs[nn]

    return snapped_vecs, snapped_idxs


def generate_latent_hop(
    rnn_model,
    latent_t,           # [B, n_q*8]
    cond_hop,           # [B, hop_size, p]
    hidden,
    cascade_mode="free",
    tau=0.0,
    top_k=0,
):
    """
    Generate hop_size frames autoregressively.

    Each step samples from the predicted logits with optional
    top-k + temperature controls.

    Args:
        tau: softmax temperature (0 = argmax).
        top_k: keep only top-k logits before sampling (0 = disabled).

    Returns:
        latents_out: [B, hop_size, n_q*8]
        vectors_out: [B, hop_size, n_q, D]   (predicted 8D vectors per codebook)
        hidden
    """
    B, hop_size, _ = cond_hop.shape
    n_q = rnn_model.config.n_q
    D = rnn_model.config.codebook_dim

    latents_out = []
    vecs_per_step = []

    for t in range(hop_size):
        cond_t = cond_hop[:, t, :]  # [B, p]

        latent_t = torch.clamp(latent_t, -1.0, 1.0)
        out = rnn_model.forward_step(
            latent_t=latent_t,
            cond_t=cond_t,
            hidden=hidden,
            cascade_mode=cascade_mode,
            tau=tau,
            top_k=top_k,
        )

        preds_per_q = out["predicted_vectors_per_codebook"]
        # Each is [B, 1, D] → squeeze to [B, D]
        vecs_q = [p.squeeze(1) for p in preds_per_q]
        hidden = out["hidden"]

        # stack → [B, n_q, D]
        vecs_t = torch.stack(vecs_q, dim=1)

        # flatten → [B, n_q*D] for next input
        latent_t = vecs_t.reshape(B, n_q * D)

        latents_out.append(latent_t)
        vecs_per_step.append(vecs_t)

    latents_out = torch.stack(latents_out, dim=1)        # [B, hop_size, n_q*D]
    vecs_per_step = torch.stack(vecs_per_step, dim=1)     # [B, hop_size, n_q, D]

    return latents_out, vecs_per_step, hidden


#------------------------------------------------------------------

def infer_streaming_with_lookahead(
    rnn_model,
    dac_model,
    cond_sequence,        # [B, T, p]
    chunk_size,
    hop_size,
    right_context,
    frame_samples=512,
    cascade_mode="free",
    tau=0.0,
    top_k=0,
):
    """
    Full streaming RNNDAC inference.

    Returns:
        audio: [B, 1, samples]
    """
    B, T_total, _ = cond_sequence.shape
    device = cond_sequence.device

    n_q = rnn_model.config.n_q
    D = rnn_model.config.codebook_dim

    left_context = chunk_size - hop_size - right_context
    assert left_context >= 0

    # --------------------------------------------------
    # Warmup: initialize latent buffer
    # --------------------------------------------------
    latent_t = torch.zeros(B, n_q * D, device=device)
    hidden = None

    latent_buffer = []

    # Fill initial chunk using first conditioning frame
    cond0 = cond_sequence[:, 0:1, :].expand(B, chunk_size, -1)

    init_latents, _, hidden = generate_latent_hop(
        rnn_model,
        latent_t,
        cond0,
        hidden,
        cascade_mode=cascade_mode,
        tau=tau,
        top_k=top_k,
    )

    latent_buffer.append(init_latents)

    outputs = []

    t = 0
    while t < T_total:
        cond_hop = cond_sequence[:, t:t+hop_size, :]

        if cond_hop.shape[1] < hop_size:
            # pad final conditioning
            pad = cond_hop[:, -1:, :].expand(B, hop_size - cond_hop.shape[1], -1)
            cond_hop = torch.cat([cond_hop, pad], dim=1)

        new_latents, _, hidden = generate_latent_hop(
            rnn_model,
            latent_buffer[-1][:, -1, :],  # last frame
            cond_hop,
            hidden,
            cascade_mode=cascade_mode,
            tau=tau,
            top_k=top_k,
        )

        latent_buffer.append(new_latents)

        # --------------------------------------------------
        # Build decode window
        # --------------------------------------------------
        all_latents = torch.cat(latent_buffer, dim=1)  # [B, T, n_q*8]

        start = t
        win_start = start - left_context
        win_end = start + hop_size + right_context

        src_start = max(0, win_start)
        src_end = min(all_latents.shape[1], win_end)

        z_chunk = all_latents[:, src_start:src_end, :]  # [B, T, D]

        # pad edges
        if src_start > win_start:
            pad = z_chunk[:, 0:1, :].expand(B, src_start - win_start, -1)
            z_chunk = torch.cat([pad, z_chunk], dim=1)

        if src_end < win_end:
            pad = z_chunk[:, -1:, :].expand(B, win_end - src_end, -1)
            z_chunk = torch.cat([z_chunk, pad], dim=1)

        # reshape → DAC format, un-normalize × clamp_val
        z_chunk = (z_chunk * rnn_model.config.clamp_val).transpose(1, 2)  # [B, n_q*8, T]

        z_proj, _, _ = dac_model.quantizer.from_latents(z_chunk)
        audio_chunk = dac_model.decode(z_proj)

        audio_start = left_context * frame_samples
        audio_end = audio_start + hop_size * frame_samples

        outputs.append(audio_chunk[:, :, audio_start:audio_end])

        t += hop_size

    return torch.cat(outputs, dim=-1)
    