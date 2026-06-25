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


def sample_logits_topk(logits, top_k=1, temperature=1.0):
    """
    logits: [N, K]
    returns indices: [N]
    """
    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None:
        values, indices = torch.topk(logits, top_k, dim=-1)
        masked = torch.full_like(logits, float('-inf'))
        masked.scatter_(dim=-1, index=indices, src=values)
        logits = masked

    probs = torch.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return idx


def generate_latent_hop(
    rnn_model,
    latent_t,           # [B, n_q*8]
    cond_hop,           # [B, hop_size, p]
    hidden,
    top_k=5,
    temperature=0.5,
    cascade_mode="soft",
    pool=None,
    pool_cond_index=0,
    pool_n_bins=100,
    pool_widen=1,
):
    """
    Generate hop_size frames autoregressively.

    If pool is provided, CB0 expected 8D vectors (from logits) are snapped
    to the nearest observed pool vector in Euclidean space.  This keeps
    CB0 in-distribution for the current conditioning value.

    Returns:
        latents_out: [B, hop_size, n_q*8]
        codes_out:   [B, hop_size, n_q]
        hidden
    """
    B, hop_size, _ = cond_hop.shape
    n_q = rnn_model.config.n_q
    D = rnn_model.config.codebook_dim

    latents_out = []
    codes_out = []

    for t in range(hop_size):
        cond_t = cond_hop[:, t, :]  # [B, p]

        # Normalize to [-1, 1] to match training-time input distribution
        normed_latent_t = torch.clamp(latent_t, -rnn_model.config.clamp_val,
                                       rnn_model.config.clamp_val) / rnn_model.config.clamp_val
        out = rnn_model.forward_step(
            latent_t=normed_latent_t,
            cond_t=cond_t,
            hidden=hidden,
            cascade_mode=cascade_mode,
        )

        logits_per_q = out["logits_per_codebook"]
        hidden = out["hidden"]

        codes_t = []
        vecs_t = []

        for q, logits_q in enumerate(logits_per_q):
            # logits_q: [B, 1, K] → flatten
            logits_q = logits_q.squeeze(1)
            codebook = rnn_model.codebook_vectors[q]  # [K, 8]

            if pool is not None and q == 0:
                # Soft expected 8D vector → snap to nearest pool vector
                probs = torch.softmax(logits_q, dim=-1)
                expected_vec = probs @ codebook  # [B, 8]
                cond_vals = cond_t[:, pool_cond_index]  # [B]
                vec_q, idx = pool_snap(
                    pool, expected_vec, cond_vals,
                    pool_n_bins, widen=pool_widen, q=0,
                )
            else:
                idx = sample_logits_topk(
                    logits_q,
                    top_k=top_k,
                    temperature=temperature,
                )
                vec_q = codebook[idx]  # [B, 8]

            codes_t.append(idx)
            vecs_t.append(vec_q)

        # stack codebooks → [B, n_q, 8]
        vecs_t = torch.stack(vecs_t, dim=1)

        # flatten → [B, n_q*8]
        latent_t = vecs_t.reshape(B, n_q * D)

        latents_out.append(latent_t)
        codes_out.append(torch.stack(codes_t, dim=1))

    latents_out = torch.stack(latents_out, dim=1)
    codes_out = torch.stack(codes_out, dim=1)

    return latents_out, codes_out, hidden


#------------------------------------------------------------------

def infer_streaming_with_lookahead(
    rnn_model,
    dac_model,
    cond_sequence,        # [B, T, p]
    chunk_size,
    hop_size,
    right_context,
    top_k=5,
    temperature=0.5,
    frame_samples=512,
    pool=None,
    pool_cond_index=0,
    pool_n_bins=100,
    pool_widen=1,
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
        top_k=top_k,
        temperature=temperature,
        pool=pool,
        pool_cond_index=pool_cond_index,
        pool_n_bins=pool_n_bins,
        pool_widen=pool_widen,
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
            top_k=top_k,
            temperature=temperature,
            pool=pool,
            pool_cond_index=pool_cond_index,
            pool_n_bins=pool_n_bins,
            pool_widen=pool_widen,
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

        # reshape → DAC format
        z_chunk = z_chunk.transpose(1, 2)  # [B, n_q*8, T]

        z_proj, _, _ = dac_model.quantizer.from_latents(z_chunk)
        audio_chunk = dac_model.decode(z_proj)

        audio_start = left_context * frame_samples
        audio_end = audio_start + hop_size * frame_samples

        outputs.append(audio_chunk[:, :, audio_start:audio_end])

        t += hop_size

    return torch.cat(outputs, dim=-1)
    