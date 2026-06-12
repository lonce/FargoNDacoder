import torch
import torch.nn.functional as F


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
    top_k=1,
    temperature=1.0,
):
    """
    Generate hop_size frames autoregressively.

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
            cascade_mode="soft",   # IMPORTANT
        )

        logits_per_q = out["logits_per_codebook"]
        hidden = out["hidden"]

        codes_t = []
        vecs_t = []

        for q, logits_q in enumerate(logits_per_q):
            # logits_q: [B, 1, K] → flatten
            logits_q = logits_q.squeeze(1)

            idx = sample_logits_topk(
                logits_q,
                top_k=top_k,
                temperature=temperature,
            )

            codes_t.append(idx)

            codebook = rnn_model.codebook_vectors[q]  # [K, 8]
            vec_q = codebook[idx]                     # [B, 8]
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
    top_k=1,
    temperature=1.0,
    frame_samples=512,
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
    