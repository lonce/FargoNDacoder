import torch
from core.inference import generate_latent_hop


class RNNSynth:
    """
    Stateful real-time inference engine.

    Maintains GRU hidden state and latent buffer across calls so
    that generate_hop() can be called repeatedly with changing
    conditioning values, producing a seamless stream of audio.

    Usage:
        synth = RNNSynth(rnn_model, dac_model)
        synth.warmup(initial_cond)          # fill lookahead buffer
        audio = synth.generate_hop(cond)    # one hop of audio
        audio = synth.generate_hop(cond)    # next hop, seamless
        synth.reset()                       # flush state

    Audio is decoded by extracting a sliding window of latents
    from the buffer, projecting them to DAC latent space with
    from_latents(), and decoding with the DAC decoder.  The
    right_context lookahead is satisfied by previously-generated
    latents already in the buffer — no future cond values needed.
    """

    def __init__(self, rnn_model, dac_model,
                 chunk_size=16, hop_size=8, right_context=4,
                 frame_samples=512, verbose=False):
        self.rnn_model = rnn_model
        self.dac_model = dac_model
        self.verbose = verbose
        self.device = next(rnn_model.parameters()).device

        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.right_context = right_context
        self.left_context = chunk_size - hop_size - right_context
        self.frame_samples = frame_samples
        self.clamp_val = rnn_model.config.clamp_val

        assert self.left_context >= 0, \
            f"chunk_size {chunk_size} must be >= hop_size + right_context"

        self.reset()

    def reset(self):
        """Clear all internal state."""
        self.hidden = None
        self.latent_buffer = []
        self.current_pos = 0
        self.warmed_up = False
        self._n_q = self.rnn_model.config.n_q
        self._D = self.rnn_model.config.codebook_dim

    def warmup(self, cond_value):
        """
        Generate chunk_size latent frames to fill the buffer.

        cond_value: [p] or [1, p] tensor — repeated across all
                    chunk_size frames.
        """
        if cond_value.dim() == 1:
            cond_value = cond_value.unsqueeze(0)
        B = cond_value.shape[0]
        cond_rep = cond_value.unsqueeze(1).expand(B, self.chunk_size, -1)

        latent_t = torch.zeros(B, self._n_q * self._D, device=self.device)

        init_latents, _, self.hidden = generate_latent_hop(
            self.rnn_model, latent_t, cond_rep, self.hidden)

        self.latent_buffer = [init_latents]
        self.current_pos = 0
        self.warmed_up = True

    @torch.no_grad()
    def generate_hop(self, cond_value):
        """
        Generate one hop of audio.

        cond_value: [p] or [1, p] tensor — current conditioning
                    values, duplicated for all hop_size frames.

        Returns: [1, hop_size * frame_samples] tensor of audio
                 samples in [-1, 1].
        """
        if not self.warmed_up:
            self.warmup(cond_value)

        if cond_value.dim() == 1:
            cond_value = cond_value.unsqueeze(0)
        B = cond_value.shape[0]

        cond_hop = cond_value.unsqueeze(1).expand(B, self.hop_size, -1)

        latent_t = self.latent_buffer[-1][:, -1, :]

        new_latents, _, self.hidden = generate_latent_hop(
            self.rnn_model, latent_t, cond_hop, self.hidden)

        self.latent_buffer.append(new_latents)

        all_latents = torch.cat(self.latent_buffer, dim=1)

        start = self.current_pos
        win_start = start - self.left_context
        win_end = start + self.hop_size + self.right_context

        src_start = max(0, win_start)
        src_end = min(all_latents.shape[1], win_end)

        z_chunk = all_latents[:, src_start:src_end, :]

        if src_start > win_start:
            pad = z_chunk[:, 0:1, :].expand(B, src_start - win_start, -1)
            z_chunk = torch.cat([pad, z_chunk], dim=1)
        if src_end < win_end:
            pad = z_chunk[:, -1:, :].expand(B, win_end - src_end, -1)
            z_chunk = torch.cat([z_chunk, pad], dim=1)

        z_chunk = (z_chunk * self.clamp_val).transpose(1, 2)
        z_proj, _, _ = self.dac_model.quantizer.from_latents(z_chunk)
        audio_chunk = self.dac_model.decode(z_proj)

        audio_start = self.left_context * self.frame_samples
        audio_end = audio_start + self.hop_size * self.frame_samples
        audio = audio_chunk[:, :, audio_start:audio_end]

        self.current_pos += self.hop_size

        if self.verbose and self.current_pos <= self.hop_size * 4:
            rms = audio.square().mean().sqrt().item()
            mx = audio.abs().max().item()
            print(f"[synth] hop_pos={self.current_pos}  audio_rms={rms:.6f}  "
                  f"max={mx:.6f}  shape={audio.shape}")

        return audio
