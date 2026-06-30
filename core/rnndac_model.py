from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CascadeMode = Literal["teacher", "free"]
CondInjection = Literal["concat", "film"]


@dataclass
class GRUModelConfig:
    # codec / input structure
    n_q: int = 9
    codebook_size: int = 1024
    codebook_dim: int = 8
    input_size: int = 72          # typically n_q * codebook_dim for stacked DAC latents
    cond_size: int = 0
    clamp_val: float = 15.0

    # embeddings / projections
    input_embed_size: int = 128
    cond_embed_size: int = 0      # deprecated — use inp_proportion/cond_proportion instead
    film_hidden_size: int = 128   # only used for film mode

    # proportion of the 128 GRU input dimensions devoted to latent vs cond (concat mode)
    inp_proportion: int = 5
    cond_proportion: int = 2

    # recurrent core
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    rnn_dropout: float = 0.0       # dropout applied to GRU output before cascade (training only)

    # classification (logits over codebook entries instead of 8D regression)
    tau: float = 1.0               # softmax temperature for codebook prediction

    # conditioning
    cond_injection: CondInjection = "concat"

    # bookkeeping
    model_version: str = "rnndac_v1"

    def __post_init__(self) -> None:
        expected_input = self.n_q * self.codebook_dim
        if self.input_size != expected_input:
            raise ValueError(
                f"input_size must equal n_q * codebook_dim ({expected_input}), got {self.input_size}"
            )
        if self.cond_size < 0:
            raise ValueError("cond_size must be >= 0")
        if self.cond_injection not in {"concat", "film"}:
            raise ValueError(f"Unsupported cond_injection: {self.cond_injection}")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "GRUModelConfig":
        return cls(**d)


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 16
    teacher_forcing_schedule: Optional[str] = None
    gradient_clip_norm: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainingConfig":
        return cls(**d)


class FiLMBlock(nn.Module):
    """Simple feature-wise affine modulation from conditioning."""

    def __init__(self, cond_size: int, hidden_size: int, film_hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_size, film_hidden_size),
            nn.ReLU(),
            nn.Linear(film_hidden_size, hidden_size * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(cond)  # [B, T, 2H]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return (1.0 + gamma) * x + beta


class RNNDACModel(nn.Module):
    """
    GRU-based DAC code predictor with per-timestep codebook cascade.

    Each cascade head predicts logits over 1024 codebook entries (classification).
    During training, softmax with temperature tau provides differentiable
    weights for codebook mixing. During inference, argmax picks the hard index.

    Input:
      latents: [B, T, n_q * 8]   (stacked, normalized/clamped)
      cond:    [B, T, p]
      target_codes: [B, T, n_q]  (required for loss computation outside the model,
                                  and required inside the model when cascade_mode='teacher')

    Output:
      predicted_logits_per_codebook: list of n_q tensors, each [B, T, 1024]

    Cascade at each timestep:
      head 0 input = rnn_out[t]
      head 1 input = concat(rnn_out[t], vec_0)
      head 2 input = concat(rnn_out[t], vec_0, vec_1)
      ...

    where vec_i is chosen according to cascade_mode:
      - teacher: target vector obtained by looking up target_codes in codebook_vectors
      - free:    softmax-weighted (train) or argmax (eval) codebook vector

    Inference uses cascade_mode='free': argmax picks the codebook index,
    and the resulting vector (normalized) is fed back to the next head.
    """

    def __init__(
        self,
        config: GRUModelConfig,
        dac_model=None,
        codebook_vectors: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.config = config

        if (dac_model is None) == (codebook_vectors is None):
            raise ValueError("Provide exactly one of `dac_model` or `codebook_vectors`.")

        if codebook_vectors is None:
            try:
                quantizers = dac_model.quantizer.quantizers
            except Exception as e:
                raise ValueError(
                    "Could not access dac_model.quantizer.quantizers. "
                    "Please provide a valid pretrained DAC model."
                ) from e

            if len(quantizers) < config.n_q:
                raise ValueError(
                    f"DAC model only has {len(quantizers)} quantizers, but config.n_q={config.n_q}."
                )

            vectors = []
            for q in range(config.n_q):
                try:
                    w = quantizers[q].codebook.weight.detach().clone()
                except Exception as e:
                    raise ValueError(f"Could not extract codebook.weight for quantizer {q}.") from e
                vectors.append(w)

            codebook_vectors = torch.stack(vectors, dim=0)

        if codebook_vectors.ndim != 3:
            raise ValueError(
                "codebook_vectors must have shape "
                f"[n_q, codebook_size, codebook_dim], got {tuple(codebook_vectors.shape)}"
            )

        nq, k, d = codebook_vectors.shape
        if nq != config.n_q or k != config.codebook_size or d != config.codebook_dim:
            raise ValueError(
                "codebook_vectors shape mismatch: expected "
                f"[{config.n_q}, {config.codebook_size}, {config.codebook_dim}], "
                f"got {tuple(codebook_vectors.shape)}"
            )

        self.register_buffer("codebook_vectors", codebook_vectors.float())

        if config.cond_injection == "concat":
            if config.cond_size > 0:
                total_slots = config.inp_proportion + config.cond_proportion
                lpn = config.inp_proportion * config.input_embed_size // total_slots
                lcn = config.input_embed_size - lpn
                self.latent_proj = nn.Linear(config.input_size, lpn)
                self.cond_proj = nn.Linear(config.cond_size, lcn)
                gru_input_size = lpn + lcn
            else:
                self.latent_proj = nn.Linear(config.input_size, config.input_embed_size)
                self.cond_proj = None
                gru_input_size = config.input_embed_size
            self.film = None
        else:  # film
            self.cond_proj = None
            self.latent_proj = nn.Linear(config.input_size, config.input_embed_size)
            gru_input_size = config.input_embed_size
            self.film = (
                FiLMBlock(config.cond_size, config.input_embed_size, config.film_hidden_size)
                if config.cond_size > 0
                else None
            )

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.heads = nn.ModuleList([
            nn.Linear(config.hidden_size + i * config.codebook_dim, config.codebook_size)
            for i in range(config.n_q)
        ])

        self.rnn_dropout = nn.Dropout(config.rnn_dropout) if config.rnn_dropout > 0 else nn.Identity()

    def _prepare_inputs(self, latents: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.latent_proj(latents)

        if self.config.cond_injection == "concat":
            if self.config.cond_size > 0:
                if cond is None:
                    raise ValueError("cond is required when cond_size > 0")
                cond_feat = self.cond_proj(cond)
                x = torch.cat([x, cond_feat], dim=-1)
        else:  # film
            if self.film is not None:
                if cond is None:
                    raise ValueError("cond is required when cond_size > 0")
                x = self.film(x, cond)
        return x

    def _teacher_vectors_from_codes(self, target_codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_codes: [B, T, n_q]
        Returns:
            teacher vectors: [B, T, n_q, D]
        """
        if target_codes.ndim != 3:
            raise ValueError(
                f"target_codes must have shape [B, T, n_q], got {tuple(target_codes.shape)}"
            )
        if target_codes.shape[-1] != self.config.n_q:
            raise ValueError(
                f"target_codes last dim must equal n_q={self.config.n_q}, got {target_codes.shape[-1]}"
            )

        vecs = []
        for q in range(self.config.n_q):
            idx_q = target_codes[:, :, q].long()        # [B, T]
            codebook_q = self.codebook_vectors[q]       # [K, D]
            vec_q = codebook_q[idx_q]                   # [B, T, D]
            vecs.append(vec_q)
        return torch.stack(vecs, dim=2)                # [B, T, n_q, D]

    def _run_codebook_cascade(
        self,
        rnn_out: torch.Tensor,
        cascade_mode: CascadeMode,
        target_codes: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Predict codebook indices via logits, with differentiable codebook
        mixing using softmax temperature tau.

        Args:
            rnn_out: [B, T, H]
            cascade_mode: 'teacher' | 'free'
            target_codes: [B, T, n_q], required when cascade_mode='teacher'
            tau: softmax temperature (default: self.config.tau)

        Returns:
            logits_per_q: list of n_q tensors, each [B, T, codebook_size]
            cascade_vectors: list of n_q tensors, each [B, T, codebook_dim]
                             (normalized to [-1,1])
        """
        cfg = self.config
        B, T, H = rnn_out.shape
        tau = tau if tau is not None else cfg.tau

        teacher_flat = None
        if cascade_mode == "teacher":
            if target_codes is None:
                raise ValueError("target_codes is required when cascade_mode='teacher'")
            teacher = self._teacher_vectors_from_codes(target_codes)  # [B, T, n_q, D]
            teacher_flat = teacher.reshape(-1, cfg.n_q, cfg.codebook_dim)  # [N, n_q, D]

        logits_per_q: List[torch.Tensor] = []
        cascade_vectors: List[torch.Tensor] = []

        for q in range(cfg.n_q):
            parts = [rnn_out]  # [B, T, H]
            if q > 0:
                prev = torch.cat(cascade_vectors, dim=-1)  # [B, T, q*D]
                parts.append(prev * cfg.clamp_val)          # undo normalization
            head_in = torch.cat(parts, dim=-1)  # [B, T, H + q*D]

            logits_q = self.heads[q](head_in)  # [B, T, K]
            logits_per_q.append(logits_q)

            # Derive cascade vectors for next head
            if cascade_mode == "teacher" and teacher_flat is not None:
                N = B * T
                vec_raw = teacher_flat[:, q, :].reshape(B, T, -1)  # [B, T, D]
            elif self.training:
                probs = F.softmax(logits_q / tau, dim=-1)           # [B, T, K]
                vec_raw = probs @ self.codebook_vectors[q]          # [B, T, D]
            else:
                idx = logits_q.argmax(dim=-1)                        # [B, T]
                vec_raw = self.codebook_vectors[q][idx]              # [B, T, D]

            cascade_vectors.append(vec_raw / cfg.clamp_val)

        return logits_per_q, cascade_vectors

    def forward_autoregressive(
        self,
        latents: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        target_codes: Optional[torch.Tensor] = None,
        warmup_steps: int = 0,
        tau: Optional[float] = None,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor] | None]:
        """
        Autoregressive forward pass for scheduled sampling during training.

        Phase 1 (warmup): batched forward with ground-truth latents for the
        first `warmup_steps` frames.  All predictions are from 'free' cascade.

        Phase 2 (autoregressive): for each remaining frame the GRU input is
        the *model's own* predicted latent from the previous frame — exactly as
        during inference.  The cascade is always 'free'.

        Args:
            latents: [B, T, n_q * codebook_dim]  (ground truth, used for warmup)
            cond: [B, T, cond_size]
            target_codes: [B, T, n_q]  (used only for loss outside, not forced)
            warmup_steps: number of initial frames that receive GT GRU input.
            tau: softmax temperature (default: self.config.tau)

        Returns:
            Same dict as forward().
        """
        B, T, D = latents.shape
        if warmup_steps < 0 or warmup_steps > T:
            raise ValueError(f"warmup_steps must be in [0, T={T}], got {warmup_steps}")
        if cond is not None:
            if cond.shape[:2] != (B, T):
                raise ValueError("cond and latents must agree on [B, T]")

        # ------------------------------------------------------------------
        # Phase 1 — warmup (batched, GT input, free cascade)
        # ------------------------------------------------------------------
        warmup_logits_per_q: List[torch.Tensor] = []
        warmup_hidden = None

        if warmup_steps > 0:
            out_warmup = self.forward(
                latents=latents[:, :warmup_steps, :],
                cond=cond[:, :warmup_steps, :] if cond is not None else None,
                cascade_mode="free",
                tau=tau,
            )
            warmup_logits_per_q = out_warmup["predicted_logits_per_codebook"]
            warmup_cascade_vectors = out_warmup["cascade_vectors"]
            warmup_hidden = out_warmup["hidden"]
        else:
            # Seed step: use first GT frame as GRU input, predict frame 0.
            cond_0 = cond[:, 0:1, :] if cond is not None else None
            x_0 = self._prepare_inputs(latents[:, 0:1, :], cond_0)
            rnn_0, hidden_seed = self.gru(x_0, None)
            rnn_0 = self.rnn_dropout(rnn_0)
            logits_0, cascade_vecs_0 = self._run_codebook_cascade(
                rnn_0, cascade_mode="free", tau=tau,
            )
            warmup_logits_per_q = logits_0          # list of [B, 1, K]
            warmup_cascade_vectors = cascade_vecs_0  # list of [B, 1, D]
            warmup_hidden = hidden_seed
            warmup_steps = 1  # step 0 is done

        # ------------------------------------------------------------------
        # Phase 2 — autoregressive
        # ------------------------------------------------------------------
        autoreg_logits: List[List[torch.Tensor]] = [
            [] for _ in range(self.config.n_q)
        ]
        hidden = warmup_hidden

        # The last cascade vectors from warmup become the first AR input
        last_pred_flat = torch.cat(
            [warmup_cascade_vectors[q][:, -1:, :] for q in range(self.config.n_q)],
            dim=-1,
        )  # [B, 1, D]  (normalized vectors)

        for t in range(warmup_steps, T):
            cond_t = cond[:, t : t + 1, :] if cond is not None else None
            x_t = self._prepare_inputs(last_pred_flat, cond_t)

            rnn_t, hidden = self.gru(x_t, hidden)
            rnn_t = self.rnn_dropout(rnn_t)

            logits_t, cascade_vecs_t = self._run_codebook_cascade(
                rnn_t, cascade_mode="free", tau=tau,
            )
            for q in range(self.config.n_q):
                autoreg_logits[q].append(logits_t[q])

            last_pred_flat = torch.cat(
                [cascade_vecs_t[q] for q in range(self.config.n_q)], dim=-1
            )

        # ------------------------------------------------------------------
        # Concatenate warmup + autoregressive logits
        # ------------------------------------------------------------------
        all_logits_per_q: List[torch.Tensor] = []
        for q in range(self.config.n_q):
            warmup_q = warmup_logits_per_q[q]  # [B, warmup_steps, K]
            autoreg_q = (
                torch.cat(autoreg_logits[q], dim=1)
                if autoreg_logits[q]
                else torch.empty(
                    B, 0, self.config.codebook_size,
                    device=latents.device,
                )
            )
            all_logits_per_q.append(
                torch.cat([warmup_q, autoreg_q], dim=1)
            )

        return {
            "predicted_logits_per_codebook": all_logits_per_q,
            "hidden": hidden,
        }

    def forward(
        self,
        latents: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        target_codes: Optional[torch.Tensor] = None,
        cascade_mode: CascadeMode = "free",
        tau: Optional[float] = None,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor] | None]:
        """
        Args:
            latents: [B, T, n_q * codebook_dim] stacked DAC latents
            cond: [B, T, cond_size]
            hidden: [num_layers, B, hidden_size]
            target_codes: [B, T, n_q], required for cascade_mode='teacher'
            cascade_mode: 'teacher' | 'free'
            tau: softmax temperature (default: self.config.tau)

        Returns dict with:
            predicted_logits_per_codebook: list[n_q] of [B, T, codebook_size]
            cascade_vectors: list[n_q] of [B, T, codebook_dim] (normalized)
            hidden: [num_layers, B, hidden_size]
            rnn_out: [B, T, hidden_size]
        """
        if latents.ndim != 3:
            raise ValueError(f"latents must have shape [B, T, D], got {tuple(latents.shape)}")
        if latents.shape[-1] != self.config.input_size:
            raise ValueError(
                f"latents last dim must equal input_size={self.config.input_size}, got {latents.shape[-1]}"
            )
        if cond is not None and cond.ndim != 3:
            raise ValueError(f"cond must have shape [B, T, C], got {tuple(cond.shape)}")
        if cond is not None and cond.shape[:2] != latents.shape[:2]:
            raise ValueError("cond and latents must agree on [B, T]")
        if cond is not None and cond.shape[-1] != self.config.cond_size:
            raise ValueError(
                f"cond last dim must equal cond_size={self.config.cond_size}, got {cond.shape[-1]}"
            )
        if target_codes is not None:
            if target_codes.ndim != 3:
                raise ValueError(
                    f"target_codes must have shape [B, T, n_q], got {tuple(target_codes.shape)}"
                )
            if target_codes.shape[:2] != latents.shape[:2]:
                raise ValueError("target_codes and latents must agree on [B, T]")
            if target_codes.shape[-1] != self.config.n_q:
                raise ValueError(
                    f"target_codes last dim must equal n_q={self.config.n_q}, got {target_codes.shape[-1]}"
                )

        x = self._prepare_inputs(latents, cond)
        rnn_out, hidden_out = self.gru(x, hidden)
        rnn_out = self.rnn_dropout(rnn_out)  # training-only dropout for head robustness
        logits_per_q, cascade_vectors = self._run_codebook_cascade(
            rnn_out,
            cascade_mode=cascade_mode,
            target_codes=target_codes,
            tau=tau,
        )

        return {
            "predicted_logits_per_codebook": logits_per_q,
            "cascade_vectors": cascade_vectors,
            "hidden": hidden_out,
            "rnn_out": rnn_out,
        }

    @torch.no_grad()
    def forward_step(
        self,
        latent_t: torch.Tensor,
        cond_t: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        target_codes_t: Optional[torch.Tensor] = None,
        cascade_mode: CascadeMode = "free",
        tau: float = 0.0,
        top_k: int = 0,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor] | None]:
        """
        Single-timestep convenience wrapper.

        Args:
            latent_t: [B, n_q * codebook_dim]
            cond_t: [B, cond_size]
            hidden: [num_layers, B, hidden_size]
            target_codes_t: [B, n_q] when cascade_mode='teacher'
            tau: softmax temperature (default 0 = argmax).
            top_k: keep only top-k logits before sampling (0 = disabled).
        """
        if latent_t.ndim != 2:
            raise ValueError(f"latent_t must have shape [B, D], got {tuple(latent_t.shape)}")
        latents = latent_t.unsqueeze(1)
        cond = cond_t.unsqueeze(1) if cond_t is not None else None
        target_codes = target_codes_t.unsqueeze(1) if target_codes_t is not None else None
        out = self.forward(
            latents=latents,
            cond=cond,
            hidden=hidden,
            target_codes=target_codes,
            cascade_mode=cascade_mode,
            tau=tau,
        )

        # Convert logits → vectors with optional top-k + temperature sampling
        logits_per_q = out["predicted_logits_per_codebook"]
        vecs = []
        for q, logits_q in enumerate(logits_per_q):
            logits_q = logits_q.squeeze(1)  # [B, K]

            if top_k > 0:
                top_k_vals, _ = logits_q.topk(top_k, dim=-1)
                logits_q = logits_q.masked_fill(
                    logits_q < top_k_vals[..., -1:], float('-inf')
                )

            if tau > 0:
                probs = F.softmax(logits_q / tau, dim=-1)
                idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                idx = logits_q.argmax(dim=-1)

            vec = self.codebook_vectors[q][idx] / self.config.clamp_val
            vecs.append(vec)
        pred_vectors = torch.stack(vecs, dim=1)  # [B, n_q, D], normalized

        return {
            "predicted_vectors_per_codebook": pred_vectors,
            "predicted_logits_per_codebook": out["predicted_logits_per_codebook"],
            "hidden": out["hidden"],
            "rnn_out": out["rnn_out"],
        }

    def compute_loss(
        self,
        predicted_logits: List[torch.Tensor],
        target_codes: torch.Tensor,
        quantizer_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        if target_codes.ndim != 3:
            raise ValueError(f"target_codes must have shape [B, T, n_q], got {tuple(target_codes.shape)}")
        if target_codes.shape[-1] != self.config.n_q:
            raise ValueError(
                f"target_codes last dim must equal n_q={self.config.n_q}, got {target_codes.shape[-1]}"
            )
        if len(predicted_logits) != self.config.n_q:
            raise ValueError(
                f"Expected {self.config.n_q} logit tensors, got {len(predicted_logits)}"
            )

        K = self.config.codebook_size
        losses: List[torch.Tensor] = []
        for q, logits_q in enumerate(predicted_logits):
            targets_q = target_codes[:, :, q].long()          # [B, T]
            loss_q = F.cross_entropy(
                logits_q.reshape(-1, K),
                targets_q.reshape(-1),
                reduction="mean",
            )
            losses.append(loss_q)

        if quantizer_weights is None:
            total_loss = torch.stack(losses).sum()
        else:
            if quantizer_weights.shape != (self.config.n_q,):
                raise ValueError(
                    f"quantizer_weights must have shape [{self.config.n_q}], got {tuple(quantizer_weights.shape)}"
                )
            total_loss = torch.stack([w * l for w, l in zip(quantizer_weights, losses)]).sum()

        return {
            "total_loss": total_loss,
            "per_codebook_losses": losses,
        }




class RNNDACModelNoCascade(RNNDACModel):
    """
    Ablation model: predicts every codebook directly from the shared RNN output,
    with no lower-order codebook information passed between heads.

    Notes
    -----
    - The GRU, conditioning injection, input projections, codebook buffers, and
      loss computation are inherited unchanged from RNNDACModel.
    - `cascade_mode` is accepted for API compatibility with the base class, but
      it does not affect behavior here because no inter-codebook cascade exists.
    - Each head sees only `rnn_out[t]`.
    """

    def __init__(
        self,
        config: GRUModelConfig,
        dac_model=None,
        codebook_vectors: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(config=config, dac_model=dac_model, codebook_vectors=codebook_vectors)

        # Fixed-width heads depending only on GRU output (cond already injected
        # into RNN input via _prepare_inputs, so it's in the hidden state).
        self.heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.codebook_size)
            for _ in range(config.n_q)
        ])

    def _run_codebook_cascade(
        self,
        rnn_out: torch.Tensor,
        cascade_mode: CascadeMode,
        target_codes: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Predict each codebook independently from the shared RNN representation.

        Args:
            rnn_out: [B, T, H]
            cascade_mode: accepted for API compatibility, ignored
            target_codes: accepted for API compatibility, ignored
            tau: softmax temperature (default: self.config.tau)

        Returns:
            logits_per_q: list of n_q tensors, each [B, T, codebook_size]
            cascade_vectors: list of n_q tensors, each [B, T, codebook_dim]
                             (normalized, used for API consistency)
        """
        cfg = self.config
        B, T, H = rnn_out.shape
        tau = tau if tau is not None else cfg.tau

        logits_per_q: List[torch.Tensor] = []
        cascade_vectors: List[torch.Tensor] = []

        for q in range(cfg.n_q):
            logits_q = self.heads[q](rnn_out)  # [B, T, K]
            logits_per_q.append(logits_q)

            if self.training:
                probs = F.softmax(logits_q / tau, dim=-1)
                vec_raw = probs @ self.codebook_vectors[q]  # [B, T, D]
            else:
                idx = logits_q.argmax(dim=-1)                # [B, T]
                vec_raw = self.codebook_vectors[q][idx]      # [B, T, D]

            cascade_vectors.append(vec_raw / cfg.clamp_val)

        return logits_per_q, cascade_vectors


__all__ = [
    "GRUModelConfig",
    "TrainingConfig",
    "RNNDACModel",
    "RNNDACModelNoCascade",
]
