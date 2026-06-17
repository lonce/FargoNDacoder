from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CascadeMode = Literal["teacher", "soft", "hard"]
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

    # conditioning
    cond_injection: CondInjection = "concat"

    # soft cascade tuning
    tau_soft: float = 1.0          # temperature for soft cascade softmax (<1 sharpens)
    top_n_soft: int = 0            # 0 = no sparsification; >0 keeps only top-k logits

    # free-run training
    free_run_window: int = 32      # frames per free-run autoregressive segment

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

    Input:
      latents: [B, T, n_q * 8]   (stacked, normalized/clamped)
      cond:    [B, T, p]
      target_codes: [B, T, n_q]  (required for loss computation outside the model,
                                  and required inside the model when cascade_mode='teacher')

    Output:
      logits_per_codebook: list of n_q tensors, each [B, T, codebook_size]

    Cascade at each timestep:
      head 0 input = rnn_out[t]
      head 1 input = concat(rnn_out[t], vec_0)
      head 2 input = concat(rnn_out[t], vec_0, vec_1)
      ...

    where vec_i is chosen according to cascade_mode:
      - teacher: target 8D vector obtained by looking up target_codes in codebook_vectors
      - soft:    expected 8D vector from logits + codebook embeddings
      - hard:    argmax-selected 8D vector from codebook embeddings

    Sampling for autoregressive deployment is intentionally kept outside this model.
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

        head_cond_dim = config.cond_size if config.cond_size > 0 else 0
        self.heads = nn.ModuleList([
            nn.Linear(config.hidden_size + head_cond_dim + i * config.codebook_dim, config.codebook_size)
            for i in range(config.n_q)
        ])

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

    def _expected_vector(self, logits: torch.Tensor, q_idx: int) -> torch.Tensor:
        logits = logits / self.config.tau_soft
        if self.config.top_n_soft > 0:
            values, indices = torch.topk(logits, self.config.top_n_soft, dim=-1)
            masked = torch.full_like(logits, float('-inf'))
            masked.scatter_(dim=-1, index=indices, src=values)
            logits = masked
        probs = F.softmax(logits, dim=-1)
        codebook = self.codebook_vectors[q_idx]  # [K, D]
        return probs @ codebook

    def _hard_vector(self, logits: torch.Tensor, q_idx: int) -> torch.Tensor:
        idx = torch.argmax(logits, dim=-1)  # [N]
        codebook = self.codebook_vectors[q_idx]  # [K, D]
        return codebook[idx]

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
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            rnn_out: [B, T, H]
            cascade_mode: 'teacher' | 'soft' | 'hard'
            target_codes: [B, T, n_q], required when teacher forcing is used
            cond: [B, T, cond_size] or None

        Returns:
            logits_per_q: list of n_q tensors, each [B, T, K]
            cascade_vectors_per_q: list of n_q tensors, each [B, T, D]
        """
        B, T, H = rnn_out.shape
        N = B * T

        rnn_flat = rnn_out.reshape(N, H)
        cond_flat = cond.reshape(N, -1) if cond is not None else None
        teacher_flat = None
        if cascade_mode == "teacher":
            if target_codes is None:
                raise ValueError("target_codes is required when cascade_mode='teacher'")
            teacher = self._teacher_vectors_from_codes(target_codes)
            teacher_flat = teacher.reshape(N, self.config.n_q, self.config.codebook_dim)

        logits_per_q_flat: List[torch.Tensor] = []
        vecs_per_q_flat: List[torch.Tensor] = []

        for q in range(self.config.n_q):
            parts = [rnn_flat]
            if cond_flat is not None:
                parts.append(cond_flat)
            if q > 0:
                prev_vecs = torch.cat(vecs_per_q_flat, dim=-1)  # [N, q * D]
                parts.append(prev_vecs)
            head_in = torch.cat(parts, dim=-1)

            logits_q = self.heads[q](head_in)  # [N, K]
            logits_per_q_flat.append(logits_q)

            if cascade_mode == "teacher":
                assert teacher_flat is not None
                vec_q = teacher_flat[:, q, :]
            elif cascade_mode == "soft":
                vec_q = self._expected_vector(logits_q, q)
            elif cascade_mode == "hard":
                vec_q = self._hard_vector(logits_q, q)
            else:
                raise ValueError(f"Unsupported cascade_mode: {cascade_mode}")

            vecs_per_q_flat.append(vec_q)

        logits_per_q = [x.reshape(B, T, self.config.codebook_size) for x in logits_per_q_flat]
        vecs_per_q = [x.reshape(B, T, self.config.codebook_dim) for x in vecs_per_q_flat]
        return logits_per_q, vecs_per_q

    def forward(
        self,
        latents: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        target_codes: Optional[torch.Tensor] = None,
        cascade_mode: CascadeMode = "soft",
        return_expected_vectors: bool = True,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor] | None]:
        """
        Args:
            latents: [B, T, n_q * codebook_dim] stacked DAC latents
            cond: [B, T, cond_size]
            hidden: [num_layers, B, hidden_size]
            target_codes: [B, T, n_q], required for cascade_mode='teacher'
            cascade_mode: 'teacher' | 'soft' | 'hard'
            return_expected_vectors: include the per-codebook 8D vectors used/generated in the cascade

        Returns dict with:
            logits_per_codebook: list[n_q] of [B, T, codebook_size]
            hidden: [num_layers, B, hidden_size]
            rnn_out: [B, T, hidden_size]
            expected_vectors_per_codebook: list[n_q] of [B, T, codebook_dim] or None
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
        logits_per_q, vecs_per_q = self._run_codebook_cascade(
            rnn_out,
            cascade_mode=cascade_mode,
            target_codes=target_codes,
            cond=cond,
        )

        return {
            "logits_per_codebook": logits_per_q,
            "hidden": hidden_out,
            "rnn_out": rnn_out,
            "expected_vectors_per_codebook": vecs_per_q if return_expected_vectors else None,
        }

    @torch.no_grad()
    def forward_step(
        self,
        latent_t: torch.Tensor,
        cond_t: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        target_codes_t: Optional[torch.Tensor] = None,
        cascade_mode: CascadeMode = "soft",
        return_expected_vectors: bool = True,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor] | None]:
        """
        Single-timestep convenience wrapper.

        Args:
            latent_t: [B, n_q * codebook_dim]
            cond_t: [B, cond_size]
            hidden: [num_layers, B, hidden_size]
            target_codes_t: [B, n_q] when cascade_mode='teacher'
        """
        if latent_t.ndim != 2:
            raise ValueError(f"latent_t must have shape [B, D], got {tuple(latent_t.shape)}")
        latents = latent_t.unsqueeze(1)
        cond = cond_t.unsqueeze(1) if cond_t is not None else None
        target_codes = target_codes_t.unsqueeze(1) if target_codes_t is not None else None
        return self.forward(
            latents=latents,
            cond=cond,
            hidden=hidden,
            target_codes=target_codes,
            cascade_mode=cascade_mode,
            return_expected_vectors=return_expected_vectors,
        )

    def free_run_forward(
        self,
        latent_seed: torch.Tensor,
        cond_seq: torch.Tensor,
        target_codes: torch.Tensor,
        quantizer_weights: Optional[torch.Tensor] = None,
        cascade_mode: CascadeMode = "soft",
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """
        Free-running autoregressive forward pass for training.

        Unlike the normal batched forward, each timestep's input is the
        previous step's predicted codebook vector with straight-through
        estimation:
          - Forward: discrete (argmax) codebook vectors → matches the
            training distribution the GRU expects.
          - Backward: gradient flows through soft expected vectors →
            fully differentiable.

        Args:
            latent_seed: [B, n_q * codebook_dim]  ground-truth first-frame
                          latents (detached, already at dataset norm scale).
            cond_seq:    [B, T, cond_size]         conditioning sequence.
            target_codes: [B, T, n_q]              ground-truth code indices.
            quantizer_weights: [n_q] or None.
            cascade_mode: "soft" or "hard" (teacher not supported here).

        Returns:
            Same dict as compute_loss: {"total_loss", "per_codebook_losses"}.
        """
        B, T, _ = cond_seq.shape
        T = min(T, self.config.free_run_window)
        n_q = self.config.n_q
        device = latent_seed.device

        all_logits: List[List[torch.Tensor]] = [[] for _ in range(n_q)]

        # Seed is already normalised by the dataset (raw_latent / clamp_val).
        # Do NOT divide again — the GRU expects the dataset-normalised scale.
        latent_t = torch.clamp(latent_seed, -1.0, 1.0)

        hidden: Optional[torch.Tensor] = None

        for t in range(T):
            cond_t = cond_seq[:, t, :]  # [B, cond_size]

            out = self.forward(
                latents=latent_t.unsqueeze(1),                # [B, 1, D]
                cond=cond_t.unsqueeze(1),                     # [B, 1, C]
                hidden=hidden,
                cascade_mode=cascade_mode,
                return_expected_vectors=True,
            )

            hidden = out["hidden"]

            for q in range(n_q):
                all_logits[q].append(out["logits_per_codebook"][q])  # [B, 1, K]

            # Build next-step GRU input using straight-through estimation.
            # Forward pass: discrete (argmax) codebook vectors → matches training distribution.
            # Backward pass: gradient flows through soft expected vectors → differentiable.
            vecs_soft = out["expected_vectors_per_codebook"]  # n_q × [B, 1, 8]
            latent_t_soft = torch.cat(vecs_soft, dim=-1).squeeze(1)  # [B, 72]

            hard_vecs = []
            for q in range(n_q):
                logits_q = out["logits_per_codebook"][q]  # [B, 1, K]
                hard_vec = self._hard_vector(logits_q.squeeze(1), q)  # [B, 8]
                hard_vecs.append(hard_vec)
            latent_t_hard = torch.cat(hard_vecs, dim=-1)  # [B, 72]

            latent_t = latent_t_hard.detach() + latent_t_soft - latent_t_soft.detach()

            latent_t = torch.clamp(
                latent_t, -self.config.clamp_val, self.config.clamp_val
            ) / self.config.clamp_val

        # Stack per-timestep logits → standard [B, T, K]
        stacked_logits = [
            torch.cat(q_list, dim=1) for q_list in all_logits
        ]
        stacked_targets = target_codes[:, :T, :]

        return self.compute_loss(stacked_logits, stacked_targets, quantizer_weights)

    def compute_loss(
        self,
        logits_per_codebook: List[torch.Tensor],
        target_codes: torch.Tensor,
        quantizer_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        if target_codes.ndim != 3:
            raise ValueError(f"target_codes must have shape [B, T, n_q], got {tuple(target_codes.shape)}")
        if target_codes.shape[-1] != self.config.n_q:
            raise ValueError(
                f"target_codes last dim must equal n_q={self.config.n_q}, got {target_codes.shape[-1]}"
            )
        if len(logits_per_codebook) != self.config.n_q:
            raise ValueError(
                f"Expected {self.config.n_q} logits tensors, got {len(logits_per_codebook)}"
            )

        losses: List[torch.Tensor] = []
        for q, logits_q in enumerate(logits_per_codebook):
            B, T, K = logits_q.shape
            target_q = target_codes[:, :, q].reshape(B * T)
            loss_q = F.cross_entropy(
                logits_q.reshape(B * T, K),
                target_q.long(),
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

        # Replace the variable-width cascade heads from the base class with
        # fixed-width heads that depend only on the GRU output and cond.
        head_cond_dim = config.cond_size if config.cond_size > 0 else 0
        self.heads = nn.ModuleList([
            nn.Linear(config.hidden_size + head_cond_dim, config.codebook_size)
            for _ in range(config.n_q)
        ])

    def _run_codebook_cascade(
        self,
        rnn_out: torch.Tensor,
        cascade_mode: CascadeMode,
        target_codes: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Predict each codebook independently from the shared RNN representation.

        Args:
            rnn_out: [B, T, H]
            cascade_mode: accepted for API compatibility, ignored
            target_codes: accepted for API compatibility, ignored
            cond: [B, T, cond_size] or None

        Returns:
            logits_per_q: list of n_q tensors, each [B, T, K]
            vecs_per_q: list of n_q tensors, each [B, T, D]
                Each vector is the expected 8D vector derived from that head's logits,
                returned only for inspection/debugging.
        """
        B, T, H = rnn_out.shape
        N = B * T
        rnn_flat = rnn_out.reshape(N, H)
        cond_flat = cond.reshape(N, -1) if cond is not None else None

        logits_per_q_flat: List[torch.Tensor] = []
        vecs_per_q_flat: List[torch.Tensor] = []

        for q in range(self.config.n_q):
            parts = [rnn_flat]
            if cond_flat is not None:
                parts.append(cond_flat)
            head_in = torch.cat(parts, dim=-1)
            logits_q = self.heads[q](head_in)  # [N, K]
            logits_per_q_flat.append(logits_q)

            # Returned for inspection/debugging only; not used by any later head.
            vec_q = self._expected_vector(logits_q, q)
            vecs_per_q_flat.append(vec_q)

        logits_per_q = [x.reshape(B, T, self.config.codebook_size) for x in logits_per_q_flat]
        vecs_per_q = [x.reshape(B, T, self.config.codebook_dim) for x in vecs_per_q_flat]
        return logits_per_q, vecs_per_q


__all__ = [
    "GRUModelConfig",
    "TrainingConfig",
    "RNNDACModel",
    "RNNDACModelNoCascade",
]
