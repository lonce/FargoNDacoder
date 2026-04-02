
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from datasets import load_from_disk
except ImportError:
    load_from_disk = None

FilterSpec = Dict[str, Set[int]]


@dataclass
class LatentDatasetConfig:
    dataset_path: str
    sequence_length: int
    add_noise: bool = False
    noise_weight: float = 0.1
    n_q: int = 9
    clamp_val: float = 15.0
    filters: Optional[FilterSpec] = None
    files_per_sequence: int = 4


class RNNDACLatentDataset(Dataset):
    """
    RNNDAC dataset that reads a HuggingFace dataset created in step 4 and returns:

        {
            "latents": [T, n_q * 8],
            "targets": [T, n_q],
            "cond":    [T, p],
        }

    Autoregressive alignment
    ------------------------
    The returned tensors are aligned so that:

        latents[t], cond[t]  ->  targets[t]

    where targets[t] is the DAC code index at the *following* timestep
    relative to the latent/conditioning input timestep.

    Notes
    -----
    - Latents are returned in the user's "stacked" format: [T, n_q * 8].
    - Conditioning uses the expanded sidecar representation already created by
      dataprep step 3 / step 4:
          * one feature per continuous parameter
          * one one-hot feature per class value
    - Filters are applied at the FILE level using parameters.json semantics.
      For class parameters, a file passes if its dominant class index belongs to
      the allowed set.
    - Each returned training sequence is built from `files_per_sequence` random
      file segments. Segment lengths are split as evenly as possible.
    """

    def __init__(
        self,
        config: LatentDatasetConfig,
        dac_model_path: Optional[Union[str, Path]] = None,
        split: str = "train",
        device: str = "cpu",
    ) -> None:
        if load_from_disk is None:
            raise ImportError("Please install datasets: pip install datasets")

        self.config = config
        self.split = split
        self.device = torch.device(device)
        self.dataset_root = Path(config.dataset_path)

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_root}")
        if config.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if config.files_per_sequence <= 0:
            raise ValueError("files_per_sequence must be > 0")
        if config.n_q <= 0:
            raise ValueError("n_q must be > 0")
        if config.clamp_val <= 0:
            raise ValueError("clamp_val must be > 0")
        if config.noise_weight < 0:
            raise ValueError("noise_weight must be >= 0")

        self.parameters_config = self._load_parameters_json(self.dataset_root)
        self.conditioning_config = self._load_conditioning_config(self.dataset_root)
        self.feature_names: List[str] = self.conditioning_config.get("feature_names", [])

        hf_obj = load_from_disk(str(self.dataset_root))
        if split not in hf_obj:
            raise KeyError(f"Split '{split}' not found in dataset at {self.dataset_root}")
        self.hf_split = hf_obj[split]

        self._dac_model = self._load_dac_model(dac_model_path)
        self._file_entries = self._build_file_entries()
        self._eligible_indices = self._filter_file_indices(self._file_entries, config.filters)

        if not self._eligible_indices:
            raise ValueError("No files remain after applying filters.")

        self._length = len(self._eligible_indices)

    def _eligible_indices_for_segment_len(self, seg_len: int) -> List[int]:
        required_len = seg_len + 1
        return [
            idx for idx in self._eligible_indices
            if int(self._file_entries[idx]["num_aligned_frames"]) >= required_len
        ]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seg_lengths = self._split_lengths(self.config.sequence_length, self.config.files_per_sequence)

        latents_parts: List[torch.Tensor] = []
        targets_parts: List[torch.Tensor] = []
        cond_parts: List[torch.Tensor] = []

        # Anchor the first segment to the requested index, but only among files
        # that are long enough to provide seg_len input frames plus one following
        # target frame.
        first_seg_len = seg_lengths[0]
        first_valid = self._eligible_indices_for_segment_len(first_seg_len)
        if not first_valid:
            raise ValueError(
                f"No eligible files are long enough for segment length {first_seg_len} "
                f"(requires at least {first_seg_len + 1} aligned frames)."
            )
        anchor_pool_pos = index % len(first_valid)
        file_choices = [first_valid[anchor_pool_pos]]

        for seg_len in seg_lengths[1:]:
            valid = self._eligible_indices_for_segment_len(seg_len)
            if not valid:
                raise ValueError(
                    f"No eligible files are long enough for segment length {seg_len} "
                    f"(requires at least {seg_len + 1} aligned frames)."
                )
            file_choices.append(random.choice(valid))

        for file_idx, seg_len in zip(file_choices, seg_lengths):
            entry = self._file_entries[file_idx]
            full_latents, full_targets, full_cond = self._load_example_tensors(entry)
            T = min(full_latents.shape[0], full_targets.shape[0], full_cond.shape[0])
            required_len = seg_len + 1

            if T < required_len:
                raise RuntimeError(
                    f"Internal error: selected file {entry['audio_path']} with only {T} aligned frames "
                    f"for required segment length {seg_len} (+1 target frame)."
                )

            start = 0 if T == required_len else random.randint(0, T - required_len)
            stop = start + required_len
            src_latents = full_latents[start:stop]
            src_targets = full_targets[start:stop]
            src_cond = full_cond[start:stop]

            seg_latents = src_latents[:-1]
            seg_targets = src_targets[1:]
            seg_cond = src_cond[:-1]

            latents_parts.append(seg_latents)
            targets_parts.append(seg_targets)
            cond_parts.append(seg_cond)

        latents = torch.cat(latents_parts, dim=0)
        targets = torch.cat(targets_parts, dim=0)
        cond = torch.cat(cond_parts, dim=0)

        if self.config.add_noise and self.config.noise_weight > 0:
            latents = latents + self.config.noise_weight * torch.randn_like(latents)
            latents = torch.clamp(latents, -1.0, 1.0)

        return {
            "latents": latents,
            "targets": targets,
            "cond": cond,
        }

    # ------------------------------------------------------------------
    # File indexing and filtering
    # ------------------------------------------------------------------
    def _build_file_entries(self) -> List[Dict]:
        entries: List[Dict] = []
        for row_idx in range(len(self.hf_split)):
            row = self.hf_split[row_idx]
            if "audio" not in row:
                raise KeyError("Expected HuggingFace split rows to contain an 'audio' field.")

            rel_audio = Path(row["audio"])
            audio_path = self.dataset_root / rel_audio
            cond_path = audio_path.with_suffix(".cond.npy")

            if not audio_path.exists():
                raise FileNotFoundError(f"Missing DAC file: {audio_path}")
            if not cond_path.exists():
                raise FileNotFoundError(f"Missing conditioning sidecar: {cond_path}")

            file_level_meta = self._summarize_conditioning_file(cond_path)
            num_aligned_frames = self._infer_num_aligned_frames(audio_path, cond_path)

            entries.append(
                {
                    "row_idx": row_idx,
                    "audio_path": audio_path,
                    "cond_path": cond_path,
                    "file_level_meta": file_level_meta,
                    "num_aligned_frames": num_aligned_frames,
                }
            )
        return entries

    def _filter_file_indices(
        self,
        entries: Sequence[Dict],
        filters: Optional[FilterSpec],
    ) -> List[int]:
        if not filters:
            return list(range(len(entries)))

        valid_param_names = {spec.get("name") for spec in self.parameters_config.values()}
        for key in filters.keys():
            if key not in valid_param_names:
                raise KeyError(
                    f"Filter key '{key}' not found in parameters.json names: {sorted(valid_param_names)}"
                )

        kept: List[int] = []
        for idx, entry in enumerate(entries):
            file_meta = entry["file_level_meta"]
            keep = True
            for param_name, allowed in filters.items():
                allowed_set = set(int(x) for x in allowed)
                if param_name not in file_meta:
                    keep = False
                    break
                value = int(file_meta[param_name])
                if value not in allowed_set:
                    keep = False
                    break
            if keep:
                kept.append(idx)
        return kept

    def _infer_num_aligned_frames(self, audio_path: Path, cond_path: Path) -> int:
        cond = np.load(cond_path)
        if cond.ndim != 2:
            raise ValueError(f"Expected conditioning array [T, p], got shape {cond.shape} for {cond_path}")
        cond_len = int(cond.shape[0])

        codes_tn = self._load_codes_tn(audio_path, self.config.n_q)
        code_len = int(codes_tn.shape[0])

        return min(code_len, cond_len)

    def _summarize_conditioning_file(self, cond_path: Path) -> Dict[str, int]:
        """
        Produce file-level metadata for filtering.

        Continuous parameters are ignored for filtering here.
        Class parameters are reconstructed from the expanded one-hot columns by
        taking the most frequent active class over the file.
        """
        cond = np.load(cond_path)
        if cond.ndim != 2:
            raise ValueError(f"Expected conditioning array [T, p], got shape {cond.shape} for {cond_path}")

        name_to_index = {name: i for i, name in enumerate(self.feature_names)}
        summary: Dict[str, int] = {}

        for _, spec in self.parameters_config.items():
            param_name = spec["name"]
            param_type = spec.get("type", "continuous")
            if param_type != "class":
                continue

            classes = spec.get("classes", [])
            if classes:
                feature_names = [f"{param_name}_{cls}" for cls in classes]
            else:
                num_classes = int(spec.get("num_classes", 2))
                feature_names = [f"{param_name}_{i}" for i in range(num_classes)]

            missing = [n for n in feature_names if n not in name_to_index]
            if missing:
                raise KeyError(
                    f"conditioning_config is missing expected class features for '{param_name}': {missing}"
                )

            idxs = [name_to_index[n] for n in feature_names]
            class_block = cond[:, idxs]
            class_ids = np.argmax(class_block, axis=1)
            if class_ids.size == 0:
                dominant = 0
            else:
                counts = np.bincount(class_ids, minlength=len(idxs))
                dominant = int(np.argmax(counts))
            summary[param_name] = dominant

        return summary

    # ------------------------------------------------------------------
    # Tensor loading
    # ------------------------------------------------------------------
    def _load_example_tensors(self, entry: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        codes_tn = self._load_codes_tn(entry["audio_path"], self.config.n_q)  # [T, n_q]
        latents_tne = self._codes_to_unstacked_latents(codes_tn)               # [T, n_q, 8]
        latents_tn8 = latents_tne.reshape(latents_tne.shape[0], -1)            # [T, n_q*8]
        latents_tn8 = torch.clamp(latents_tn8, -self.config.clamp_val, self.config.clamp_val)
        latents_tn8 = latents_tn8 / self.config.clamp_val

        cond = np.load(entry["cond_path"]).astype(np.float32)
        cond_t = torch.from_numpy(cond)
        targets_t = torch.as_tensor(codes_tn, dtype=torch.long)

        T = min(latents_tn8.shape[0], targets_t.shape[0], cond_t.shape[0])
        return latents_tn8[:T], targets_t[:T], cond_t[:T]

    def _load_codes_tn(self, dac_path: Path, n_q: int) -> np.ndarray:
        raw_codes = self._read_dac_codes(dac_path)
        codes = self._coerce_codes_to_nq_t(raw_codes)
        if codes.shape[0] < n_q:
            raise ValueError(
                f"Requested n_q={n_q}, but file {dac_path.name} only contains {codes.shape[0]} quantizers"
            )
        codes = codes[:n_q]
        return codes.T.copy()  # [T, n_q]

    def _read_dac_codes(self, dac_path: Path):
        """
        Read codes from a .dac file.

        The implementation is intentionally tolerant because DAC packaging has
        varied slightly across versions. It first tries DAC-native file loading,
        then falls back to torch.load for dict-like payloads.
        """
        try:
            from dac import DACFile  # type: ignore
            dac_obj = DACFile.load(dac_path)
            for attr in ("codes", "audio_codes"):
                if hasattr(dac_obj, attr):
                    value = getattr(dac_obj, attr)
                    if value is not None:
                        return value
            if isinstance(dac_obj, dict):
                for key in ("codes", "audio_codes"):
                    if key in dac_obj:
                        return dac_obj[key]
        except Exception:
            pass

        obj = torch.load(dac_path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("codes", "audio_codes"):
                if key in obj:
                    return obj[key]
        raise ValueError(f"Could not find codes inside DAC file: {dac_path}")

    def _coerce_codes_to_nq_t(self, codes_obj) -> np.ndarray:
        """Convert many possible code layouts into a numpy array [n_q, T]."""
        if isinstance(codes_obj, (list, tuple)):
            if len(codes_obj) == 0:
                raise ValueError("Empty codes list")
            if len(codes_obj) == 1:
                return self._coerce_codes_to_nq_t(codes_obj[0])
            coerced = [self._coerce_codes_to_nq_t(x) for x in codes_obj]
            nq = coerced[0].shape[0]
            if not all(x.shape[0] == nq for x in coerced):
                raise ValueError("Inconsistent n_q across code chunks")
            return np.concatenate(coerced, axis=1)

        if isinstance(codes_obj, np.ndarray):
            arr = codes_obj
        elif isinstance(codes_obj, torch.Tensor):
            arr = codes_obj.detach().cpu().numpy()
        else:
            arr = np.asarray(codes_obj)

        arr = np.array(arr)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        while arr.ndim > 2 and arr.shape[1] == 1:
            arr = arr.squeeze(1)

        if arr.ndim == 1:
            raise ValueError(f"Codes must have at least 2 dimensions, got {arr.shape}")
        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])

        if arr.shape[0] <= arr.shape[1]:
            nq_t = arr
        else:
            nq_t = arr.T

        return nq_t.astype(np.int64, copy=False)

    def _codes_to_unstacked_latents(self, codes_tn: np.ndarray) -> torch.Tensor:
        """
        Convert discrete codes [T, n_q] to unstacked quantized latents [T, n_q, 8].
        """
        codes_nqt = torch.as_tensor(codes_tn.T[None, ...], dtype=torch.long, device=self.device)

        q = getattr(self._dac_model, "quantizer", None)
        if q is None:
            raise AttributeError("Loaded DAC model does not expose a quantizer attribute")
        if not hasattr(q, "from_codes"):
            raise AttributeError("DAC quantizer does not expose from_codes(...)")

        with torch.no_grad():
            out = q.from_codes(codes_nqt)

        if not isinstance(out, (tuple, list)) or len(out) == 0:
            raise ValueError("Unexpected return value from DAC quantizer.from_codes(...)")

        latents = None
        for candidate in out[1:]:
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 3:
                if candidate.shape[1] == self.config.n_q * 8:
                    latents = candidate
                    break
                if candidate.shape[-1] == self.config.n_q * 8:
                    latents = candidate.transpose(1, 2)
                    break
        if latents is None:
            tensor_candidates = [x for x in out if isinstance(x, torch.Tensor) and x.ndim == 3]
            for cand in tensor_candidates:
                if cand.shape[1] == self.config.n_q * 8:
                    latents = cand
                    break
                if cand.shape[-1] == self.config.n_q * 8:
                    latents = cand.transpose(1, 2)
                    break

        if latents is None:
            raise ValueError(
                "Could not identify low-D quantized latents in quantizer.from_codes(...) output"
            )

        latents = latents[0].transpose(0, 1).contiguous()  # [T, n_q*8]
        T = latents.shape[0]
        latents = latents.reshape(T, self.config.n_q, 8)
        return latents.cpu().float()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def _load_dac_model(self, dac_model_path: Optional[Union[str, Path]]):
        try:
            from dac import DAC  # type: ignore
            from dac.utils import download  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Could not import DAC. Make sure descript-audio-codec is installed in the current environment."
            ) from e

        if dac_model_path is None:
            model_path = download(model_type="44khz")
        else:
            model_path = str(dac_model_path)

        if hasattr(DAC, "load"):
            model = DAC.load(model_path)
        elif hasattr(DAC, "load_model"):
            model = DAC.load_model(model_path)
        else:
            raise AttributeError("Could not find a DAC model loading method")

        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _load_conditioning_config(dataset_root: Path) -> Dict:
        candidate_paths = [
            dataset_root / "conditioning_config.json",
            dataset_root / "hf_dataset" / "conditioning_config.json",
        ]
        for path in candidate_paths:
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        raise FileNotFoundError(
            f"Could not find conditioning_config.json under {dataset_root}"
        )

    @staticmethod
    def _load_parameters_json(dataset_root: Path) -> Dict:
        candidate_paths = [
            dataset_root / "parameters.json",
            dataset_root.parent / "parameters.json",
            dataset_root / "raw" / "parameters.json",
            dataset_root.parent / "raw" / "parameters.json",
        ]
        for path in candidate_paths:
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        raise FileNotFoundError(
            f"Could not find parameters.json under or near {dataset_root}"
        )

    # ------------------------------------------------------------------
    # Sequence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_lengths(total_length: int, n_parts: int) -> List[int]:
        base = total_length // n_parts
        remainder = total_length % n_parts
        return [base + (1 if i < remainder else 0) for i in range(n_parts)]



EnCodecLatentDataset_dynamic = RNNDACLatentDataset
