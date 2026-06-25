from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# pool[b][q] = (vectors, indices) — matches CodePool from inference.py
CodePool = List[List[Tuple[torch.Tensor, torch.Tensor]]]


def pool_summary(
    pool: CodePool,
    model,
    n_bins: int = 100,
) -> Dict:
    """Print and return per-bin and global statistics for a code pool.

    Args:
        pool: from build_code_pool.
        model: RNNDACModel (provides codebook_vectors, codebook_size).
        n_bins: number of bins (must match pool length).

    Returns:
        dict with keys: n_bins_with_codes, total_pairs, codes_per_bin,
        unique_codes_global, codebook_size.
    """
    K = model.config.codebook_size

    per_bin = [pool[b][0][0].shape[0] for b in range(n_bins)]
    n_active = sum(1 for c in per_bin if c > 0)

    all_idxs: List[int] = []
    for b in range(n_bins):
        _, idxs = pool[b][0]
        all_idxs.extend(idxs.tolist())
    n_unique = len(set(all_idxs))

    print(f"=== Pool summary ===")
    print(f"  Active bins:         {n_active}/{n_bins}")
    print(f"  Total vectors stored:   {sum(per_bin)}")
    print(f"  Unique CB0 codes across all bins: {n_unique} / {K}")
    print(f"  Coverage:            {100 * n_unique / K:.1f}% of codebook")
    print(f"  Vectors per bin:      min={min(per_bin)}, "
          f"max={max(per_bin)}, mean={sum(per_bin)/n_active:.1f}")
    print()

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    bins_hist = min(50, max(per_bin) + 1) if max(per_bin) > 0 else 1
    axes[0].hist(per_bin, bins=bins_hist, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Unique CB0 vectors per bin")
    axes[0].set_ylabel("Number of bins")
    axes[0].set_title("Vectors per bin (histogram)")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(n_bins), per_bin, width=0.8)
    axes[1].set_xlabel("Cond bin index")
    axes[1].set_ylabel("Unique CB0 vectors")
    axes[1].set_title("Vectors per bin (by bin index)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "n_bins_with_codes": n_active,
        "total_pairs": sum(per_bin),
        "codes_per_bin": per_bin,
        "unique_codes_global": n_unique,
        "codebook_size": K,
    }


def intra_bin_distances(
    pool: CodePool,
    model,
    n_bins: int = 100,
    n_example_bins: int = 5,
) -> Dict:
    """Compute within-bin 8D vector similarity.

    For each bin, computes pairwise cosine similarity between all
    codebook vectors stored in that bin.  The vectors are already
    stored in the pool (no codebook lookup needed).

    Args:
        pool: from build_code_pool.
        model: RNNDACModel (used for codebook_size only).
        n_bins: number of bins.
        n_example_bins: number of example bins to print (centered at middle).

    Returns:
        dict with keys: mean_intra_similarity, std_intra_similarity,
        per_bin_similarity.
    """
    per_bin_sim: List[float] = []

    for b in range(n_bins):
        vecs, _ = pool[b][0]
        if vecs.shape[0] < 2:
            continue

        normed = vecs / vecs.norm(dim=1, keepdim=True).clamp(min=1e-12)  # [N, 8]
        sim = normed @ normed.T  # [N, N]
        tri = sim.triu(diagonal=1)
        n_pairs = tri.numel()
        if n_pairs > 0:
            per_bin_sim.append(tri.sum().item() / n_pairs)

    mean_sim = np.mean(per_bin_sim) if per_bin_sim else 0.0
    std_sim = np.std(per_bin_sim) if per_bin_sim else 0.0

    print(f"=== Intra-bin cosine similarity (CB0 vectors) ===")
    print(f"  Bins with >=2 vectors analyzed: {len(per_bin_sim)}")
    print(f"  Mean intra-bin cosine sim:   {mean_sim:.4f} +- {std_sim:.4f}")
    print(f"  (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")
    print()

    if n_example_bins > 0:
        print("Example bins:")
        bin_indices = np.linspace(0, n_bins - 1, n_example_bins, dtype=int).tolist()
        for b in bin_indices:
            vecs, _ = pool[b][0]
            if vecs.shape[0] < 2:
                print(f"  bin {b:3d} (cond~{b/n_bins:.2f}): {vecs.shape[0]} vector(s) -- too few to pair")
                continue
            normed = vecs / vecs.norm(dim=1, keepdim=True).clamp(min=1e-12)
            sim = normed @ normed.T
            tri = sim.triu(diagonal=1)
            vals = tri[tri > -1.9].tolist()
            if vals:
                print(f"  bin {b:3d} (cond~{b/n_bins:.2f}): {vecs.shape[0]} vectors, "
                      f"sim mean={np.mean(vals):.3f}  range=[{np.min(vals):.3f}, {np.max(vals):.3f}]")
    print()

    return {
        "mean_intra_similarity": mean_sim,
        "std_intra_similarity": std_sim,
        "per_bin_similarity": per_bin_sim,
    }


def plot_pool_projections(
    pool: CodePool,
    model,
    n_bins: int = 100,
    max_bins_to_color: int = 10,
):
    """2D PCA projection of CB0 codebook vectors, colored by bin membership.

    Args:
        pool: from build_code_pool.
        model: RNNDACModel (provides codebook_vectors).
        n_bins: number of bins.
        max_bins_to_color: color only the top-N bins by vector count.
    """
    codebook = model.codebook_vectors[0].cpu().numpy()  # [K, 8]
    K = codebook.shape[0]

    # PCA via SVD on centered data
    X = codebook - codebook.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ Vt[:2].T  # [K, 2]

    per_bin = [pool[b][0][0].shape[0] for b in range(n_bins)]
    top_bins = np.argsort(per_bin)[-max_bins_to_color:]

    code_to_color: Dict[int, int] = {}
    for b in reversed(top_bins):
        _, idxs = pool[b][0]
        for c in idxs.tolist():
            if c not in code_to_color:
                code_to_color[c] = int(b)

    colors = [code_to_color.get(i, -1) for i in range(K)]
    uncolored = [i for i, c in enumerate(colors) if c == -1]
    colored_mask = [c >= 0 for c in colors]

    fig, ax = plt.subplots(figsize=(10, 8))

    if uncolored:
        ax.scatter(proj[uncolored, 0], proj[uncolored, 1],
                   c="lightgray", s=8, alpha=0.5, label="unobserved")

    if any(colored_mask):
        c_arr = np.array(colors)
        sc = ax.scatter(proj[colored_mask, 0], proj[colored_mask, 1],
                        c=c_arr[colored_mask], cmap="tab10", s=20, alpha=0.8)

        handles = []
        for b in top_bins:
            vecs, _ = pool[b][0]
            if vecs.shape[0] == 0:
                continue
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=sc.cmap(sc.norm(b)),
                                      markersize=6,
                                      label=f"bin {b} (cond~{b/n_bins:.2f}, {vecs.shape[0]} vecs)"))
        ax.legend(handles=handles, fontsize=8, loc="best")

    ax.set_title("CB0 codebook vectors: PCA projection colored by cond bin")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
