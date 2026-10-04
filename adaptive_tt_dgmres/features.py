"""Compact physical-cut spectrum features for a pure tensor train.

No dense tensor or full unfolding is constructed. Right QR orthogonalization
followed by a left SVD sweep gives the singular values of each physical cut.
The spectral features are invariant under nonsingular internal TT gauge changes
up to floating-point error. Stored ranks describe the implementation: they need
not be invariant under redundant rank enlargement. Mode order is significant.
"""
from __future__ import annotations

import math
from typing import List

import torch


TT_FEATURE_NAMES = (
    "log1p_stored_rank",
    "log1p_left_mode_size",
    "relative_cut_position",
    "top1_energy_fraction",
    "top2_energy_fraction",
    "tail_after_half_energy_fraction",
    "log1p_effective_rank_relative_error_1e2",
    "log1p_effective_rank_relative_error_1e4",
    "normalized_spectral_entropy",
    "log1p_mode_count",
)


def _spectrum_summary(singular_values: torch.Tensor) -> List[float]:
    """Return six scale-free summaries, including two effective TT ranks."""
    values = singular_values.detach().abs()
    maximum = float(values.max()) if values.numel() else 0.0
    if maximum == 0.0:
        return [0.0] * 6
    # Scale before squaring to avoid overflow when tensor amplitudes are large.
    energy = (values / maximum).square()
    energy = energy / energy.sum()
    count = int(energy.numel())
    cumulative = torch.cumsum(energy, dim=0)

    def effective_rank(tolerance: float) -> int:
        target = 1.0 - tolerance * tolerance
        return min(count, int(torch.searchsorted(cumulative, target).item()) + 1)

    positive = energy[energy > 0]
    entropy = float(-(positive * torch.log(positive)).sum())
    entropy = entropy / math.log(count) if count > 1 else 0.0
    half = max(1, (count + 1) // 2)
    return [
        float(energy[0]),
        float(energy[:2].sum()),
        float(energy[half:].sum()),
        math.log1p(effective_rank(1e-2)),
        math.log1p(effective_rank(1e-4)),
        entropy,
    ]


@torch.no_grad()
def tt_interface_tokens(tensor) -> List[List[float]]:
    """Return one 10-feature row per internal cut (one sentinel row for d=1).

    Features, in order, are exactly ``TT_FEATURE_NAMES``. Effective rank is the
    smallest rank whose discarded singular energy is at most tolerance squared
    times total energy. Zero tensors have zero spectral summaries. For a single
    mode there is no internal cut; its row has rank one, position zero, and a
    one-value spectrum. The input cores are never modified. Tucker mode factors
    are absorbed locally if present; CP/mixed two-dimensional cores are rejected.
    """
    if not getattr(tensor, "cores", None):
        raise ValueError("Expected a tensor train with at least one core.")
    factors = getattr(tensor, "Us", [None] * len(tensor.cores))
    cores = []
    for core, factor in zip(tensor.cores, factors):
        if core.ndim != 3:
            raise ValueError("Features require pure three-dimensional TT cores.")
        if not (core.is_floating_point() or core.is_complex()):
            raise ValueError("TT spectrum features require floating-point cores.")
        copied = core.detach().clone()
        if factor is not None:
            copied = torch.einsum("anb,in->aib", copied, factor.detach())
        if not bool(torch.isfinite(copied).all()):
            raise ValueError("Nonfinite TT cores cannot be encoded.")
        cores.append(copied)
    order = len(cores)
    if len(cores) != len(tensor.cores):
        raise ValueError("TT core and mode-factor counts disagree.")
    if cores[0].shape[0] != 1 or cores[-1].shape[-1] != 1:
        raise ValueError("TT boundary ranks must be one.")
    for left, right in zip(cores[:-1], cores[1:]):
        if left.shape[-1] != right.shape[0]:
            raise ValueError("Adjacent TT ranks disagree.")
    stored_ranks = [int(core.shape[-1]) for core in cores[:-1]]
    sizes = [int(core.shape[1]) for core in cores]

    if order == 1:
        spectrum = torch.linalg.vector_norm(cores[0]).reshape(1)
        return [[math.log1p(1), math.log1p(sizes[0]), 0.0,
                 *_spectrum_summary(spectrum), math.log1p(order)]]

    # At each step the suffix map has orthonormal rows. Carry its QR factor
    # into the preceding core without dropping even tiny singular directions.
    for position in range(order - 1, 0, -1):
        left_rank, size, right_rank = cores[position].shape
        matrix = cores[position].reshape(left_rank, size * right_rank)
        q, r = torch.linalg.qr(matrix.mH, mode="reduced")
        rank = q.shape[1]
        cores[position] = q.mH.reshape(rank, size, right_rank)
        cores[position - 1] = torch.tensordot(
            cores[position - 1], r.mH, dims=([-1], [0])
        )

    rows = []
    for position in range(order - 1):
        left_rank, size, right_rank = cores[position].shape
        matrix = cores[position].reshape(left_rank * size, right_rank)
        u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        rows.append([
            math.log1p(stored_ranks[position]),
            math.log1p(sizes[position]),
            (position + 1) / order,
            *_spectrum_summary(singular_values),
            math.log1p(order),
        ])
        cores[position] = u.reshape(left_rank, size, u.shape[1])
        carry = singular_values.unsqueeze(1) * vh
        cores[position + 1] = torch.tensordot(
            carry, cores[position + 1], dims=([1], [0])
        )
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("TT spectrum feature extraction produced nonfinite values.")
    return rows
