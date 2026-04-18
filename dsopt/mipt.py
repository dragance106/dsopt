"""
Implementation of Crombecq's MIPT sampling.

This version keeps the public API intact while speeding up the hottest kernels
(`inter_dist` and `proj_dist`) and vectorizing interval sampling inside `mipt`.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BLOCK_SIZE = 4096


def _as_2d_float(array) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _nearest_sorted_distance(values: np.ndarray, sorted_points: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_points, values, side="left")
    right = np.clip(idx, 0, len(sorted_points) - 1)
    left = np.clip(idx - 1, 0, len(sorted_points) - 1)
    return np.minimum(np.abs(values - sorted_points[left]), np.abs(values - sorted_points[right]))


def inter_dist(candidates, sampled, block_size: int = DEFAULT_BLOCK_SIZE):
    """
    Computes intersite distance from each candidate point to the set of already sampled points.
    For the candidate point c, its intersite distance to sampled points is
    min_j sqrt(sum_k (c_k - sampled_{j,k})^2)
    """
    cand = _as_2d_float(candidates)
    seen = _as_2d_float(sampled)

    if len(cand) == 0:
        return np.empty((0,), dtype=float)
    if len(seen) == 0:
        return np.full((len(cand),), np.inf, dtype=float)

    block_size = max(1, int(block_size))
    seen_sq = np.einsum("ij,ij->i", seen, seen)
    out = np.empty((len(cand),), dtype=float)

    for start in range(0, len(cand), block_size):
        stop = min(start + block_size, len(cand))
        block = cand[start:stop]
        block_sq = np.einsum("ij,ij->i", block, block)[:, None]
        dist_sq = block_sq + seen_sq[None, :] - 2.0 * block.dot(seen.T)
        np.maximum(dist_sq, 0.0, out=dist_sq)
        out[start:stop] = np.sqrt(np.min(dist_sq, axis=1))
    return out


def proj_dist(candidates, sampled):
    """
    Computes projected distance from each candidate point to the set of already sampled points.
    For the candidate point c, its projected distance to sampled points is
    min_j min_k |c_k - sampled_{j,k}|.
    """
    cand = _as_2d_float(candidates)
    seen = _as_2d_float(sampled)

    if len(cand) == 0:
        return np.empty((0,), dtype=float)
    if len(seen) == 0:
        return np.full((len(cand),), np.inf, dtype=float)

    out = np.full((len(cand),), np.inf, dtype=float)
    for dim in range(cand.shape[1]):
        sorted_points = np.sort(seen[:, dim])
        dim_dist = _nearest_sorted_distance(cand[:, dim], sorted_points)
        np.minimum(out, dim_dist, out=out)
    return out


def _sample_from_intervals(intervals, n_samples: int, rng) -> np.ndarray:
    if not intervals:
        return rng.random((n_samples,))

    bounds = np.asarray(intervals, dtype=float)
    lengths = bounds[:, 1] - bounds[:, 0]
    cum = np.cumsum(lengths)
    total = float(cum[-1])

    draws = total * rng.random((n_samples,))
    which = np.searchsorted(cum, draws, side="right")
    prev = np.zeros_like(draws)
    mask = which > 0
    prev[mask] = cum[which[mask] - 1]
    return bounds[which, 0] + (draws - prev)


def mipt(n, dim, rng, alpha=0.5, k=100, negligible=1e-6):
    """
    Implementation of the Crombecq mc-intersite-proj-th sampling scheme.
    """
    n = int(n)
    dim = int(dim)
    if n <= 0:
        return np.zeros((0, dim), dtype=float)

    sample = np.zeros((n, dim), dtype=float)
    sample[0] = rng.random((dim,))

    for s in range(1, n):
        dmin = alpha / (s + 1)
        n_candidates = max(1, int(k * s))
        candidates = np.zeros((n_candidates, dim), dtype=float)

        for x in range(dim):
            start_intervals = [(0.0, 1.0)]
            for j in range(s):
                l2 = float(sample[j, x] - dmin)
                u2 = float(sample[j, x] + dmin)
                end_intervals = []
                for (l1, u1) in start_intervals:
                    if u2 < l1 + negligible:
                        end_intervals.append((l1, u1))
                    elif u1 < l2 + negligible:
                        end_intervals.append((l1, u1))
                    elif l2 < l1 + negligible and l1 < u2 + negligible and u2 < u1 + negligible:
                        end_intervals.append((u2, u1))
                    elif l1 < l2 + negligible and l2 < u1 + negligible and u1 < u2 + negligible:
                        end_intervals.append((l1, l2))
                    elif l1 < l2 + negligible and u2 < u1 + negligible:
                        end_intervals.append((l1, l2))
                        end_intervals.append((u2, u1))
                start_intervals = end_intervals
            candidates[:, x] = _sample_from_intervals(start_intervals, n_candidates, rng)

        ind = inter_dist(candidates, sample[:s])
        sample[s] = candidates[int(np.argmax(ind))]

    return sample
