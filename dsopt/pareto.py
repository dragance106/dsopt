"""
Selection of candidate points for expensive function evaluation.

This version restores the intended `omega_percentage` cutoff and replaces the
DataFrame-heavy implementation with a NumPy-based one.
"""
from __future__ import annotations

import numpy as np

EPSILON = 1e-12


def _prediction_cutoff(best_simulated_value: float, opt: str, omega_percentage: float) -> float:
    scale = max(abs(float(best_simulated_value)), EPSILON)
    allowed_gap = scale * float(omega_percentage) / 100.0
    if opt == 'min':
        max_allowed_value = float(best_simulated_value) + allowed_gap
        return -max_allowed_value
    return float(best_simulated_value) - allowed_gap


def select_candidates(cand_values, cand_expl, opt, max_new_points,
                      best_simulated_value, omega_percentage=5,
                      tau_percentage=10, upsilon_percentage=25,
                      sigma_t=37.5, sigma_u=12.5,
                      rng=np.random.default_rng()):
    cand_values = np.asarray(cand_values, dtype=float)
    cand_expl = np.asarray(cand_expl, dtype=float)
    n = len(cand_values)
    if n == 0 or max_new_points <= 0:
        return np.empty((0,), dtype=int)

    cand_opt_values = -cand_values if opt == 'min' else cand_values
    best_prediction_cutoff = _prediction_cutoff(best_simulated_value, opt, omega_percentage)

    order = np.lexsort((-cand_opt_values, cand_expl))
    sorted_values = cand_opt_values[order]
    sorted_expl = cand_expl[order]
    sorted_indices = order.astype(int)

    low_boundary = min(max(int(round(tau_percentage * n / 100.0)), 0), n)
    high_boundary = min(max(int(round(upsilon_percentage * n / 100.0)), 0), n)

    select_low_best = min(low_boundary, int(round(0.5 + sigma_t * max_new_points / 100.0)))
    select_high_best = max(0, min(high_boundary, int(round(sigma_u * max_new_points / 100.0))))
    if high_boundary > 0 and sigma_u > 0:
        select_high_best = max(select_high_best, 1)

    selected_parts = []

    if low_boundary > 0 and select_low_best > 0:
        low_values = sorted_values[:low_boundary]
        low_orig = sorted_indices[:low_boundary]
        admissible = low_values >= best_prediction_cutoff
        low_values = low_values[admissible]
        low_orig = low_orig[admissible]
        if len(low_orig) > 0:
            take = min(select_low_best, len(low_orig))
            keep = np.argsort(low_values)[-take:]
            selected_parts.append(low_orig[keep])

    if high_boundary > 0 and select_high_best > 0:
        high_slice = slice(n - high_boundary, n)
        high_expl = sorted_expl[high_slice]
        high_orig = sorted_indices[high_slice]
        if len(high_orig) > 0:
            take = min(select_high_best, len(high_orig))
            keep = np.argsort(high_expl)[-take:]
            selected_parts.append(high_orig[keep])

    already_selected = sum(len(part) for part in selected_parts)
    remaining_points = max(0, int(max_new_points) - already_selected)

    medium_start = low_boundary
    medium_stop = max(low_boundary, n - high_boundary)
    if remaining_points > 0 and medium_stop > medium_start:
        medium_values = sorted_values[medium_start:medium_stop]
        medium_orig = sorted_indices[medium_start:medium_stop]
        pareto = np.ones((len(medium_values),), dtype=bool)
        if len(medium_values) > 0:
            prev_max_value = medium_values[0]
            for i in range(1, len(medium_values)):
                if prev_max_value > medium_values[i]:
                    pareto[i] = False
                else:
                    prev_max_value = medium_values[i]
        # admissible = medium_values >= best_prediction_cutoff
        # pareto_indices = medium_orig[pareto & admissible]
        pareto_indices = medium_orig[pareto]
        if len(pareto_indices) <= remaining_points:
            selected_parts.append(pareto_indices)
        elif len(pareto_indices) > 0:
            keep = rng.permutation(len(pareto_indices))[:remaining_points]
            selected_parts.append(pareto_indices[keep])

    if not selected_parts:
        return np.empty((0,), dtype=int)

    selected = np.concatenate(selected_parts, axis=None)
    _, keep = np.unique(selected, return_index=True)
    return selected[np.sort(keep)].astype(int)
