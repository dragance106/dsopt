"""
Implementations of uncertainty models for discrete surrogate optimization.

This version keeps the existing API but adds a cached Voronoi uncertainty model
that is fitted only when the set of computed points changes.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import xgboost as xgb

from dsopt.mipt import inter_dist, proj_dist

DEFAULT_BLOCK_SIZE = 4096


def _make_xgb_model(**kwargs):
    params = dict(tree_method="hist", learning_rate=0.1, verbosity=0, n_jobs=1)
    params.update(kwargs)
    return xgb.XGBRegressor(**params)


def _array_digest(array) -> bytes:
    arr = np.ascontiguousarray(np.asarray(array, dtype=float))
    return hashlib.blake2b(arr.view(np.uint8), digest_size=16).digest()


def _closest_sample_indices(candidates, sampled, block_size: int = DEFAULT_BLOCK_SIZE) -> np.ndarray:
    cand = np.atleast_2d(np.asarray(candidates, dtype=float))
    seen = np.atleast_2d(np.asarray(sampled, dtype=float))
    if len(cand) == 0:
        return np.empty((0,), dtype=int)
    if len(seen) == 0:
        return np.full((len(cand),), -1, dtype=int)

    seen_sq = np.einsum("ij,ij->i", seen, seen)
    out = np.empty((len(cand),), dtype=int)
    block_size = max(1, int(block_size))
    for start in range(0, len(cand), block_size):
        stop = min(start + block_size, len(cand))
        block = cand[start:stop]
        block_sq = np.einsum("ij,ij->i", block, block)[:, None]
        dist_sq = block_sq + seen_sq[None, :] - 2.0 * block.dot(seen.T)
        np.maximum(dist_sq, 0.0, out=dist_sq)
        out[start:stop] = np.argmin(dist_sq, axis=1)
    return out


class MIPTUncertainty:
    def __init__(self, dim):
        self.dim = dim

    def predict(self, cand_hyper, computed_hyper):
        cand_inter = inter_dist(cand_hyper, computed_hyper)
        cand_proj = proj_dist(cand_hyper, computed_hyper)
        evaluated = len(computed_hyper)
        return ((evaluated + 1) ** (1.0 / self.dim) - 1) / 2 * cand_inter + (evaluated + 1) / 2 * cand_proj

    def update(self, *args, **kwargs):
        pass


class APosterioriUncertainty:
    EPSILON = 1e-10

    def __init__(self, init_sample, init_values, alpha):
        init_values = np.asarray(init_values, dtype=float)
        self.alpha = float(alpha)
        self.uncertain_xgb = _make_xgb_model(enable_categorical=True, max_cat_to_onehot=1)

        if len(init_sample) < 4:
            self.rel_errors = np.zeros((len(init_sample),), dtype=float)
            self.uncertain_xgb.fit(init_sample, self.rel_errors, verbose=False)
            return

        k = len(init_sample) // 2
        div1_sample = init_sample.iloc[:k]
        div1_values = init_values[:k]
        div2_sample = init_sample.iloc[k:]
        div2_values = init_values[k:]

        uncertain_xgb1 = _make_xgb_model(enable_categorical=True, max_cat_to_onehot=1)
        uncertain_xgb1.fit(div1_sample, div1_values, verbose=False)
        uncertain_xgb2 = _make_xgb_model(enable_categorical=True, max_cat_to_onehot=1)
        uncertain_xgb2.fit(div2_sample, div2_values, verbose=False)

        init_predictions = np.concatenate((uncertain_xgb2.predict(div1_sample), uncertain_xgb1.predict(div2_sample)))
        self.rel_errors = np.abs(init_predictions - init_values) / (np.abs(init_values) + self.EPSILON)
        self.uncertain_xgb.fit(init_sample, self.rel_errors, verbose=False)

    def predict(self, cand_sample):
        pred = np.asarray(self.uncertain_xgb.predict(cand_sample), dtype=float)
        return np.maximum(pred, 0.0)

    def update(self, computed_sample, predicted_values, computed_values):
        predicted_values = np.asarray(predicted_values, dtype=float)
        computed_values = np.asarray(computed_values, dtype=float)
        rel_errors_new = np.abs(predicted_values - computed_values) / (np.abs(computed_values) + self.EPSILON)
        self.rel_errors = np.concatenate((self.alpha * self.rel_errors, rel_errors_new))
        self.uncertain_xgb.fit(computed_sample, self.rel_errors, verbose=False)


class VoronoiUncertainty:
    def __init__(
        self,
        reference_multiplier: int = 64,
        min_reference_size: int = 2048,
        max_reference_size: int = 20000,
        block_size: int = DEFAULT_BLOCK_SIZE,
        seed: Optional[int] = None,
    ):
        self.uncertain_xgb = _make_xgb_model()
        self.reference_multiplier = int(reference_multiplier)
        self.min_reference_size = int(min_reference_size)
        self.max_reference_size = int(max_reference_size)
        self.block_size = int(block_size)
        self.rng = np.random.default_rng(seed)
        self._fit_key: Optional[bytes] = None
        self._const_prediction: Optional[float] = None

    def _fit_if_needed(self, computed_hyper) -> None:
        hyper = np.atleast_2d(np.asarray(computed_hyper, dtype=float))
        if len(hyper) == 0:
            self._fit_key = b"empty"
            self._const_prediction = 0.0
            return

        fit_key = _array_digest(hyper)
        if self._fit_key == fit_key:
            return
        self._fit_key = fit_key

        if len(hyper) == 1:
            self._const_prediction = 1.0
            return

        self._const_prediction = None
        dim = hyper.shape[1]
        ref_size = max(self.min_reference_size, self.reference_multiplier * len(hyper))
        ref_size = min(self.max_reference_size, ref_size)
        ref = self.rng.random((ref_size, dim))
        nearest = _closest_sample_indices(ref, hyper, block_size=self.block_size)
        counts = np.zeros((len(hyper),), dtype=float)
        np.add.at(counts, nearest, 1.0)
        counts /= max(1, ref_size)
        self.uncertain_xgb.fit(hyper, counts, verbose=False)

    def predict(self, cand_hyper, computed_hyper):
        self._fit_if_needed(computed_hyper)
        cand = np.atleast_2d(np.asarray(cand_hyper, dtype=float))
        if len(cand) == 0:
            return np.empty((0,), dtype=float)
        if self._const_prediction is not None:
            return np.full((len(cand),), float(self._const_prediction), dtype=float)
        pred = np.asarray(self.uncertain_xgb.predict(cand), dtype=float)
        return np.maximum(pred, 0.0)

    def update(self, computed_hyper=None):
        if computed_hyper is None:
            self._fit_key = None
            self._const_prediction = None
            return
        self._fit_if_needed(computed_hyper)
