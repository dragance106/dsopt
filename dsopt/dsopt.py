"""
Implementation of discrete surrogate optimization.

This patched version keeps the original public interface but improves the main
performance bottlenecks:
  - candidate deduplication and already-evaluated filtering are done in a
    compact integer encoding instead of by building full Python dictionaries for
    every candidate;
  - categorical columns are created with a stable full vocabulary, which is
    safer for native categorical boosting;
  - distance-based uncertainty uses deterministic cell centers instead of
    random points inside the same discrete cell;
  - expensive function evaluations use a reusable ProcessPoolExecutor bounded by
    the intended batch size / parallelism;
  - XGBoost uses the faster `hist` tree method.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from dsopt.arguments import analyze_arguments, hypercube_to_arguments
from dsopt.mipt import mipt
from dsopt.pareto import select_candidates
from dsopt.uncertainty import MIPTUncertainty, APosterioriUncertainty, VoronoiUncertainty

DEFAULT_REFILL_ROUNDS = 3


def compute_exp_func(args):
    exp_func = args[0]
    point = args[1]
    return exp_func(**point)


def _hypercube_to_indices(hyperpoints, arg_len: np.ndarray) -> np.ndarray:
    hyper = np.atleast_2d(np.asarray(hyperpoints, dtype=float))
    idx = np.floor(hyper * arg_len.reshape(1, -1)).astype(int)
    np.clip(idx, 0, arg_len.reshape(1, -1) - 1, out=idx)
    return idx


def _indices_to_hyper_centers(indices, arg_len: np.ndarray) -> np.ndarray:
    idx = np.atleast_2d(np.asarray(indices, dtype=float))
    return (idx + 0.5) / arg_len.reshape(1, -1)


def _indices_to_sample(indices, arg_type: np.ndarray, arg_array: np.ndarray) -> pd.DataFrame:
    idx = np.atleast_2d(np.asarray(indices, dtype=int))
    data = np.empty(idx.shape, dtype=object)
    for j in range(idx.shape[1]):
        data[:, j] = arg_array[j][idx[:, j]]
    df = pd.DataFrame(data=data)
    for j in range(idx.shape[1]):
        if arg_type[j] == 'c':
            df[j] = pd.Categorical(df[j], categories=list(arg_array[j]))
        else:
            df[j] = df[j].astype(float)
    return df


def _row_tuple(row: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(x) for x in row.tolist())


def _point_key_set(indices) -> set[Tuple[int, ...]]:
    arr = np.atleast_2d(np.asarray(indices, dtype=int))
    return {_row_tuple(row) for row in arr}


def _unique_row_indices(indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((0,), dtype=int)
    _, keep = np.unique(indices, axis=0, return_index=True)
    return np.sort(keep).astype(int)


def _make_xgb_model(random_state: Optional[int] = None) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        tree_method='hist',
        enable_categorical=True,
        max_cat_to_onehot=1,
        learning_rate=0.1,
        verbosity=0,
        n_jobs=1,
        random_state=random_state,
    )


def _fit_surrogate(model, computed_sample: pd.DataFrame, computed_values: np.ndarray,
                   test_set_percentage: float, rng: np.random.Generator) -> None:
    y = np.asarray(computed_values, dtype=float)
    if test_set_percentage < 1 or len(y) < 4:
        model.fit(computed_sample, y, verbose=False)
        return

    rnd_indices = rng.permutation(len(y))
    train_size = int(len(y) * (100.0 - test_set_percentage) / 100.0)
    train_size = min(max(train_size, 1), len(y) - 1)
    train_idx = rnd_indices[:train_size]
    test_idx = rnd_indices[train_size:]
    model.fit(
        computed_sample.iloc[train_idx, :],
        y[train_idx],
        eval_set=[(computed_sample.iloc[test_idx, :], y[test_idx])],
        verbose=False,
    )


def _evaluate_parallel(executor: ProcessPoolExecutor, exp_func, points: Sequence[Dict]) -> np.ndarray:
    material = [(exp_func, point) for point in points]
    return np.asarray(list(executor.map(compute_exp_func, material)), dtype=float)


def dsopt(exp_func,
          arg_dict,
          opt='min',
          initial_sample_size=50,
          iterative_sample_size=8,
          max_evaluations=1000,
          k=100,
          uncertainty_metric='aposteriori',
          alpha=1.0,
          test_set_percentage=0,
          omega_percentage=10,
          tau_percentage=10,
          upsilon_percentage=25,
          sigma_t=50,
          sigma_u=10,
          gamma=0.01,
          r=10,
          verbose_level=0,
          seed=None,
          filename=None,
          parallel_tasks=None):
    opt = str(opt).lower()
    uncertainty_metric = str(uncertainty_metric).lower()
    if opt not in {'min', 'max'}:
        raise ValueError("opt must be one of {'min', 'max'}")
    if uncertainty_metric not in {'mipt', 'aposteriori', 'voronoi'}:
        raise ValueError("uncertainty_metric must be one of {'mipt', 'aposteriori', 'voronoi'}")

    if filename is None:
        exp_name = getattr(exp_func, '__name__', exp_func.__class__.__name__)
        filename = 'report_' + exp_name \
                   + '_' + opt \
                   + '_eval_' + str(max_evaluations) \
                   + '_init_' + str(initial_sample_size) \
                   + '_iter_' + str(iterative_sample_size) \
                   + '_' + uncertainty_metric \
                   + '_omega_' + str(omega_percentage) \
                   + '_tau_' + str(tau_percentage) \
                   + '_upsilon_' + str(upsilon_percentage) \
                   + '_sigma_t_' + str(sigma_t) \
                   + '_sigma_u_' + str(sigma_u) \
                   + '_gamma_' + str(gamma) \
                   + '_r_' + str(r) + '.csv'

    rng = np.random.default_rng(seed)
    dim, arg_type, arg_array, arg_len = analyze_arguments(arg_dict)

    max_workers = iterative_sample_size if parallel_tasks is None else parallel_tasks
    max_workers = max(1, min(int(max_workers), os.cpu_count() or 1))

    iteration = 0
    total_evaluated = 0
    total_best_value = None
    total_best_point = None
    header_written = False

    if verbose_level >= 1:
        with open(filename, 'w'):
            pass

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        while total_evaluated < max_evaluations:
            tic = time.perf_counter()
            n_init = int(min(initial_sample_size, max_evaluations - total_evaluated))
            if n_init <= 0:
                break

            init_hyper = mipt(n_init, dim, rng)
            computed_indices = _hypercube_to_indices(init_hyper, arg_len)
            computed_indices = computed_indices[_unique_row_indices(computed_indices)]
            computed_hyper = _indices_to_hyper_centers(computed_indices, arg_len)
            computed_sample = _indices_to_sample(computed_indices, arg_type, arg_array)
            computed_points = hypercube_to_arguments(computed_hyper, arg_dict)
            evaluated_set = _point_key_set(computed_indices)

            toc = time.perf_counter()
            if verbose_level >= 3:
                print(f'  timing: initial sample created in {toc - tic:.4f} seconds')
            tic = toc

            computed_values = _evaluate_parallel(executor, exp_func, computed_points)
            evaluated = len(computed_values)
            total_evaluated += evaluated

            toc = time.perf_counter()
            if verbose_level >= 3:
                print(f'  timing: initial sample evaluated in {toc - tic:.4f} seconds')
            tic = toc

            opt_func = np.argmax if opt == 'max' else np.argmin
            index = int(opt_func(computed_values))
            best_sample = computed_sample.iloc[index]
            best_point = computed_points[index]
            best_value = float(computed_values[index])

            if opt == 'max':
                if total_best_value is None or total_best_value < best_value:
                    total_best_value = best_value
                    total_best_point = best_point
            else:
                if total_best_value is None or total_best_value > best_value:
                    total_best_value = best_value
                    total_best_point = best_point

            iter_last_improvement = iteration
            previous_best_value = best_value

            if verbose_level >= 2:
                print(f'evaluated the initial sample of {evaluated} points (total evaluated: {total_evaluated})')
                print(f'  best value={best_value} for point: {best_point}')

            xgb_model = _make_xgb_model(random_state=seed)
            _fit_surrogate(xgb_model, computed_sample, computed_values, test_set_percentage, rng)

            toc = time.perf_counter()
            if verbose_level >= 3:
                print(f'  timing: initial surrogate model trained in {toc - tic:.4f} seconds')
            tic = toc

            if uncertainty_metric == 'mipt':
                uncertainty_model = MIPTUncertainty(dim)
            elif uncertainty_metric == 'aposteriori':
                uncertainty_model = APosterioriUncertainty(computed_sample, computed_values, alpha)
            else:
                uncertainty_model = VoronoiUncertainty(seed=seed)
                uncertainty_model.update(computed_hyper)

            toc = time.perf_counter()
            if verbose_level >= 3:
                print(f'  timing: initial uncertainty estimated in {toc - tic:.4f} seconds')
            tic = toc

            if verbose_level >= 1:
                with open(filename, 'a') as f:
                    if not header_written:
                        line = 'iteration'
                        for i in range(dim):
                            line += f', arg{i}'
                        line += ', computed, exploitation, uncertainty, best value'
                        for i in range(dim):
                            line += f', best_arg{i}'
                        f.write(line + '\n')
                        header_written = True
                    for i in range(evaluated):
                        line = str(iteration)
                        for j in range(dim):
                            line += f', {computed_sample.iat[i, j]}'
                        line += f', {computed_values[i]}'
                        line += ', 0, 0'
                        line += f', {best_value}'
                        for j in range(dim):
                            line += f', {best_sample.iat[j]}'
                        f.write(line + '\n')

            while total_evaluated < max_evaluations:
                iteration += 1
                max_new_points = int(min(max_evaluations - total_evaluated, iterative_sample_size))
                if max_new_points <= 0:
                    break

                how_many = max(int(k * (evaluated + iterative_sample_size)), max_new_points)
                cand_indices = _hypercube_to_indices(rng.random((how_many, dim)), arg_len)
                cand_indices = cand_indices[_unique_row_indices(cand_indices)]

                if len(cand_indices) > 0:
                    mask = np.array([_row_tuple(row) not in evaluated_set for row in cand_indices], dtype=bool)
                    cand_indices = cand_indices[mask]

                # top up conservatively when discretization collapses onto many duplicates
                refill_rounds = 0
                while len(cand_indices) < max_new_points and refill_rounds < DEFAULT_REFILL_ROUNDS:
                    refill_rounds += 1
                    extra = rng.random((max_new_points * 8, dim))
                    extra_indices = _hypercube_to_indices(extra, arg_len)
                    extra_indices = extra_indices[_unique_row_indices(extra_indices)]
                    if len(extra_indices) > 0:
                        mask = np.array([_row_tuple(row) not in evaluated_set for row in extra_indices], dtype=bool)
                        extra_indices = extra_indices[mask]
                        cand_indices = np.vstack([cand_indices, extra_indices]) if len(cand_indices) > 0 else extra_indices
                        cand_indices = cand_indices[_unique_row_indices(cand_indices)]

                if len(cand_indices) == 0:
                    return total_best_value, total_best_point

                cand_hyper = _indices_to_hyper_centers(cand_indices, arg_len)
                cand_sample = _indices_to_sample(cand_indices, arg_type, arg_array)
                cand_values = np.asarray(xgb_model.predict(cand_sample), dtype=float)
                if uncertainty_metric == 'mipt':
                    cand_expl = np.asarray(uncertainty_model.predict(cand_hyper, computed_hyper), dtype=float)
                elif uncertainty_metric == 'aposteriori':
                    cand_expl = np.asarray(uncertainty_model.predict(cand_sample), dtype=float)
                else:
                    cand_expl = np.asarray(uncertainty_model.predict(cand_hyper, computed_hyper), dtype=float)

                toc = time.perf_counter()
                if verbose_level >= 3:
                    print(f'  timing: values and uncertainties for candidate sample predicted in {toc - tic:.4f} seconds')
                tic = toc

                select_indices = np.asarray(
                    select_candidates(cand_values, cand_expl, opt, max_new_points, best_value,
                                      omega_percentage, tau_percentage, upsilon_percentage,
                                      sigma_t, sigma_u, rng),
                    dtype=int,
                )
                if len(select_indices) == 0:
                    return total_best_value, total_best_point

                select_cand_indices = cand_indices[select_indices]
                select_hyper = cand_hyper[select_indices]
                select_sample = cand_sample.iloc[select_indices].reset_index(drop=True)
                select_points = hypercube_to_arguments(select_hyper, arg_dict)
                select_pred = cand_values[select_indices]
                select_unc = cand_expl[select_indices]

                toc = time.perf_counter()
                if verbose_level >= 3:
                    print(f'  timing: a subset of candidates selected in {toc - tic:.4f} seconds')
                tic = toc

                select_values = _evaluate_parallel(executor, exp_func, select_points)
                evaluated += len(select_values)
                total_evaluated += len(select_values)

                toc = time.perf_counter()
                if verbose_level >= 3:
                    print(f'  timing: a subset of candidates evaluated in {toc - tic:.4f} seconds')
                tic = toc

                computed_indices = np.vstack([computed_indices, select_cand_indices])
                computed_hyper = _indices_to_hyper_centers(computed_indices, arg_len)
                computed_sample = _indices_to_sample(computed_indices, arg_type, arg_array)
                computed_points.extend(select_points)
                computed_values = np.concatenate((computed_values, select_values), axis=None)
                evaluated_set.update(_point_key_set(select_cand_indices))

                index = int(opt_func(computed_values))
                best_point = computed_points[index]
                best_value = float(computed_values[index])
                best_sample = computed_sample.iloc[index]

                if verbose_level >= 2:
                    print(f'evaluated so far: {len(select_values)} selected, {evaluated} evaluated since last restart, {total_evaluated} in total')
                    print(f'  best value={best_value} for point:  {best_point}')

                if opt == 'max':
                    if total_best_value is None or total_best_value < best_value:
                        total_best_value = best_value
                        total_best_point = best_point
                else:
                    if total_best_value is None or total_best_value > best_value:
                        total_best_value = best_value
                        total_best_point = best_point

                if opt == 'max':
                    if best_value > previous_best_value * (100 + gamma) / 100:
                        iter_last_improvement = iteration
                        previous_best_value = best_value
                else:
                    if best_value < previous_best_value * (100 - gamma) / 100:
                        iter_last_improvement = iteration
                        previous_best_value = best_value

                if iteration - iter_last_improvement > r:
                    if verbose_level >= 2:
                        print(f'restart: no improvement since iteration {iter_last_improvement}, now iteration={iteration}')
                    break

                _fit_surrogate(xgb_model, computed_sample, computed_values, test_set_percentage, rng)

                toc = time.perf_counter()
                if verbose_level >= 3:
                    print(f'  timing: new surrogate model trained in {toc - tic:.4f} seconds')
                tic = toc

                if uncertainty_metric == 'aposteriori':
                    uncertainty_model.update(computed_sample, select_pred, select_values)
                elif uncertainty_metric == 'voronoi':
                    uncertainty_model.update(computed_hyper)

                toc = time.perf_counter()
                if verbose_level >= 3:
                    print(f'  timing: model for uncertainty predictions updated in {toc - tic:.4f} seconds')
                tic = toc

                if verbose_level >= 1:
                    with open(filename, 'a') as f:
                        for i in range(len(select_values)):
                            line = str(iteration)
                            for j in range(dim):
                                line += f', {select_sample.iat[i, j]}'
                            line += f', {select_values[i]}'
                            line += f', {select_pred[i]}'
                            line += f', {select_unc[i]}'
                            line += f', {best_value}'
                            for j in range(dim):
                                line += f', {best_sample.iat[j]}'
                            f.write(line + '\n')

    return total_best_value, total_best_point


if __name__ == '__main__':
    pass
