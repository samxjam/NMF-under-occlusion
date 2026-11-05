"""Supporting studies on exactly the inputs of the main study (see `prepare_run`).

* ``lambda``  RNMF-L1 with lam from Huber's rule (the main study) against an
              oracle lam = 0.05 that was tuned on the clean images, each run to
              convergence and stopped after 500 iterations. Shows how much of
              RNMF-L1's robustness comes from tuning and from early stopping.
* ``rank``    all four algorithms with rank k/2, k and 2k (k = number of classes).
* ``sigma``   CIM-NMF with its kernel width from the mean squared residual (the
              main study) against a robust median-based (MAD) estimate.

Usage (from the `code` folder):

    python algorithm/extra_studies.py                  # all studies
    python algorithm/extra_studies.py --studies sigma  # rerun one study only

Results go to results/extra_studies.csv; rows of studies that are not rerun
are kept.
"""

import os

for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(var, '1')

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments import RESULTS_DIR, prepare_run
from metrics import clustering_scores, relative_reconstruction_error
from nmf import ALGORITHMS, huber_lambda, nmf_cim, nmf_rnmf, random_init

ORACLE_LAM = 0.05
RUNS = range(5)


def _row(study, dataset, setting, run, algorithm, variant, k, V, Y, W, H, history):
    acc, nmi = clustering_scores(H, Y, seed=run)
    return {'study': study, 'dataset': dataset, 'block_size': setting[1], 'n_blocks': setting[2],
            'run': run, 'algorithm': algorithm, 'variant': variant, 'k': k,
            'rre': relative_reconstruction_error(V, W, H), 'acc': acc, 'nmi': nmi,
            'iterations': len(history)}


def lambda_job(dataset, setting, run):
    V, Y, V_noisy, _, W0, H0 = prepare_run(dataset, setting, run)
    rows = []
    for lam_name, lam in (('huber', huber_lambda(V_noisy, W0, H0)), ('oracle', ORACLE_LAM)):
        for budget_name, budget in (('converged', 3000), ('500 it.', 500)):
            W, H, history = nmf_rnmf(V_noisy, W0, H0, lam=lam, max_iter=budget)
            rows.append(_row('lambda', dataset, setting, run, 'RNMF-L1', f'{lam_name}, {budget_name}',
                             W0.shape[1], V, Y, W, H, history))
    return rows


def rank_job(dataset, setting, run, factor):
    V, Y, V_noisy, _, W0, _ = prepare_run(dataset, setting, run)
    k = int(round(factor * len(set(Y))))
    W0, H0 = random_init(V.shape[0], V.shape[1], k, V_noisy.mean(), np.random.default_rng(1000 + run))
    rows = []
    for name, solver in ALGORITHMS.items():
        hyper = {'lam': huber_lambda(V_noisy, W0, H0)} if name == 'RNMF-L1' else {}
        W, H, history = solver(V_noisy, W0, H0, **hyper)
        rows.append(_row('rank', dataset, setting, run, name, f'{factor:g}k', k, V, Y, W, H, history))
    return rows


def sigma_job(dataset, setting, run):
    V, Y, V_noisy, _, W0, H0 = prepare_run(dataset, setting, run)
    rows = []
    for rule in ('mean', 'mad'):
        W, H, history = nmf_cim(V_noisy, W0, H0, sigma_rule=rule)
        rows.append(_row('sigma', dataset, setting, run, 'CIM-NMF', rule, W0.shape[1], V, Y, W, H, history))
    return rows


JOBS = {'lambda': lambda_job, 'rank': rank_job, 'sigma': sigma_job}


def _dispatch(job):
    kind, args = job
    return JOBS[kind](*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--studies', nargs='+', default=list(JOBS), choices=list(JOBS))
    args = parser.parse_args()
    jobs = []
    if 'lambda' in args.studies:
        jobs += [('lambda', (d, ('all', b, 2, 1.0), r)) for d in ('ORL', 'YaleB') for b in (8, 12) for r in RUNS]
    if 'rank' in args.studies:
        jobs += [('rank', (d, ('all', 8, 2, 1.0), r, f)) for d in ('ORL', 'YaleB') for f in (0.5, 1, 2) for r in RUNS]
    if 'sigma' in args.studies:
        jobs += [('sigma', (d, (('all', 0, 0, 0.0) if b == 0 else ('all', b, 2, 1.0)), r))
                 for d in ('ORL', 'YaleB') for b in (0, 8) for r in RUNS]

    out = os.path.join(RESULTS_DIR, 'extra_studies.csv')
    kept = []
    if os.path.exists(out):
        with open(out) as f:
            kept = [r for r in csv.DictReader(f) if r['study'] not in args.studies]
    new_rows = []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i, rows in enumerate(pool.map(_dispatch, jobs), 1):
            new_rows += rows
            print(f'[{i}/{len(jobs)}] {rows[0]["study"]} {rows[0]["dataset"]} done', flush=True)
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(new_rows[0]))
        writer.writeheader()
        writer.writerows(kept + new_rows)
    print('wrote', out)


if __name__ == '__main__':
    main()
