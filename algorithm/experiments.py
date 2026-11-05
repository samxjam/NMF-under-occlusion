"""Run the robustness experiments and write one CSV row per (setting, run, algorithm).

Usage (from the `code` folder):

    python algorithm/experiments.py              # full experiments
    python algorithm/experiments.py --quick      # 1 run, 100 iterations, smoke test

Two noise protocols are evaluated on both datasets:

* ``all``      every image receives `n_blocks` white squares of side `block_size`;
               the grid covers block_size x n_blocks.
* ``partial``  only a fraction of the images is occluded (sample-wise outliers),
               the case the L2,1-norm is designed for.

Each run draws a random 90% subset of the images, a fresh noise pattern and a
random initialisation shared by all algorithms, so algorithms are compared on
identical inputs. Every algorithm runs until the relative change of its
objective falls below `--tol` (default 1e-4), capped at `--max-iter`
iterations. RNMF-L1 chooses lam by Huber's rule (`nmf.huber_lambda`), which
never sees the clean images. Results are appended to the CSV as each job finishes, and
jobs already present in the CSV are skipped, so an interrupted run resumes.
"""

import os

# One BLAS thread per worker; parallelism comes from the process pool.
for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(var, '1')

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from data import add_block_occlusion, load_dataset
from metrics import clustering_scores, relative_reconstruction_error
from nmf import ALGORITHMS, huber_lambda, random_init

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))


# (protocol, block_size, n_blocks, fraction of images occluded)
SETTINGS = [('all', 0, 0, 0.0)]
SETTINGS += [('all', b, n, 1.0) for b in (4, 8, 12) for n in (1, 2, 3)]
SETTINGS += [('partial', 16, 3, f) for f in (0.1, 0.3, 0.5)]

FIELDS = ['dataset', 'protocol', 'block_size', 'n_blocks', 'fraction', 'run', 'algorithm',
          'rre', 'rre_clean_imgs', 'acc', 'nmi', 'iterations', 'seconds', 'lam']

_cache = {}


def _dataset(name):
    if name not in _cache:
        _cache[name] = load_dataset(name)
    return _cache[name]


def corrupt(V, shape, block_size, n_blocks, fraction, rng):
    """Occlude a random `fraction` of the columns of V. Returns (V_noisy, occluded_cols)."""
    cols = rng.random(V.shape[1]) < fraction
    V_noisy = V.copy()
    if block_size and cols.any():
        V_noisy[:, cols], _ = add_block_occlusion(V[:, cols], shape, block_size, n_blocks, rng)
    return V_noisy, cols


def prepare_run(dataset, setting, run):
    """The 90% subset, noisy data and shared initialisation for one run.

    Fully determined by (dataset, setting, run), so other scripts can reproduce
    exactly the inputs used here. Returns (V, Y, V_noisy, cols, W0, H0).
    """
    protocol, block_size, n_blocks, fraction = setting
    V_full, Y_full, shape = _dataset(dataset)
    rng = np.random.default_rng([run, block_size, n_blocks, int(fraction * 100)])
    subset = np.random.default_rng(run).choice(V_full.shape[1], int(0.9 * V_full.shape[1]), replace=False)
    V, Y = V_full[:, subset], Y_full[subset]
    V_noisy, cols = corrupt(V, shape, block_size, n_blocks, fraction, rng)
    k = len(set(Y))
    W0, H0 = random_init(V.shape[0], V.shape[1], k, V_noisy.mean(), np.random.default_rng(1000 + run))
    return V, Y, V_noisy, cols, W0, H0


def run_job(dataset, setting, run, max_iter, tol):
    """Fit every algorithm on one noisy 90% subset; return a list of result rows."""
    protocol, block_size, n_blocks, fraction = setting
    V, Y, V_noisy, cols, W0, H0 = prepare_run(dataset, setting, run)

    rows = []
    for name, solver in ALGORITHMS.items():
        start = time.time()
        hyper = {'lam': huber_lambda(V_noisy, W0, H0)} if name == 'RNMF-L1' else {}
        W, H, history = solver(V_noisy, W0, H0, max_iter=max_iter, tol=tol, **hyper)
        seconds = time.time() - start
        acc, nmi = clustering_scores(H, Y, seed=run)
        clean = ~cols if cols.any() and not cols.all() else np.ones_like(cols)
        rows.append({
            'dataset': dataset, 'protocol': protocol, 'block_size': block_size,
            'n_blocks': n_blocks, 'fraction': fraction, 'run': run, 'algorithm': name,
            'rre': relative_reconstruction_error(V, W, H),
            'rre_clean_imgs': relative_reconstruction_error(V[:, clean], W, H[:, clean]),
            'acc': acc, 'nmi': nmi, 'iterations': len(history), 'seconds': seconds,
            'lam': hyper.get('lam', ''),
        })
    return rows


def _job_key(dataset, setting, run):
    protocol, b, n, f = setting
    return (dataset, protocol, str(b), str(n), f'{f:.2f}', str(run))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='1 run and 100 iterations, for a fast check')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--max-iter', type=int, default=3000)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--datasets', nargs='+', default=['ORL', 'YaleB'])
    parser.add_argument('--out', default=None, help='CSV path (default: results/results[_quick].csv)')
    args = parser.parse_args()
    if args.quick:
        args.runs, args.max_iter = 1, 100
    out = args.out or os.path.join(RESULTS_DIR, 'results_quick.csv' if args.quick else 'results.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for r in csv.DictReader(f):
                done.add((r['dataset'], r['protocol'], r['block_size'], r['n_blocks'],
                          f"{float(r['fraction']):.2f}", r['run']))
    jobs = [(d, s, r) for d in args.datasets for s in SETTINGS for r in range(args.runs)
            if _job_key(d, s, r) not in done]
    print(f'{len(jobs)} jobs to run ({len(done)} already in {out})')

    new_file = not os.path.exists(out)
    with open(out, 'a', newline='') as f, ProcessPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        futures = {pool.submit(run_job, d, s, r, args.max_iter, args.tol): (d, s, r) for d, s, r in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            rows = fut.result()
            writer.writerows(rows)
            f.flush()
            d, s, r = futures[fut]
            summary = '  '.join(f"{row['algorithm']}={row['rre']:.3f}" for row in rows)
            print(f'[{i}/{len(jobs)}] {d} {s} run={r}  RRE: {summary}', flush=True)


if __name__ == '__main__':
    main()
