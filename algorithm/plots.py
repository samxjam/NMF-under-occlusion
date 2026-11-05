"""Produce the report's figures and LaTeX tables from results/results.csv.

Usage (from the `code` folder, after experiments.py has finished):

    python algorithm/plots.py [--out ../report]

Quantitative figures and tables come from the CSV. The qualitative figures
(noise examples, reconstructions, outlier maps, basis images) run a few
extra single fits, which takes about a minute. Only outputs the report uses
are produced.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data import add_block_occlusion, load_dataset
from experiments import RESULTS_DIR, corrupt
from nmf import ALGORITHMS, cim_weights, huber_lambda, nmf_rnmf, random_init, rnmf_outliers

ALGS = list(ALGORITHMS)
# Display names: subscripted norms, matching the report text.
LABEL = {'L2-NMF': 'L$_2$-NMF', 'L21-NMF': 'L$_{2,1}$-NMF', 'RNMF-L1': 'RNMF-L$_1$', 'CIM-NMF': 'CIM-NMF'}
COLORS = dict(zip(ALGS, ['#4C72B0', '#DD8452', '#55A868', '#C44E52']))
MARKERS = dict(zip(ALGS, ['o', 's', '^', 'D']))
DATASETS = ['ORL', 'YaleB']

plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 9,
                     'legend.fontsize': 8, 'savefig.bbox': 'tight', 'savefig.dpi': 200})


def load_results(path):
    """Group rows by (dataset, protocol, b, n, fraction, algorithm) -> metric -> list of values."""
    groups = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r['dataset'], r['protocol'], int(r['block_size']), int(r['n_blocks']),
                   round(float(r['fraction']), 2), r['algorithm'])
            for m in ('rre', 'rre_clean_imgs', 'acc', 'nmi', 'seconds', 'iterations'):
                groups[key][m].append(float(r[m]))
    return groups


def stat(groups, key, metric):
    vals = np.array(groups[key][metric])
    return vals.mean(), vals.std()


# ----------------------------------------------------------------------------- tables

def _cells(stats, lower_is_better):
    """Format mean +- std cells; every mean that ties the best at 3 decimals is bold."""
    rounded = [round(m, 3) for m, _ in stats]
    best = min(rounded) if lower_is_better else max(rounded)
    cells = []
    for r, (m, s) in zip(rounded, stats):
        cell = f'{m:.3f} $\\pm$ {s:.3f}'
        cells.append(r'\textbf{' + cell + '}' if r == best else cell)
    return cells


def _header(first_cols):
    return first_cols + ' & ' + ' & '.join(LABEL[a] for a in ALGS) + r' \\'


def write_main_table(groups, metric, path, caption, label, lower_is_better):
    """Mean +- std over runs for every occlusion setting; best mean per row in bold."""
    settings = [(0, 0)] + [(b, n) for b in (4, 8, 12) for n in (1, 2, 3)]
    lines = [r'\begin{table}[htbp]', r'\caption{' + caption + '}', r'\label{' + label + '}',
             r'\begin{center}', r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{ll' + 'c' * len(ALGS) + '}', r'\hline',
             _header(r'Dataset & $(b, n)$'), r'\hline']
    for d in DATASETS:
        for i, (b, n) in enumerate(settings):
            proto, frac = ('all', 0.0) if b == 0 else ('all', 1.0)
            stats = [stat(groups, (d, proto, b, n, frac, a), metric) for a in ALGS]
            cells = _cells(stats, lower_is_better)
            name = d if i == 0 else ''
            setting = 'clean' if b == 0 else f'({b}, {n})'
            lines.append(f'{name} & {setting} & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\hline')
    lines += [r'\end{tabular}}', r'\end{center}', r'\end{table}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_partial_table(groups, path):
    lines = [r'\begin{table}[htbp]',
             r'\caption{Sample-wise corruption ($b=16$, $n=3$ on a fraction $p$ of the images). '
             r'RRE measured on the uncorrupted images only (mean $\pm$ std over 5 runs; best in bold).}',
             r'\label{tab:partial}', r'\begin{center}', r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{ll' + 'c' * len(ALGS) + '}', r'\hline',
             _header(r'Dataset & $p$'), r'\hline']
    for d in DATASETS:
        for i, p in enumerate((0.1, 0.3, 0.5)):
            stats = [stat(groups, (d, 'partial', 16, 3, p, a), 'rre_clean_imgs') for a in ALGS]
            cells = _cells(stats, lower_is_better=True)
            lines.append(f"{d if i == 0 else ''} & {p:.1f} & " + ' & '.join(cells) + r' \\')
        lines.append(r'\hline')
    lines += [r'\end{tabular}}', r'\end{center}', r'\end{table}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _extra(path_csv):
    vals = defaultdict(lambda: defaultdict(list))
    with open(path_csv) as f:
        for r in csv.DictReader(f):
            key = (r['study'], r['dataset'], int(r['block_size']), r['algorithm'], r['variant'])
            for m in ('rre', 'acc', 'iterations'):
                vals[key][m].append(float(r[m]))
    return vals


def _ms(values):
    return f'{np.mean(values):.3f} $\\pm$ {np.std(values):.3f}'


def write_lambda_table(groups, extra, path):
    """RNMF-L1 RRE: Huber-rule vs oracle lam, converged vs stopped at 500 iterations."""
    variants = ['huber, converged', 'huber, 500 it.', 'oracle, converged', 'oracle, 500 it.']
    heads = [r'Huber $\lambda$', r'Huber $\lambda$, 500 it.', r'oracle $\lambda$', r'oracle $\lambda$, 500 it.']
    lines = [r'\begin{table}[htbp]',
             r"\caption{RRE of \Rnmf{} with $\lambda$ from Huber's rule (main study) and with an oracle "
             r'$\lambda=0.05$ tuned on the clean images, each run to convergence or stopped after 500 '
             r'iterations, on the inputs of the main study. CIM-NMF (converged) for reference. '
             r'Mean $\pm$ std over 5 runs; best in bold.}',
             r'\label{tab:lambda}', r'\begin{center}', r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{ll' + 'c' * (len(variants) + 1) + '}', r'\hline',
             r'Dataset & $(b, n)$ & ' + ' & '.join(heads) + r' & CIM-NMF \\', r'\hline']
    for d in DATASETS:
        for i, b in enumerate((8, 12)):
            stats = [(np.mean(v), np.std(v)) for v in
                     [extra[('lambda', d, b, 'RNMF-L1', var)]['rre'] for var in variants]]
            stats.append(stat(groups, (d, 'all', b, 2, 1.0, 'CIM-NMF'), 'rre'))
            lines.append(f"{d if i == 0 else ''} & ({b}, 2) & " + ' & '.join(_cells(stats, True)) + r' \\')
        lines.append(r'\hline')
    lines += [r'\end{tabular}}', r'\end{center}', r'\end{table}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------- figures

def plot_metric_vs_block(groups, metric, ylabel, path):
    """Rows: datasets. Columns: number of blocks. x: block size (0 = clean)."""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharey='row')
    for r, d in enumerate(DATASETS):
        for c, n in enumerate((1, 2, 3)):
            ax = axes[r, c]
            for a in ALGS:
                m0, s0 = stat(groups, (d, 'all', 0, 0, 0.0, a), metric)
                xs, ms, ss = [0], [m0], [s0]
                for b in (4, 8, 12):
                    m, s = stat(groups, (d, 'all', b, n, 1.0, a), metric)
                    xs.append(b), ms.append(m), ss.append(s)
                ax.errorbar(xs, ms, yerr=ss, label=LABEL[a], color=COLORS[a], marker=MARKERS[a],
                            ms=4, lw=1.3, capsize=2)
            ax.set_title(f'{d}, {n} block{"s" if n > 1 else ""}')
            ax.set_xticks([0, 4, 8, 12])
            ax.grid(alpha=0.3)
            if r == 1:
                ax.set_xlabel('block size $b$ (pixels)')
            if c == 0:
                ax.set_ylabel(ylabel)
    axes[0, 0].legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_noise_examples(path):
    rng = np.random.default_rng(7)
    configs = [(4, 1), (8, 1), (12, 1), (4, 3), (8, 3), (12, 3)]
    fig, axes = plt.subplots(2, 1 + len(configs), figsize=(7.2, 2.9))
    for r, d in enumerate(DATASETS):
        V, _, shape = load_dataset(d)
        v = V[:, [3 if d == 'ORL' else 100]]
        panels = [('original', v)] + [(f'$b$={b}, $n$={n}', add_block_occlusion(v, shape, b, n, rng)[0])
                                      for b, n in configs]
        for c, (title, img) in enumerate(panels):
            axes[r, c].imshow(img.reshape(shape), cmap='gray', vmin=0, vmax=1)
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(title, fontsize=8)
        axes[r, 0].set_ylabel(d)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def plot_partial_examples(path, fraction=0.3, n_show=8):
    """Sample-wise occlusion: a random subset of images gets three 16x16 blocks, the rest stay clean."""
    fig, axes = plt.subplots(2, n_show, figsize=(7.2, 2.6))
    for r, d in enumerate(DATASETS):
        V, _, shape = load_dataset(d)
        pick = np.random.default_rng(3).choice(V.shape[1], n_show, replace=False)
        V_noisy, cols = corrupt(V[:, pick], shape, 16, 3, fraction, np.random.default_rng(4))
        for c in range(n_show):
            ax = axes[r, c]
            ax.imshow(V_noisy[:, c].reshape(shape), cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title('occluded' if cols[c] else 'clean', fontsize=8)
        axes[r, 0].set_ylabel(d)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def _fit_all(V_noisy, k, seed):
    """Fit every algorithm to convergence. Returns (results, RNMF lam, (W0, H0))."""
    W0, H0 = random_init(V_noisy.shape[0], V_noisy.shape[1], k, V_noisy.mean(), np.random.default_rng(seed))
    lam = huber_lambda(V_noisy, W0, H0)
    res = {a: f(V_noisy, W0, H0, **({'lam': lam} if a == 'RNMF-L1' else {})) for a, f in ALGORITHMS.items()}
    return res, lam, (W0, H0)


def plot_outlier_maps(fits, examples, path):
    """What each robust method treats as an outlier: RNMF's |E| and CIM's weights Omega.

    Also prints, per dataset, how well each map separates occluded from clean
    pixels (over all images), for quoting in the report.
    """
    fig, axes = plt.subplots(2, 5, figsize=(6.4, 2.9))
    titles = ['occluded input', 'occlusion mask', r'RNMF-L$_1$: $|E|$', r'CIM-NMF: $\Omega$', 'CIM-NMF recon.']
    for r, d in enumerate(DATASETS):
        V, V_noisy, shape, res, lam, _ = fits[d]
        mask = V_noisy != V
        E = np.abs(rnmf_outliers(V_noisy, *res['RNMF-L1'][:2], lam))
        Om = cim_weights(V_noisy, *res['CIM-NMF'][:2])
        e_share = E[mask].sum() / E.sum()
        print(f'{d}: occluded pixels {mask.mean():.1%} of all; |E| mass inside mask {e_share:.1%}; '
              f'mean Omega occluded {Om[mask].mean():.3f} vs clean {Om[~mask].mean():.3f}')
        j = examples[d][0]
        W, H = res['CIM-NMF'][:2]
        panels = [(V_noisy[:, j], 'gray', (0, 1)), (mask[:, j], 'gray', (0, 1)),
                  (E[:, j], 'magma', (0, None)), (Om[:, j], 'viridis', (0, 1)),
                  (np.clip(W @ H[:, j], 0, 1), 'gray', (0, 1))]
        for c, (img, cmap, (lo, hi)) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img.reshape(shape), cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
        axes[r, 0].set_ylabel(d)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def plot_early_stopping(fits, examples, path):
    """RNMF-L1 reconstructions after 500 iterations and at convergence, same inputs."""
    fig, axes = plt.subplots(2, 4, figsize=(5.2, 2.9))
    titles = ['clean', 'occluded', r'RNMF-L$_1$, 500 it.', r'RNMF-L$_1$, converged']
    for r, d in enumerate(DATASETS):
        V, V_noisy, shape, res, lam, (W0, H0) = fits[d]
        W5, H5, _ = nmf_rnmf(V_noisy, W0, H0, lam=lam, max_iter=500)
        Wc, Hc = res['RNMF-L1'][:2]
        rre = lambda W, H: np.linalg.norm(V - W @ H) / np.linalg.norm(V)
        print(f'{d}: RNMF-L1 RRE after 500 it. {rre(W5, H5):.3f}, converged '
              f'({len(res["RNMF-L1"][2])} it.) {rre(Wc, Hc):.3f}')
        j = examples[d][0]
        panels = [V[:, j], V_noisy[:, j], W5 @ H5[:, j], Wc @ Hc[:, j]]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1).reshape(shape), cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
        axes[r, 0].set_ylabel(d)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def qualitative(out_dir):
    """Outlier maps, reconstructions and basis images for one noise setting per dataset."""
    examples = {'ORL': [3, 120, 251], 'YaleB': [100, 900, 1700]}
    fits = {}
    for d in DATASETS:
        V, Y, shape = load_dataset(d)
        rng = np.random.default_rng(11)
        V_noisy, _ = corrupt(V, shape, 8, 2, 1.0, rng)
        fits[d] = (V, V_noisy, shape, *_fit_all(V_noisy, len(set(Y)), seed=5))

    plot_outlier_maps(fits, examples, os.path.join(out_dir, 'fig_weights.pdf'))
    plot_early_stopping(fits, examples, os.path.join(out_dir, 'fig_early_stop.pdf'))

    # Reconstructions: rows = example images, columns = clean, noisy, each algorithm.
    fig, axes = plt.subplots(6, 2 + len(ALGS), figsize=(6.4, 7.6))
    for di, d in enumerate(DATASETS):
        V, V_noisy, shape, res, _, _ = fits[d]
        for ei, j in enumerate(examples[d]):
            row = 3 * di + ei
            panels = [('clean', V[:, j]), ('occluded', V_noisy[:, j])]
            panels += [(LABEL[a], (res[a][0] @ res[a][1][:, j])) for a in ALGS]
            for c, (title, img) in enumerate(panels):
                ax = axes[row, c]
                ax.imshow(np.clip(img, 0, 1).reshape(shape), cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([]), ax.set_yticks([])
                if row == 0:
                    ax.set_title(title, fontsize=8)
            axes[row, 0].set_ylabel(d if ei == 1 else '')
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(out_dir, 'fig_reconstructions.pdf'))
    plt.close(fig)

    # Basis images learned on occluded ORL.
    V, V_noisy, shape, res, _, _ = fits['ORL']
    fig, axes = plt.subplots(len(ALGS), 10, figsize=(7.2, 4.0))
    for r, a in enumerate(ALGS):
        W = res[a][0]
        for c in range(10):
            ax = axes[r, c]
            ax.imshow(W[:, c].reshape(shape), cmap='gray')
            ax.set_xticks([]), ax.set_yticks([])
        axes[r, 0].set_ylabel(LABEL[a], fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(out_dir, 'fig_basis.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=os.path.join(RESULTS_DIR, 'results.csv'))
    parser.add_argument('--out', default=os.path.join(RESULTS_DIR, 'figures'))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    groups = load_results(args.results)
    write_main_table(groups, 'rre', os.path.join(args.out, 'tab_rre.tex'),
                     r'Relative reconstruction error (lower is better) with $n$ blocks of side $b$ '
                     r'in every image. Mean $\pm$ std over 5 runs; best in bold.', 'tab:rre', True)
    write_main_table(groups, 'acc', os.path.join(args.out, 'tab_acc.tex'),
                     r'Clustering accuracy (higher is better) of K-means on $H$. '
                     r'Mean $\pm$ std over 5 runs; best in bold.', 'tab:acc', False)
    write_partial_table(groups, os.path.join(args.out, 'tab_partial.tex'))
    extra_csv = os.path.join(os.path.dirname(args.results), 'extra_studies.csv')
    if os.path.exists(extra_csv):
        extra = _extra(extra_csv)
        write_lambda_table(groups, extra, os.path.join(args.out, 'tab_lambda.tex'))
    plot_metric_vs_block(groups, 'rre', 'RRE', os.path.join(args.out, 'fig_rre.pdf'))
    plot_noise_examples(os.path.join(args.out, 'fig_noise_examples.pdf'))
    plot_partial_examples(os.path.join(args.out, 'fig_noise_partial.pdf'))
    qualitative(args.out)
    print('figures and tables written to', os.path.abspath(args.out))


if __name__ == '__main__':
    main()
