# NMF robustness to block occlusion

Four NMF algorithms implemented with NumPy only (`algorithm/nmf.py`):
L2-NMF, L2,1-NMF, L1-regularised robust NMF (RNMF-L1) and correntropy NMF (CIM-NMF).

## Setup

Put the datasets in `data/` (not included in the submission):

```
data/ORL/s1 ... s40
data/CroppedYaleB/yaleB01 ... yaleB39
```

```
pip install numpy pillow matplotlib scikit-learn
```

## Run (from this `code` folder)

```
python algorithm/experiments.py --quick   # smoke test, ~1 minute
python algorithm/experiments.py           # full experiments (~19 min) -> results/results.csv
python algorithm/extra_studies.py         # lambda, rank and sigma studies (~10 min) -> results/extra_studies.csv
python algorithm/plots.py                 # figures + LaTeX tables -> results/figures/
```

`experiments.py` resumes from an existing CSV if interrupted. Options: `--datasets ORL`,
`--runs N`, `--max-iter N`, `--workers N`. scikit-learn is used only for evaluation
(K-means, NMI) in `algorithm/metrics.py`.
