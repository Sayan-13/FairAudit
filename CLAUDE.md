# CLAUDE.md — Fairness Audit project (Intro to ML, Group 17)

## What this is
Audit of fairness/bias in classical ML models (LogReg, linear SVM, RF, gradient boosting) on the
UCI Adult income dataset. Deliverables: 6-page IEEE two-column report (`Fairness_Audit_IEEE_Report.pdf`),
live Streamlit demo (`demo/app.py`), viva guide (`docs/Viva_Guide.pdf`).
Report deadline: Fri Nov 13 2026 (midday). Presentations: Nov 16–17.
README.md explains the folder map and how to run everything.

## Hard rules
- Classical ML only. No neural nets, PyTorch, TensorFlow. fairlearn is OK (wraps sklearn).
- RANDOM_STATE = 42 everywhere. Split 60/15/25 train/val/test, stratified (test = 25% held out first).
- Never tune on the test set. All preprocessing inside sklearn Pipelines.
- Every number in `report/main.tex` must come from `experiments/results/`. After re-running experiments,
  run `experiments/check_report_numbers.py` and fix the tex by hand (no templating).
- Report must stay 4–6 pages. Compile with `cd report && tectonic main.tex` (needs `[T1]{fontenc}`, already in).
- Do not invent references or results. State missing things as limitations.

## Where things are
- `experiments/common.py` — shared data loading, split, preprocessing and fairness metrics (imported by both scripts)
- `experiments/run_experiments.py` — part 1 (data, models, loss curves, audit, stats, reweighing, thresholds)
- `experiments/run_extensions.py` — part 2 (age, PPV, intersectional, DI remover, Hardt post-proc, EG in-processing)
- `notebooks/fairness_audit_full.ipynb` — generated from common.py + the two scripts for Colab (regenerate it after changing them)
- `presentation/extra_figures/` — plots not used in the report, for slides (the scripts no longer generate these)
- Code style: keep the code short and plain, like student work. Short one-line comments only, no banner comments or long docstrings.
