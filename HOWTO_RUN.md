# Running STGFIP

There is no `main` in the individual scripts — they are small stages that
must run in order, because each consumes the previous one's output. Use the
driver:

```bash
python3 run_all.py            # whole pipeline, ~12 min
python3 run_all.py --list     # show the 18 stages
python3 run_all.py --stage 11 # re-run one stage
python3 run_all.py --from 11  # resume from a stage
```

## Requirements

```bash
pip install numpy scipy scikit-learn matplotlib
```
Python 3.9+. Verified on numpy 2.2, scipy 1.15, scikit-learn 1.7.

## What it writes

* `/tmp/stgfip_data_main/`  — coupled corpus (raw + labelled), ~1.2 GB
* `/tmp/stgfip_data_nocpl/` — the C = 0 ablation corpus
* `/tmp/ff_<tag>_<cfg>.npz` — feature matrices
* `results_*.json`          — every number reported in the paper
* `figs2/`                  — the figures

Nothing is written outside `/tmp` and this directory.

## Common errors

**`FileNotFoundError: /tmp/stgfip_data_main/raw.npz`**
A stage was run out of order, or `/tmp` was cleared (it is wiped on reboot).
Run `python3 run_all.py` from the start, or `--from <n>` at the first stage
whose inputs still exist.

**`ConvergenceWarning: Maximum iterations (40) reached`** (stage 14, MLP)
Expected. `max_iter` is fixed at 40 so the published numbers are
reproducible; the warning is not an error and the MLP results are the ones
in Table 2. Raising it changes the reported figures.

**Stage 1 or 3 appears to hang**
Each integrates 96 episodes over 5738 RK4 steps and takes 50–100 s with no
intermediate output. Pass a smaller chunk size to checkpoint more often:
`python3 gen_corpus.py main 1 2000` (resumable — re-run to continue).

## Expected output

Stage 2 must print:

```
episodes 96 steps 5738
sample class counts [398370  38680  43213   4522   3895  62168      0]
cascaded samples    10819 (7.1% of faulty)
cross-subsystem channel hits 22  median latency 2138.8 s
episodes that cascaded: 11 / 80
```

Stage 4 (the C = 0 ablation) must print `cascaded samples 0` and
`episodes that cascaded: 0 / 80`. That contrast is the result in Sec. V-B.
