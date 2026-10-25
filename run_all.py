#!/usr/bin/env python3
"""
STGFIP - single entry point. Reproduces every number in the paper.

    python3 run_all.py              # full pipeline (~8 min)
    python3 run_all.py --stage 3    # run one stage only
    python3 run_all.py --list       # show stages and exit

The individual scripts are deliberately small and take positional
arguments; they must be run in order because each consumes the previous
one's output. This driver enforces that order, checks the inputs exist
before each step, and fails with a readable message instead of a
traceback from inside NumPy.

Intermediate data goes to /tmp/stgfip_data_<tag>/ and /tmp/ff_<tag>_<cfg>.npz
(roughly 1.5 GB total). Results are written next to this file as
results_*.json. Nothing is written outside /tmp and this directory.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


# (label, script, args, produces) -- `produces` is checked afterwards so a
# silent failure is caught at the step that caused it, not three steps later.
STAGES = [
    ("Generate coupled corpus (96 episodes, C != 0)",
     "gen_corpus.py", ["main", "1", "6000"],
     ["/tmp/stgfip_data_main/raw.npz"]),

    ("Label coupled corpus (paired-trajectory differencing)",
     "build_dataset.py", ["main"],
     ["/tmp/stgfip_data_main/labelled.npz"]),

    ("Generate ablation corpus (C = 0, same seeds and faults)",
     "gen_corpus.py", ["nocpl", "0", "6000"],
     ["/tmp/stgfip_data_nocpl/raw.npz"]),

    ("Label ablation corpus",
     "build_dataset.py", ["nocpl"],
     ["/tmp/stgfip_data_nocpl/labelled.npz"]),

    ("Label statistics: kappa sensitivity and censoring",
     "stats_labels.py", [],
     ["results_labels.json"]),

    ("CCSDS conformance: round-trip 250 CADUs / 2000 packets",
     "exp_conformance.py", ["nominal"],
     ["results_conformance.json"]),

    ("Build feature matrices - full",
     "prep_final.py", ["main", "full"], ["/tmp/ff_main_full.npz"]),
    ("Build feature matrices - raw channels",
     "prep_final.py", ["main", "raw"], ["/tmp/ff_main_raw.npz"]),
    ("Build feature matrices - without rho",
     "prep_final.py", ["main", "norho"], ["/tmp/ff_main_norho.npz"]),
    ("Build feature matrices - C = 0 ablation",
     "prep_final.py", ["nocpl", "full"], ["/tmp/ff_nocpl_full.npz"]),

    # Split per model rather than one "all" run: each fits in a couple of
    # minutes, so a failure costs one model instead of the whole table, and
    # results_final.json is merged incrementally.
    ("Detectors: static threshold + Isolation Forest (Table 2)",
     "exp_final.py", ["main", "full", "base"], ["results_final.json"]),
    ("Detectors: Random Forest (Table 2)",
     "exp_final.py", ["main", "full", "rf"], ["results_final.json"]),
    ("Detectors: histogram gradient boosting (Table 2)",
     "exp_final.py", ["main", "full", "hgb"], ["results_final.json"]),
    ("Detectors: MLP (Table 2)",
     "exp_final.py", ["main", "full", "mlp"], ["results_final.json"]),
    ("Ablation A1: C = 0",
     "exp_final.py", ["nocpl", "full", "rf"], ["results_final.json"]),
    ("Ablation A2: raw channels",
     "exp_final.py", ["main", "raw", "rf"], ["results_final.json"]),
    ("Ablation A2: without rho",
     "exp_final.py", ["main", "norho", "rf"], ["results_final.json"]),

    ("Root-cause attribution, leave-one-episode-out (Q3)",
     "exp_rootcause.py", [], ["results_rootcause.json"]),

    ("Export released artifacts (splits, channels, schedule, CADUs)",
     "export_artifacts.py", [], ["artifacts/splits.json",
                                 "artifacts/sample_250.cadu"]),

    ("Figures",
     "make_figs2.py", [], []),
]


def check_deps():
    missing = []
    for mod in ("numpy", "scipy", "sklearn", "matplotlib"):
        try:
            __import__(mod)
        except ImportError:
            missing.append({"sklearn": "scikit-learn"}.get(mod, mod))
    if missing:
        sys.exit("Missing packages: %s\n  pip install %s"
                 % (", ".join(missing), " ".join(missing)))


def run(stage_no, stage):
    label, script, args, produces = stage
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        sys.exit("[%02d] missing script: %s" % (stage_no, script))

    print("\n[%02d/%02d] %s" % (stage_no, len(STAGES), label))
    print("        %s %s" % (script, " ".join(args)))
    t0 = time.time()
    r = subprocess.run([PY, path] + args, cwd=HERE)
    if r.returncode != 0:
        sys.exit("\n[%02d] FAILED: %s %s (exit %d)\n"
                 "Re-run just this stage with:\n"
                 "    python3 run_all.py --stage %d"
                 % (stage_no, script, " ".join(args), r.returncode, stage_no))

    for f in produces:
        target = f if f.startswith("/tmp") else os.path.join(HERE, f)
        if not os.path.exists(target):
            sys.exit("[%02d] %s exited 0 but did not produce %s"
                     % (stage_no, script, f))
    print("        ok (%.1fs)" % (time.time() - t0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int,
                    help="run only this stage number (see --list)")
    ap.add_argument("--from", dest="start", type=int, default=1,
                    help="start from this stage and continue")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    a = ap.parse_args()

    if a.list:
        for i, (label, script, args, _) in enumerate(STAGES, 1):
            print("%2d  %-52s %s %s" % (i, label, script, " ".join(args)))
        return

    check_deps()

    if a.stage:
        run(a.stage, STAGES[a.stage - 1])
    else:
        t0 = time.time()
        for i, stage in enumerate(STAGES, 1):
            if i < a.start:
                continue
            run(i, stage)
        print("\nAll stages complete in %.1f min." % ((time.time() - t0) / 60))
        summarise()


def summarise():
    """Print the handful of numbers that appear in the paper."""
    def load(n):
        try:
            return json.load(open(os.path.join(HERE, n)))
        except Exception:
            return None

    lab, fin = load("results_labels.json"), load("results_final.json")
    print("\n" + "=" * 62)
    print("Key results (compare against the paper)")
    print("=" * 62)
    if lab:
        c = lab.get("censoring", {})
        print("  cascading episodes      %s / 80          (paper: 11 / 80)"
              % c.get("cascaded"))
        print("  censoring rate          %-15s (paper: 0.633)"
              % c.get("censoring_rate"))
    if fin:
        for name in ("Static threshold", "MLP"):
            r = (fin.get("main_full") or {}).get(name)
            if r:
                print("  %-22s F1 %-5s R@10fa %-6s delay %s min"
                      % (name, r.get("f1"), r.get("recall@10fa"),
                         r.get("median_delay_min")))
        a1 = (fin.get("nocpl_full") or {}).get("Random Forest")
        if a1:
            print("  A1 (C = 0) RF           F1 %s" % a1.get("f1"))
    rc = load("results_rootcause.json")
    if rc:
        print("  LOEO attribution        %d/%d  CI %s   (paper: 11/11)"
              % (round(rc["loeo_acc"] * rc["n_episodes"]), rc["n_episodes"],
                 rc["ci"]))
    print("=" * 62)


if __name__ == "__main__":
    main()
