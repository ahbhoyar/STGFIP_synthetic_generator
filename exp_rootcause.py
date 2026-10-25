#!/usr/bin/env python3
"""
Q3: root-cause attribution by leave-one-episode-out.

The single 60/15/25 split reported in the submitted version was not a valid
evaluation: it put origin classes {AOCS, THM} in train and {AOCS, EPS} in
test, so most test cascaded samples belonged to a class absent from training.
Here the unit of evaluation is the EPISODE, as the reviewer asked: each of the
cascading episodes is held out in turn, a classifier is fitted on the
remaining ones, and the episode is attributed by majority vote over its
cascaded samples. The majority baseline is recomputed per fold from the
training episodes only.

    python3 exp_rootcause.py          # writes results_rootcause.json
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stgfip.batch import CHANNELS          # noqa: E402
from features import build                 # noqa: E402

DATA = "/tmp/stgfip_data_main"
SUBS = ["none", "EPS", "AOCS", "THM", "PRP", "OBC", "COM", "NAV", "PLD"]
SUB = 12          # same 12 s decision cadence as the detectors


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(round(max(0.0, c - h), 3)), float(round(min(1.0, c + h), 3)))


def main():
    if not os.path.exists(DATA + "/labelled.npz"):
        sys.exit("missing %s/labelled.npz\n  run: python3 run_all.py --stage 2"
                 % DATA)

    lab = np.load(DATA + "/labelled.npz")
    X, _ = build({c: lab["F_" + c] for c in CHANNELS})        # (T, N, F)
    X = X[::SUB]
    casc = lab["casc"][::SUB]
    origin = lab["origin"][::SUB]

    # Episodes that actually cascade are the only ones attribution applies to.
    eps = [j for j in range(X.shape[1]) if (casc[:, j] > 0).any()]
    if not eps:
        sys.exit("no cascading episodes found")

    rows, truth = {}, {}
    for j in eps:
        m = casc[:, j] > 0
        rows[j] = X[m, j, :]
        o = origin[m, j]
        truth[j] = int(Counter(o[o > 0]).most_common(1)[0][0])

    correct, per_ep, maj_hits = 0, [], 0
    for j in eps:
        tr = [k for k in eps if k != j]
        Xtr = np.concatenate([rows[k] for k in tr])
        ytr = np.concatenate([np.full(len(rows[k]), truth[k]) for k in tr])

        clf = RandomForestClassifier(n_estimators=300, random_state=0,
                                     n_jobs=-1).fit(Xtr, ytr)
        pred = Counter(clf.predict(rows[j])).most_common(1)[0][0]
        hit = int(pred) == truth[j]
        correct += hit
        per_ep.append([j, SUBS[truth[j]], SUBS[int(pred)], bool(hit)])

        # per-fold majority baseline: most common origin among TRAIN episodes
        maj = Counter(truth[k] for k in tr).most_common(1)[0][0]
        maj_hits += (maj == truth[j])

    n = len(eps)
    res = {"n_episodes": n,
           "loeo_acc": round(correct / n, 3),
           "majority_acc": maj_hits / n,
           "ci": list(wilson(correct, n)),
           "per_episode": per_ep}
    json.dump(res, open(os.path.join(HERE, "results_rootcause.json"), "w"),
              indent=1)

    print("LOEO attribution over %d cascading episodes" % n)
    for j, t, p, hit in per_ep:
        print("  ep %-3d true %-5s pred %-5s %s" % (j, t, p, "ok" if hit else "MISS"))
    print("\n  accuracy %d/%d = %.3f   Wilson 95%% CI %s"
          % (correct, n, correct / n, res["ci"]))
    print("  majority baseline (per fold) %.3f" % res["majority_acc"])


if __name__ == "__main__":
    main()
