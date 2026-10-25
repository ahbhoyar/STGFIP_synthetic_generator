"""Detectors with matched operating points, event-level metrics, delay,
by-severity breakdown and episode-level bootstrap CIs."""
import sys, json, numpy as np
from sklearn.ensemble import (RandomForestClassifier, IsolationForest,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

TAG, CFG, WHICH = sys.argv[1], sys.argv[2], sys.argv[3]
d = np.load(f"/tmp/ff_{TAG}_{CFG}.npz")
Xtr, ytr = d["tr_X"], d["tr_y"]
Xte, yte = d["te_X"], d["te_y"]
casc, lam, ep, tt = d["te_casc"], d["te_lam"], d["te_ep"], d["te_t"]
tf, NCHA = d["tf"], int(d["ncha"])

SUB_S = 12                      # one decision per 12 s of simulated time
DEC_PER_DAY = 86400 / SUB_S     # = 7200 decisions/day
bt = (yte > 0).astype(int)
nom = bt == 0


def fa_per_day(pred):
    return float(pred[nom].sum() / max(nom.sum(), 1) * DEC_PER_DAY)


def at_matched_far(score, target):
    """Threshold the score to hit `target` false alarms/day; return recall."""
    q = 1.0 - target / DEC_PER_DAY
    thr = np.quantile(score[nom], np.clip(q, 0, 1))
    pred = (score > thr).astype(int)
    return (float(pred[bt == 1].mean()), float(pred[casc > 0].mean())
            if (casc > 0).sum() else float("nan"), fa_per_day(pred))


def event_metrics(pred):
    """Per-episode detection and delay from true onset."""
    det, delays = 0, []
    fault_eps = [e for e in np.unique(ep) if tf[e] >= 0]
    for e in fault_eps:
        m = (ep == e) & (tt > tf[e])
        if not m.any():
            continue
        if pred[m].any():
            det += 1
            delays.append(float(tt[m][np.argmax(pred[m])] - tf[e]))
    return (det / max(len(fault_eps), 1),
            float(np.median(delays)) / 60 if delays else float("nan"))


TYPE_NAME = {1: "bias", 2: "drift", 3: "transient", 4: "intermittent",
             5: "degradation"}


def by_type(pred):
    """Detection recall stratified by fault type (the class label)."""
    return {TYPE_NAME[k]: round(float(pred[yte == k].mean()), 3)
            for k in TYPE_NAME if (yte == k).sum()}


def by_severity(pred):
    return {int(l): round(float(pred[(lam == l) & (bt == 1)].mean()), 3)
            for l in range(1, 7) if ((lam == l) & (bt == 1)).sum()}


def boot_ci(pred, n=400, seed=0):
    """Bootstrap over EPISODES, not rows."""
    rng = np.random.default_rng(seed); ue = np.unique(ep); out = []
    for _ in range(n):
        s = rng.choice(ue, len(ue), replace=True)
        m = np.concatenate([np.where(ep == e)[0] for e in s])
        out.append(f1_score((yte[m] > 0).astype(int), (pred[m] > 0).astype(int),
                            zero_division=0))
    return [round(float(np.percentile(out, 2.5)), 3),
            round(float(np.percentile(out, 97.5)), 3)]


def report(name, pred, score=None):
    r = dict(f1=round(float(f1_score(bt, pred, zero_division=0)), 3),
             f1_ci=boot_ci(pred),
             f1_6=round(float(f1_score(yte, pred_multi, average="macro",
                                       zero_division=0)), 3) if pred_multi is not None else None,
             casc_recall=round(float(pred[casc > 0].mean()), 3) if (casc > 0).sum() else None,
             fa_day=round(fa_per_day(pred), 1),
             by_severity=by_severity(pred),
             by_type=by_type(pred))
    r["event_recall"], r["median_delay_min"] = [round(v, 3) for v in event_metrics(pred)]
    if score is not None:
        for tgt in (10, 50):
            rec, cr, fa = at_matched_far(score, tgt)
            r[f"recall@{tgt}fa"] = round(rec, 3)
            r[f"casc_recall@{tgt}fa"] = round(cr, 3)
    print(f"  {name:22s} F1 {r['f1']:.3f} CI{r['f1_ci']}  FA/day {r['fa_day']:>6.1f}"
          f"  evRec {r['event_recall']:.2f}  delay {r['median_delay_min']:.1f}min"
          + (f"  R@10fa {r.get('recall@10fa')}" if score is not None else ""))
    return r

res, pred_multi = {}, None
raw_tr, raw_te = Xtr[:, :NCHA], Xte[:, :NCHA]

if WHICH in ("base", "all"):
    nomtr = raw_tr[ytr == 0]
    lo, hi = nomtr.min(0), nomtr.max(0); pad = 0.02 * (hi - lo)
    viol = np.maximum(raw_te - (hi + pad), (lo - pad) - raw_te).max(1)
    res["Static threshold"] = report("Static threshold",
                                     (viol > 0).astype(int), viol)
    iso = IsolationForest(n_estimators=60, random_state=3, contamination=0.15)
    iso.fit(Xtr[ytr == 0][:8000])
    res["Isolation Forest"] = report("Isolation Forest",
                                     (iso.predict(Xte) == -1).astype(int),
                                     -iso.score_samples(Xte))
if WHICH in ("rf", "all"):
    rf = RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=3,
                                min_samples_leaf=3).fit(Xtr, ytr)
    pred_multi = rf.predict(Xte)
    res["Random Forest"] = report("Random Forest", (pred_multi > 0).astype(int),
                                  1 - rf.predict_proba(Xte)[:, 0])
    pred_multi = None
if WHICH in ("hgb", "all"):
    h = HistGradientBoostingClassifier(max_iter=60, random_state=3).fit(Xtr, ytr)
    pred_multi = h.predict(Xte)
    res["Hist. grad. boosting"] = report("Hist. grad. boosting",
                                         (pred_multi > 0).astype(int),
                                         1 - h.predict_proba(Xte)[:, 0])
    pred_multi = None
if WHICH in ("mlp", "all"):
    m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=40, random_state=3,
                      early_stopping=True).fit(Xtr, ytr)
    pred_multi = m.predict(Xte)
    res["MLP"] = report("MLP", (pred_multi > 0).astype(int),
                        1 - m.predict_proba(Xte)[:, 0])
    pred_multi = None

try:
    allr = json.load(open("results_final.json"))
except Exception:
    allr = {}
allr.setdefault(f"{TAG}_{CFG}", {}).update(res)
json.dump(allr, open("results_final.json", "w"), indent=1)
