"""Label-side statistics the reviewer asked for: kappa sensitivity, censoring,
effective episode counts per split."""
import pickle, json
import numpy as np
from stgfip.batch import CHANNELS, CH_SUB
from features import episode_split

D = "/tmp/stgfip_data_main"
NOISE = dict(zip(CHANNELS, [0.045,0.030,0.0022,1.10,0.06,0.06,0.06,3.2,0.012,
                            0.020,0.11,0.13,0.20,0.11,0.040,0.060,0.90,0.22,
                            0.28,0.035,0.32,0.060]))
# params with an outgoing edge in C (mdot_scale has none -> cannot cascade)
CASCADE_CAPABLE = {"b_visc", "r_int0", "sa_loss", "k_rad"}

raw = np.load(f"{D}/raw.npz")
eps = pickle.load(open(f"{D}/episodes.pkl", "rb"))
t = raw["t"]; N = 96; T = len(t)
F = {c: raw[c][:, :N].astype(float) for c in CHANNELS}
Nm = {c: raw[c][:, N:].astype(float) for c in CHANNELS}

out = {}

# ---------------- kappa sensitivity ----------------
print("kappa | casc.episodes | cross-hits | median cross latency (min)")
ks = {}
for kappa in (2.0, 3.0, 5.0):
    ce, lat = set(), []
    for j, e in enumerate(eps):
        if e["kind"] == "nominal":
            continue
        after = t > e["t_f"]
        for c in CHANNELS:
            r = np.abs(F[c][:, j] - Nm[c][:, j])
            hit = after & (r > kappa * NOISE[c])
            if hit.sum() < 5:
                continue
            if CH_SUB[c] != e["host"]:
                ce.add(j); lat.append(t[int(np.argmax(hit))] - e["t_f"])
    med = float(np.median(lat))/60 if lat else float("nan")
    ks[kappa] = dict(episodes=len(ce), hits=len(lat), median_min=round(med,1))
    print(f" {kappa:>4.1f} | {len(ce):>13d} | {len(lat):>10d} | {med:>10.1f}")
out["kappa_sensitivity"] = ks

# ---------------- censoring ----------------
t2 = [j for j, e in enumerate(eps) if e["kind"] == "t2"]
cap = [j for j in t2 if eps[j]["par"] in CASCADE_CAPABLE]
casc = ks[3.0]["episodes"]
# min-5-sample rule
dropped = 0
for j, e in enumerate(eps):
    if e["kind"] == "nominal":
        continue
    after = t > e["t_f"]
    for c in CHANNELS:
        r = np.abs(F[c][:, j] - Nm[c][:, j])
        h = after & (r > 3.0 * NOISE[c])
        if 0 < h.sum() < 5:
            dropped += 1
obs_window = [float(t[-1] - eps[j]["t_f"]) for j in cap]
out["censoring"] = dict(
    tier2_episodes=len(t2), cascade_capable=len(cap), cascaded=casc,
    right_censored=len(cap) - casc,
    censoring_rate=round((len(cap) - casc) / max(len(cap), 1), 3),
    dropped_by_min5_rule=dropped,
    median_observation_window_min=round(float(np.median(obs_window)) / 60, 1),
    episode_length_min=round(float(t[-1]) / 60, 1))
print("\ncensoring:", json.dumps(out["censoring"], indent=1))

# ---------------- effective episode counts per split ----------------
lab = np.load(f"{D}/labelled.npz")
casc_mask = lab["casc"]
tr, va, te = episode_split(N, seed=3)
sp = {}
for name, idx in (("train", tr), ("val", va), ("test", te)):
    ep_with = [int(j) for j in idx if casc_mask[:, j].sum() > 0]
    origins = sorted({eps[j]["host"] for j in ep_with})
    sp[name] = dict(episodes=len(idx), cascaded_episodes=len(ep_with),
                    cascaded_samples=int(casc_mask[:, idx].sum()),
                    origin_subsystems=origins)
out["splits"] = sp
print("\nper-split:", json.dumps(sp, indent=1))
json.dump(out, open("results_labels.json", "w"), indent=1)
