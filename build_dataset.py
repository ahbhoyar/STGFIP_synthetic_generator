"""Apply Tier-1 measurement faults, add noise, label via Eq. (19), build features."""
import sys, pickle
import numpy as np
from stgfip.batch import CHANNELS, CH_SUB

TAG = sys.argv[1] if len(sys.argv) > 1 else "main"
D = f"/tmp/stgfip_data_{TAG}"
SEED = 20260726
KAPPA = 3.0
AMP = np.array([0.06, 0.14, 0.28, 0.50, 0.80, 1.35])
NOISE = dict(zip(CHANNELS, [0.045, 0.03, 0.0022, 1.1, 0.06, 0.06, 0.06, 3.2, 0.012, 0.02, 0.11, 0.13, 0.2, 0.11, 0.04, 0.06, 0.9, 0.22, 0.28, 0.035, 0.32, 0.06]))
# class ids: 0 nominal, 1 bias(step), 2 drift, 3 transient(soft),
#            4 intermittent, 5 degradation(tier2), 6 protocol
FT2CLS = {"step": 1, "drift": 2, "soft": 3, "intermittent": 4}


def fval(e, t):
    if e["kind"] == "nominal":
        return np.zeros_like(t)
    a = AMP[e["lam"] - 1]
    dt = t - e["t_f"]
    on = dt >= 0
    ft, d = e["ftype"], e["dur"]
    if ft == "step":
        return a * on
    if ft == "drift":
        return a * np.minimum(np.maximum(dt, 0) / d, 1.6) * on
    if ft == "soft":
        return a * np.exp(-(dt ** 2) / (2 * e["sigma"] ** 2)) * on
    if ft == "intermittent":
        return a * on * (dt <= d) * (np.sin(2*np.pi*np.maximum(dt, 0)/180.) > .35)
    return np.zeros_like(t)


def main():
    raw = np.load(f"{D}/raw.npz")
    eps = pickle.load(open(f"{D}/episodes.pkl", "rb"))
    t = raw["t"]; T, N2 = raw[CHANNELS[0]].shape; N = N2 // 2
    rng = np.random.default_rng(SEED + 7)

    F = {c: raw[c][:, :N].astype(np.float64) for c in CHANNELS}   # faulted
    Nm = {c: raw[c][:, N:].astype(np.float64) for c in CHANNELS}  # paired nominal

    # Tier-1 faults act on the measurement only (Sec. IV-B)
    for j, e in enumerate(eps):
        if e["kind"] != "t1":
            continue
        c = e["ch"]
        span = Nm[c][:, j].std() or 1.0
        F[c][:, j] += span * 6.0 * fval(e, t)

    # identical noise realisation on both members of the pair
    for c in CHANNELS:
        z = rng.normal(0, NOISE[c], (T, N))
        F[c] += z; Nm[c] += z

    # ---- Eq. (19): 3-sigma first crossing against the paired nominal -----
    y = np.zeros((T, N), np.int8)
    tier = np.zeros((T, N), np.int8)        # injected tier of the active fault
    casc = np.zeros((T, N), np.int8)        # 1 once it has left its subsystem
    lam = np.zeros((T, N), np.int8)
    origin = np.zeros((T, N), np.int8)
    SUBS = ["none", "EPS", "AOCS", "THM", "PRP", "OBC", "COM", "NAV", "PLD"]
    lat_records = []
    for j, e in enumerate(eps):
        if e["kind"] == "nominal":
            continue
        host = e["host"]
        cls = FT2CLS[e["ftype"]] if e["kind"] == "t1" else 5
        inj_tier = 1 if e["kind"] == "t1" else 2
        after = t > e["t_f"]
        any_hit = np.zeros(T, bool)         # affected anywhere
        out_hit = np.zeros(T, bool)         # affected outside the host
        for c in CHANNELS:
            r = np.abs(F[c][:, j] - Nm[c][:, j])
            # sigma_c^0 is the nominal RESIDUAL std (measurement noise), not
            # the std of the signal, which contains the orbital swing.
            hit = after & (r > KAPPA * max(NOISE[c], 1e-9))
            if hit.sum() < 5:
                continue
            i0 = int(np.argmax(hit))
            cross = CH_SUB[c] != host
            any_hit |= hit
            if cross:
                out_hit |= hit
            lat_records.append(dict(ep=j, ch=c, host=host, cross=bool(cross),
                                    tier=inj_tier, lam=e["lam"],
                                    latency=float(t[i0] - e["t_f"])))
        y[any_hit, j] = cls
        tier[any_hit, j] = inj_tier
        lam[any_hit, j] = e["lam"]
        origin[any_hit, j] = SUBS.index(host)
        casc[out_hit, j] = 1

    np.savez(f"{D}/labelled.npz", t=t,
             **{f"F_{c}": F[c].astype(np.float32) for c in CHANNELS},
             y=y, tier=tier, lam=lam, origin=origin, casc=casc)
    pickle.dump(lat_records, open(f"{D}/latency.pkl", "wb"))
    print("episodes", N, "steps", T)
    print("sample class counts", np.bincount(y.ravel(), minlength=7))
    print("tier counts        ", np.bincount(tier.ravel(), minlength=4))
    print("cascaded samples   ", int(casc.sum()),
          "(%.1f%% of faulty)" % (100*casc.sum()/max((y > 0).sum(), 1)))
    cr = np.array([r["latency"] for r in lat_records if r["cross"]])
    ho = np.array([r["latency"] for r in lat_records if not r["cross"]])
    print("cross-subsystem channel hits %d  median latency %.1f s"
          % (len(cr), np.median(cr) if len(cr) else -1))
    print("in-host channel hits        %d  median latency %.1f s"
          % (len(ho), np.median(ho) if len(ho) else -1))
    ne = len(set(r["ep"] for r in lat_records if r["cross"]))
    print("episodes that cascaded: %d / %d" % (ne, sum(1 for e in eps if e["kind"] != "nominal")))


if __name__ == "__main__":
    main()
