"""Generate the STGFIP reference corpus with paired-trajectory labelling.

Chunked so it fits sandbox time limits; state is checkpointed to disk.
Usage: python3 gen_corpus.py <tag> <coupling 0|1> <chunk_steps>
"""
import os, sys, pickle
import numpy as np
from stgfip.batch import init_state, init_theta, rk4, CHANNELS, CH_SUB
from stgfip.models import T_ORBIT

TAG = sys.argv[1] if len(sys.argv) > 1 else "main"
COUPLING = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

N_EP, DUR, H = 96, int(T_ORBIT), 1.0
SEED = 20260726
OUT = f"/tmp/stgfip_data_{TAG}"
os.makedirs(OUT, exist_ok=True)
CKPT = f"{OUT}/ckpt.pkl"

TIER1_CH = ["eps_v_batt", "aocs_wheel_rpm", "thm_t_batt", "prp_p_tank",
            "com_snr", "nav_pos_err", "obc_t_cpu", "pld_data_rate"]
TIER2_PAR = {"b_visc": ("AOCS", 8.0), "r_int0": ("EPS", 8.0),
             "sa_loss": ("EPS", 0.50), "mdot_scale": ("PRP", -0.70),
             "k_rad": ("THM", -0.45)}
AMP = np.array([0.06, 0.14, 0.28, 0.50, 0.80, 1.35])
FTYPES = ["step", "drift", "soft", "intermittent"]
NOISE = dict(zip(CHANNELS, [0.045, 0.03, 0.0022, 1.1, 0.06, 0.06, 0.06, 3.2, 0.012, 0.02, 0.11, 0.13, 0.2, 0.11, 0.04, 0.06, 0.9, 0.22, 0.28, 0.035, 0.32, 0.06]))


def build_episodes():
    rng = np.random.default_rng(SEED)
    eps = []
    for i in range(N_EP):
        if i < 16:                                  # nominal-only episodes
            eps.append(dict(kind="nominal")); continue
        tier = 1 if rng.random() < 0.50 else 2
        lam = int(rng.integers(1, 7))
        t_f = float(rng.uniform(0.15, 0.65) * DUR)
        if tier == 1:
            ch = str(rng.choice(TIER1_CH))
            ft = str(rng.choice(FTYPES))
            eps.append(dict(kind="t1", ch=ch, ftype=ft, t_f=t_f, lam=lam,
                            host=CH_SUB[ch],
                            dur=float(rng.uniform(600, 2400)),
                            sigma=float(rng.uniform(60, 300))))
        else:
            par = str(rng.choice(list(TIER2_PAR)))
            eps.append(dict(kind="t2", par=par, t_f=t_f, lam=lam,
                            host=TIER2_PAR[par][0], ftype="drift",
                            dur=float(rng.uniform(900, 3000)), sigma=200.0))
    return eps


def fval(e, t):
    if e["kind"] == "nominal" or t < e["t_f"]:
        return 0.0
    a = AMP[e["lam"] - 1]
    ft, d = e["ftype"], e["dur"]
    if ft == "step":
        return a
    if ft == "drift":
        return a * min((t - e["t_f"]) / d, 1.6)
    if ft == "soft":
        return a * np.exp(-((t - e["t_f"]) ** 2) / (2 * e["sigma"] ** 2))
    if ft == "intermittent":
        if t > e["t_f"] + d:
            return 0.0
        return a if np.sin(2 * np.pi * (t - e["t_f"]) / 180.0) > 0.35 else 0.0
    return 0.0


def main():
    eps = build_episodes()
    if os.path.exists(CKPT):
        with open(CKPT, "rb") as f:
            st = pickle.load(f)
    else:
        rng = np.random.default_rng(SEED + 1)
        x1 = init_state(N_EP, rng)             # [0:N] faulted, [N:2N] nominal
        # The paired nominal run MUST start from identical initial conditions,
        # else Eq. (19) measures IC divergence rather than fault propagation.
        x = {k: np.concatenate([v, v.copy()], axis=0) for k, v in x1.items()}
        st = dict(i=0, x=x, buf={c: [] for c in CHANNELS}, t=[])
    x, i0 = st["x"], st["i"]
    th0 = init_theta(2 * N_EP)
    nsteps = min(CHUNK, DUR - i0)
    for k in range(nsteps):
        t = (i0 + k) * H
        th = {p: v.copy() for p, v in th0.items()}
        for j, e in enumerate(eps):             # Tier-2 acts on theta
            if e["kind"] == "t2":
                sc = TIER2_PAR[e["par"]][1]
                # sa_loss has a zero baseline, so a multiplicative perturbation
                # would be identically zero; it is applied additively instead.
                if e["par"] == "sa_loss":
                    th[e["par"]][j] += sc * fval(e, t)
                else:
                    th[e["par"]][j] *= (1.0 + sc * fval(e, t))
        x, obs = rk4(x, th, t, H, coupling=COUPLING)
        for c in CHANNELS:
            st["buf"][c].append(obs[c].astype(np.float32))
        st["t"].append(t)
    st["i"] = i0 + nsteps
    st["x"] = x
    if st["i"] >= DUR:
        arr = {c: np.stack(st["buf"][c]) for c in CHANNELS}   # (T, 2N)
        np.savez(f"{OUT}/raw.npz", t=np.array(st["t"]),
                            **{c: arr[c] for c in CHANNELS})
        with open(f"{OUT}/episodes.pkl", "wb") as f:
            pickle.dump(eps, f)
        if os.path.exists(CKPT):
            os.remove(CKPT)
        print(f"DONE {st['i']}/{DUR}  saved {OUT}/raw.npz")
    else:
        with open(CKPT, "wb") as f:
            pickle.dump(st, f)
        print(f"progress {st['i']}/{DUR}")


if __name__ == "__main__":
    main()
