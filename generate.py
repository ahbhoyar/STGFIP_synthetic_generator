"""Layer-2 driver: paired-trajectory generation and labelling (Eq. 19)."""
import numpy as np, pandas as pd
from .models import P, T_ORBIT
from .sim import derivs, State, Theta, CHANNELS
from .faults import Fault, sample_fault, apply_tier2, TIER2_PARAMS

H, FS = 0.5, 1.0        # integration step (s), telemetry rate (Hz)
KAPPA = 3.0             # sigma multiplier in Eq. (19)
NOISE = {c: v for c, v in zip(CHANNELS, [
    0.012, 0.010, 0.0008, 0.35,          # eps
    0.02, 0.02, 0.02, 0.9, 0.004, 0.006, # aocs
    0.030, 0.035, 0.055, 0.030,          # thermal
    0.012, 0.020,                        # prop
    0.30, 0.060,                         # obc
    0.090, 0.010,                        # com
    0.100, 0.020])}                      # nav, pld


def duty_cycle(t):
    ph = np.mod(t / T_ORBIT, 1.0)
    return {"tx": 1.0 if 0.30 < ph < 0.40 else 0.0,
            "thrust": 1.0 if 0.80 < ph < 0.83 else 0.0}


def _rk4(s, th, dc):
    x0 = s.vec(); t0 = s.t
    k1, _ = derivs(s, th, dc)
    for k, frac in ((k1, 0.5),):
        pass
    s.load(x0 + 0.5 * H * k1); s.t = t0 + 0.5 * H
    k2, _ = derivs(s, th, dc)
    s.load(x0 + 0.5 * H * k2)
    k3, _ = derivs(s, th, dc)
    s.load(x0 + H * k3); s.t = t0 + H
    k4, _ = derivs(s, th, dc)
    s.load(x0 + H / 6.0 * (k1 + 2*k2 + 2*k3 + k4)); s.t = t0 + H
    s.soc = float(np.clip(s.soc, 0.0, 1.0))
    return s


def run(duration_s, faults=None, seed=0, coupling=True):
    """Integrate faulted and paired-nominal trajectories under one seed."""
    faults = faults or []
    rng = np.random.default_rng(seed)
    sf, sn = State(), State()
    th0 = Theta()
    rows_f, rows_n, t_out = [], [], []
    n = int(duration_s / H)
    stride = int(1.0 / (FS * H))
    for i in range(n):
        t = i * H
        dc = duty_cycle(t)
        thf = th0.copy(); apply_tier2(thf, faults, t)
        thn = th0.copy()
        if not coupling:
            # A1 ablation: sever the state-dependent coupling channels
            thf.k_rad = thn.k_rad = 1.0
        _, of = derivs(sf, thf, dc)
        _, on = derivs(sn, thn, dc)
        if i % stride == 0:
            rows_f.append(of); rows_n.append(on); t_out.append(t)
        _rk4(sf, thf, dc); _rk4(sn, thn, dc)

    df = pd.DataFrame(rows_f); dn = pd.DataFrame(rows_n)
    df.insert(0, "t", t_out); dn.insert(0, "t", t_out)

    # Tier-1 faults act on the measurement only
    for f in faults:
        if f.tier != 1:
            continue
        span = df[f.target].std() or 1.0
        df[f.target] = df[f.target] + span * 6.0 * np.array(
            [f.value(t) for t in df.t])

    # measurement noise, common seed for the paired pair
    for c in CHANNELS:
        e = rng.normal(0, NOISE[c], len(df))
        df[c] = df[c] + e; dn[c] = dn[c] + e
    return df, dn


def label(df, dn, faults):
    """Eq. (19): 3-sigma first crossing against the paired nominal run."""
    N = len(df)
    y = np.zeros(N, int); tier = np.zeros(N, int)
    lam = np.zeros(N, int); origin = np.array(["none"] * N, dtype=object)
    lat = np.full(N, np.nan)
    if not faults:
        return dict(y=y, tier=tier, lam=lam, origin=origin, latency=lat)
    resid = {c: (df[c].values - dn[c].values) for c in CHANNELS}
    sig = {c: max(dn[c].values.std(), 1e-9) for c in CHANNELS}
    for f in faults:
        host = f.subsystem
        for c in CHANNELS:
            cross = np.where((df.t.values > f.t_f) &
                             (np.abs(resid[c]) > KAPPA * sig[c]))[0]
            if not len(cross):
                continue
            i0 = cross[0]
            aff = np.zeros(N, bool); aff[i0:] = np.abs(resid[c][i0:]) > KAPPA*sig[c]
            if not aff.any():
                continue
            csub = c.split("_")[0].upper()
            csub = {"EPS":"EPS","AOCS":"AOCS","THM":"THM","PRP":"PRP",
                    "OBC":"OBC","COM":"COM","NAV":"NAV","PLD":"PLD"}.get(csub, csub)
            # Tier 3 = the fault shows up outside the subsystem it started in
            t_here = 3 if csub != host else f.tier
            upd = aff & (t_here >= tier)
            y[upd] = f.klass(); tier[upd] = t_here
            lam[upd] = f.lam; origin[upd] = host
            lat[upd] = df.t.values[i0] - f.t_f
    return dict(y=y, tier=tier, lam=lam, origin=origin, latency=lat)
