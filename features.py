"""Feature engineering (Sec. IV-C) and episode-wise splits."""
import numpy as np
from stgfip.batch import CHANNELS

W = 60                     # rolling window, s
COUPLED_PAIRS = [("aocs_i_wheel", "eps_i_load"), ("eps_i_load", "thm_t_elec"),
                 ("thm_t_batt", "eps_v_batt"), ("thm_t_tank", "prp_p_tank"),
                 ("aocs_point_err", "com_snr"), ("eps_p_sa", "eps_soc")]


def _roll_mean(x, w):
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    out = np.empty_like(x)
    out[:w] = c[1:w + 1] / np.arange(1, w + 1)[:, None]
    out[w:] = (c[w + 1:] - c[1:-w]) / w
    return out


def build(F, use_rho=True, use_theta=True, use_stats=True):
    """F: dict channel -> (T, N). Returns (X of shape (T, N, D), names).

    Channel raw values always come first, one per channel, so the static
    threshold baseline can recover them by slicing.
    """
    raw, extra, names = [], [], []
    for c in CHANNELS:
        raw.append(F[c].astype(np.float32)); names.append(c)
    if use_stats:
        for c in CHANNELS:
            x = F[c].astype(np.float32)
            mu = _roll_mean(x, W)
            sd = np.sqrt(np.maximum(_roll_mean(x ** 2, W) - mu ** 2, 0))
            d1 = np.diff(x, axis=0, prepend=x[:1])
            d2 = np.diff(d1, axis=0, prepend=d1[:1])
            extra += [mu, sd, d1, d2]
            names += [f"{c}_mu", f"{c}_sd", f"{c}_d1", f"{c}_d2"]
    if use_theta:
        for c in CHANNELS:
            x = F[c].astype(np.float32)
            hp = x - _roll_mean(x, W)              # high-pass residual
            extra.append(_roll_mean(hp ** 2, W))   # band energy
            names.append(f"{c}_spec")
    if use_rho:
        for a, b in COUPLED_PAIRS:
            xa, xb = F[a].astype(np.float32), F[b].astype(np.float32)
            ma, mb = _roll_mean(xa, W), _roll_mean(xb, W)
            va = np.sqrt(np.maximum(_roll_mean(xa ** 2, W) - ma ** 2, 1e-12))
            vb = np.sqrt(np.maximum(_roll_mean(xb ** 2, W) - mb ** 2, 1e-12))
            extra.append((_roll_mean(xa * xb, W) - ma * mb) / (va * vb))
            names.append(f"rho_{a}_{b}")
    X = np.stack(raw + extra, axis=-1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def episode_split(n_ep, seed=3):
    """Split BY EPISODE, never by row, to avoid sliding-window leakage."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_ep)
    n_tr = int(0.60 * n_ep)
    n_va = int(0.15 * n_ep)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
