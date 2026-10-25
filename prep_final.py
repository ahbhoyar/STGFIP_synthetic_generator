"""Cache episode-split feature matrices (+ metadata needed for event metrics)."""
import sys, pickle, numpy as np
from stgfip.batch import CHANNELS
from features import build, episode_split

tag = sys.argv[1]; cfg = sys.argv[2] if len(sys.argv) > 2 else "full"
KW = {"full": {}, "raw": dict(use_rho=False, use_theta=False, use_stats=False),
      "norho": dict(use_rho=False)}[cfg]
D = f"/tmp/stgfip_data_{tag}"
lab = np.load(f"{D}/labelled.npz"); eps = pickle.load(open(f"{D}/episodes.pkl","rb"))
X, names = build({c: lab["F_"+c] for c in CHANNELS}, **KW)
SUB = 12
t = lab["t"][::SUB]
y, tier, lam, casc = (lab[k][::SUB] for k in ("y","tier","lam","casc"))
X = X[::SUB]
tr, va, te = episode_split(X.shape[1], seed=3)
rng = np.random.default_rng(3); MAXTR = 20000

def pack(idx, cap=None):
    Xf = X[:, idx, :].reshape(-1, X.shape[-1])
    d = dict(X=Xf, y=y[:, idx].ravel(), tier=tier[:, idx].ravel(),
             lam=lam[:, idx].ravel(), casc=casc[:, idx].ravel(),
             ep=np.repeat(idx[None, :], X.shape[0], 0).ravel(),
             t=np.repeat(t[:, None], len(idx), 1).ravel())
    if cap and len(Xf) > cap:
        s = rng.choice(len(Xf), cap, replace=False)
        d = {k: v[s] for k, v in d.items()}
    return d

TR, TE = pack(tr, MAXTR), pack(te)
mu, sg = TR["X"].mean(0), TR["X"].std(0) + 1e-9
TR["X"] = ((TR["X"]-mu)/sg).astype(np.float32)
TE["X"] = ((TE["X"]-mu)/sg).astype(np.float32)
np.savez(f"/tmp/ff_{tag}_{cfg}.npz",
         **{f"tr_{k}": v for k, v in TR.items()},
         **{f"te_{k}": v for k, v in TE.items()},
         tf=np.array([eps[j]["t_f"] if eps[j]["kind"]!="nominal" else -1
                      for j in range(X.shape[1])]), ncha=len(CHANNELS))
print(f"{tag}/{cfg}: train {TR['X'].shape}  test {TE['X'].shape}  "
      f"classes present {sorted(set(np.r_[TR['y'],TE['y']].tolist()))}")
