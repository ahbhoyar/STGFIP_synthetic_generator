"""Publication-quality figures, all from the released runs."""
import json, pickle
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stgfip.batch import CHANNELS, CH_SUB
from stgfip.models import T_ORBIT, eclipse
import os; os.makedirs("figs2", exist_ok=True)

plt.rcParams.update({
    "font.size": 6.5, "axes.linewidth": .8, "lines.linewidth": 1.2,
    "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": .02,
    "axes.grid": True, "grid.alpha": .22, "grid.linewidth": .5,
    "legend.frameon": False, "font.family": "DejaVu Sans",
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False})
C = dict(f="#b2182b", n="#2166ac", a="#1a9850", o="#e08214", g="#636363")

D = np.load("/tmp/stgfip_data_main/labelled.npz")
RAW = np.load("/tmp/stgfip_data_main/raw.npz")
EPS = pickle.load(open("/tmp/stgfip_data_main/episodes.pkl", "rb"))
LAT = pickle.load(open("/tmp/stgfip_data_main/latency.pkl", "rb"))
T = D["t"]; NEP = 96
NOISE = dict(zip(CHANNELS, [0.045,0.030,0.0022,1.10,0.06,0.06,0.06,3.2,0.012,
                            0.020,0.11,0.13,0.20,0.11,0.040,0.060,0.90,0.22,
                            0.28,0.035,0.32,0.060]))

# ---------------- Fig: paired-trajectory mechanism (the money figure) --------
def fig_cascade():
    cross = [r for r in LAT if r["cross"]]
    cand = [e for e in {r["ep"] for r in cross} if EPS[e].get("par") == "b_visc"]
    ep = max(cand or list({r["ep"] for r in cross}), key=lambda e: EPS[e]["lam"])
    e = EPS[ep]
    hits = sorted([r for r in cross if r["ep"] == ep], key=lambda r: r["latency"])
    host_ch, cross_ch = "aocs_i_wheel", hits[0]["ch"]
    m = T / 60
    fig, ax = plt.subplots(2, 2, figsize=(3.45, 2.75), sharex=True)
    for col, (ch, ttl) in enumerate([(host_ch, f"host subsystem — {e['host']}"),
                                     (cross_ch, f"downstream — {CH_SUB[cross_ch]}")]):
        fa = RAW[ch][:, ep].astype(float); no = RAW[ch][:, NEP + ep].astype(float)
        ax[0, col].plot(m, no, color=C["n"], lw=1.0, label="paired nominal $y^0$")
        ax[0, col].plot(m, fa, color=C["f"], lw=1.0, label="faulted $y^\\phi$")
        ax[0, col].set_title(f"{ttl}\n{ch}", fontsize=5.6)
        ax[0, col].legend(fontsize=4.4, loc="upper left", handlelength=1.0)
        r = np.abs(fa - no); sg = NOISE[ch]
        ax[1, col].semilogy(m, np.maximum(r, 1e-6), color=C["a"], lw=1.0)
        ax[1, col].axhline(3 * sg, color=C["f"], ls="--", lw=1.0)
        ax[1, col].text(2, 4.5 * sg, "$\\kappa\\sigma_c^0$", fontsize=5.5, color=C["f"])
        ax[1, col].axvline(e["t_f"] / 60, color=C["g"], ls="--", lw=1.0)
        ax[1, col].text(e["t_f"]/60 + 1.5, 2e-5, "$t_f$", fontsize=5.5, color=C["g"])
        idx = np.where((T > e["t_f"]) & (r > 3 * sg))[0]
        if len(idx):
            ax[1, col].axvline(T[idx[0]]/60, color=C["o"], ls=":", lw=1.4)
            ax[1, col].annotate(f"$\\Delta$ = {(T[idx[0]]-e['t_f'])/60:.0f} min",
                                xy=(T[idx[0]]/60, 4*sg), fontsize=5.3, color=C["o"],
                                xytext=(4, 16), textcoords="offset points",
                                arrowprops=dict(arrowstyle="->", lw=.7, color=C["o"]))
        ax[1, col].set_xlabel("time (min)", fontsize=6)
        ax[1, col].set_ylabel("$|y^{\\phi}-y^{0}|$", fontsize=6)
    nice = {"b_visc": "bearing wear ($b_i$)", "r_int0": "battery ageing ($R_{int}$)",
            "k_rad": "radiator loss ($k_{rad}$)", "sa_loss": "array loss",
            "mdot_scale": "thruster efficiency"}.get(e["par"], e["par"])
    fig.suptitle(f"Tier-2 {nice} fault, $\\lambda$={e['lam']}, injected in {e['host']}",
                 fontsize=6.2, y=1.02)
    fig.savefig("figs2/cascade.png"); plt.close(fig)

# ---------------- Fig: severity + latency (two panels) -----------------------
def fig_sev_lat():
    res = json.load(open("results_final.json"))["main_full"]
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.55))
    lams = list(range(1, 7))
    style = {"Static threshold": ("o-", C["g"]), "MLP": ("s-", C["f"]),
             "Hist. grad. boosting": ("^-", C["n"]), "Random Forest": ("v-", C["a"])}
    for k, (mk, col) in style.items():
        bs = res[k]["by_severity"]
        ax[0].plot(lams, [bs.get(str(l), np.nan) for l in lams], mk, color=col,
                   ms=3.5, label=k, lw=1.1)
    ax[0].set_xlabel("fault severity $\\lambda$"); ax[0].set_ylabel("detection recall")
    ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=4.6, loc="lower right", handlelength=1.2)
    ax[0].set_title("(a) recall vs severity", fontsize=6.5)
    ho = np.array([r["latency"] for r in LAT if not r["cross"]]) / 60
    cr = np.array([r["latency"] for r in LAT if r["cross"]]) / 60
    bins = np.linspace(0, 96, 25)
    ax[1].hist(ho, bins=bins, color=C["n"], alpha=.8, label=f"in-host (n={len(ho)})")
    ax[1].hist(cr, bins=bins, color=C["f"], alpha=.8, label=f"cross-subsystem (n={len(cr)})")
    ax[1].axvline(np.median(ho), color=C["n"], ls="--", lw=1.1)
    ax[1].axvline(np.median(cr), color=C["f"], ls="--", lw=1.1)
    ax[1].set_xlabel("propagation latency $\\Delta_{r,c}$ (min)")
    ax[1].set_ylabel("channel detections"); ax[1].legend(fontsize=4.6, handlelength=1.0)
    ax[1].set_title(f"(b) latency (min)", fontsize=6.5)
    fig.savefig("figs2/severity_latency.png"); plt.close(fig)

# ---------------- Fig: nominal orbit ----------------------------------------
def fig_orbit():
    j = next(i for i, e in enumerate(EPS) if e["kind"] == "nominal")
    m = T / 60; e = eclipse(T)
    fig, ax = plt.subplots(4, 1, figsize=(3.45, 3.6), sharex=True)
    d = np.diff(np.r_[1.0, e])
    for s in np.where(d < 0)[0]:
        f = np.where((np.arange(len(e)) > s) & (e > .5))[0]
        for a in ax:
            a.axvspan(m[s], m[f[0]] if len(f) else m[-1], color="#bdbdbd",
                      alpha=.3, lw=0)
    for a, (ch, lb, col) in zip(ax, [("eps_p_sa", "$P_{sa}$ (W)", C["o"]),
                                     ("eps_soc", "SoC (%)", C["a"]),
                                     ("thm_t_batt", "$T_{batt}$ (°C)", C["f"]),
                                     ("prp_p_tank", "$P_{tank}$ (bar)", C["n"])]):
        v = D["F_" + ch][:, j]
        a.plot(m, v * (100 if ch == "eps_soc" else 1), color=col)
        a.set_ylabel(lb, fontsize=7.5)
    ax[-1].set_xlabel("time (min)")
    ax[0].set_title("nominal orbit (shaded = eclipse)", fontsize=8)
    fig.align_ylabels(ax); fig.savefig("figs2/orbit.png"); plt.close(fig)

if __name__ == "__main__":
    fig_cascade(); print("cascade.png")
    fig_sev_lat(); print("severity_latency.png")
    fig_orbit();   print("orbit.png")
