"""Batched coupled simulator: advances N episodes on a shared time grid.

Mathematically identical to sim._core (Sec. IV-A of the paper); vectorised so
that a corpus of episodes integrates in one pass.
"""
import math
import numpy as np
from .models import P, T_ORBIT, SIGMA_SB, eclipse

JINV = np.linalg.inv(P.J)
KROW = P.K_th.sum(axis=1)
CHANNELS = [
    "eps_v_batt", "eps_i_load", "eps_soc", "eps_p_sa",
    "aocs_wx", "aocs_wy", "aocs_wz", "aocs_wheel_rpm", "aocs_i_wheel",
    "aocs_point_err",
    "thm_t_batt", "thm_t_elec", "thm_t_rad", "thm_t_tank",
    "prp_p_tank", "prp_thrust",
    "obc_cpu_load", "obc_t_cpu",
    "com_snr", "com_ber",
    "nav_pos_err", "pld_data_rate",
]
CH_SUB = {c: c.split("_")[0].upper() for c in CHANNELS}
PARAMS = ["b_visc", "r_int0", "sa_loss", "mdot_scale", "k_rad"]


def init_state(N, rng):
    return dict(
        soc=np.full(N, 0.82) + rng.normal(0, .01, N),
        w=np.tile([1.1e-4, -0.8e-4, 0.6e-4], (N, 1)) + rng.normal(0, 4e-6, (N, 3)),
        h_w=np.tile([0.021, -0.014, 0.008], (N, 1)) + rng.normal(0, 3e-4, (N, 3)),
        T=np.tile([291., 297., 268., 289.], (N, 1)) + rng.normal(0, .7, (N, 4)),
        m_prop=np.full(N, P.m_prop0), cpu=np.full(N, 0.34))


def init_theta(N):
    return dict(b_visc=np.full(N, P.b_visc0), r_int0=np.full(N, P.r_int0),
                sa_loss=np.zeros(N), mdot_scale=np.ones(N),
                k_rad=np.ones(N))


def duty(t):
    ph = (t / T_ORBIT) % 1.0
    return (1.0 if 0.30 < ph < 0.40 else 0.0,
            1.0 if 0.80 < ph < 0.88 else 0.0)


def core(x, th, t, coupling=True):
    """Batched RHS of Eq. (1). Returns (dx dict, obs dict)."""
    beta = eclipse(t)
    d_tx, d_th = duty(t)
    soc, w, h_w, T, m_prop, cpu = (x["soc"], x["w"], x["h_w"], x["T"],
                                   x["m_prop"], x["cpu"])
    T_batt, T_elec, T_rad, T_tank = T[:, 0], T[:, 1], T[:, 2], T[:, 3]
    _p = 2 * math.pi * t / T_ORBIT

    # --- AOCS, Eq. (8)-(10) ---
    tau_d = P.tau_dist * np.array([math.sin(_p), math.cos(_p),
                                   0.5 * math.sin(2 * _p)])
    tau_cmd = -2.2e-3 * w - 1.0e-4 * h_w        # attitude control torque
    omega_w = h_w * (1.0 / 2.6e-5)
    # The wheel is speed-regulated: the motor supplies the commanded torque
    # PLUS the friction torque, so growth of b_visc raises motor current at
    # constant wheel speed instead of spinning the wheel down.
    tau_fric = (th["b_visc"][:, None] * omega_w
                + P.tau_coul * np.tanh(omega_w * 0.2))
    dh_w = tau_cmd
    Jw = w @ P.J.T + h_w
    cr = np.cross(w, Jw)
    dw = (-cr - dh_w + tau_d) @ JINV.T
    i_wheel = np.abs(tau_cmd + tau_fric).sum(1) / P.kt
    point_err = np.degrees(np.linalg.norm(w, axis=1) * 180.0)

    # --- EPS, Eq. (5)-(7) ---
    inc = 0.35 * math.sin(_p) + 0.12 * math.sin(3 * _p)
    p_sa = (beta * P.sa_eff * P.solar_const * P.sa_area * max(0.0, math.cos(inc))
            * np.clip(1.0 - th["sa_loss"], 0.0, 1.0))
    # A1 ablation: with C = 0 no cross-subsystem term may enter the bus load.
    if coupling:
        p_load = P.p_base + 11.0 * i_wheel + 26.0 * cpu + d_tx * 18.0
    else:
        p_load = P.p_base + 11.0 * 0.207 + 26.0 * 0.34 + d_tx * 18.0 \
            + np.zeros_like(soc)
    Tc = np.clip(T_batt if coupling else np.full_like(T_batt, 293.15), 220, 340)
    r_i = th["r_int0"] * np.exp(1400.0 * (1.0 / Tc - 1.0 / 293.15))
    v_b = P.v_oc0 + P.v_oc_span * np.clip(soc, 0, 1)
    i_b = (p_load - p_sa) / np.maximum(v_b, 1.0)
    i_b = np.where((soc >= 1.0) & (i_b < 0), 0.0, i_b)
    v_batt = v_b - i_b * r_i
    dsoc = -i_b / P.q_cap

    # --- Thermal, Eq. (11) ---
    p_diss = p_load * 0.62 + i_b ** 2 * r_i
    q_int = (p_diss[:, None] * P.zeta) if coupling else \
            (np.full_like(p_diss, 45.0)[:, None] * P.zeta)
    cond = T @ P.K_th.T - T * KROW
    rad = (th["k_rad"][:, None] * P.emis * SIGMA_SB * P.area_rad
           * (P.t_sink ** 4 - T ** 4))
    dT = (cond + rad + q_int + beta * P.q_sol) / P.C_th

    # --- Propulsion, Eq. (12)-(13) ---
    dm = -d_th * P.mdot_nom * th["mdot_scale"]
    Tt = T_tank if coupling else np.full_like(T_tank, 293.15)
    p_tank = (P.p_tank0 * (np.maximum(m_prop, 1e-3) / P.m_prop0) ** P.gamma_bd
              * Tt / 293.15)
    thrust = np.abs(dm) * P.ve

    # --- OBC / COM / NAV / PLD ---
    dcpu = (0.34 + 0.22 * d_tx + 0.18 * d_th - cpu) / 40.0
    # THM -> OBC runs through the OBSERVATION map, not the dynamics; it
    # must be severed under C = 0 as well (this was the A1 residue).
    T_amb = T_elec if coupling else np.full_like(T_elec, 297.0)
    t_cpu = T_amb + 14.0 * cpu
    pe = point_err if coupling else np.zeros_like(point_err)
    g_tx = P.g_tx_peak - 12.0 * (pe / P.hpbw) ** 2
    slant = (P.alt + 1.1e6 * (1 - beta)) / 1e3
    l_fs = 20*math.log10(max(slant, 300.)) + 20*math.log10(2.2e3) + 32.45
    snr = (P.p_tx_dbm + g_tx + P.g_rx - l_fs - 1.8
           - (-174 + P.nf_db + 10 * math.log10(P.bw)))
    ebn0 = np.clip(10.0 ** (snr / 10.0), 1e-3, 700.0)
    ber = np.log10(np.maximum(0.5 * np.exp(-ebn0 / 2.0), 1e-12))
    pos_err = (3.2 + 1.4 * math.sin(_p) + 8.0 * (1 - beta) * 0.3
               + 0.9 * pe + np.zeros_like(soc))
    data_rate = beta * P.img_rate * P.bpp / P.compression / 1e6 * np.ones_like(soc)

    dx = dict(soc=dsoc, w=dw, h_w=dh_w, T=dT, m_prop=dm, cpu=dcpu)
    obs = dict(eps_v_batt=v_batt, eps_i_load=p_load / np.maximum(v_batt, 1.0),
               eps_soc=soc, eps_p_sa=p_sa,
               aocs_wx=w[:, 0]*1e4, aocs_wy=w[:, 1]*1e4, aocs_wz=w[:, 2]*1e4,
               aocs_wheel_rpm=omega_w.mean(1)*60/(2*math.pi),
               aocs_i_wheel=i_wheel, aocs_point_err=point_err,
               thm_t_batt=T_batt-273.15, thm_t_elec=T_elec-273.15,
               thm_t_rad=T_rad-273.15, thm_t_tank=T_tank-273.15,
               prp_p_tank=p_tank/1e5, prp_thrust=thrust*1e3,
               obc_cpu_load=cpu*100, obc_t_cpu=t_cpu-273.15,
               com_snr=snr, com_ber=ber,
               nav_pos_err=pos_err, pld_data_rate=data_rate)
    return dx, obs


def _add(x, dx, h):
    return {k: x[k] + h * dx[k] for k in x}


def rk4(x, th, t, h, coupling=True):
    k1, obs = core(x, th, t, coupling)
    k2, _ = core(_add(x, k1, h/2), th, t + h/2, coupling)
    k3, _ = core(_add(x, k2, h/2), th, t + h/2, coupling)
    k4, _ = core(_add(x, k3, h), th, t + h, coupling)
    nx = {k: x[k] + h/6.0*(k1[k] + 2*k2[k] + 2*k3[k] + k4[k]) for k in x}
    nx["soc"] = np.clip(nx["soc"], 0.0, 1.0)
    return nx, obs
