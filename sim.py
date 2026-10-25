"""STGFIP coupled simulator (Layer 1) + fault injection (Layer 2)."""
import math
import numpy as np
from .models import (P, T_ORBIT, SIGMA_SB, eclipse, sun_incidence, v_oc,
                     r_internal)

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
SUBSYSTEMS = ["EPS", "AOCS", "THM", "PRP", "OBC", "COM", "NAV", "PLD"]
FAULT_TYPES = ["step", "drift", "soft", "intermittent"]

# Coupling edges actually realised in derivs(), for the paper's Fig. 1.
COUPLING_EDGES = [("AOCS", "EPS"), ("THM", "EPS"), ("EPS", "THM"),
                  ("OBC", "THM"), ("THM", "PRP"), ("AOCS", "COM"),
                  ("AOCS", "PLD")]


class State:
    """Joint state vector x of Eq. (1)."""
    __slots__ = ("soc", "w", "h_w", "q", "T", "m_prop", "cpu", "t")

    def __init__(self):
        self.soc = 0.82
        self.w = np.array([1.1e-4, -0.8e-4, 0.6e-4])
        self.h_w = np.array([0.021, -0.014, 0.008])
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.T = np.array([291.0, 297.0, 268.0, 289.0])
        self.m_prop = P.m_prop0
        self.cpu = 0.34
        self.t = 0.0

    def vec(self):
        return np.concatenate(([self.soc], self.w, self.h_w, self.T,
                               [self.m_prop], [self.cpu]))

    def load(self, v):
        self.soc = v[0]; self.w = v[1:4]; self.h_w = v[4:7]
        self.T = v[7:11]; self.m_prop = v[11]; self.cpu = v[12]

    def copy(self):
        s = State.__new__(State)
        for k in State.__slots__:
            val = getattr(self, k)
            setattr(s, k, val.copy() if isinstance(val, np.ndarray) else val)
        return s


class Theta:
    """Mutable parameter vector; Tier-2 faults perturb these."""
    def __init__(self):
        self.b_visc = P.b_visc0      # wheel viscous friction  (bearing wear)
        self.r_int0 = P.r_int0       # battery internal resistance
        self.kappa_d = P.kappa_d     # solar array degradation
        self.mdot_scale = 1.0        # thruster efficiency
        self.k_rad = 1.0             # radiator effectiveness

    def copy(self):
        t = Theta.__new__(Theta)
        t.__dict__.update(self.__dict__)
        return t


JINV = np.linalg.inv(P.J)


def _core(s, th, duty):
    """Shared physics for Eq. (1); returns (dx, locals needed for obs)."""
    t = s.t
    beta = eclipse(t)
    T_batt, T_elec, T_rad, T_tank = s.T

    # ---- AOCS (Eq. 8-10) ------------------------------------------------
    _p = 2*math.pi*t/T_ORBIT
    tau_d = P.tau_dist * np.array([math.sin(_p), math.cos(_p),
                                   0.5*math.sin(2*_p)])
    tau_c = -2.2e-3 * s.w - 1.0e-4 * s.h_w              # simple rate damping
    omega_w = s.h_w * (1.0/2.6e-5)                       # wheel rate, rad/s
    dh_w = tau_c - th.b_visc*omega_w - P.tau_coul*np.tanh(omega_w*0.2)
    Jw = P.J @ s.w + s.h_w
    w = s.w
    cr = np.array([w[1]*Jw[2]-w[2]*Jw[1], w[2]*Jw[0]-w[0]*Jw[2],
                   w[0]*Jw[1]-w[1]*Jw[0]])
    dw = JINV @ (-cr - dh_w + tau_d)
    # observable wheel current -> EPS  (edge AOCS->EPS)
    i_wheel = float(np.abs(tau_c + th.b_visc * omega_w).sum()) / P.kt
    point_err = math.degrees(float(np.linalg.norm(s.w)) * 180.0)

    # ---- EPS (Eq. 5-7) --------------------------------------------------
    inc = sun_incidence(t)
    p_sa = (beta * P.sa_eff * P.solar_const * P.sa_area
            * max(0.0, math.cos(inc)) * (1.0 - th.kappa_d * t / P.t_life))
    p_load = P.p_base + 11.0 * i_wheel + 26.0 * s.cpu + duty["tx"] * 18.0
    r_i = r_internal(T_batt, th.r_int0)                  # edge THM->EPS
    v_b = v_oc(s.soc)
    i_b = (p_load - p_sa) / max(v_b, 1.0)
    if s.soc >= 1.0 and i_b < 0:      # charge regulator: taper at full
        i_b = 0.0
    v_batt = v_b - i_b * r_i
    dsoc = -i_b / P.q_cap

    # ---- Thermal (Eq. 11) ----------------------------------------------
    p_diss = p_load * 0.62 + i_b ** 2 * r_i              # edge EPS,OBC->THM
    q_int = P.zeta * p_diss
    cond = P.K_th @ s.T - s.T * P.K_th.sum(axis=1)
    rad = th.k_rad * P.emis * SIGMA_SB * P.area_rad * (P.t_sink**4 - s.T**4)
    dT = (cond + rad + q_int + beta * P.q_sol) / P.C_th

    # ---- Propulsion (Eq. 12-13) ----------------------------------------
    dm = -duty["thrust"] * P.mdot_nom * th.mdot_scale
    p_tank = (P.p_tank0 * (max(s.m_prop, 1e-3) / P.m_prop0) ** P.gamma_bd
              * T_tank / 293.15)                          # edge THM->PRP
    thrust = abs(dm) * P.ve

    # ---- OBC / COM / NAV / PLD -----------------------------------------
    dcpu = (0.34 + 0.22*duty["tx"] + 0.18*duty["thrust"] - s.cpu) / 40.0
    t_cpu = T_elec + 14.0 * s.cpu
    g_tx = P.g_tx_peak - 12.0 * (point_err / P.hpbw) ** 2   # edge AOCS->COM
    slant = (P.alt + 1.1e6 * (1 - beta)) / 1e3
    l_fs = 20*math.log10(max(slant, 300.0)) + 20*math.log10(2.2e3) + 32.45
    snr = (P.p_tx_dbm + g_tx + P.g_rx - l_fs - 1.8
           - (-174 + P.nf_db + 10*math.log10(P.bw)))
    ebn0 = 10.0 ** (snr / 10.0)
    ber = 0.5 * math.exp(-max(min(ebn0, 700.0), 1e-3) / 2.0)
    pos_err = 3.2 + 1.4*math.sin(_p) + 8.0*(1-beta)*0.3
    data_rate = beta * P.img_rate * P.bpp / P.compression / 1e6

    dx = np.concatenate(([dsoc], dw, dh_w, dT, [dm], [dcpu]))
    return dx, (v_batt, p_load, p_sa, i_wheel, omega_w, point_err, t_cpu,
                snr, ber, pos_err, data_rate, p_tank, thrust, T_batt,
                T_elec, T_rad, T_tank, i_b)


def dx_only(s, th, duty):
    return _core(s, th, duty)[0]


def derivs(s: State, th: Theta, duty):
    dx, k = _core(s, th, duty)
    (v_batt, p_load, p_sa, i_wheel, omega_w, point_err, t_cpu, snr, ber,
     pos_err, data_rate, p_tank, thrust, T_batt, T_elec, T_rad, T_tank,
     i_b) = k
    obs = dict(eps_v_batt=v_batt, eps_i_load=p_load/max(v_batt,1.0),
               eps_soc=s.soc, eps_p_sa=p_sa,
               aocs_wx=s.w[0]*1e4, aocs_wy=s.w[1]*1e4, aocs_wz=s.w[2]*1e4,
               aocs_wheel_rpm=float(np.mean(omega_w)*60/(2*np.pi)),
               aocs_i_wheel=i_wheel, aocs_point_err=point_err,
               thm_t_batt=T_batt-273.15, thm_t_elec=T_elec-273.15,
               thm_t_rad=T_rad-273.15, thm_t_tank=T_tank-273.15,
               prp_p_tank=p_tank/1e5, prp_thrust=thrust*1e3,
               obc_cpu_load=s.cpu*100, obc_t_cpu=t_cpu-273.15,
               com_snr=snr, com_ber=math.log10(max(ber,1e-12)),
               nav_pos_err=pos_err, pld_data_rate=data_rate)
    return dx, obs
