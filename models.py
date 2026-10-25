"""Coupled spacecraft subsystem models (STGFIP Layer 1).

Implements the equations of Sec. IV-A of the paper. All parameters are for a
representative 150 kg LEO smallsat in a 550 km sun-synchronous orbit.
"""
import math
import numpy as np

MU_E, R_E = 3.986004418e14, 6378.137e3
SIGMA_SB, G0 = 5.670374419e-8, 9.80665


class Params:
    # --- orbit / environment ---
    alt = 550e3
    solar_const = 1361.0            # W/m^2
    # --- EPS (Eq. 5-7) ---
    sa_area, sa_eff = 0.32, 0.29    # m^2, - (sized for ~7% DoD per orbit)
    kappa_d = 0.03                  # array degradation over life
    t_life = 5 * 365 * 86400.0
    q_cap = 15.0 * 3600.0           # A.s  (15 Ah)
    v_oc0, v_oc_span = 22.0, 6.0    # V
    r_int0 = 0.085                  # ohm at 293 K
    p_base = 28.0                   # W housekeeping
    # --- AOCS (Eq. 8-10) ---
    J = np.diag([8.2, 9.1, 6.4])
    b_visc0 = 4.0e-6                # N.m.s viscous friction per wheel
    tau_coul = 1.2e-4               # N.m Coulomb friction
    kt = 0.032                      # N.m/A motor constant
    tau_dist = 6.0e-6               # N.m disturbance amplitude
    # --- Thermal (Eq. 11) : nodes = [battery, elec box, radiator, tank] ---
    C_th = np.array([9.0e3, 5.5e3, 3.0e3, 4.2e3])       # J/K
    K_th = np.array([[0, 1.4, 0.6, 0.2],
                     [1.4, 0, 1.1, 0.3],
                     [0.6, 1.1, 0, 0.4],
                     [0.2, 0.3, 0.4, 0]])                # W/K
    emis = np.array([0.10, 0.12, 0.86, 0.10])
    area_rad = np.array([0.10, 0.14, 0.42, 0.12])        # m^2
    q_sol = np.array([2.0, 3.0, 34.0, 2.5])              # W in sunlight
    zeta = np.array([0.35, 0.45, 0.05, 0.15])            # dissipation split
    t_sink = 2.7                                          # K deep space
    # --- Propulsion (Eq. 12-13) ---
    p_tank0, m_prop0 = 22.0e5, 4.5                        # Pa, kg
    gamma_bd, ve = 1.28, 2150.0
    mdot_nom = 1.40e-4
    # --- Comms (Eq. 15) ---
    p_tx_dbm, g_rx, nf_db, bw = 33.0, 42.0, 2.5, 2.0e6
    g_tx_peak, hpbw = 8.0, 32.0                           # dBi, deg
    # --- Payload ---
    img_rate, bpp, compression = 1.4e6, 12.0, 3.2


P = Params()
T_ORBIT = 2 * np.pi * np.sqrt((R_E + P.alt) ** 3 / MU_E)   # ~5735 s


_r = R_E + P.alt
_ECL_HALF = math.acos(min(1.0, math.sqrt(max(_r**2 - R_E**2, 0.0))
                          / (_r * math.cos(math.radians(22.0))))) / (2*math.pi)


def eclipse(t):
    """Cylindrical shadow indicator, Eq. (2). 1 = sunlit. Scalar fast path."""
    if np.isscalar(t):
        return 1.0 if abs((t / T_ORBIT) % 1.0 - 0.5) > _ECL_HALF else 0.0
    phase = np.mod(np.asarray(t) / T_ORBIT, 1.0)
    return (np.abs(phase - 0.5) > _ECL_HALF).astype(float)


def sun_incidence(t):
    """Angle between array normal and Sun line, driven by AOCS pointing."""
    return (0.35 * math.sin(2*math.pi*t/T_ORBIT)
            + 0.12 * math.sin(6*math.pi*t/T_ORBIT))


def v_oc(soc):
    return P.v_oc0 + P.v_oc_span * (0.0 if soc < 0.0 else 1.0 if soc > 1.0 else soc)


def r_internal(T_batt, r_int0):
    """Arrhenius-type rise of internal resistance at low temperature."""
    T = 220.0 if T_batt < 220.0 else 340.0 if T_batt > 340.0 else T_batt
    return r_int0 * math.exp(1400.0 * (1.0 / T - 1.0 / 293.15))
