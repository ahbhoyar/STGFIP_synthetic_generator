"""Multi-tier fault model, Sec. IV-B. Severity lambda in 1..6."""
import numpy as np

TIER1_CHANNELS = ["eps_v_batt", "aocs_wheel_rpm", "thm_t_batt", "prp_p_tank",
                  "com_snr", "nav_pos_err", "obc_t_cpu", "pld_data_rate"]
TIER2_PARAMS = {"b_visc": ("AOCS", 9.0), "r_int0": ("EPS", 6.0),
                "kappa_d": ("EPS", 14.0), "mdot_scale": ("PRP", 0.55),
                "k_rad": ("THM", -0.42)}
CLASS_NAMES = ["nominal", "sensor_bias", "sensor_drift", "transient",
               "intermittent", "degradation", "protocol"]


class Fault:
    def __init__(self, subsystem, target, ftype, t_f, lam, tier, rng,
                 duration=None):
        self.subsystem, self.target, self.ftype = subsystem, target, ftype
        self.t_f, self.lam, self.tier = t_f, lam, tier
        self.sigma = duration if duration else rng.uniform(60, 300)
        self.duration = duration if duration else rng.uniform(600, 2400)
        self.rng = rng

    def amp(self):
        """Amplitude monotone in severity; lambda=1 sub-threshold, 6 hard."""
        return np.array([0.06, 0.14, 0.28, 0.50, 0.80, 1.35])[self.lam - 1]

    def value(self, t):
        """F_phi(t) of Eq. (14), normalised (dimensionless)."""
        if t < self.t_f:
            return 0.0
        a = self.amp()
        if self.ftype == "step":
            return a
        if self.ftype == "drift":
            return a * (t - self.t_f) / self.duration
        if self.ftype == "soft":
            return a * np.exp(-((t - self.t_f) ** 2) / (2 * self.sigma ** 2))
        if self.ftype == "intermittent":
            if t > self.t_f + self.duration:
                return 0.0
            on = (np.sin(2 * np.pi * (t - self.t_f) / 180.0) > 0.35)
            return a * on * (0.7 + 0.3 * np.sin(t))
        return 0.0

    def klass(self):
        return {"step": 1, "drift": 2, "soft": 3, "intermittent": 4}[self.ftype] \
            if self.tier == 1 else 5


def sample_fault(rng, t_hi):
    """Uniform over tier, subsystem/target, type and severity."""
    tier = rng.choice([1, 2], p=[0.62, 0.38])
    lam = int(rng.integers(1, 7))
    t_f = float(rng.uniform(0.12 * t_hi, 0.72 * t_hi))
    if tier == 1:
        ch = str(rng.choice(TIER1_CHANNELS))
        ftype = str(rng.choice(["step", "drift", "soft", "intermittent"]))
        return Fault(ch.split("_")[0].upper(), ch, ftype, t_f, lam, 1, rng)
    par = str(rng.choice(list(TIER2_PARAMS)))
    sub = TIER2_PARAMS[par][0]
    return Fault(sub, par, "drift", t_f, lam, 2, rng)


def apply_tier2(theta, faults, t):
    """theta_s(t) = theta_s^0 (1 + F_phi(t)), Sec. IV-B."""
    for f in faults:
        if f.tier != 2:
            continue
        scale = TIER2_PARAMS[f.target][1]
        setattr(theta, f.target,
                getattr(theta, f.target) * (1.0 + scale * f.value(t)))
