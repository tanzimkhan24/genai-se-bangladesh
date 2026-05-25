"""
Bass diffusion model fitting and policy-scenario projection.

The Bass model (Bass 1969) describes the cumulative adoption F(t) of an
innovation as

    dF/dt = (p + (q/m) F)(m - F),     F(0) = 0,

where p is the coefficient of innovation, q is the coefficient of imitation,
and m is the asymptotic market potential. Discrete-time version is used here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class BassParameters:
    p: float
    q: float
    m: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.p, self.q, self.m


def bass_cumulative(p: float, q: float, m: float, t: np.ndarray) -> np.ndarray:
    """Closed-form cumulative Bass diffusion."""
    t = np.asarray(t, dtype=float)
    num = 1.0 - np.exp(-(p + q) * t)
    den = 1.0 + (q / p) * np.exp(-(p + q) * t)
    return m * num / den


def fit_bass(t: np.ndarray, y: np.ndarray) -> BassParameters:
    """Nonlinear least squares fit of the Bass model.

    Parameters
    ----------
    t : array of time points
    y : observed cumulative adoption fractions, in [0, 1].
    """
    def loss(theta):
        p, q, m = theta
        if p <= 0 or q <= 0 or m <= 0 or m > 1.5:
            return 1e10
        y_hat = bass_cumulative(p, q, m, t)
        return float(np.mean((y_hat - y) ** 2))

    x0 = np.array([0.01, 0.4, 0.9])
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 5000})
    p_hat, q_hat, m_hat = res.x
    return BassParameters(p=float(p_hat), q=float(q_hat), m=float(m_hat))


def project_adoption(params: BassParameters, t_future: np.ndarray) -> np.ndarray:
    """Project cumulative adoption over a future time horizon."""
    return bass_cumulative(params.p, params.q, params.m, t_future)


def policy_scenarios(base: BassParameters,
                      horizon_months: int = 60) -> pd.DataFrame:
    """Generate policy scenarios.

    Three scenarios:
      - Baseline (current trajectory).
      - Digital-literacy investment: raises p by 50% (national programmes).
      - Industry mentorship: raises q by 30% (peer effects accelerate).

    Returns long-format frame: month, scenario, adoption_rate.
    """
    months = np.arange(0, horizon_months + 1, 1, dtype=float)
    scenarios = {
        "Baseline":              BassParameters(base.p, base.q, base.m),
        "Digital literacy":      BassParameters(base.p * 1.5, base.q, base.m),
        "Industry mentorship":   BassParameters(base.p, base.q * 1.3, base.m),
        "Combined policy":       BassParameters(base.p * 1.5, base.q * 1.3,
                                                  min(base.m * 1.05, 0.99)),
    }
    rows = []
    for name, par in scenarios.items():
        F = bass_cumulative(par.p, par.q, par.m, months)
        for t, f in zip(months, F):
            rows.append({"month": int(t), "scenario": name,
                          "adoption_rate": float(f)})
    return pd.DataFrame(rows)
