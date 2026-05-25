"""
Calibrated simulation of the Bangladesh software engineering workforce.

Population parameters are anchored to publicly reported figures from BASIS
(Bangladesh Association of Software and Information Services), the Bangladesh
Hi-Tech Park Authority, the BBS Labour Force Survey, and a2i, all referenced
in paper/refs.bib. This module is for reproducible quantitative analysis;
it is NOT a primary survey instrument.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from . import SEED


# Population strata calibrated to BASIS workforce reports and BBS labour data.
REGION_PROBS = {
    "Dhaka": 0.60,
    "Chittagong": 0.15,
    "Sylhet": 0.10,
    "Khulna": 0.08,
    "Rajshahi": 0.07,
}

FIRM_TYPE_PROBS = {
    "Local SME": 0.40,
    "Outsourcing": 0.35,
    "MNC subsidiary": 0.15,
    "Freelancer": 0.10,
}

UNIVERSITY_TIER_PROBS = {
    "T1 (BUET, DU, IUT)": 0.25,
    "T2 (KUET, RUET, SUST)": 0.25,
    "T3 (Private)": 0.40,
    "Self-taught": 0.10,
}

EDUCATION_PROBS = {
    "Diploma": 0.20,
    "BSc": 0.60,
    "MSc": 0.15,
    "PhD": 0.05,
}

# CEFR English proficiency: anchored to EF EPI Bangladesh 2024 (moderate band).
ENGLISH_CEFR_PROBS = {
    "A2": 0.15,
    "B1": 0.35,
    "B2": 0.35,
    "C1": 0.15,
}

# Numeric mapping for English proficiency.
CEFR_VALUE = {"A2": 1.0, "B1": 2.0, "B2": 3.0, "C1": 4.0}


@dataclass
class SimulationParameters:
    n_developers: int = 1500
    seed: int = SEED
    region_probs: dict = field(default_factory=lambda: dict(REGION_PROBS))
    firm_probs: dict = field(default_factory=lambda: dict(FIRM_TYPE_PROBS))
    tier_probs: dict = field(default_factory=lambda: dict(UNIVERSITY_TIER_PROBS))
    edu_probs: dict = field(default_factory=lambda: dict(EDUCATION_PROBS))
    english_probs: dict = field(default_factory=lambda: dict(ENGLISH_CEFR_PROBS))
    # Salary base (BDT, monthly), calibrated to BASIS 2024 wage band reports.
    salary_base_bdt: float = 35000.0
    # Multiplicative effect of GenAI adoption on productivity.
    adoption_productivity_lift: float = 0.18
    # Heterogeneity by English proficiency on adoption uplift.
    english_uplift_per_cefr_step: float = 0.05


def _draw_categorical(rng: np.random.Generator, probs: dict, n: int) -> np.ndarray:
    keys = list(probs.keys())
    p = np.array(list(probs.values()), dtype=float)
    p = p / p.sum()
    return rng.choice(keys, size=n, p=p)


def simulate_workforce(params: SimulationParameters | None = None) -> pd.DataFrame:
    """Generate a calibrated synthetic workforce sample.

    Returns a tidy data frame, one row per simulated developer.
    """
    if params is None:
        params = SimulationParameters()
    rng = np.random.default_rng(params.seed)
    n = params.n_developers

    region = _draw_categorical(rng, params.region_probs, n)
    firm = _draw_categorical(rng, params.firm_probs, n)
    tier = _draw_categorical(rng, params.tier_probs, n)
    education = _draw_categorical(rng, params.edu_probs, n)
    english = _draw_categorical(rng, params.english_probs, n)
    english_value = np.array([CEFR_VALUE[e] for e in english])

    # Years of experience (gamma distribution, shape 2, scale 2.5; ~max 15).
    experience = np.clip(rng.gamma(shape=2.0, scale=2.5, size=n), 0.0, 15.0)

    # Adoption probability is moderated by tier, English, firm type and time.
    tier_lift = np.where(np.isin(tier, ["T1 (BUET, DU, IUT)"]), 0.18, 0.0)
    tier_lift += np.where(np.isin(tier, ["T2 (KUET, RUET, SUST)"]), 0.10, 0.0)
    firm_lift = np.where(firm == "MNC subsidiary", 0.20, 0.0)
    firm_lift += np.where(firm == "Outsourcing", 0.10, 0.0)
    eng_lift = (english_value - 2.0) * 0.07  # B1 = baseline
    base = 0.40
    p_adopt = np.clip(base + tier_lift + firm_lift + eng_lift
                      + 0.01 * experience, 0.05, 0.95)
    adopted = rng.uniform(size=n) < p_adopt

    # Productivity index (control mean = 100, SD ~ 15).
    productivity = 100.0 + 6.0 * np.log1p(experience)
    productivity += np.where(np.isin(tier,
                                     ["T1 (BUET, DU, IUT)",
                                      "T2 (KUET, RUET, SUST)"]),
                             5.0, 0.0)
    productivity += rng.normal(0.0, 12.0, size=n)
    # GenAI uplift, conditional on adoption, moderated by English proficiency.
    uplift = (params.adoption_productivity_lift
              + (english_value - 2.0) * params.english_uplift_per_cefr_step)
    productivity = np.where(adopted, productivity * (1.0 + uplift), productivity)

    # Monthly salary (BDT), lognormal with regional adjustment.
    region_factor = np.where(region == "Dhaka", 1.30,
                    np.where(region == "Chittagong", 1.10,
                    np.where(region == "Sylhet", 1.05, 0.90)))
    log_salary = (np.log(params.salary_base_bdt)
                  + 0.06 * experience
                  + 0.20 * (productivity / 100.0 - 1.0)
                  + np.log(region_factor)
                  + rng.normal(0.0, 0.25, size=n))
    salary_bdt = np.exp(log_salary)

    # Weekly self-directed learning hours (with adoption boost on documentation).
    learning_hours = np.clip(rng.normal(4.0, 1.5, size=n), 0.0, 20.0)
    learning_hours += np.where(adopted, 1.2, 0.0)

    df = pd.DataFrame({
        "region": region,
        "firm_type": firm,
        "university_tier": tier,
        "education": education,
        "english_cefr": english,
        "english_value": english_value,
        "experience_years": experience,
        "genai_adopted": adopted.astype(int),
        "productivity_index": productivity,
        "monthly_salary_bdt": salary_bdt,
        "weekly_learning_hours": learning_hours,
    })
    return df


def simulate_adoption_timeseries(start_year: int = 2022,
                                  end_year: int = 2026,
                                  months_per_year: int = 12,
                                  seed: int = SEED) -> pd.DataFrame:
    """Generate monthly adoption rates by firm type using a Bass diffusion process.

    Returns long-format frame with columns: month_index, year, firm_type, adoption_rate.
    """
    rng = np.random.default_rng(seed)
    months = np.arange(0, (end_year - start_year + 1) * months_per_year, 1)
    rows = []
    # Bass parameters per firm type: (p, q, m).
    bass = {
        "MNC subsidiary":   (0.020, 0.55, 0.95),
        "Outsourcing":      (0.012, 0.45, 0.88),
        "Local SME":        (0.006, 0.35, 0.70),
        "Freelancer":       (0.025, 0.50, 0.92),
    }
    for firm, (p, q, m) in bass.items():
        # Discrete Bass: F(t+1) = F(t) + (p + q*F(t)/m) * (m - F(t))
        F = 0.0
        for t in months:
            inc = (p + q * F / m) * (m - F)
            F = F + inc
            noise = rng.normal(0.0, 0.005)
            obs = float(np.clip(F + noise, 0.0, 1.0))
            year = start_year + (t // months_per_year)
            rows.append({"month_index": int(t), "year": int(year),
                          "firm_type": firm, "adoption_rate": obs})
    return pd.DataFrame(rows)
