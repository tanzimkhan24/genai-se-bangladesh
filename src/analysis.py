"""Statistical analyses for the Bangladesh GenAI workforce study."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def descriptive_by_firm(df: pd.DataFrame) -> pd.DataFrame:
    """Mean productivity, salary, and adoption rate by firm type."""
    g = df.groupby("firm_type").agg(
        n=("genai_adopted", "size"),
        adoption_rate=("genai_adopted", "mean"),
        mean_productivity=("productivity_index", "mean"),
        median_salary_bdt=("monthly_salary_bdt", "median"),
        mean_experience=("experience_years", "mean"),
    ).reset_index()
    return g


def descriptive_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Adoption and salary statistics stratified by region."""
    g = df.groupby("region").agg(
        n=("genai_adopted", "size"),
        adoption_rate=("genai_adopted", "mean"),
        median_salary_bdt=("monthly_salary_bdt", "median"),
        mean_productivity=("productivity_index", "mean"),
    ).reset_index()
    return g


def descriptive_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Adoption and productivity by university tier."""
    g = df.groupby("university_tier").agg(
        n=("genai_adopted", "size"),
        adoption_rate=("genai_adopted", "mean"),
        mean_productivity=("productivity_index", "mean"),
        mean_salary_bdt=("monthly_salary_bdt", "mean"),
    ).reset_index()
    return g


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1)
                       + (len(b) - 1) * b.var(ddof=1))
                      / (len(a) + len(b) - 2))
    if pooled == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def adoption_treatment_effect(df: pd.DataFrame,
                               outcome: str = "productivity_index") -> dict:
    """Welch's t-test and Cohen's d for adoption effect on a continuous outcome."""
    treated = df.loc[df["genai_adopted"] == 1, outcome].to_numpy()
    control = df.loc[df["genai_adopted"] == 0, outcome].to_numpy()
    t, p = stats.ttest_ind(treated, control, equal_var=False)
    d = cohens_d(treated, control)
    return {
        "outcome": outcome,
        "n_treated": int(len(treated)),
        "n_control": int(len(control)),
        "mean_treated": float(treated.mean()),
        "mean_control": float(control.mean()),
        "t_statistic": float(t),
        "p_value": float(p),
        "cohens_d": float(d),
    }


def productivity_regression(df: pd.DataFrame) -> pd.DataFrame:
    """OLS regression of productivity on adoption with controls."""
    y = df["productivity_index"].to_numpy()
    X = pd.get_dummies(
        df[["firm_type", "university_tier", "english_cefr"]],
        drop_first=True, dtype=float)
    X["experience_years"] = df["experience_years"].to_numpy()
    X["genai_adopted"] = df["genai_adopted"].to_numpy()
    X = sm.add_constant(X)
    model = sm.OLS(y, X.astype(float)).fit()
    out = pd.DataFrame({
        "term": model.params.index,
        "estimate": model.params.values,
        "std_error": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    return out


def salary_log_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Log-salary regression to estimate adoption salary premium."""
    y = np.log(df["monthly_salary_bdt"].to_numpy())
    X = pd.get_dummies(
        df[["firm_type", "region", "university_tier"]],
        drop_first=True, dtype=float)
    X["experience_years"] = df["experience_years"].to_numpy()
    X["genai_adopted"] = df["genai_adopted"].to_numpy()
    X = sm.add_constant(X)
    model = sm.OLS(y, X.astype(float)).fit()
    out = pd.DataFrame({
        "term": model.params.index,
        "estimate": model.params.values,
        "std_error": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    return out


def english_moderator_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Heterogeneous treatment effect by English proficiency stratum."""
    rows = []
    for cefr in sorted(df["english_cefr"].unique()):
        sub = df[df["english_cefr"] == cefr]
        if sub["genai_adopted"].nunique() < 2:
            continue
        res = adoption_treatment_effect(sub, "productivity_index")
        res["english_cefr"] = cefr
        rows.append(res)
    return pd.DataFrame(rows)


def skills_gap_index(df: pd.DataFrame) -> pd.DataFrame:
    """Skills gap index by university tier.

    Defined as the standardised distance between a tier's mean productivity
    and the asymptotic competence ceiling (set at the 95th percentile of the
    overall productivity distribution).
    """
    ceiling = float(np.percentile(df["productivity_index"], 95))
    overall_sd = float(df["productivity_index"].std(ddof=1))
    rows = []
    for tier, sub in df.groupby("university_tier"):
        gap = (ceiling - sub["productivity_index"].mean()) / overall_sd
        rows.append({
            "university_tier": tier,
            "mean_productivity": float(sub["productivity_index"].mean()),
            "ceiling": ceiling,
            "skills_gap_index": float(gap),
            "n": int(len(sub)),
        })
    return pd.DataFrame(rows)


def workforce_projection(adoption_rate_2030: float,
                          base_workforce_2026: int = 1_500_000,
                          annual_growth_rate: float = 0.08,
                          horizon_years: int = 4) -> pd.DataFrame:
    """Projection of total Bangladeshi SE workforce and adopters to 2030.

    The base workforce figure is anchored to BASIS / a2i estimates for 2025-26.
    """
    rows = []
    workforce = float(base_workforce_2026)
    rate = adoption_rate_2030 / horizon_years
    for k in range(horizon_years + 1):
        adoption = min(0.05 + rate * k, adoption_rate_2030)
        rows.append({
            "year": 2026 + k,
            "workforce": int(workforce),
            "adoption_rate": float(adoption),
            "adopters": int(workforce * adoption),
        })
        workforce = workforce * (1.0 + annual_growth_rate)
    return pd.DataFrame(rows)
