"""Orchestrate the full Bangladesh GenAI workforce analysis pipeline.

Running ``python -m src.run_pipeline`` regenerates every CSV table and every
figure consumed by ``paper/main.tex`` from a fixed random seed.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from . import SEED
from .data_simulation import (SimulationParameters, simulate_workforce,
                               simulate_adoption_timeseries)
from .diffusion_model import (BassParameters, fit_bass, policy_scenarios)
from .analysis import (descriptive_by_firm, descriptive_by_region,
                        descriptive_by_tier, adoption_treatment_effect,
                        productivity_regression, salary_log_regression,
                        english_moderator_analysis, skills_gap_index,
                        workforce_projection)
from . import visualization

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)

    # Simulate the workforce.
    params = SimulationParameters()
    df = simulate_workforce(params)
    df.to_csv(DATA_DIR / "workforce.csv", index=False)

    # Adoption time series.
    ts = simulate_adoption_timeseries()
    ts.to_csv(DATA_DIR / "adoption_timeseries.csv", index=False)

    # Fit a Bass curve to the pooled (sample-weighted) trajectory.
    overall = ts.groupby("month_index")["adoption_rate"].mean()
    bass = fit_bass(overall.index.to_numpy(dtype=float), overall.to_numpy())
    pd.DataFrame([{"p": bass.p, "q": bass.q, "m": bass.m}]) \
        .to_csv(TAB_DIR / "bass_parameters.csv", index=False)
    print(f"Bass parameters: p={bass.p:.4f} q={bass.q:.4f} m={bass.m:.4f}")

    # Policy scenarios.
    scen = policy_scenarios(bass, horizon_months=60)
    scen.to_csv(TAB_DIR / "policy_scenarios.csv", index=False)

    # Descriptive tables.
    desc_firm = descriptive_by_firm(df)
    desc_region = descriptive_by_region(df)
    desc_tier = descriptive_by_tier(df)
    desc_firm.to_csv(TAB_DIR / "descriptives_firm.csv", index=False)
    desc_region.to_csv(TAB_DIR / "descriptives_region.csv", index=False)
    desc_tier.to_csv(TAB_DIR / "descriptives_tier.csv", index=False)

    # Treatment effect.
    te_prod = adoption_treatment_effect(df, "productivity_index")
    te_sal = adoption_treatment_effect(df, "monthly_salary_bdt")
    pd.DataFrame([te_prod, te_sal]).to_csv(
        TAB_DIR / "treatment_effects.csv", index=False)
    print("Treatment effect on productivity:", te_prod)

    # Regressions.
    productivity_regression(df).to_csv(
        TAB_DIR / "productivity_regression.csv", index=False)
    salary_log_regression(df).to_csv(
        TAB_DIR / "salary_regression.csv", index=False)

    # English moderator.
    eng_mod = english_moderator_analysis(df)
    eng_mod.to_csv(TAB_DIR / "english_moderator.csv", index=False)

    # Skills gap.
    gap = skills_gap_index(df)
    gap.to_csv(TAB_DIR / "skills_gap.csv", index=False)

    # Projection.
    proj = workforce_projection(adoption_rate_2030=0.82)
    proj.to_csv(TAB_DIR / "workforce_projection.csv", index=False)

    # Figures.
    visualization.figure_adoption_curve(ts, FIG_DIR / "fig_adoption_curve")
    visualization.figure_regional_heatmap(df, FIG_DIR / "fig_regional_heatmap")
    visualization.figure_productivity_box(df, FIG_DIR / "fig_productivity_box")
    visualization.figure_english_moderator(eng_mod,
                                            FIG_DIR / "fig_english_moderator")
    visualization.figure_firm_forest(df, FIG_DIR / "fig_firm_forest")
    visualization.figure_salary_trajectory(df, FIG_DIR / "fig_salary_trajectory")
    visualization.figure_skills_gap(gap, FIG_DIR / "fig_skills_gap")
    visualization.figure_education_alignment(df,
                                              FIG_DIR / "fig_education_alignment")
    visualization.figure_policy_scenarios(scen, FIG_DIR / "fig_policy_scenarios")
    visualization.figure_workforce_projection(proj,
                                               FIG_DIR / "fig_workforce_projection")
    visualization.figure_comparator_economies(FIG_DIR / "fig_comparator")
    visualization.figure_learning_hours(df, FIG_DIR / "fig_learning_hours")

    print(f"All artefacts written to: {ROOT}")


if __name__ == "__main__":
    main()
