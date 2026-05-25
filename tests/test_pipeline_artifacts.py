"""Smoke test for the full pipeline."""
from pathlib import Path

import pandas as pd

from src.run_pipeline import main, FIG_DIR, TAB_DIR


def test_pipeline_generates_expected_artifacts():
    main()
    expected_figs = [
        "fig_adoption_curve.pdf",
        "fig_regional_heatmap.pdf",
        "fig_productivity_box.pdf",
        "fig_english_moderator.pdf",
        "fig_firm_forest.pdf",
        "fig_salary_trajectory.pdf",
        "fig_skills_gap.pdf",
        "fig_education_alignment.pdf",
        "fig_policy_scenarios.pdf",
        "fig_workforce_projection.pdf",
        "fig_comparator.pdf",
        "fig_learning_hours.pdf",
    ]
    for name in expected_figs:
        assert (FIG_DIR / name).exists(), f"Missing figure {name}"

    expected_tabs = [
        "bass_parameters.csv",
        "policy_scenarios.csv",
        "descriptives_firm.csv",
        "descriptives_region.csv",
        "descriptives_tier.csv",
        "treatment_effects.csv",
        "productivity_regression.csv",
        "salary_regression.csv",
        "english_moderator.csv",
        "skills_gap.csv",
        "workforce_projection.csv",
    ]
    for name in expected_tabs:
        path = TAB_DIR / name
        assert path.exists(), f"Missing table {name}"
        assert pd.read_csv(path).shape[0] > 0
