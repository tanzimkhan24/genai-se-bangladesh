"""Tests for the analysis layer."""
import numpy as np
import pandas as pd
import pytest

from src.data_simulation import simulate_workforce, SimulationParameters
from src.analysis import (descriptive_by_firm, descriptive_by_region,
                           descriptive_by_tier, cohens_d,
                           adoption_treatment_effect, productivity_regression,
                           salary_log_regression, english_moderator_analysis,
                           skills_gap_index, workforce_projection)


@pytest.fixture(scope="module")
def df():
    return simulate_workforce(SimulationParameters(seed=42))


class TestDescriptives:
    def test_by_firm_has_all_firms(self, df):
        out = descriptive_by_firm(df)
        assert len(out) == df["firm_type"].nunique()
        assert "adoption_rate" in out.columns

    def test_by_region_has_all_regions(self, df):
        out = descriptive_by_region(df)
        assert len(out) == df["region"].nunique()

    def test_by_tier_has_all_tiers(self, df):
        out = descriptive_by_tier(df)
        assert len(out) == df["university_tier"].nunique()


class TestCohensD:
    def test_zero_for_identical_distributions(self):
        a = np.array([1, 2, 3, 4, 5], dtype=float)
        assert abs(cohens_d(a, a)) < 1e-9

    def test_positive_when_a_greater(self):
        a = np.array([10, 11, 12, 13, 14], dtype=float)
        b = np.array([1, 2, 3, 4, 5], dtype=float)
        assert cohens_d(a, b) > 1.0


class TestTreatmentEffect:
    def test_positive_on_productivity(self, df):
        res = adoption_treatment_effect(df, "productivity_index")
        assert res["cohens_d"] > 0.0
        assert res["p_value"] < 0.05

    def test_returns_expected_keys(self, df):
        res = adoption_treatment_effect(df, "productivity_index")
        for k in ("outcome", "n_treated", "n_control", "cohens_d", "p_value"):
            assert k in res


class TestRegressions:
    def test_productivity_regression_has_genai_term(self, df):
        out = productivity_regression(df)
        assert "genai_adopted" in out["term"].tolist()

    def test_productivity_regression_genai_effect_positive(self, df):
        out = productivity_regression(df).set_index("term")
        assert out.loc["genai_adopted", "estimate"] > 0.0

    def test_salary_log_regression_runs(self, df):
        out = salary_log_regression(df)
        assert "genai_adopted" in out["term"].tolist()


class TestEnglishModerator:
    def test_runs_and_returns_rows(self, df):
        out = english_moderator_analysis(df)
        assert len(out) >= 3
        assert "cohens_d" in out.columns


class TestSkillsGap:
    def test_all_tiers_present(self, df):
        out = skills_gap_index(df)
        assert len(out) == df["university_tier"].nunique()

    def test_self_taught_gap_is_largest(self, df):
        out = skills_gap_index(df).set_index("university_tier")
        assert out.loc["Self-taught", "skills_gap_index"] > \
            out.loc["T1 (BUET, DU, IUT)", "skills_gap_index"]


class TestWorkforceProjection:
    def test_projection_rows(self):
        proj = workforce_projection(adoption_rate_2030=0.80)
        assert len(proj) == 5  # 2026 to 2030 inclusive
        assert proj["year"].tolist() == [2026, 2027, 2028, 2029, 2030]

    def test_workforce_grows(self):
        proj = workforce_projection(adoption_rate_2030=0.80)
        assert proj["workforce"].is_monotonic_increasing
