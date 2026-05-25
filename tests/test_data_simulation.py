"""Tests for the workforce simulation and adoption time series."""
import pandas as pd
import pytest

from src.data_simulation import (SimulationParameters, simulate_workforce,
                                  simulate_adoption_timeseries,
                                  REGION_PROBS, FIRM_TYPE_PROBS, CEFR_VALUE)


class TestSimulateWorkforce:
    def test_returns_dataframe(self):
        df = simulate_workforce()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == SimulationParameters().n_developers

    def test_expected_columns(self):
        df = simulate_workforce()
        expected = {"region", "firm_type", "university_tier", "education",
                    "english_cefr", "english_value", "experience_years",
                    "genai_adopted", "productivity_index", "monthly_salary_bdt",
                    "weekly_learning_hours"}
        assert expected.issubset(df.columns)

    def test_adoption_is_binary(self):
        df = simulate_workforce()
        assert set(df["genai_adopted"].unique()).issubset({0, 1})

    def test_regions_match_calibration(self):
        df = simulate_workforce(SimulationParameters(n_developers=5000))
        rate = df["region"].value_counts(normalize=True)
        for region, p in REGION_PROBS.items():
            assert abs(rate.get(region, 0.0) - p) < 0.05

    def test_firm_types_match_calibration(self):
        df = simulate_workforce(SimulationParameters(n_developers=5000))
        rate = df["firm_type"].value_counts(normalize=True)
        for firm, p in FIRM_TYPE_PROBS.items():
            assert abs(rate.get(firm, 0.0) - p) < 0.05

    def test_seed_is_deterministic(self):
        df1 = simulate_workforce(SimulationParameters(seed=42))
        df2 = simulate_workforce(SimulationParameters(seed=42))
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = simulate_workforce(SimulationParameters(seed=1))
        df2 = simulate_workforce(SimulationParameters(seed=2))
        assert not df1["productivity_index"].equals(df2["productivity_index"])

    def test_salary_is_positive(self):
        df = simulate_workforce()
        assert (df["monthly_salary_bdt"] > 0).all()

    def test_experience_in_bounds(self):
        df = simulate_workforce()
        assert (df["experience_years"] >= 0).all()
        assert (df["experience_years"] <= 15).all()

    def test_adopters_have_higher_mean_productivity(self):
        df = simulate_workforce()
        treated = df.loc[df["genai_adopted"] == 1, "productivity_index"].mean()
        control = df.loc[df["genai_adopted"] == 0, "productivity_index"].mean()
        assert treated > control

    def test_cefr_value_mapping_is_monotone(self):
        assert CEFR_VALUE["A2"] < CEFR_VALUE["B1"] < CEFR_VALUE["B2"] < CEFR_VALUE["C1"]


class TestAdoptionTimeseries:
    def test_returns_long_format(self):
        ts = simulate_adoption_timeseries()
        assert {"month_index", "year", "firm_type", "adoption_rate"} \
            <= set(ts.columns)

    def test_monotone_increasing_per_firm(self):
        ts = simulate_adoption_timeseries(seed=999)
        for firm, sub in ts.groupby("firm_type"):
            adopt = sub.sort_values("month_index")["adoption_rate"].to_numpy()
            # Allow tiny dips from noise but global trend must be up.
            assert adopt[-1] > adopt[0] + 0.3

    def test_rates_bounded(self):
        ts = simulate_adoption_timeseries()
        assert (ts["adoption_rate"] >= 0.0).all()
        assert (ts["adoption_rate"] <= 1.0).all()

    def test_mnc_adopts_faster_than_local_sme(self):
        ts = simulate_adoption_timeseries(seed=1)
        last = ts.groupby("firm_type")["adoption_rate"].last()
        assert last["MNC subsidiary"] > last["Local SME"]
