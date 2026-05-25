"""Tests for the Bass diffusion fitting and policy scenarios."""
import numpy as np
import pytest

from src.diffusion_model import (BassParameters, bass_cumulative, fit_bass,
                                  project_adoption, policy_scenarios)


class TestBassCumulative:
    def test_starts_at_zero(self):
        f = bass_cumulative(0.01, 0.4, 0.9, np.array([0.0]))
        assert abs(f[0]) < 1e-6

    def test_asymptotes_to_m(self):
        f = bass_cumulative(0.02, 0.5, 0.9, np.array([500.0]))
        assert abs(f[0] - 0.9) < 1e-3

    def test_monotone_increasing(self):
        t = np.linspace(0, 60, 200)
        f = bass_cumulative(0.01, 0.4, 0.9, t)
        diffs = np.diff(f)
        assert (diffs >= -1e-9).all()


class TestFitBass:
    def test_recovers_known_parameters(self):
        t = np.arange(0, 60, 1, dtype=float)
        y = bass_cumulative(0.012, 0.42, 0.88, t)
        hat = fit_bass(t, y)
        assert abs(hat.p - 0.012) < 0.01
        assert abs(hat.q - 0.42) < 0.05
        assert abs(hat.m - 0.88) < 0.05

    def test_returns_bass_parameters(self):
        t = np.arange(0, 30, 1, dtype=float)
        y = bass_cumulative(0.01, 0.4, 0.9, t)
        hat = fit_bass(t, y)
        assert isinstance(hat, BassParameters)


class TestProjectAdoption:
    def test_future_above_present(self):
        params = BassParameters(p=0.01, q=0.4, m=0.9)
        present = project_adoption(params, np.array([30.0]))[0]
        future = project_adoption(params, np.array([120.0]))[0]
        assert future >= present


class TestPolicyScenarios:
    def test_returns_four_scenarios(self):
        base = BassParameters(p=0.01, q=0.4, m=0.9)
        scen = policy_scenarios(base, horizon_months=24)
        assert set(scen["scenario"].unique()) == {
            "Baseline", "Digital literacy",
            "Industry mentorship", "Combined policy"}

    def test_combined_dominates_baseline(self):
        base = BassParameters(p=0.01, q=0.4, m=0.9)
        scen = policy_scenarios(base, horizon_months=36)
        last = scen[scen["month"] == scen["month"].max()] \
                  .set_index("scenario")["adoption_rate"]
        assert last["Combined policy"] >= last["Baseline"]

    def test_digital_literacy_speeds_early_adoption(self):
        base = BassParameters(p=0.01, q=0.4, m=0.9)
        scen = policy_scenarios(base, horizon_months=60)
        early = scen[scen["month"] == 12].set_index("scenario")["adoption_rate"]
        assert early["Digital literacy"] > early["Baseline"]
