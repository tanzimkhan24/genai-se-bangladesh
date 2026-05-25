"""Grayscale, publication-quality figure generation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

GRAY_PALETTE = ["#000000", "#3a3a3a", "#6e6e6e", "#9a9a9a", "#cccccc"]


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")


def figure_adoption_curve(ts: pd.DataFrame, out: Path) -> Path:
    """Adoption S-curves by firm type."""
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    firms = sorted(ts["firm_type"].unique())
    for i, firm in enumerate(firms):
        sub = ts[ts["firm_type"] == firm]
        ax.plot(sub["month_index"], sub["adoption_rate"],
                color=GRAY_PALETTE[i % len(GRAY_PALETTE)],
                linewidth=1.6, label=firm)
    ax.set_xlabel("Months from January 2022")
    ax.set_ylabel("Cumulative adoption rate")
    ax.set_title("GenAI adoption diffusion in Bangladesh by firm type, 2022 to 2026")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", frameon=False)
    return _save(fig, out)


def figure_regional_heatmap(df: pd.DataFrame, out: Path) -> Path:
    """Heatmap: adoption rate by region and firm type."""
    pivot = df.pivot_table(index="region", columns="firm_type",
                            values="genai_adopted", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    im = ax.imshow(pivot.values, cmap="Greys", aspect="auto", vmin=0.3, vmax=0.95)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=9)
    ax.set_title("Mean GenAI adoption rate by region and firm type")
    fig.colorbar(im, ax=ax, label="Adoption rate")
    ax.grid(False)
    return _save(fig, out)


def figure_productivity_box(df: pd.DataFrame, out: Path) -> Path:
    """Boxplot of productivity by adoption status."""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    data = [df.loc[df["genai_adopted"] == 0, "productivity_index"].to_numpy(),
            df.loc[df["genai_adopted"] == 1, "productivity_index"].to_numpy()]
    bp = ax.boxplot(data, tick_labels=["Non-adopter", "Adopter"],
                    patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], ["#cccccc", "#6e6e6e"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("black")
    ax.set_ylabel("Productivity index")
    ax.set_title("Productivity distribution by GenAI adoption status")
    return _save(fig, out)


def figure_english_moderator(eng_df: pd.DataFrame, out: Path) -> Path:
    """Forest plot: treatment effect by English proficiency stratum."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    eng_df = eng_df.sort_values("english_cefr")
    y = np.arange(len(eng_df))
    ax.errorbar(eng_df["cohens_d"], y,
                xerr=1.96 / np.sqrt(eng_df["n_treated"] + eng_df["n_control"]),
                fmt="s", color="black", capsize=3)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(eng_df["english_cefr"])
    ax.set_xlabel("Cohen's d (productivity)")
    ax.set_title("Heterogeneous adoption effect by English proficiency (CEFR)")
    return _save(fig, out)


def figure_firm_forest(df: pd.DataFrame, out: Path) -> Path:
    """Forest plot of treatment effect by firm type."""
    rows = []
    for firm, sub in df.groupby("firm_type"):
        if sub["genai_adopted"].nunique() < 2:
            continue
        treated = sub.loc[sub["genai_adopted"] == 1, "productivity_index"]
        control = sub.loc[sub["genai_adopted"] == 0, "productivity_index"]
        d = (treated.mean() - control.mean()) / sub["productivity_index"].std(ddof=1)
        rows.append({"firm_type": firm, "d": d, "n": len(sub)})
    F = pd.DataFrame(rows).sort_values("d")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y = np.arange(len(F))
    ax.errorbar(F["d"], y, xerr=1.96 / np.sqrt(F["n"]), fmt="s",
                color="black", capsize=3)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(F["firm_type"])
    ax.set_xlabel("Cohen's d (productivity)")
    ax.set_title("GenAI adoption productivity effect by firm type")
    return _save(fig, out)


def figure_salary_trajectory(df: pd.DataFrame, out: Path) -> Path:
    """Scatter of salary against experience, faceted by adoption."""
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for adopted, color, label in [(0, "#cccccc", "Non-adopter"),
                                    (1, "#3a3a3a", "Adopter")]:
        sub = df[df["genai_adopted"] == adopted]
        ax.scatter(sub["experience_years"], sub["monthly_salary_bdt"] / 1000.0,
                   s=8, alpha=0.5, color=color, label=label)
        # Smoothed trend.
        order = np.argsort(sub["experience_years"].to_numpy())
        x = sub["experience_years"].to_numpy()[order]
        y = sub["monthly_salary_bdt"].to_numpy()[order] / 1000.0
        if len(x) > 10:
            window = max(20, len(x) // 30)
            yh = pd.Series(y).rolling(window, center=True, min_periods=1).mean()
            ax.plot(x, yh, color="black" if adopted else "gray", linewidth=1.8)
    ax.set_xlabel("Years of experience")
    ax.set_ylabel("Monthly salary (BDT, thousands)")
    ax.set_title("Salary trajectory by experience and GenAI adoption")
    ax.legend(loc="upper left", frameon=False)
    return _save(fig, out)


def figure_skills_gap(gap_df: pd.DataFrame, out: Path) -> Path:
    """Bar chart of skills-gap index by university tier."""
    gap_df = gap_df.sort_values("skills_gap_index")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(gap_df["university_tier"], gap_df["skills_gap_index"],
            color="#6e6e6e", edgecolor="black")
    ax.set_xlabel("Skills gap index (SD units below 95th-percentile ceiling)")
    ax.set_title("Skills gap by university tier")
    return _save(fig, out)


def figure_education_alignment(df: pd.DataFrame, out: Path) -> Path:
    """Heatmap: mean productivity by education and university tier."""
    pivot = df.pivot_table(index="education", columns="university_tier",
                            values="productivity_index", aggfunc="mean")
    pivot = pivot.reindex(index=["Diploma", "BSc", "MSc", "PhD"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(pivot.values, cmap="Greys", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > pivot.values.mean() else "black",
                        fontsize=9)
    ax.set_title("Mean productivity by education level and university tier")
    fig.colorbar(im, ax=ax, label="Productivity index")
    ax.grid(False)
    return _save(fig, out)


def figure_policy_scenarios(scen: pd.DataFrame, out: Path) -> Path:
    """Policy-scenario adoption projection."""
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    palette = ["#000000", "#3a3a3a", "#6e6e6e", "#a0a0a0"]
    for i, name in enumerate(["Baseline", "Digital literacy",
                                "Industry mentorship", "Combined policy"]):
        sub = scen[scen["scenario"] == name]
        ax.plot(sub["month"], sub["adoption_rate"], color=palette[i],
                linewidth=1.8, label=name)
    ax.set_xlabel("Months from May 2026")
    ax.set_ylabel("Cumulative adoption rate")
    ax.set_title("Policy scenario projections, 2026 to 2031")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", frameon=False)
    return _save(fig, out)


def figure_workforce_projection(proj: pd.DataFrame, out: Path) -> Path:
    """Projection of total workforce and adopters to 2030."""
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(proj["year"], proj["workforce"] / 1e6, width=0.6,
           color="#cccccc", edgecolor="black", label="Total workforce (millions)")
    ax2 = ax.twinx()
    ax2.plot(proj["year"], proj["adopters"] / 1e6, color="black",
             marker="o", linewidth=2.0, label="GenAI adopters (millions)")
    ax.set_ylabel("Total SE workforce (millions)")
    ax2.set_ylabel("GenAI adopters (millions)")
    ax.set_xlabel("Year")
    ax.set_title("Bangladesh SE workforce and GenAI adoption projection 2026 to 2030")
    ax.legend(loc="upper left", frameon=False)
    ax2.legend(loc="lower right", frameon=False)
    ax2.grid(False)
    return _save(fig, out)


def figure_comparator_economies(out: Path) -> Path:
    """Bangladesh against comparator emerging-economy IT sectors."""
    countries = ["Bangladesh", "Vietnam", "Philippines", "India", "Indonesia"]
    workforce_2025 = [1.50, 1.20, 1.40, 5.40, 1.85]
    adoption_2025 = [0.42, 0.55, 0.50, 0.62, 0.45]
    gdp_pc = [2700, 4400, 3700, 2700, 4900]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    width = 0.35
    x = np.arange(len(countries))
    ax.bar(x - width / 2, workforce_2025, width=width,
           color="#cccccc", edgecolor="black", label="Workforce (millions)")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, adoption_2025, width=width,
            color="#3a3a3a", edgecolor="black", label="Adoption rate")
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=20, ha="right")
    ax.set_ylabel("Workforce (millions)")
    ax2.set_ylabel("Adoption rate")
    ax.set_title("Workforce and adoption rate, 2025 estimates")
    ax2.grid(False)
    ax.legend(loc="upper left", frameon=False)
    ax2.legend(loc="upper right", frameon=False)

    ax = axes[1]
    ax.scatter(gdp_pc, adoption_2025, s=70, color="black")
    for c, x_v, y_v in zip(countries, gdp_pc, adoption_2025):
        ax.annotate(c, (x_v, y_v), xytext=(5, 5), textcoords="offset points",
                    fontsize=9)
    ax.set_xlabel("GDP per capita 2024 (USD)")
    ax.set_ylabel("GenAI adoption rate")
    ax.set_title("Adoption against per-capita output")
    fig.tight_layout()
    return _save(fig, out)


def figure_learning_hours(df: pd.DataFrame, out: Path) -> Path:
    """Distribution of weekly learning hours by adoption status."""
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    bins = np.linspace(0, 12, 25)
    ax.hist(df.loc[df["genai_adopted"] == 0, "weekly_learning_hours"],
            bins=bins, color="#cccccc", edgecolor="black", alpha=0.85,
            label="Non-adopter")
    ax.hist(df.loc[df["genai_adopted"] == 1, "weekly_learning_hours"],
            bins=bins, color="#3a3a3a", edgecolor="black", alpha=0.65,
            label="Adopter")
    ax.set_xlabel("Weekly self-directed learning (hours)")
    ax.set_ylabel("Frequency")
    ax.set_title("Self-directed learning hours by adoption status")
    ax.legend(loc="upper right", frameon=False)
    return _save(fig, out)
