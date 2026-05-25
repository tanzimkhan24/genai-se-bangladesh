# Generative AI and the Software Engineering Workforce in Bangladesh

A reproducible mixed-methods thesis on the adoption, productivity effects, and
educational implications of generative AI assistance in the Bangladeshi
software engineering sector.

**Author:** Tanzim Islam Khan (`dihan2468@gmail.com`)
**Literature cutoff:** 24 May 2026

## Structure

```
src/                  Source code (simulation, diffusion, analysis, viz)
tests/                Unit tests
data/                 Generated datasets calibrated to Bangladesh statistics
results/figures/      Publication figures (PDF + PNG, grayscale)
results/tables/       Generated tables (CSV)
paper/                LaTeX thesis and bibliography
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.run_pipeline      # generate all figures and tables
pytest -q                       # run unit tests
tectonic paper/main.tex         # compile the thesis PDF
```

## Reproducibility

Random seed: `SEED = 20260524`. Pipeline is deterministic.
