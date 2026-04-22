# Impact Analyzer — Formula Transparency App

An interactive Streamlit application that visualizes every statistical calculation
behind Pega's Impact Analyzer, step by step.

Use this app to understand **exactly** how Impact Analyzer computes engagement lift,
value lift, confidence intervals, and statistical significance for each experiment.

---

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## What This App Shows

### Overall Health
A gauge summarizing how well your experiments are performing, with a
point-based scoring breakdown you can expand and inspect.

### Experiment Cards
Each of the five Impact Analyzer experiments is displayed as a card showing:

| Element | Description |
|---------|-------------|
| **Engagement Lift** | Relative improvement in accept rate: test vs control |
| **Value Lift** | Relative improvement in value per impression: test vs control |
| **Arm Comparison** | Side-by-side impressions, accepts, and accept rates |
| **Lift Chart** | How lift evolves over time with a 95 % confidence band |
| **CI Bar** | Point estimate and confidence interval at a glance |
| **Lift Details** | Expandable step-by-step formula derivation (see below) |

### Step-by-Step Formulas (inside each card)

| Step | Calculation |
|------|-------------|
| 1 — Accept Rates | `p = Accepts / Impressions` for each arm |
| 2 — Standard Errors | Binomial SE: `√(p(1−p) / n)` |
| 3 — Engagement Lift | Relative lift: `(p_test − p_ctrl) / p_ctrl` |
| 4 — Confidence Interval | Delta-method SE and 95 % CI; significance check |
| 5 — Value Lift | VPI = rate × action value; delta-method CI for value lift |

Every formula is rendered in LaTeX with the actual numerical values substituted
so you can trace each result from raw counts to final output.

### Summary Table
All five experiments in one table with lift, standard error, and significance.

### Full Formula Reference
A collapsible section at the bottom with every formula used across the system,
including required sample size estimation.

---

## Data Sources

| Mode | Description |
|------|-------------|
| **Sample data** | Pre-loaded events for immediate exploration |
| **Live generator** | Simulate streaming data with four scenario profiles |
| **File upload** | Load a PDC JSON export or event array |
| **Live bridge** | Poll a REST endpoint for real-time events |

The live generator includes four profiles — *Realistic NBA*, *Clear NBA winner*,
*Neck-and-neck*, and *NBA losing* — so you can see how the metrics and
significance verdicts change under different conditions.

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `ia_engine.py` | Computation engine (rates, SE, lift, CI, VPI, health) |
| `sample_data.py` | Sample payload and generator profiles |
| `requirements.txt` | Python dependencies |
| `REPORT_Impact_Analyzer_Formulas_and_Validation.md` | Detailed formula reference |

---

## Requirements

- Python 3.10+
- Streamlit
- Plotly
- Requests

All dependencies are listed in `requirements.txt`.

---

## Learn More

- [Pega Impact Analyzer documentation](https://docs.pega.com)
- [pdstools on GitHub](https://github.com/pegasystems/pega-datascientist-tools)
