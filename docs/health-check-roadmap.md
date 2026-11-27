# ADM Health Check roadmap

Open follow-ups for the ADM Health Check and its supporting library.
P2 is medium priority; P3 is exploratory.

## Correctness

- **P2 — Active models:** Review whether action overview counts should apply
  the same active-model filter as the rest of the report (#594).

## Plots and reports

- **P2 — Plotly template:** Consolidate repeated report layout defaults into
  the `"pega"` template where appropriate (#600).
- **P3 — Report plot helpers:** Move cumulative gains, lift, and
  base-propensity overlays from Model Report into reusable library plots.
- **P3 — Long labels:** Ensure abbreviated bin labels remain distinguishable;
  the shared abbreviation helper does not guarantee uniqueness.
