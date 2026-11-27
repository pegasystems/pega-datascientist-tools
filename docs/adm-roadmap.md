# ADM roadmap

Open follow-ups for the ADM library. P2 is medium priority; P3 is exploratory.

## Architecture and reliability

- **P2 — Plotly template registration:** Remove the reliance on
  `prediction.Plots` importing the `"pega"` template as a side effect.
  Ensure ADM plots register it independently before removing the shim.
