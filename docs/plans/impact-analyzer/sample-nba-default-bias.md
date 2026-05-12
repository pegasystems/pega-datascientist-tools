# Sample data — default profile makes NBA win by traffic, not lift

**Priority:** P2
**Touches:** `python/pdstools/app/impact_analyzer/sample_data.py`

The default `"Realistic NBA (asymmetric traffic)"` profile gives NBA ~5000
impressions and `PropensityPriority` ~400 impressions. Even though
`PropensityPriority` has a higher per-impression accept rate (≈0.020 vs
NBA's ≈0.0115), NBA dominates absolute accepts by sheer volume. The
"NBA vs Propensity Only" experiment then shows NBA as the winner, which
is the opposite of what users expect when they read the experiment name
("does NBA beat a propensity-only ranking?").

## Approach

- Rebalance the default profile so that, for the experiments the sample
  is meant to demonstrate, the "winner" matches the experiment's
  conceptual intent:
  - "NBA vs Random": NBA should win.
  - "NBA vs Propensity Only": Propensity Only should win on engagement
    lift (it picks the highest-propensity item) — NBA wins on value lift.
  - "NBA vs No Levers" / "NBA vs Only Eligibility Rules": NBA should win.
- Document the expected outcomes in `GENERATOR_PROFILES[...]["description"]`
  so users can verify the numbers against their reading.
- Optionally add a profile tailored for each experiment.
