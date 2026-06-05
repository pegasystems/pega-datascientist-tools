"""Build a synthetic ImpactAnalyzer Excel export fixture.

Run once to (re)generate
``python/tests/data/ia/ImpactAnalyzerExport_minimal.xlsx``.

The fixture mirrors the real Pega Infinity Impact Analyzer Excel export's
``Data`` sheet schema (21 columns) but with hand-crafted, public-safe
numbers: 2 dates, 2 channels, 2 actions, all 5 standard experiments —
enough to exercise NBA dedup across experiments, channel/direction
concatenation, ModelControl arm splitting, and an unknown-experiment
warning path.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

OUT = Path(__file__).resolve().parent.parent / "python" / "tests" / "data" / "ia" / "ImpactAnalyzerExport_minimal.xlsx"


def _row(
    *,
    date: str,
    experiment: str,
    issue: str,
    group: str,
    action: str,
    treatment: str,
    direction: str,
    channel: str,
    impressions_test: int,
    accepts_test: int,
    impressions_control: int,
    accepts_control: int,
    value_test: float = 0.0,
    value_control: float = 0.0,
    value_imp_test: float = 0.0,
    value_imp_control: float = 0.0,
) -> dict:
    return {
        "Date": date,
        "Experiment Name": experiment,
        "Issue": issue,
        "Issue Name": issue,
        "Group": group,
        "Group Name": group,
        "Action": action,
        "Action Name": action,
        "Treatment": treatment,
        "Treatment Name": treatment,
        "Direction": direction,
        "Channel": channel,
        "TreatmentSubScript": f"{issue}{group}{action}{channel}",
        "Impressions_Test": impressions_test,
        "Accepts_Test": accepts_test,
        "Impressions_Control": impressions_control,
        "Accepts_Control": accepts_control,
        "ActionValue_Test": value_test,
        "ActionValue_Control": value_control,
        "ActionValueImpression_Test": value_imp_test,
        "ActionValueImpression_Control": value_imp_control,
    }


# Two dates, two channels (Web/Inbound and Email/Outbound), two actions.
# The NBA test arm shows up across all four NBA experiments with identical
# numbers per (date, channel, action) — dedup must collapse to one row.
rows: list[dict] = []

for date in ("2026-01-01", "2026-01-02"):
    for channel, direction, ch_mult in (("Web", "Inbound", 1), ("Email", "Outbound", 2)):
        for action, act_mult in (("Action_A", 1), ("Action_B", 3)):
            # NBA traffic is the same regardless of experiment.
            nba_imp = 1000 * ch_mult * act_mult
            nba_acc = 50 * ch_mult * act_mult
            nba_val_imp = 200.0 * ch_mult * act_mult

            # NBA-vs-X experiments: identical Test (NBA) numbers, distinct controls.
            for experiment, ctrl_imp, ctrl_acc, ctrl_val_imp in (
                (
                    "NBA vs Random relevant action",
                    100 * ch_mult * act_mult,
                    4 * ch_mult * act_mult,
                    10.0 * ch_mult * act_mult,
                ),
                (
                    "NBA vs Arbitrating with propensity only",
                    110 * ch_mult * act_mult,
                    5 * ch_mult * act_mult,
                    12.0 * ch_mult * act_mult,
                ),
                (
                    "NBA vs NBA without levers",
                    120 * ch_mult * act_mult,
                    6 * ch_mult * act_mult,
                    14.0 * ch_mult * act_mult,
                ),
                (
                    "NBA vs NBA with eligibility polices only",
                    130 * ch_mult * act_mult,
                    7 * ch_mult * act_mult,
                    16.0 * ch_mult * act_mult,
                ),
            ):
                rows.append(
                    _row(
                        date=date,
                        experiment=experiment,
                        issue="Sales",
                        group="Cards",
                        action=action,
                        treatment=f"{action}_T",
                        direction=direction,
                        channel=channel,
                        impressions_test=nba_imp,
                        accepts_test=nba_acc,
                        impressions_control=ctrl_imp,
                        accepts_control=ctrl_acc,
                        value_imp_test=nba_val_imp,
                        value_imp_control=ctrl_val_imp,
                    )
                )

            # AdaptiveModel experiment: distinct test & control arms.
            rows.append(
                _row(
                    date=date,
                    experiment="AdaptiveModel (p) vs Random (p)",
                    issue="Sales",
                    group="Cards",
                    action=action,
                    treatment=f"{action}_T",
                    direction=direction,
                    channel=channel,
                    impressions_test=200 * ch_mult * act_mult,
                    accepts_test=8 * ch_mult * act_mult,
                    impressions_control=210 * ch_mult * act_mult,
                    accepts_control=9 * ch_mult * act_mult,
                    value_imp_test=20.0 * ch_mult * act_mult,
                    value_imp_control=22.0 * ch_mult * act_mult,
                )
            )

# A row for an unknown experiment — must trigger a warning and be dropped.
rows.append(
    _row(
        date="2026-01-01",
        experiment="Some Future Unknown Experiment",
        issue="Sales",
        group="Cards",
        action="Action_A",
        treatment="Action_A_T",
        direction="Inbound",
        channel="Web",
        impressions_test=1,
        accepts_test=0,
        impressions_control=1,
        accepts_control=0,
    )
)

df = pl.DataFrame(rows)
print("Built fixture:", df.shape)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.write_excel(OUT, worksheet="Data")
print("Wrote", OUT)
