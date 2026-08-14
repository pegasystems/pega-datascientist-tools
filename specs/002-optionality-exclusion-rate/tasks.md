# Tasks: Exclusion Rate Distribution (Decision Analyzer)

**Input**: Design documents from `/specs/002-optionality-exclusion-rate/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, and `contracts/exclusion-rate-contract.md`

**Tests**: Included because the feature specification defines independent tests and exact-value acceptance scenarios.

**Organization**: Tasks are grouped by user story. The core library work is placed in User Story 1 because it delivers the metric independently; stage controls and page presentation follow as independently testable increments.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm the existing project environment and implementation surfaces; no new dependencies or project scaffolding are required.

- [ ] T001 Verify the existing `uv` test environment and Decision Analyzer sample fixtures are available for this feature in `pyproject.toml`, `data/da/sample_eev2_minimal.csv`, and `python/tests/streamlit_apps/conftest.py`
- [ ] T002 [P] Confirm the current Optionality Analysis integration points and existing plot registration in `python/pdstools/app/decision_analyzer/pages/7_Optionality_Analysis.py` and `python/pdstools/decision_analyzer/plots/__init__.py`

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the shared stage-count contract before story-specific work.

- [ ] T003 Document and preserve the relationship between optionality counts and exclusion-rate counts in `specs/002-optionality-exclusion-rate/data-model.md`, using `aggregate_remaining_per_stage` as the shared calculation boundary
- [ ] T004 [P] Add the exclusion-rate API contract, including output columns, stage validation, zero-baseline behavior, `return_df`, and 2.5% plot bands, to `specs/002-optionality-exclusion-rate/contracts/exclusion-rate-contract.md`
- [ ] T005 Define the validation matrix for ordered stages, equal stages, inverted stages, unknown stages, zero baseline, and empty filtered data in `specs/002-optionality-exclusion-rate/quickstart.md`

**Checkpoint**: Shared data semantics, API expectations, and edge cases are defined; User Story 1 can begin.

## Phase 3: User Story 1 - See Excluded Actions Before Arbitration (Priority: P1) MVP

**Goal**: Compute exact per-interaction exclusion rates and render a distribution showing the share of interactions in each 2.5% exclusion-rate band.

**Independent Test**: On the minimal EEV2 dataset, compute `Eligibility → Output` and verify the exact rates `0.60`, `1.00`, and `2/3`; build the plot and verify one bar trace with 40 bands from `0-2.5%` through `97.5-100%`, green-to-red bar colors, and no cumulative line.

### Tests for User Story 1

- [ ] T006 [P] [US1] Add exact aggregate tests for `5→2`, `4→0`, and `3→1` action counts, expected `Actions From`, `Actions To`, `Excluded`, and `Exclusion Rate` columns in `python/tests/decision_analyzer/test_DecisionAnalyzer.py`
- [ ] T007 [P] [US1] Add aggregate edge-case tests for equal stages, zero-baseline interaction omission, inverted stage errors, and unknown stage errors in `python/tests/decision_analyzer/test_DecisionAnalyzer.py`
- [ ] T008 [P] [US1] Add plot tests for `return_df`, 20 5% bands, hidden-by-default optionality-style propensity trace, endpoint labels, and distinct green-to-red marker colors in `python/tests/decision_analyzer/test_da_plots.py`

### Implementation for User Story 1

- [ ] T009 [US1] Implement `Aggregates.get_exclusion_rate_data(from_stage, to_stage, df)` in `python/pdstools/decision_analyzer/_aggregates.py` by reusing `aggregate_remaining_per_stage`, validating stage order, omitting zero-baseline interactions, and returning exact per-interaction counts and rates
- [ ] T010 [US1] Implement `exclusion_rate_distribution(from_stage, to_stage, df, return_df)` in `python/pdstools/decision_analyzer/plots/_optionality.py` with 5% bands, percentage-of-interactions bars, a green-to-red color scale, and hidden-by-default propensity on a secondary axis
- [ ] T011 [US1] Register `exclusion_rate_distribution` on the `Plot` namespace in `python/pdstools/decision_analyzer/plots/__init__.py`
- [ ] T012 [US1] Run the focused aggregate and plot tests with `uv run pytest python/tests/decision_analyzer/test_DecisionAnalyzer.py python/tests/decision_analyzer/test_da_plots.py -k "ExclusionRate or exclusion" -q` and resolve defects in the User Story 1 implementation files

**Checkpoint**: User Story 1 is independently usable through the library API and plot namespace, without the Streamlit app.

## Phase 4: User Story 2 - Override Baseline and Measurement Stages (Priority: P2)

**Goal**: Allow analysts to choose the baseline stage while the measurement stage follows the existing Optionality selector, with a clear warning for inverted ranges.

**Independent Test**: Change the baseline selector on Page 7 and verify session state and plot rerendering; choose a baseline after the measurement stage and verify a warning appears without a misleading plot.

### Tests for User Story 2

- [ ] T013 [P] [US2] Add an AppTest state-transition test that changes `exclusion_from_stage` and verifies the selected value persists in `python/tests/streamlit_apps/decision_analyzer/test_exclusion_rate_selectbox.py`
- [ ] T014 [P] [US2] Add an AppTest invalid-range test that selects a baseline after `optionality_stage` and verifies the warning path in `python/tests/streamlit_apps/decision_analyzer/test_exclusion_rate_selectbox.py`
- [ ] T015 [P] [US2] Update the Page 7 smoke-test widget expectation for the additional baseline selectbox in `python/tests/streamlit_apps/decision_analyzer/test_da_pages.py`

### Implementation for User Story 2

- [ ] T016 [US2] Add the baseline stage selector keyed `exclusion_from_stage`, defaulting to `Engagement Policies` with existing stage-option fallback, in `python/pdstools/app/decision_analyzer/pages/7_Optionality_Analysis.py`
- [ ] T017 [US2] Wire the existing `optionality_stage` session-state value as the exclusion-rate measurement stage and pass the page's filtered frame into the plot from `python/pdstools/app/decision_analyzer/pages/7_Optionality_Analysis.py`
- [ ] T018 [US2] Add the invalid-stage-order warning and suppress plot rendering when the baseline follows the measurement stage in `python/pdstools/app/decision_analyzer/pages/7_Optionality_Analysis.py`
- [ ] T019 [US2] Run focused Page 7 tests with `uv run pytest python/tests/streamlit_apps/decision_analyzer/test_da_pages.py::test_da_page_renders[7_Optionality_Analysis] python/tests/streamlit_apps/decision_analyzer/test_exclusion_rate_selectbox.py -q`

**Checkpoint**: User Stories 1 and 2 work independently; stage overrides are validated through Streamlit state transitions and invalid ranges are safe.

## Phase 5: User Story 3 - Read Exclusion Rate Alongside Optionality (Priority: P3)

**Goal**: Present the exclusion-rate distribution on the existing Optionality Analysis page for v2 extracts, alongside optionality, while preserving filters and v1 behavior.

**Independent Test**: Render Page 7 with the seeded v2 fixture, verify the Exclusion Rate section and filtered plot surface, and render a v1 extract to verify the section is absent.

### Tests for User Story 3

- [ ] T020 [P] [US3] Add or update the Page 7 render test to verify the Exclusion Rate section renders for v2 data and the page remains exception-free in `python/tests/streamlit_apps/decision_analyzer/test_da_pages.py`
- [ ] T021 [P] [US3] Add a v1 Page 7 coverage case confirming the exclusion-rate section is not rendered for `extract_type == "explainability_extract"` in `python/tests/streamlit_apps/decision_analyzer/test_da_pages.py`
- [ ] T022 [P] [US3] Add a filtered-data AppTest case confirming the exclusion plot receives the same global/contextual filters as optionality in `python/tests/streamlit_apps/decision_analyzer/test_channel_filter.py` or a focused new test file under `python/tests/streamlit_apps/decision_analyzer/`

### Implementation for User Story 3

- [ ] T023 [US3] Add the v2-only Exclusion Rate section, explanatory caption, and plot call below the Optionality section in `python/pdstools/app/decision_analyzer/pages/7_Optionality_Analysis.py`
- [ ] T024 [US3] Preserve v1 gating and empty-filter handling so `explainability_extract` data does not show an invalid pre-arbitration metric in `python/pdstools/app/decision_analyzer/pages/7_Optionality_Analysis.py`
- [ ] T025 [US3] Run the related Decision Analyzer regression tests for optionality, exclusion, plot rendering, and Page 7 behavior with `uv run pytest python/tests/decision_analyzer/test_DecisionAnalyzer.py python/tests/decision_analyzer/test_da_plots.py python/tests/streamlit_apps/decision_analyzer/test_da_pages.py -k "Optionality or Exclusion or exclusion or optionality" -q`

**Checkpoint**: All three user stories are independently testable; v2 analysts see filtered exclusion-rate distributions beside optionality, while v1 behavior remains unchanged.

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate documentation, quality, performance assumptions, and the final user-facing workflow.

- [ ] T026 [P] Update the worked examples and plot-band description to match the final 2.5% green-to-red presentation in `specs/002-optionality-exclusion-rate/data-model.md`, `specs/002-optionality-exclusion-rate/research.md`, and `specs/002-optionality-exclusion-rate/quickstart.md`
- [ ] T027 [P] Run `uv run ruff check` on changed Python files including `python/pdstools/decision_analyzer/_aggregates.py`, `python/pdstools/decision_analyzer/plots/_optionality.py`, and `python/tests/decision_analyzer/test_da_plots.py`, preserving unrelated pre-existing lint findings outside the feature slice
- [ ] T028 [P] Run `uv run pyright` against changed Python files including `python/pdstools/decision_analyzer/_aggregates.py`, `python/pdstools/decision_analyzer/plots/_optionality.py`, and `python/pdstools/app/decision_analyzer/pages/7_Optionality_Analysis.py`, resolving newly introduced errors
- [ ] T029 Run the quickstart validation scenarios from `specs/002-optionality-exclusion-rate/quickstart.md`, including a real v2 sample and the invalid-stage case
- [ ] T030 Review the final diff and `./AGENTS.md` for accidental generated artifacts, private-data paths, unrelated changes, and repository-policy violations; leave no customer data or preview files tracked

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No code dependencies; confirms existing surfaces.
- **Foundational (Phase 2)**: Depends on Setup and defines the shared metric contract.
- **User Story 1 (Phase 3)**: Depends on Foundational; delivers the library MVP.
- **User Story 2 (Phase 4)**: Depends on User Story 1's aggregate and plot APIs because the selectors call them.
- **User Story 3 (Phase 5)**: Depends on User Story 2's selector/session-state wiring and adds v2 presentation coverage.
- **Polish (Phase 6)**: Depends on all desired story phases.

### User Story Dependencies

- **US1 (P1)**: No dependency on another user story after Foundational; this is the MVP.
- **US2 (P2)**: Depends on US1's public aggregate and plot methods.
- **US3 (P3)**: Depends on US2's Page 7 selector wiring, but remains independently testable through AppTest.

### Parallel Opportunities

- T002, T004, and T005 can run in parallel because they inspect or update separate design/integration files.
- T006, T007, and T008 can be written in parallel before implementation.
- T013, T014, and T015 can be written in parallel because they touch separate test concerns, though T015 shares the Page 7 smoke-test file with no implementation task.
- T020, T021, and T022 can be written in parallel after the Page 7 surface exists.
- T026, T027, and T028 can run in parallel during final polish.

## Parallel Execution Examples

### User Story 1

```text
Parallel: T006 exact aggregate tests
Parallel: T007 aggregate edge-case tests
Parallel: T008 plot tests
Then:    T009 aggregate implementation
Then:    T010 plot implementation
Then:    T011 plot namespace registration
Then:    T012 focused validation
```

### User Story 2

```text
Parallel: T013 baseline selector state-transition test
Parallel: T014 inverted-range warning test
Parallel: T015 Page 7 widget-count update
Then:    T016-T018 Page 7 implementation
Then:    T019 focused AppTest validation
```

### User Story 3

```text
Parallel: T020 v2 render coverage
Parallel: T021 v1 gating coverage
Parallel: T022 filtered-data coverage
Then:    T023-T024 Page 7 integration and gating
Then:    T025 related regression suite
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Setup and Foundational phases.
2. Implement and test `get_exclusion_rate_data`.
3. Implement and register `exclusion_rate_distribution` with 5% green-to-red bars and optional hidden propensity.
4. Run the exact-value and plot tests.
5. Stop at the US1 checkpoint for a notebook/library demonstration.

### Incremental Delivery

1. Add US1 for the reusable metric and plot API.
2. Add US2 for baseline-stage overrides and invalid-range protection.
3. Add US3 for v2-only Page 7 presentation and filter parity.
4. Complete Polish validation and review the final diff.

## Notes

- Every task uses the required checklist format: checkbox, sequential ID, optional `[P]`, required story label in story phases, and an exact file path.
- `[P]` means the task can proceed independently without waiting on an incomplete task in the same phase.
- The plot intentionally returns exact per-interaction data with `return_df=True`; 2.5% binning and color are presentation concerns.
- Do not commit private sample paths, generated HTML, screenshots, or customer data to the repository.

## Remediation Tasks Added by Consistency Analysis

- [ ] T031 [P] [US2] Add an AppTest measurement-selector transition that changes `optionality_stage` and verifies the exclusion section follows the selected measurement stage in `python/tests/streamlit_apps/decision_analyzer/test_exclusion_rate_selectbox.py`
- [ ] T032 [P] [US3] Add an empty-filter AppTest case confirming Page 7 warns and stops before rendering charts when `page_channel_expr` matches no rows in `python/tests/streamlit_apps/decision_analyzer/test_exclusion_rate_selectbox.py`
