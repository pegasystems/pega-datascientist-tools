# Changelog

All notable changes to pdstools are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

(Pre-release notes accumulate here until the next tag.)

## [5.0.0] — TBD

Major release. **Breaking changes** — see
[`docs/migration-v4-to-v5.md`](docs/migration-v4-to-v5.md) for the upgrade
guide.

### Removed

- `ADMTrees(...)` polymorphic factory shim. Use `ADMTreesModel.from_file`,
  `ADMTreesModel.from_url`, `ADMTreesModel.from_dict`, or
  `MultiTrees.from_datamart` instead.
- `ADMTreesModel(file)` deprecated polymorphic constructor. Use the
  explicit `from_*` classmethods.
- `ADMTreesModel.parse_split_values` /
  `ADMTreesModel.parse_split_values_with_spaces` deprecated helpers.
  Use `ADMTreesModel.parse_split` instead.
- `pdstools.adm.ADMTrees` re-export module (file).  Import from
  `pdstools.adm.trees` directly.
- `pdstools.app.decision_analyzer.da_streamlit_utils.get_current_index`
  back-compat re-export.
- Private `_read_client_credential_file` removed from
  `pdstools.pega_io.__all__`. Still importable for now but no longer a
  documented entry point.

### Changed

- (To be populated by the v5 cleanup PRs.)

### Added

- `pdstools.valuefinder.__init__` now re-exports `ValueFinder` (was empty).
- `CHANGELOG.md` (this file).
- `docs/migration-v4-to-v5.md`.

---

## [4.7.1] and earlier

See git history (`git log --oneline V4.7.1..HEAD`) for pre-CHANGELOG
release notes. Highlights from the 4.x line are summarised in the
migration guide where they matter for v5 users.
