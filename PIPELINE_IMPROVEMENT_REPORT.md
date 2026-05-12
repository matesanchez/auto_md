# AutoMD Pipeline Improvement Report

Date: 2026-05-12  
Objective: review the repo, manual, and pipeline goals; identify at least 10 improvements; complete all 10 to move AutoMD toward a more automated, near-professional MD pipeline.

## Summary

AutoMD already had a functioning smoke pipeline, but the manual calls for more automation, fewer free-form user decisions, stronger provenance, better high-throughput triage, and clearer production-readiness boundaries. The improvements below focus on those gaps.

All ten improvements listed here are implemented in the current codebase and covered by targeted tests or command validation.

## Completed Improvements

### 1. Streamlined inline auto input

Type: streamlining

Problem: the one-command `automd auto` lane accepted comma-separated `SMILES:ratio` strings, but not semicolon or newline separated entries. That made pasting formulations from notes or spreadsheets brittle.

Implementation:

- `parse_auto_input()` now accepts comma, semicolon, and newline separated inline components.
- Ratios are still normalized and hashed into the same provenance path.

Evidence:

- `tests/unit/test_workflow.py::test_auto_inline_parser_accepts_semicolon_and_newline`

### 2. Metadata-preserving CSV/YAML/JSON auto input

Type: streamlining

Problem: file-based auto input collapsed useful user hints such as component name, role, local ID, and topology hints.

Implementation:

- Auto input now preserves `local_id`, `name`, `role`, `topology_hint`, `topology_source_hint`, and `topology_id` when present.
- The expanded formulation uses that metadata instead of turning every component into an anonymous `component_###`.

Evidence:

- `tests/unit/test_workflow.py::test_auto_csv_parser`
- `tests/unit/test_workflow.py::test_auto_preserves_input_metadata_pipeline_trace_and_seed`

### 3. Auto-input validation warnings

Type: automation/provenance

Problem: duplicate SMILES and zero-ratio components were not surfaced early in the automation manifest.

Implementation:

- Auto input now records `input_validation` with component count and warnings.
- Zero-ratio and duplicate-SMILES warnings are preserved in `automation_manifest.yaml`.

Evidence:

- `parse_auto_input()` emits `validation.warnings`.
- `automation_manifest.yaml` records `input_validation`.

### 4. Machine-readable pipeline step trace

Type: automation observability

Problem: a user could inspect individual manifests, but there was no single ordered trace of which automation stages passed, blocked, skipped, or failed.

Implementation:

- `command_auto()` now records `pipeline_steps` into `manifests/automation_manifest.yaml`.
- Steps include intake, descriptors, role inference, topology generation, topology review, template selection, build, preflight, smoke, QC, metrics, production plan, and report.

Evidence:

- `tests/unit/test_workflow.py::test_auto_preserves_input_metadata_pipeline_trace_and_seed`

### 5. Reproducible random-seed propagation

Type: reproducibility

Problem: the manual requires random seeds to be recorded. The build manifest recorded the simulation request, but MDP files used a hardcoded seed path.

Implementation:

- MDP generation now accepts a seed.
- `command_build_smoke()` reads the seed from the template/intake assumptions.
- `build_manifest.yaml` records a `reproducibility` section.

Evidence:

- `tests/unit/test_workflow.py::test_auto_preserves_input_metadata_pipeline_trace_and_seed`

### 6. Production-readiness planning

Type: significant high-level improvement

Problem: the pipeline had smoke readiness but no explicit production-readiness plan, even though the manual separates smoke triage from production MD.

Implementation:

- Added `automd production plan RUN_DIR`.
- Added `command_production_plan()`.
- Production plans block unresolved, placeholder, review-required, non-production-eligible, or QC-failing runs.
- Smoke workflows now write `manifests/production_plan_manifest.yaml`.

Evidence:

- `tests/unit/test_workflow.py::test_auto_preserves_input_metadata_pipeline_trace_and_seed`
- CLI exposes `production`.

### 7. Feature-table export across runs and batches

Type: significant high-level improvement

Problem: high-throughput MD triage needs a single table that joins descriptors, topology readiness, QC, blockers, and metrics.

Implementation:

- Added `automd features build RUN_OR_BATCH ... --out OUT_DIR`.
- Added `command_features_build()`.
- Supports individual run directories and batch directories with `batch_status.yaml`.
- Writes `features.csv` plus `feature_manifest.yaml`.

Evidence:

- `tests/unit/test_workflow.py::test_batch_auto_input_priority_features_and_summary`
- CLI exposes `features`.

### 8. Smarter prioritization scoring

Type: automation/triage

Problem: prioritization only used descriptor coverage and unresolved topology count.

Implementation:

- Prioritization now combines descriptor coverage, smoke readiness, QC status, qualitative metric availability, and blocker count.
- The manifest schema was advanced to `automd.priority_manifest.v0.2`.
- Recommended next actions now distinguish `production_plan`, `run_smoke_test`, and `resolve_topology`.

Evidence:

- `tests/unit/test_workflow.py::test_batch_auto_input_priority_features_and_summary`

### 9. Auto-input batch planning and execution

Type: streamlining/high-throughput automation

Problem: batch execution expected formulation-file paths, but the user goal is to supply starting SMILES and ratios.

Implementation:

- `batch plan` now accepts `auto_input`, `smiles_ratio`, or `components` columns.
- `batch smoke` routes those rows through `command_auto()`.
- Batch statuses now include readiness, QC status, audit status, blocker count, and report path.

Evidence:

- `tests/unit/test_workflow.py::test_batch_auto_input_priority_features_and_summary`

### 10. Richer batch summaries and user-facing reports

Type: reporting/professionalization

Problem: batch summaries only counted statuses, and run reports did not show the automation trace or production-readiness result.

Implementation:

- `batch_summary.md` now includes status counts, readiness counts, and total blockers.
- Run reports now include automation trace, blockers, and production-readiness status when available.

Evidence:

- `tests/unit/test_workflow.py::test_batch_auto_input_priority_features_and_summary`
- `command_report_run()` includes automation, blocker, and production sections.

## Validation Plan

The implementation should be considered complete only if all of the following pass:

```bash
ruff check automd tests scripts
python -m compileall -q automd tests scripts
pytest -q
automd auto "CCCCCCCCCCCCN(CCCCCCCC)CCCCCCCC:100" --out /tmp/automd_improvement_auto
automd audit run /tmp/automd_improvement_auto
automd production plan /tmp/automd_improvement_auto
automd features build /tmp/automd_improvement_auto --out /tmp/automd_improvement_features
```

## Remaining Future Opportunities

These are intentionally not counted among the completed ten:

- split `automd/core.py` into package modules,
- add SLURM submit/status/collect commands,
- add real Packmol/insane/Polyply builders behind the existing plugin boundary,
- add HTML reports,
- add a formal topology registry ingestion command for public Martini source metadata.
