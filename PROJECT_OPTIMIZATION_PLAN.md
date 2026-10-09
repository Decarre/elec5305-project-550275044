# Project optimization and continuation plan

**Purpose:** Keep the ELEC5305 project resumable if a Codex session or usage allowance ends. Work is divided into independently testable Git checkpoints. Do not claim full-data performance until a held-out MedleyDB or duplicate-safe Slakh evaluation has run.

## Resume procedure

```powershell
Set-Location F:\USYD\ELEC5305\elec5305-project-550275044
git status --short
git log -5 --oneline
.\.venv\Scripts\python.exe -m pytest
```

Read this file, `README.md`, `docs/index.md`, and the latest Git commit before editing. Full audio must remain outside Git.

## Phase 1: experiment semantics and evaluation foundations

**Status:** completed and pushed as commit `d8ef6b3` on 4 October 2026.

- Central instrument taxonomy shared by metadata and audio pipelines.
- Mean and attention models aggregate the same frame-probability quantity so pooling comparisons are controlled.
- Optional max-pooling baseline with the same encoder/classifier structure.
- Deterministic class-aware artist split helper and disjointness validation.
- Activation-confidence parsing, stem-to-family aggregation, and fixed reporting-grid conversion.
- Dependency-light multi-label metrics, validation-only F1 threshold selection, and attention-mass alignment metric.
- Training/evaluation configuration fields for seed, model size, optimizer settings, and reporting grid.
- Meaningful unit tests for the new scientific invariants.

Acceptance gate:

- `pytest` passes.
- `git diff --check` passes.
- Controlled model semantics are covered by the test suite.

## Phase 2: manifests and leakage barriers

**Status:** completed and pushed as commit `172c04b` on 4 October 2026.

Deliverables:

- `manifest.py` dataclasses/read-write helpers for tracks and clips.
- A script that joins official metadata, annotation availability, audio paths and the frozen artist split.
- Clip windows created only after the artist split.
- A single target-family list and column order written into every manifest.
- Checks for missing WAVs, duplicated track IDs, artist leakage, annotation coverage and class coverage.
- Saved split summary with a stable manifest hash.

Implementation note: the manifest contains paths, windows and clip-level identity only. Temporal reference arrays remain in the evaluation path and are not fields in `ClipManifestRow`.

Acceptance gate:

- Public sample produces a valid manifest.
- Synthetic metadata test proves that the same artist cannot cross partitions.
- No temporal reference arrays are returned by the training dataset interface.

## Phase 3: training and validation CLI

**Status:** completed in commits `233f330` and `ad61382` on 4 October 2026.

Deliverables:

- Reusable dataset loader and mini-batch training loop.
- `train`, `tune`, and `evaluate` CLI commands while retaining config validation.
- Matched mean/max/attention model construction.
- BCE loss, train-only class weighting if needed, validation early stopping, checkpointing and deterministic seeds.
- Validation-only threshold and post-processing selection.
- Run metadata: source commit, config, split hash, seed, best epoch, runtime and hardware.

Acceptance gate:

- Tiny synthetic dataset can overfit.
- Validation thresholds are saved and test evaluation cannot modify them.
- Checkpoint reload reproduces predictions.

## Phase 4: public-sample rerun and documentation sync

**Status:** completed on 5 October 2026.

Deliverables:

- Refactor `run_sample_pilot.py` to use the shared taxonomy, metrics and model builder.
- Rerun the two-song sample under the controlled pooling protocol.
- Replace saved sample CSV/JSON/figure outputs.
- Update README and Project Site so every number matches saved artifacts.
- Add an AttentionMIC implementation map explaining reproduced and modified components.

Acceptance gate:

- Sample command is reproducible from a clean environment with separately obtained sample audio.
- Results remain labelled training-set engineering diagnostics.
- Links on the Project Site resolve to committed artifacts.

## Phase 5: Colab and full-data execution readiness

**Status:** pending.

Deliverables:

- Colab notebook for Drive paths, environment setup, manifest construction, training and artifact export.
- Resume-from-checkpoint support.
- Disk-space and archive-integrity checks.
- Instructions that keep restricted audio and checkpoints outside Git.

Acceptance gate:

- Notebook dry run completes without full audio.
- A small authorized subset can complete train/tune/evaluate end to end.

## Phase 6A: full MedleyDB experiment after approval

**Status:** blocked only on dataset-owner approval.

Execution:

1. Download v1 and v2 outside Git and record archive hashes privately.
2. Audit real audio availability and revise the final classes; strings is provisional.
3. Freeze the artist split and manifest before feature extraction.
4. Train matched mean and attention models; max pooling is optional.
5. Select all thresholds and temporal post-processing on validation artists only.
6. Evaluate held-out artists once with frozen settings.
7. Report clip/frame micro, macro and per-class precision/recall/F1.
8. Evaluate attention as a relative/ranking signal and analyse instrument co-occurrence and failures.
9. Publish aggregate tables and derived figures only.

## Phase 6B: Slakh2100 fallback if MedleyDB remains pending

**Status:** prepared; activation gate is 16 October 2026 at 18:00 Sydney.

Use the duplicate-free `Slakh2100-redux` split as the main quantitative dataset.
Keep the weak-supervision protocol unchanged: train from clip-level labels, hold
aligned MIDI/stem activity out of the training interface, tune on validation
tracks and evaluate the test split once. Replace the artist barrier with a
composition/UUID barrier.

The complete decision rule, implementation checkpoints, evaluation references
and reporting limitations are in `DATASET_FALLBACK_PLAN.md`.

## Deferred extensions

- Strongly supervised frame model as an upper bound.
- OpenMIC-to-MedleyDB transfer.
- Pretrained music/audio embeddings.
- Detailed event onset/offset evaluation.

These begin only after the main weak-supervision experiment is complete.

## Checkpoint log

- 2026-10-04: Project Feedback Two site and two-song engineering pilot already published in commit `7207229`.
- 2026-10-04: Phase 1 committed and pushed as `d8ef6b3`; 10 tests pass.
- 2026-10-04: Phase 2 committed and pushed as `172c04b`; public-sample manifest smoke test found 2/2 mixes and 2/2 activation files while keeping `generalisation_valid=false`.
- 2026-10-04: Phase 3 training/checkpoint engine committed and pushed as `233f330`; validation thresholds are stored in the checkpoint and reused unchanged for evaluation.
- 2026-10-04: Phase 3 validation/train/evaluate CLI committed and pushed as `ad61382`; legacy dry-run configuration validation remains available.
- 2026-10-05: Phase 4 controlled public-sample rerun completed; all 13 tests pass and the README, Project Site and AttentionMIC implementation map were synchronised to the saved artifacts.
- 2026-10-09: Phase 6B contingency prepared. If MedleyDB is still pending at the 16 October gate, Slakh2100-redux becomes the frozen main dataset; the final Project is due 6 November 2026 at 23:59 Sydney.
