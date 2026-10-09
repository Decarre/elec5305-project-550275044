# Dataset fallback plan

**Prepared:** 9 October 2026  
**Final ELEC5305 Project deadline:** 6 November 2026, 23:59 Australia/Sydney  
**Purpose:** preserve a valid weakly supervised temporal-localisation experiment
if the MedleyDB v1/v2 audio requests remain unapproved.

## Decision gates

1. Keep both existing Zenodo requests open. Do not cancel or submit duplicates.
2. Begin the fallback adapter and a tiny-data smoke test immediately; this work
   is useful even if MedleyDB is later approved.
3. If full MedleyDB audio is unavailable by **16 October 2026, 18:00 Sydney**,
   freeze **Slakh2100-redux** as the main quantitative dataset.
4. If MedleyDB arrives after that gate, use it as a real-recording external
   validation or qualitative case study. Do not replace the frozen main split
   close to the final deadline.

This leaves three weeks for full training, evaluation, error analysis and final
reporting after the fallback decision.

## Primary fallback: Slakh2100-redux

Slakh2100 contains aligned mixtures, isolated instrument stems, separated MIDI
and per-track metadata. `Slakh2100-redux` removes repeated compositions so each
underlying MIDI occurs once. Its official split contains 1289 training, 270
validation and 151 test tracks.

Official resources:

- Dataset and tiny subset: <https://www.slakh.com/>
- Structure, metadata and duplicate-safe splits:
  <https://github.com/ethman/slakh-utils>
- Generation details: <https://github.com/ethman/slakh-generation>

Audio and MIDI remain outside Git. Only code, manifests, aggregate tables and
derived figures may be committed.

## Research design preserved

The research question remains:

> When trained using only clip-level instrument labels, how accurately do
> frame-level class scores and instrument-specific attention signals recover
> the true temporal activity of instruments in polyphonic music?

The replacement protocol is:

1. Use `mix.flac` as model input.
2. Map `metadata.yaml` `inst_class` values to the existing families: drums,
   bass, guitar, piano and strings.
3. Create ten-second clips only after applying the official Redux split.
4. Derive one binary clip-level vector from instrument MIDI activity inside the
   clip. The training NPZ contains only log-mel features and this vector.
5. Keep MIDI frame rolls and stem-energy activity outside the training path.
6. Select thresholds and post-processing on validation tracks only.
7. Freeze the checkpoint and thresholds, then evaluate the test split once.

### Temporal reference hierarchy

- **Primary:** aligned per-instrument MIDI note activity on a 0.5-second grid.
- **Secondary:** activity derived from rendered stem energy on the same grid.
- **Diagnostic:** attention mass placed on active reference frames.

MIDI activity is precise for the synthesized performance, while stem energy
better represents audible sustain and release. Reporting both prevents one
reference definition from determining the conclusion.

### Leakage barrier

Slakh has no meaningful artist identity. The MedleyDB artist barrier is replaced
by a composition barrier:

- use the official Redux split;
- record the `redux.json` hash;
- use metadata `UUID` as the composition-group identifier;
- assert that no UUID occurs in more than one partition;
- create clip windows only after the track partition is frozen.

Never use the original Slakh split without duplicate checking. The official
utilities document repeated MIDI files crossing the original partitions.

## Controlled comparison

Keep the current scientific controls unchanged:

- one shared convolutional frame encoder;
- mean pooling over bounded frame probabilities;
- optional max pooling over the same probabilities;
- instrument-specific attention pooling;
- identical features, classes, optimizer search and early-stopping rule;
- clip micro/macro F1 plus frame micro/macro and per-class F1;
- attention weights evaluated separately from frame probabilities.

Run at least three fixed seeds if compute permits. Report individual seeds and
mean plus standard deviation; do not hide unstable runs.

## Implementation checkpoints

### F1 - data audit and adapter

- Parse `train`, `validation` and `test` track directories.
- Validate `mix.flac`, `metadata.yaml`, MIDI and rendered stems.
- Produce class coverage, duration and co-occurrence tables.
- Confirm UUID disjointness and save source archive/split hashes.

Acceptance gate: the tiny subset produces a deterministic audit without loading
temporal targets into the training interface.

### F2 - manifests and weak labels

- Add a Slakh track/clip manifest adapter behind the existing manifest schema.
- Add a Slakh instrument-class mapping separate from the MedleyDB taxonomy.
- Generate clip labels from MIDI presence and separate evaluation references.

Acceptance gate: synthetic fixtures prove clips inherit the parent track split,
UUIDs are disjoint and frame references are absent from training NPZ files.

### F3 - tiny-subset end-to-end run

- Run mean and attention models on the official tiny subset.
- Confirm loss reduction, checkpoint reload and frozen validation thresholds.
- Label all results as engineering diagnostics if the subset cannot support
  adequate class coverage.

Acceptance gate: the complete command chain is reproducible before the full
Redux archive is processed.

### F4 - full Redux experiment

- Freeze final five-class coverage and the official split manifest.
- Extract/cache features, train matched models and select validation settings.
- Evaluate the held-out test split once.
- Run co-occurrence and representative failure-case analysis.

Acceptance gate: aggregate results can be regenerated from a saved config,
manifest hash, checkpoint, thresholds and seed.

### F5 - reporting

- State prominently that Slakh is synthesized audio.
- Avoid claims about generalisation to human performances or studio recordings.
- Use the MedleyDB public sample for qualitative real-audio examples only.
- If full MedleyDB becomes available, add it as external validation and report
  the domain shift separately.

## Secondary datasets

### OpenMIC-2018

Use only for clip-level real-audio recognition or AttentionMIC reproduction.
It provides audio, crowd-sourced instrument labels and official partitions, but
does not provide the frame-level activity reference required for the main
localisation experiment.

Official resource: <https://github.com/cosmir/openmic-2018>

### MUSDB18

Do not use as the primary fallback. It also requires access approval and exposes
only drums, bass, vocals and a heterogeneous `other` stem, so it does not solve
the five-family temporal-localisation requirement cleanly.

Official resource: <https://sigsep.github.io/datasets/musdb.html>

## Minimum viable fallback if storage or compute becomes limiting

1. Use the official Slakh tiny subset to validate the full pipeline.
2. Download/process only a deterministic class-covered subset of Redux tracks.
3. Preserve the official train/validation/test membership and UUID barrier.
4. Reduce model width, epochs or seeds before weakening split integrity.
5. Report the smaller sample size and confidence limits explicitly.

The minimum viable result must still have separate train, validation and test
tracks. A two-song engineering check cannot be presented as generalisation.

## Trigger checklist

When the 16 October decision gate is reached:

- [ ] Recheck both Zenodo requests and the registered email address.
- [ ] If approved, continue Phase 6A (MedleyDB).
- [ ] If pending, activate Phase 6B (Slakh2100-redux).
- [ ] Save the chosen dataset, split definition and hashes before preprocessing.
- [ ] Update README and Project Site only after the fallback smoke test passes.
- [ ] Keep the alternative dataset and its limitations explicit in the report.
