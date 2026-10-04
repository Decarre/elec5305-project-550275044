# AttentionMIC implementation map

This note records which parts of the published AttentionMIC system are used in
this project and which parts are intentionally changed for the ELEC5305
experiment.

## Aggregation reproduced

For every time frame `t` and instrument class `c`, the local attention model
produces:

- a bounded class score `p[t,c] = sigmoid(class_logit[t,c])`;
- a non-negative attention value, normalised over time so that
  `sum_t attention[t,c] = 1`;
- a clip prediction `sum_t attention[t,c] * p[t,c]`.

This follows the aggregation structure in the published
[AttentionMIC code](https://github.com/SiddGururani/AttentionMIC). The local
model exposes frame probabilities, attention weights and clip probabilities
separately so they cannot be confused during temporal evaluation.

## Controlled baseline

The mean-pooling baseline uses the same encoder, class head and bounded frame
probabilities, then computes `mean_t p[t,c]`. The optional max baseline computes
`max_t p[t,c]`. This keeps the compared quantity fixed while changing only the
temporal aggregation rule.

An earlier engineering pilot averaged frame logits before applying a sigmoid.
The saved Phase 4 results replace that pilot with the controlled probability
pooling definition above.

## Deliberate differences

The local implementation uses a compact shared Conv1d frame encoder and 64-bin
log-mel input so it can support a controlled, reproducible course experiment.
It does not claim to reproduce the full published feature pipeline, network
capacity or OpenMIC benchmark result. The main scientific question is whether
the attention aggregation provides useful temporal localisation under matched
conditions.

## Interpretation and evaluation

- `frame_probabilities` are candidate per-frame activity estimates.
- `attention_weights` describe relative contribution to the clip decision and
  are not calibrated activity probabilities.
- their product is a contribution signal and will be evaluated separately.
- all thresholds and temporal post-processing choices will be selected on
  validation artists and frozen before held-out evaluation.

The current two-song public-sample run is a training-set engineering check. It
cannot support an artist-disjoint split or a generalisation claim.
