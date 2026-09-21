# Frozen best baseline and next work

## Authoritative best

- Version: V28
- Leaderboard score: **0.2977**
- Submission: `outputs/leaderboard/submission_v28_v20_qwen3b_fixed.zip`
- Samples: 9,555
- Registry: `outputs/leaderboard/best_baseline.json`

The registry, rather than a directory name or modification time, is the sole source of truth.
Run `python scripts/verify_best_baseline.py` before every future submission experiment.

## Disallowed route

The RefCOCO V7 global reranker is not valid for competition inference. The failed 0.2544
submission changed 2,048 V28 boxes, with a median area expansion of 46.1x. Do not tune
its margin or submit any derived global-reranking variant.

## Experiment policy after V28

All future experiments start from the frozen V28 prediction. V20/V7 outputs may be used
for diagnostics or as candidate sources, but never as the prediction base. Unselected
samples must remain byte-for-byte equivalent to V28.

### Rejected experiment: joint assignment V31

- Automatic competition changes: 119
- RefCOCO ACC: 27.55% to 27.58% (+0.03 pp)
- RefCOCO new correct / new wrong: 6 / 3
- Required gate: +1.00 pp
- Decision: **REJECT; do not submit**

The manually reviewed 26-query derivative also lacks independent validation and is not
accepted merely because its change count is small.

## Next experiment: V28 same-object boundary refinement

Objective: improve boundary IoU without changing the selected object.

Hard gates:

1. V28 is the immutable fallback.
2. A proposed box must overlap the V28 box at IoU >= 0.70.
3. Proposed/V28 area ratio must stay within [0.67, 1.50].
4. The candidate needs agreement from at least two prompts or detector views.
5. Dev and holdout must both improve ACC@0.5 and mean IoU.
6. The competition change set is capped at 50 and exported for visual audit.
7. No submission is created until the offline gate and visual audit both pass.

This experiment directly addresses the observed failure mode: the 0.2544 route selected
nearly disjoint boxes with a median 46.1x area expansion. The new gates make such target
switches impossible by construction.

### Result

The mechanical gate passed, but the decision gate did not. Dev and holdout each gained
only one net hit; only 2/24 parameter configurations were positive on both splits. No
submission was produced.

## Following experiment: V28 replacement precision

Optimize whether to change a V28 prediction, rather than increasing candidate recall.
Use the existing Qwen candidate selector only on an ambiguity subset and require
multi-run agreement before a replacement is accepted.

Acceptance targets:

1. Dev ACC improvement >= 0.10 pp and at least 8 net new correct samples.
2. Holdout ACC improvement >= 0.10 pp and at least 4 net new correct samples.
3. Mean IoU must improve on both splits.
4. At least 70% of changed labeled samples must improve IoU.
5. No query-family subgroup with >=20 changes may have negative net accuracy.
6. Competition changes capped at 50, with unanimous selector agreement and visual audit.
