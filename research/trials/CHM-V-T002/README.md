# Venture Trial T2

Venture Trial T2 qualifies a separate exploratory proposal policy for the
unchanged Chimera Venture M0 T1 checkpoint. No weights were trained or modified.

## Result

The trial status is `passed`. Validation selected `explore-50`, which mixes the
model distribution with a legal-uniform distribution at rate 0.50.

| Check | Result |
| --- | --- |
| T1 checkpoint SHA-256 | Passed before and after execution |
| Train exact-graph reconstruction | Passed: 99.22% versus 95% threshold |
| Test unique-graph rate | Passed: 94.01% versus 80% threshold |
| Test changed-candidate rate | Passed: 100% versus 95% threshold |
| Test invalid-candidate rate | Passed: 0% versus 1% maximum |
| Test feasibility guardrail | Passed: 0.5875 versus 0.5719 baseline |
| Deterministic replay | Passed |

## Selection

Each policy generated 64 candidates per source case under seeds 1702, 1703 and
1704. Validation was used for selection. Test was opened once for `model-only`
and the selected policy.

| Split | Policy | Unique graphs | Changed | Invalid | Median feasibility | Operations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validation | `model-only` | 12.24% | 100% | 0% | 0.6167 | 3 |
| Validation | `explore-50` | 88.02% | 100% | 0% | 0.6750 | 8 |
| Test | `model-only` | 27.08% | 100% | 0% | 0.5719 | 3 |
| Test | `explore-50` | 94.01% | 100% | 0% | 0.5875 | 8 |

The selected test policy also reached 52.08% mean MAP-Elites coverage versus
31.25% for `model-only`.

## Diagnosis

The committed train-only diagnostic verified that T1 concentrated 45.63% of
programs into `CONNECT>STOP`. Legal-uniform exploration restored all eight
non-terminal operations while every sampled program remained validity-constrained.

## Artifacts

- `diagnostic.json`: train-only regression decomposition and policy sweep;
- `protocol.yaml`: frozen selection and acceptance rules;
- `result.json`: qualification decision and split metrics;
- `validation_candidates.jsonl`: policy-selection outputs;
- `test_candidates.jsonl`: final baseline and selected-policy outputs;
- `policy_bundle.json`: portable inference policy bound to the T1 checkpoint;
- `policy_manifest.json`: bundle size, SHA-256 and release tag;
- `environment.json`: execution environment.

T2 does not evaluate semantic novelty, commercial utility or the CHM-V-H001
language hypothesis.
