# Chimera Venture Corpus C0

Corpus C0 is the first source-grounded structural dataset for Venture M0.

| Item | Value |
| --- | ---: |
| Canonical business graphs | 10 |
| Training transitions | 384 |
| Validation transitions | 128 |
| Test transitions | 128 |
| Total transitions | 640 |
| Node capacity | 64 |
| Edit capacity | 8 |

The ten graphs are manually structured from public SEC filings. Each graph is
deterministically corrupted through edge removal, relation inversion or node
type substitution. The target edit program restores the canonical graph.
Variants of one company never cross split boundaries.

## Files

| File | Purpose |
| --- | --- |
| `source_graphs.yaml` | Human-auditable annotations and immutable SEC accessions |
| `canonical_graphs.jsonl` | Resolved features plus language sidecars |
| `records.jsonl` | Record IDs, split membership and corruption provenance |
| `train.npz` | Numeric-only training tensors |
| `validation.npz` | Numeric-only validation tensors |
| `test.npz` | Numeric-only test tensors |
| `manifest.json` | Counts, hashes, split boundaries and claim boundary |
| `quality_report.json` | Reproducible integrity and distribution profile |
| `target_quality_report.json` | Input-target ambiguity and loss-field relevance |

## Primary Sources

- [Adobe 2024 Form 10-K](https://www.sec.gov/Archives/edgar/data/796343/000079634325000004/adbe-20241129.htm)
- [Airbnb 2024 Form 10-K](https://www.sec.gov/Archives/edgar/data/1559720/000155972025000010/abnb-20241231.htm)
- [Costco 2024 Form 10-K](https://www.sec.gov/Archives/edgar/data/909832/000090983224000049/cost-20240901.htm)
- [Visa 2024 Form 10-K](https://www.sec.gov/Archives/edgar/data/1403161/000140316124000058/v-20240930.htm)
- [Amazon 2024 Form 10-K](https://www.sec.gov/Archives/edgar/data/1018724/000101872425000004/amzn-20241231.htm)
- [Marriott 2024 Form 10-K](https://www.sec.gov/Archives/edgar/data/1048286/000162828025004818/mar-20241231.htm)
- [Netflix 2024 Form 10-K](https://www.sec.gov/Archives/edgar/data/1065280/000106528025000044/nflx-20241231.htm)
- [Tesla 2024 Form 10-K](https://www.sec.gov/Archives/edgar/data/1318605/000162828025003063/tsla-20241231.htm)
- [Uber 2024 Form 10-K](https://www.sec.gov/Archives/edgar/data/1543151/000154315125000008/uber-20241231.htm)
- [Deere 2024 Form 10-K](https://www.sec.gov/Archives/edgar/data/315189/000155837024016169/de-20241027x10k.htm)

Only structured facts, accessions and original annotations are committed. SEC
filing text is not redistributed.

## Rebuild

```powershell
chimera build-corpus
chimera validate-corpus
python scripts/profile_venture_corpus.py --output datasets/venture_corpus_c0/quality_report.json
python scripts/profile_venture_targets.py --output datasets/venture_corpus_c0/target_quality_report.json
```

The committed shards contain no source text, labels or language embeddings.
Human-readable metadata is isolated in JSONL and YAML sidecars.

The target-quality profile found no conflicting target graphs and no input
overlap across splits. Exact target-graph reconstruction is identifiable.
Registered edit-program reconstruction is not fully identifiable because a
small number of identical inputs have alternative valid programs. Operation-
conditioned argument masks are required because 41.4% of raw argument slots are
placeholders unused by their operation.

## Claim Boundary

Corpus C0 teaches structural grammar through denoising. It does not establish
novelty, commercial usefulness, causal business performance or superiority to
a language model.
