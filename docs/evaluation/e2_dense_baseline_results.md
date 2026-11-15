# E2 results note — retrieval-only dense baseline (dev split)

Metrics-only note for checkpoint E2.6. Values and claims in this note are drawn from the E2.5 report,
frozen development inputs, freeze record and operator-reported execution evidence. Their sources are
identified in the relevant sections. Nothing here comes from the held-out test or audit splits.

> **Scope.** This is a *retrieval-only* dense baseline on the *development* split. It is **not** the
> full V1 RAG baseline (that belongs to C11) and it says nothing about generation, answer quality,
> latency, cost or held-out performance. The report labels itself
> "Retrieval-only dense V1 baseline (dev split) — not the full V1 RAG baseline", with
> `is_v1_baseline = false`.

## 1. Purpose

Establish a first, reproducible reference for how well dense retrieval (one embedding model, no
re-ranking, no hybrid search, no LLM) finds the labelled evidence of the 26 answerable dev
questions in the six-document evaluation corpus. Later retrieval changes (hybrid retrieval and so on)
are to be compared against this reference using the same frozen inputs.

## 2. Run identity and provenance

| Item | Value | Source |
|---|---|---|
| Report kind / format | `retrieval_baseline` / `e2.4-report-1` | report |
| Run time (UTC) | 2026-11-15T11:19:21Z | report |
| Code commit | `c06374796dc408366827225c38278132dda9301d` (`c063747`), tracked working tree clean | report |
| Python | 3.12.4 | report |
| Code identity (combined SHA-256 of 25 modules) | `0846bbb8c394d3b25f2758756c4814c6fe6373a464cb6090db2fa9c061aeda26` | report |
| Embedder | `intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3` | report |
| Embedder setup | CPU, float32, `sdpa` attention, 384 dimensions, mean pooling, L2 normalisation, prefixes `query: ` / `passage: `, torch 2.15.0+cpu, transformers 5.22.1 | report signature |
| Chunking / index build | chunker `c2.1-v3` (target 300, max 400, overlap 50 tokens); index build `c2.5-v1` | report signature |
| Retrieval | dense, top 20 per question; cutoffs 1, 3, 5, 10, 20 | report settings |
| Random reference | 1,000 seeds (0..999), algorithm `seeded_partial_fisher_yates_rejection_v1` | report settings |
| Statistics | bootstrap seed 20261114, 10,000 resamples, 95% intervals, nearest-rank percentiles | report settings |
| Command | `python scripts\e24_run_baseline.py` with the database name, the evaluation owner, the evaluation-email digest, the freeze-record SHA-256 and `--out` (no `--adjudicate`) | operator; not recorded in the report |
| Exit code | 0 | operator; not recorded in the report |

The evaluation-email digest and the database credentials are deliberately not recorded here.

## 3. Frozen inputs

| Input | Identity |
|---|---|
| Dataset version / corpus version | `eval-dataset-v1.0.0` (frozen 2026-11-14) / `eval-corpus-v1` |
| Freeze record (raw file SHA-256) | `5c4a5a05b62337f17ec9148c3d5b19c2e960ab51815c19906815e5ed40a3a47d` |
| Dev file `dev.jsonl` (raw SHA-256) | `09169e4f0bef3520fcbee1643af7d181681293201b76064f991744504a8795c0` |
| Dev label-binding hash (the report's `dev_dataset.sha256`) | `860078e2fdb8bfabcfaddc4f184eb6af941d52bfb004f8a61d8a7f6b8416951c` |
| `dataset.json` (raw SHA-256) | `3f23c286be3f4010d9595b971ebc1c9ae05efc882d6ec4406014c45a356436fa` |
| Manifest file (raw SHA-256) | `1902c8cc1491cb3239617edc88fb70c401964bf46bb3ce4e3b9ee6edc88e6422` |
| Canonical manifest hash | `7f6d7473e5b632a4ba09f980c671f6970597e0b836e9a5a27875f360234c9cc9` |
| Candidate pool | 176 chunks, SHA-256 `8227a96c9085365dac12e565034032390f94c73fefc3dae991730d2429478acc` |
| Corpus | 6 documents (manifest stage `ingested`) |
| Dev questions | 26 answerable (27 required evidence units) and 10 unanswerable |

The report records `test_split_read = false`. The 10 unanswerable questions were retrieved only
for score diagnostics; no retrieval metric is computed or claimed for them.

## 4. Evidence classes

**Hard-gate results** (a failure of any of these prevents the report or makes the CLI exit non-zero;
all passed). Each item is marked by its source: *(report)* means the E2.5 report records it;
*(implementation)* means it is enforced by the E2.4 CLI, as verified in its code and tests, and is
not recorded in the report. The exit code 0 is operator-reported.

- **Freeze identity.** *(report)* The report records the freeze identity: the freeze-record SHA-256
  and the input hashes bound to the run (section 3). *(implementation)* The comparison with the
  externally supplied SHA-256 and the re-verification of the freeze record immediately before the
  report was rendered are enforced by the CLI and are not recorded in the report.
- **Git state.** *(report)* The report records the run commit and `working_tree_clean = true`.
  *(implementation)* The tracked-state recheck immediately before writing is enforced by the CLI and
  is not recorded in the report.
- **Ranking determinism (D7)** *(report)*: two complete retrieval passes in one read-only transaction
  produced identical answerable rankings. Ranking SHA-256
  `446a275fff051e59fc140b3b104bae839e734d1e2f6bd45b2530f880393105d7` (report:
  `passes = 2`, `rankings_identical_across_passes = true`).
- **Database snapshots** *(report)*: the before and after fingerprints are identical
  (`b95747bf4807d2e4d2b94b1cfb1214e596b118cb86cc3e7714b5658c56f3a6dd`).
- **Oracle validity** *(report)*: `ok = true` (method `deterministic_greedy_reference`, rule `half_coverage`);
  26 checks, 0 adjudicated units, 0 unadjudicated units, 0 units unreachable under `half_coverage`;
  every question had Hit@1 = 1.0 and Unit Recall@20 = 1.0 for the oracle.

**Diagnostic evidence** (descriptive only, not a gate):

- Score digests identical across the two passes
  (`361d41fb2a3e7ac8a6155da22f3aea89fcdef3e00203d3f444ce4d06928c7d74` for both;
  `scores_identical_across_passes = true`). A mismatch would have been reported but would not have
  blocked the report.

**Observed results:** section 6. **Interpretation:** section 7.

## 5. Metrics (primary reading: `raw` family, `half_coverage` rule)

Definitions are those of `app/rag/evaluation/metrics.py` (E1.3.1) and `statistics.py`; they are
summarised here, not redefined.

- **Evidence structure.** A question's gold evidence is a set of *units*. All units are required
  (AND); a unit is satisfied when any one of its alternative spans is satisfied (OR).
- **`half_coverage` relevance.** A chunk is relevant to a gold span when its overlap with that span
  covers at least half of the span (exact integer arithmetic; overlap from several chunks is never
  added together).
- **Raw family.** Metrics are computed on the ranking exactly as retrieved, not on the
  distinct-passage collapse.
- **Hit@k.** 1 if at least one required unit is satisfied by the first k chunks, else 0.
  Hit@k therefore does **not** mean the whole evidence was retrieved.
- **Unit Recall@k.** Satisfied units divided by required units within the first k chunks, computed per
  question. Its denominator is therefore the question's own unit count, not 26: 25 of the 26
  questions have one required unit and one has two (27 units in total), and every unit has exactly
  one span. For single-unit questions Unit Recall@k equals Hit@k.
- **Aggregation.** Macro average over the 26 answerable questions (the mean of the 26 per-question
  values; for Unit Recall this is the mean of per-question fractions, not satisfied units over total
  units). Hit@k carries a 95% Wilson score
  interval; Unit Recall@k a 95% percentile bootstrap over questions (10,000 resamples, seed
  20261114). Both describe this sample only; questions about the same documents are not
  independent.

## 6. Observed results (dev split, n = 26)

| Metric (raw, `half_coverage`) | Dense | 95% interval | Oracle reference | Random reference (1,000 seeds) |
|---|---|---|---|---|
| Hit@1 | **0.577** (15 / 26) | 0.389 – 0.745 (Wilson) | 1.000 | mean 0.007; per-seed 2.5th–97.5th percentile 0.000 – 0.077 |
| Hit@5 | **1.000** (26 / 26) | 0.871 – 1.000 (Wilson) | 1.000 | mean 0.036; per-seed 2.5th–97.5th percentile 0.000 – 0.115 |
| Unit Recall@5 | **1.000** | 1.000 – 1.000 (bootstrap; degenerate, every question scored 1.0) | 1.000 | mean 0.035; per-seed 2.5th–97.5th percentile 0.000 – 0.115 |

Counts behind the table: Hit@1 = 15 of 26 questions; Hit@5 = 26 of 26 questions; Unit Recall@5 =
the mean of 26 per-question values, each 1.0 (27 of 27 required units satisfied, so the unit-level
and the question-level averages agree). Of the 26 questions, the first relevant chunk is at rank 1
for 15, rank 2 for 8, rank 4 for 2 and rank 5 for 1. The values were recomputed independently from
the report's stored rankings and the gold spans of `dev.jsonl` (separate implementation of the
half-coverage rule, not the repository code) and matched the report for every question.

Reported slices with at least 10 questions (descriptive only):

| Retrieval type | n | Hit@1 | Hit@5 | Unit Recall@5 |
|---|---|---|---|---|
| `en_to_en` | 16 | 0.625 | 1.000 | 1.000 |
| `hi_to_en` | 10 | 0.500 | 1.000 | 1.000 |

The `hi_to_hi` and `en_to_hi` slices have no questions and are suppressed. The report holds the
other cutoffs, rules, distinct-passage metrics and MRR; this note does not summarise them. The
oracle is a deterministic greedy reference, not a proven upper bound, and the report states that no
dense-versus-random pass/fail threshold is approved or applied: the reference rows are context, not a
criterion.

## 7. Limitations and interpretation

Interpretation (not an observation): on this sample the dense retriever surfaced a chunk relevant to
at least one required unit within the top 5 for all 26 questions, and all required units within the
top 5 for every question, while the top-ranked chunk was relevant for 15 of 26. Over the same
176-chunk pool the random reference is near zero at these cutoffs.

Do not read more into it than that.

- **Small sample.** 26 answerable questions (27 units). The Hit@1 interval spans roughly 0.39 to
  0.74. A Hit@5 of 1.000 has a Wilson lower bound of 0.871, and the degenerate Unit Recall interval
  reflects the sample, not certainty.
- **Single-unit dominance.** 25 of 26 questions have one required unit with a single span, so Unit
  Recall@5 carries almost no information beyond Hit@5; only one question tests retrieval of more than
  one unit.
- **Thin margin at k = 5.** One question has its first relevant chunk at rank 5. If that chunk moves
  below rank 5 without another relevant chunk entering the top five, Hit@5 would decrease. Stability
  under alternative embeddings, chunking or corpus configurations was not tested.
- **Question provenance.** All 26 answerable dev questions are LLM-assisted (none used retrieval
  assistance for labelling). The dev dataset carries direct/paraphrase annotations (13 each among the
  26 answerable questions), but this evaluation does not analyse them as a measure of wording overlap,
  so wording overlap with the evidence is not quantified here.
- **Small, easy-to-saturate pool.** The pool is 176 chunks from six documents, so the top 5 is about
  3% of the pool. Saturation at k = 5 is not evidence of performance on a larger or noisier corpus.
- **Not held-out.** This is the development split, which the E1.7 validity pilot also used. It
  establishes nothing about the held-out test split and nothing about generalisation or production
  readiness.
- **Retrieval only.** No generation quality, answer faithfulness, latency, cost or hybrid/re-ranked
  retrieval is measured; this is not the V1 RAG baseline.
- **Label provenance.** The labels carry the disclosures recorded in `freeze_record.json` (including
  a non-independent second-pass audit and a single Hindi reviewer); they were not independently
  re-verified for this note.
- **Language coverage.** Only `en_to_en` and `hi_to_en` have questions; the Hindi-corpus slices are
  empty.
- **Determinism scope.** D7 compares two passes inside one process on one machine and one database
  state; it does not show cross-machine or cross-library-version reproducibility.

## 8. Reproducibility

- **Report file** (stored outside the repository, not committed):
  `E:\DocuMind_eval_reports\e25_dense_baseline_report.json`
- **Size:** 1,454,659 bytes. **SHA-256:**
  `cfb3be7829c351f0947d956f6231adf6d98fc32a21e884d2d84567ce3ce4ee99`
- **Verify the file:**
  `(Get-FileHash E:\DocuMind_eval_reports\e25_dense_baseline_report.json -Algorithm SHA256).Hash.ToLower()`
- **Load and validate** from the project root with
  `app.rag.evaluation.report.load_report(...)`, which recomputes the ranking hash from the report's
  own rankings, and check `oracle.validity.ok` separately (a loader accepts an oracle-invalid
  report).
- **Re-run** from the project root with `scripts\e24_run_baseline.py` at commit `c063747` or a
  descendant that leaves the evaluation code unchanged, the frozen inputs above, the pinned embedder
  in the local Hugging Face cache and the same database state (compare the snapshot fingerprint in
  section 4). A new `--out` path is required; the tool never overwrites a report.

## 9. Not established by the report

- The process exit code, the exact command line and the wall-clock duration are not recorded in the
  report (the exit code and command are operator-reported).
- Hardware beyond "CPU", operating system version and library versions other than those in the
  embedder signature are not recorded.
- Nothing about unanswerable-question behaviour beyond the descriptive score diagnostics in the
  report, which this note does not interpret.
