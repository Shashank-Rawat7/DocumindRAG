# Public release of the E1 evaluation dataset

This repository is a **partial public release** of the frozen evaluation dataset `eval-dataset-v1.0.0`
(frozen on 2026-11-14). It does **not** contain the complete frozen evaluation dataset.

## What is published

- The corpus (`eval/corpus/`), with its manifest and licences.
- The development split (`eval/dataset/dev.jsonl`) and the draft sources of the development questions
  (`eval/dataset/draft/dev_answerable.json`, `dev_unanswerable.json`).
- The labelling guidelines (`eval/dataset/guidelines.md`), the dataset metadata (`eval/dataset/dataset.json`)
  and the freeze record (`eval/dataset/freeze_record.json`).
- The evaluation, labelling, ingestion and pilot implementation, its scripts, and the approved tests.

## What is intentionally withheld

The held-out test split, and every artifact that exposes its questions, answers, evidence or label metadata,
are not published. The project owner keeps them outside this repository.

| Withheld file | Why |
|---|---|
| `eval/dataset/test.jsonl` | the held-out test split (questions, expected answers and evidence) |
| `eval/dataset/audit.jsonl` | the review and audit log; it contains test evidence offsets, correction notes and sampled test question IDs |
| `eval/dataset/draft/test_batch2.json`, `eval/dataset/draft/test_batch3.json` | draft sources of test questions |
| `eval/dataset/draft/human_evidence_review.md`, `eval/dataset/draft/review_packet.md` | review packets that contain test answers and evidence |
| `eval/dataset/draft/README.md` | it exposes test-question IDs together with difficulty and evidence-scope changes |

The private alias map and the auditor's files from the second-pass audit are not part of this repository.

## How the freeze is anchored

`eval/dataset/freeze_record.json` records the SHA-256 hash of the seven frozen files, including the withheld
files, and the label-binding hashes of the development and test splits. The dataset is **hash-anchored**: it is
identified by those hashes, not by a clean Git commit.

A matching hash shows that a file is the frozen file. It verifies the file's identity, not the correctness of its
labels.

## Verifying a file

If you hold a copy of a frozen file, compare its SHA-256 with the value recorded in
`eval/dataset/freeze_record.json`.

PowerShell:

```powershell
Get-FileHash .\path\to\file -Algorithm SHA256
```

Linux:

```bash
sha256sum path/to/file
```

## What a public checkout cannot do

- It cannot reproduce the frozen dataset, because the test split, the audit log and the withheld review
  material are missing.
- Some evaluation tests require withheld private artifacts. A public checkout is therefore not a complete
  evaluation environment, and the full evaluation test suite is not expected to pass without restoring the
  private artifacts.

## Status of the E1.7 pilot

The E1.7 pilot was a validity pilot. It is **not** the official V1 baseline, and the official baseline has not
been created yet.

## Limitations of the frozen dataset

- The second-pass audit was non-independent.
- The Hindi review was done by a single reviewer.
- The 10 unanswerable questions have AI-only semantic review provenance.
