# Evaluation dataset: labelling guidelines (E1.6)

These rules describe how questions, expected answers and evidence are labelled for the DocuMind
retrieval evaluation. They restate the approved E1.1 contract and the checks that the code already
enforces (`app/rag/evaluation/dataset.py`, `anchoring.py`); where a rule here is only guidance, it says so.

Files, all in this folder: `dev.jsonl`, `test.jsonl`, `dataset.json` (dataset metadata: five fields, created 2026-11-14 for `eval-dataset-v1.0.0`; see section 10),
`audit.jsonl` (review and audit log, append-only) and this file. The corpus is the six documents in
`eval/corpus/`, identified by `corpus_key`, never by a database id.

## 1. What is labelled

| Kind | Split | Content |
|---|---|---|
| Answerable | `test` (50, locked) or `dev` (20 to 30) | A question, an expected answer, and one or more **evidence units** |
| Unanswerable | `dev` only (about 10) | A question that the corpus does not answer, with a type and a semantic-review record |

An answerable question has 1 to 20 evidence units. A **unit** is one fact that a correct answer needs
(all units are required). A unit has one or more **spans**; the spans of a unit are alternative places that
each state that fact (any one is enough). Do not put two different facts in one unit, and do not split one
fact over two units.

## 2. Writing a question

1. Write it yourself from the document text. Never copy a question or an answer from DocuMind's own output,
   and do not look at retrieval results before writing an answerable question.
2. It must be answerable from the corpus alone, in one language, with no outside knowledge.
3. Make it self-contained: no "this page", "above" or "the document".
4. One question, one answer target. Avoid yes/no questions and questions with a list of unrelated parts.
5. `difficulty`: `direct` if the question reuses the wording of the evidence, `paraphrase` if it asks for
   the same fact in clearly different words.
6. `close_distractor`: `true` if another passage of the corpus is plausibly confusable with the evidence
   (same topic, a different value). Judge by reading, not by running retrieval.
7. Do not repeat a question. Exact repeats (after lower-casing and collapsing whitespace) are errors.
   Very similar questions are warnings that a person must look at.
8. Dev and test must not share evidence. A test span may not overlap, or touch, any dev span of the same
   document (the validator enforces this). Author the test split first; draw dev evidence from other places.

## 3. Writing the expected answer

A short, complete statement of the answer in your own words or in the document's words. It is for reviewers
and for the later judge audit. Retrieval metrics use the evidence spans, not the answer text.

## 4. Choosing evidence

Evidence is a **quote**: the exact characters of the document's extracted text. The tool finds the quote, and
the labeller never types an offset.

1. Quote the smallest passage that states the fact completely, usually one sentence or one table row.
   A quote must be non-empty. Do not start or end in the middle of a word.
2. Copy the quote exactly: punctuation, case, the ligatures that the extracted text contains (for example `ﬁ`),
   line breaks and spacing. Nothing is normalised. If the quote is not found the tool refuses it.
3. Use `show` to read the document slice and `show --find` to see how often a text occurs.
4. If a quote occurs more than once the tool refuses it. Either lengthen the quote until it is unique, or
   name the occurrence (`"occurrence": 2`, counting from 1, overlapping matches included) after checking
   with `show --find` that it is the intended place.
5. A unit may list several spans when the same fact is genuinely stated in several places. Do not list a
   second span merely to enlarge the target. Two spans of one unit may not be identical. The same span in two
   units is a warning that a person must justify.
6. All spans of one question must come from documents of one language, and `source_language` must match.
   (The current corpus is English only.)
7. Never edit an offset, a hash or a quote in `dev.jsonl` or `test.jsonl` by hand. Every offset and hash is
   produced by the tool from the published manifest. If the corpus or the manifest changes, labels become stale
   and `check` fails: they are rebuilt from their quotes, never relocated silently.

## 5. Languages

`query_language` is the language of the question and `source_language` the language of the evidence.

| `retrieval_type` | Possible now |
|---|---|
| `en_to_en` | yes |
| `hi_to_en` | yes (Hindi question, English evidence) |
| `en_to_hi`, `hi_to_hi` | no: the corpus has no Hindi document; they are never faked |

A slice is reported only with at least 10 questions. The mix follows what the corpus supports and is not
balanced artificially.

A Hindi question must be written in Hindi (Devanagari), not transliterated. It must mean the same as an
English reading of the same fact. A named Hindi reviewer must check the wording and the meaning; the tool
records `hindi_reviewed: true` and that reviewer only when the person running it gives `--hindi-reviewer`.
An English question never has a Hindi reviewer.

## 6. Unanswerable questions

Two types:

* `off_topic`: nothing in the corpus relates to the question.
* `on_topic_not_covered`: the corpus is about the subject, but no passage answers this question.

Write each one so that a careful reader could not answer it from the corpus. Its semantic review is recorded either by a named human second reader (`--semantic-reviewer`) or as AI-only provenance (`--ai-semantic-basis`, an `ai_semantic_review` event; see section 7). Either a human semantic review or AI-only semantic provenance satisfies the semantic-provenance requirement for promotion; the separate human evidence review (`--evidence-reviewer`) is still required. Unanswerable
questions are always in `dev` and are reported separately from the answerable metrics.

If retrieval output was used at any point while checking one (for example to look for a covering passage),
record it in `retrieval_assistance` (date, method, run reference if any, a short description). Never use
retrieval output to write an answerable question.

## 7. Author and reviewer fields

* `author_type`: `human` if a person wrote the question and the evidence; `llm_assisted` if a model drafted or
  proposed any part of either. Be honest: when unsure, use `llm_assisted`.
* **A human reviews every evidence label** before it enters the dataset. The reviewer's name is given with
  `--evidence-reviewer`, and the tool writes it to `audit.jsonl`. A question with no current review is reported by
  `readiness` and cannot be promoted; `check` does not fail it (see "Check, readiness and promotion" below).
* Reviewer names are never read from a spec file, so a draft cannot claim a review it has not had.
* A reviewer name must be the name of a person. A blank name, an obvious placeholder (`DRYRUN`, `TBD`, ...) or an AI
  name (`ChatGPT`, `Claude Sonnet`, `AI Assistant`, ...) is refused by the tool and is an error in `check`. A single
  word is a valid name; no surname is required. **A name is attribution only.** The tool does not verify who
  typed it, and a recorded name is not proof of identity; it is a guard against obvious placeholders. A review by a model is an AI-assisted review: it is never recorded as a
  human review, and a question it reviewed stays `llm_assisted`.
* **Hindi review.** The tool never marks a question Hindi-reviewed because of its language. A Hindi question is
  added only when the person running the tool names the native speaker who reviewed its wording
  (`--hindi-reviewer`), which sets `hindi_reviewed: true` and writes a `hindi_review` event. Until that review
  has happened the question stays out of the dataset files; nobody is named for a review that has not happened.
  Evidence review (`label_review`) and Hindi review (`hindi_review`) are separate events and separate facts.
  The `hindi_review` event takes the date of `--date`, or of `--hindi-review-date` when the native speaker reviewed the
  wording on a different day from the evidence (or human semantic) review. That option changes only the date of the
  `hindi_review` event, is checked like `--date`, and is refused when no question in the batch is a Hindi question.

### Check, readiness and promotion

* `check` is **structural**: schema, corpus references, offsets and hashes, event schemas, id uniqueness and split
  integrity. It never fails a record only because a human review is pending, and it never marks anything reviewed.
* `readiness` reports review state and changes nothing: questions with no current human evidence review, stale
  reviews (with the id and reason), Hindi questions with no native-speaker review, and unanswerable questions
  with no semantic-review provenance at all. Questions whose only semantic review is an AI review are listed as informational and do not block readiness.
  It exits 1 while any evidence review is missing or stale, any Hindi review is pending, or any unanswerable question has no semantic-review provenance.
* **Promotion** (`add-answerable`, `add-unanswerable`) enforces the human reviews: a real `--evidence-reviewer`, a
  named native speaker for every Hindi question, and, for unanswerable questions, either a human semantic reader (`--semantic-reviewer`) or AI-only provenance (`--ai-semantic-basis`). A Hindi
  question cannot be promoted without a recorded native-speaker review.
* An evidence review is bound to the question as it was reviewed. The `label_review` event carries
  `content_sha256`, a hash of the question text, expected answer, evidence units and span offsets, source hashes
  and quote hashes. If any of these changes the review is **stale** and `readiness` reports it. `difficulty`,
  `close_distractor` and `notes` are not covered. A later `label_review` supersedes an earlier one; use
  `record-evidence-review --questions ID... --evidence-reviewer NAME` once a person has re-read the questions
  as they now are. Earlier events are never changed. A review recorded before this field existed has no hash; it is
  treated as stale only when a later `label_correction` records an evidence change for that question.
* **AI semantic review.** `semantic_review` in a record is for a **human** second reader only (reviewer, date,
  result) and is optional for an unanswerable question. An independent AI review is never written there. It is the
  `ai_semantic_review` audit event, the authoritative provenance record, written either by
  `record-ai-semantic-review --questions ID... --basis "where it is documented"` or, at promotion, by
  `add-unanswerable --ai-semantic-basis "..."` (instead of `--semantic-reviewer`). The event has no reviewer, no model
  and no date unless a date is given (never guess one), and `human_review` is always `false`. An AI-reviewed record
  stores `semantic_review: null`. Promotion needs a human `--semantic-reviewer` or an `--ai-semantic-basis`; a human
  second reader is not required. `check` fails a record whose `semantic_review.reviewer` is a placeholder or AI name
  (`SEMANTIC_REVIEWER_NOT_A_PERSON`). `readiness` lists AI-only questions separately from human-reviewed ones (informational: they do not block readiness), and
  lists a question with neither as having no semantic-review provenance (structurally valid, not reviewed).
* **When an old review is stale.** A review with `content_sha256` is stale exactly when the hash no longer matches
  the question. A review without one (the original 14) cannot be compared, so it is stale when a later
  `label_correction` for that question records an `evidence` change, and not otherwise. Corrections that change only
  `difficulty`, and corrections logged before the review, do not stale it. Limit: an edit that was never logged as a
  correction cannot be detected for a hashless review. The fix is a new review (`record-evidence-review`), which
  carries a hash.

### Events in `audit.jsonl`

The log is append-only. Every line has an `event`; any other event type, or a malformed event, makes `check` fail.

| `event` | Action | Required fields |
|---|---|---|
| `label_review` | **Human**: a named person reviewed the evidence labels of the question | `question_id`, `reviewer`, `date`, `author_type`; optional `content_sha256` |
| `hindi_review` | **Human**: a named native speaker reviewed the Hindi wording of the question | `question_id`, `reviewer`, `date` |
| `ai_semantic_review` | **AI, not human**: provenance of an independent AI review of an unanswerable question | `question_id`, `review_type` (`independent_ai_semantic_review`), `human_review` (`false`), `result`, `basis`; optional `date`, `note` |
| `label_correction` | **Not a review.** A change made to a record after it was reviewed | `question_id`, `date`, `changes`, `basis`, `review_type`, `human_review` |
| `audit_sample` | Procedure record of the second-pass audit draw (section 9); the draw is seeded, the audit itself was not blind | written by `audit-sample` |
| `audit_result` | The auditor's comparison for one question (section 9). `outcome` is derived by the tool and stays `disagree` even where a person later adjudicated it | written by `audit-record` |

`label_correction` has no command; the person applying the correction appends it. `changes` is a non-empty object
with `difficulty` (`{"from", "to"}`, two different permitted values) and/or `evidence` (a description of the
evidence change). `human_review` must be `true` or `false`: `false` means no person reviewed the correction, and
then there is no `reviewer`; `true` needs a `reviewer` that is a person. `note` is optional. A correction does not
extend an earlier review to what it changed: a question whose evidence was corrected needs a new evidence review
before it counts as reviewed for the changed evidence. `check` also compares the latest `difficulty` correction
with the record and fails if they disagree. Never edit or delete an earlier event.

## 8. Adding questions

Questions are written in a JSON spec (one object or an array) and added in batches:

```json
{
  "question_id": "t-compose-health-01",
  "split": "test",
  "question": "…",
  "query_language": "en",
  "author_type": "human",
  "expected_answer": "…",
  "difficulty": "direct",
  "close_distractor": false,
  "units": [
    {"unit_id": "u1", "spans": [{"corpus_key": "docker-compose-startup-order", "quote": "…"}]}
  ]
}
```

`question_id` matches `[a-z0-9][a-z0-9_-]{0,63}` and is unique across both splits. `source_language` is
optional (derived from the documents). An unanswerable spec has `unanswerable_type` and optionally
`retrieval_assistance` instead of the answer fields. A batch is added all together or not at all, and a batch
that makes `check` fail changes nothing. Run `check` after every batch; errors must be fixed, warnings reviewed.

## 9. Second-pass audit

After labelling, a sample is drawn for a second-pass audit. **The E1.8 audit was a non-independent second pass by
`Shark`, who had already seen the labels, expected answers, evidence and earlier review decisions. It is a consistency
check of the labels, not an independent or blind validation, and its results must not be described as one.** The
bundle withholds the labels and answerability from the files the auditor receives, but it cannot hide them from a
person who wrote or reviewed them. A truly independent audit would need an auditor who has seen none of the labels, expected answers,
evidence spans, difficulty labels, review decisions or review documents. The sample is drawn only after every question of
the dataset is in `dev.jsonl` or `test.jsonl` and `check` passes. From the draw until the audit is adjudicated, labels
change only through an adjudication (step 6).

1. `audit-sample --seed N --out QUESTIONS --private-dir PRIVATE` draws 20% of the answerable questions (at least 10, or all
   if fewer) plus **every** unanswerable question, with a recorded seed, and records one `audit_sample` event that holds
   the **real** question ids. Only one sample is recorded. It writes two things, which must be in different places:
   * `QUESTIONS`: for each sampled question only an opaque audit id (`A01`, `A02`, ...), the question text and the query
     language, in a shuffled order that follows neither the question ids, the split nor whether a question is
     answerable. The expected answer, answerability, unanswerable type, difficulty, retrieval type, evidence, reviewers,
     dates and real question ids are never in it.
   * `PRIVATE/alias_map.json` (**private**): audit id to real question id, and neutral document id (`D01`, `D02`, ...) to
     real corpus key, bound to the recorded sample by a hash of its ids. `PRIVATE` must not exist: the tool creates it
     with access for the current account only and **verifies** that (Windows: inheritance off and a single Allow rule for
     this account's SID, read back with `Get-Acl`; elsewhere: owner-only mode bits). If the restriction cannot be applied
     and verified the folder is removed and nothing is written, so never point `--private-dir` at a FAT/exFAT drive or a
     synced folder. It is never given to the auditor, never printed, and refused inside the dataset or corpus folder.
     Keep it, and a backup copy that is also private, until the results are recorded: without it the audit ids cannot be
     translated.
   * The sample is recorded last. The tool never deletes audit files; see "Interrupted sampling" below.
2. `audit-bundle --questions QUESTIONS --private-map MAP --out-dir BUNDLE` builds the auditor's folder: the instructions,
   the questions, the manifest-verified extracted corpus text files under neutral names (`D01.txt`, `D02.txt`, ...: the
   corpus keys name the topics, so they stay in the private map), a document list (neutral key and language only), a blank
   response worksheet and a declaration. The folder is validated before it is moved into place, and a failed validation deletes it: its files must
   be exactly the expected ones, equal to what the generator produces, free of real question ids, internal label names,
   reviewer names and hashes, and in an order that does not reveal split or answerability. The map is never copied into it.
3. The auditor, working only from the bundle, finds the evidence from the corpus text alone and writes it as exact quotes,
   with a status (`Answered`, `Not answered`, `Unsure`), a confidence and notes in the worksheet. The status, confidence
   and notes are kept with the audit documentation; the audit log has no field for them.
4. `audit-record --spec SPEC --auditor NAME --alias-map PRIVATE/alias_map.json` resolves the auditor's quotes with the
   same anchoring, translates audit ids and document ids (`D01` ...) through the map and records one `audit_result` per
   question under its **real** id, with spans in the real corpus keys. The spec is produced from the auditor's worksheet
   by a temporary script (there is no permanent converter); quotes are normalised only
   for line endings, never the corpus text. For the E1.8 audit that conversion was done mechanically by Claude, an AI
   model; see "Outcome of the E1.8 second-pass audit" below for exactly what it did and did not do. Each audit id
   may appear once. They **agree** when every required unit has a labelled span that overlaps one of the auditor's spans,
   and every auditor span overlaps some labelled span. An unanswerable question agrees only if the auditor found nothing.
   An empty `spans` list means "no supporting passage".
5. A disagreement, and every `Unsure` the auditor reported, is reviewed by a person. The decision is recorded as a
   `resolution` by recording that question's result again (latest result wins). An unresolved disagreement makes `check`
   fail; a sampled question without a result is a warning.
6. If a label was wrong or incomplete, remove that question's line from its file and add the corrected question
   through the tool (offsets are never edited by hand). It gets a new evidence review, and its audit result is recorded
   again from the auditor's preserved quotes so that it is compared with the corrected label. Then run `check` again.

The log keeps the seed, the auditor, what each side found and every resolution. The auditor questions and the worksheet
are kept outside the dataset folder; the private map is not part of the dataset.

### Outcome of the E1.8 second-pass audit (2026-11-14)

* **Sample and auditor.** Seed `20261114`: 16 of the 76 answerable questions and all 10 unanswerable questions, 26 in all.
  Auditor `Shark`; non-independent second pass (see above).
* **Machine outcomes.** The log holds 26 `audit_result` events: **21 `agree` and 5 `disagree`**. The five disagreements
  are audit ids A06, A07, A11, A13 and A15 (the real question ids are in `audit.jsonl`); all five are answerable test
  questions. The `disagree` outcomes are the machine results and were **not changed**: a person adjudicated each one,
  recorded the decision as its `resolution`, and kept the canonical label. No label, question or evidence span was changed
  by the audit. The audit log is 159 events in all (the 133 earlier events, including the `audit_sample` event, plus the 26 results).
* **Why they disagreed.** A06 and A13: the auditor's passage supported the complete answer but did not overlap the labelled
  span (A13's passage is adjacent to it, with no character in common). A15: the auditor's quotation overlapped only one of
  the two labelled evidence units. A07 and A11: the auditor reported `Unsure` and recorded no evidence (an empty `spans` list,
  which is always a machine disagreement for an answerable question). In both cases the resolution says the passage could not be
  located or did not establish the answer, that this was unable to be independently verified, and that it does **not**
  establish that the answer is absent from the corpus; the answerable label was retained.
* **What the audit log cannot hold.** It has no field for the worksheet status (`Answered`, `Not answered`, `Unsure`),
  confidence, notes, the non-independence of the audit, or the identity of the person who adjudicated. The `Unsure` status of
  A07 and A11 exists only in their `resolution` text and in this documentation. Results must be reported as "21 of 26 machine
  agreements; 5 disagreements, each adjudicated by a person", never as 26 of 26.
* **Specification.** The spec used for the recorded results is `audit_spec_v2.json`, SHA-256
  `c46a61d7c1ed69f3d726433bfcb0d607a554b9a20b6fe3bb99e8cbacdef8aae4` (CRLF line endings, no final newline). Its content is
  identical, apart from line endings and the final newline, to a copy with SHA-256
  `3886f3a7bc7d145ddcc8377d560280cd089fa468a8171a46885f9220e533b2c4`. It adds a `resolution` to exactly those five items
  and leaves every span as in the first spec (`audit_spec_v1.json`).
* **What Claude did, and did not do.** Claude, an AI model, (1) converted the auditor's completed Markdown worksheet into the
  JSON spec mechanically, copying the recorded quotes without paraphrase, (2) ran format and exact-quote checks and a
  read-only comparison of offsets to explain why each disagreement arose, and (3) wrote the five `resolution` texts as
  wording for the decisions the project owner stated. Claude did **not** find evidence, judge whether any quote or
  finding was correct, decide any adjudication, or change any label. Evidence discovery, the status and confidence of
  each finding, and every adjudication were the human work of the auditor and the project owner. The auditor's declaration
  that no AI assistant was used concerns finding the evidence in the bundle; it does not cover this later conversion.

**Interrupted sampling.** `audit-sample` writes in this order: the private folder (verified), the map, the auditor
questions, and last the `audit_sample` event. If it stops, it prints what to check and deletes nothing. Recover by hand:
1. Look in `audit.jsonl` for an `audit_sample` event. The map, the questions and the event belong together.
2. Event present: the sample exists. Keep the map and the questions; if a file is missing, stop and ask, because the
   sample cannot be drawn again (`AUDIT_SAMPLE_REPEATED`) and the aliases cannot be rebuilt without the map.
3. No event: nothing was sampled. Move (do not delete) the private folder and the questions file out of the way, then run
   the command again, with the same `--seed`.

## 10. Freeze and baseline

* **Freeze preparation.** The dataset is prepared for freeze as `eval-dataset-v1.0.0`, with an intended freeze date of 2026-11-14. The freeze becomes final after `freeze_record.json` is created and validated. Metadata (`dataset.json`: `schema_version` `eval-dataset-v1`,
  `corpus_version` `eval-corpus-v1`, `created` 2026-11-14, with the manifest hash recorded there). Per-file hashes are not
  kept in this guideline or in `dataset.json`, which has exactly five fields.
* **Counts.** 86 questions: dev 36 (26 answerable and 10 unanswerable), test 50 (all answerable); 76 answerable and 10
  unanswerable in all; 53 English and 33 Hindi. The audit log holds 159 events.
* **Review provenance.** All 86 questions have a current human evidence review (`label_review`, `Shark`). Hindi review was
  by a single reviewer (`Shark`). The 10 unanswerable questions (`du-01` to `du-10`) have AI-only semantic review
  provenance (`ai_semantic_review`, `human_review: false`); it is informational and does not block readiness. The
  second-pass audit was non-independent (section 9). Questions are `llm_assisted`.
* **A frozen dataset is not a baseline.** Freezing the dataset does not create the official V1 evaluation baseline; that
  is created later (E2).
* **E1.7 pilot.** The pilot was a "Validity pilot — not the V1 baseline" (report SHA-256
  `4a7418cd524a5c19f84a127803a8654da339e69380a59cd84d5dd08a36c0200f`; test split not read). It reported
  `working_tree_clean: false` and was initially provisional pending the second-pass audit. The audit is now complete and
  changed no label, and the dev label-binding hash
  `860078e2fdb8bfabcfaddc4f184eb6af941d52bfb004f8a61d8a7f6b8416951c` was recomputed with the project's own function and
  still matches the pilot's binding. It remains a validity pilot and is not the official V1 baseline.
