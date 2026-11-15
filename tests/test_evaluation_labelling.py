"""Tests for the E1.6 labelling library and CLI (app/rag/evaluation/labelling*.py).

A small synthetic corpus (one English and one Hindi Markdown document, with a ligature and a
repeated sentence) is written to a temporary folder together with a matching manifest. The real
published corpus is covered by one separate test at the end.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import itertools
import json
import os
import random
import re
import stat
import subprocess
from pathlib import Path

import pytest

from app.rag.evaluation import audit_bundle, labelling, labelling_cli
from app.rag.evaluation.parsing import parse_questions
from app.rag.evaluation.schema import EvaluationDataError

DATE = "2026-11-13"
REVIEW = labelling.Review(evidence_reviewer="Reviewer One", date=DATE)
HINDI_REVIEW = labelling.Review(evidence_reviewer="Reviewer One", date=DATE, hindi_reviewer="Hindi Reviewer")
SEMANTIC_REVIEW = labelling.Review(evidence_reviewer="Reviewer One", date=DATE, semantic_reviewer="Second Reader")

FACTS = [f"Fact number {k} about widget-{k} is value-{k}." for k in range(70)]
ENGLISH = "\n".join(FACTS) + "\nThe ﬁle system is ﬁxed.\nRepeat me.\nRepeat me.\n"
HINDI = "यह एक परीक्षण दस्तावेज़ है।\nकृपया पहला वाक्य ध्यान से पढ़ें।\nदूसरा वाक्य कुछ अलग कहता है।\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _document(key: str, filename: str, text: str, language: str, directory: Path) -> dict:
    data = text.encode("utf-8")
    (directory / filename).write_bytes(data)
    return {
        "corpus_key": key, "title": key, "publisher": "Test", "source_url": "https://example.com/" + key,
        "license": "MIT", "license_url": "https://example.com/license", "format": "md", "language": language,
        "source_file_sha256": hashlib.sha256(data).hexdigest(), "extracted_text_sha256": _sha(text),
        "extracted_text_length": len(text), "extraction": {"processing_ref": "a" * 40, "python": "3.12.4"},
        "redistributable": True,
    }


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    directory = tmp_path / "corpus"
    directory.mkdir()
    manifest = {
        "schema_version": "eval-manifest-v1", "corpus_version": "test-v1", "stage": "ingested",
        "documents": [
            _document("doc-en", "doc-en.md", ENGLISH, "en", directory),
            _document("doc-hi", "doc-hi.md", HINDI, "hi", directory),
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "dataset"
    directory.mkdir()
    return directory


def answerable(n: int, *, split: str = "test", language: str = "en", key: str = "doc-en", **extra) -> dict:
    item = {
        "question_id": f"q{n}", "split": split,
        "question": f"What value does widget number {FACTS[n].split()[3]} hold, question {n}?",
        "query_language": language, "author_type": "human", "expected_answer": f"value-{n}",
        "difficulty": "direct", "close_distractor": False,
        "units": [{"unit_id": "u1", "spans": [{"corpus_key": key, "quote": FACTS[n]}]}],
    }
    item.update(extra)
    return item


def unanswerable(n: int, **extra) -> dict:
    item = {
        "question_id": f"n{n}", "split": "dev", "question": f"Which colour is gadget {n} painted in the factory?",
        "query_language": "en", "author_type": "human", "unanswerable_type": "off_topic",
    }
    item.update(extra)
    return item


def add(dataset_dir: Path, corpus: Path, items, *, review: labelling.Review = REVIEW, is_answerable: bool = True):
    return labelling.add_questions(dataset_dir, corpus, items, answerable=is_answerable, review=review)


def issue_codes(issues) -> set[str]:
    return {item.code for item in issues}


# ---------- quotes and spans ----------

def test_a_unique_quote_resolves_to_its_exact_offsets():
    start, end = labelling.find_quote(ENGLISH, FACTS[7])
    assert ENGLISH[start:end] == FACTS[7]


def test_a_missing_quote_is_refused():
    with pytest.raises(labelling.LabellingError, match="does not occur"):
        labelling.find_quote(ENGLISH, "this sentence is not there")


def test_an_ambiguous_quote_needs_an_explicit_occurrence():
    with pytest.raises(labelling.LabellingError, match="occurs 2 times"):
        labelling.find_quote(ENGLISH, "Repeat me.")
    first = labelling.find_quote(ENGLISH, "Repeat me.", 1)
    second = labelling.find_quote(ENGLISH, "Repeat me.", 2)
    assert first[0] < second[0]
    assert ENGLISH[first[0]:first[1]] == ENGLISH[second[0]:second[1]] == "Repeat me."


@pytest.mark.parametrize("occurrence", [0, 3, -1, True, 1.0, "1"])
def test_an_invalid_occurrence_is_refused(occurrence):
    with pytest.raises(labelling.LabellingError, match="occurrence must be"):
        labelling.find_quote(ENGLISH, "Repeat me.", occurrence)


def test_overlapping_matches_are_counted_as_ambiguous():
    with pytest.raises(labelling.LabellingError, match="occurs 2 times"):
        labelling.find_quote("aaa", "aa")


def test_an_empty_quote_is_refused():
    with pytest.raises(labelling.LabellingError, match="empty"):
        labelling.find_quote(ENGLISH, "")


def test_error_messages_never_contain_the_quote():
    secret = "needle-that-is-absent"
    with pytest.raises(labelling.LabellingError) as info:
        labelling.find_quote(ENGLISH, secret)
    assert secret not in str(info.value)


def test_the_ligature_is_matched_exactly_and_never_normalised():
    start, end = labelling.find_quote(ENGLISH, "ﬁle system")
    assert ENGLISH[start:end] == "ﬁle system"
    with pytest.raises(labelling.LabellingError, match="does not occur"):
        labelling.find_quote(ENGLISH, "file system")


def test_a_devanagari_quote_uses_code_point_offsets(corpus):
    manifest = labelling.load_manifest(corpus)
    texts = labelling.load_texts(corpus, manifest)
    span = labelling.make_span(manifest, texts, "doc-hi", "पहला वाक्य")
    assert HINDI[span.start:span.end] == span.quote
    assert span.end - span.start == len("पहला वाक्य")
    assert span.source_sha256 == _sha(HINDI)
    assert span.quote_sha256 == _sha("पहला वाक्य")


def test_make_span_refuses_a_key_outside_the_manifest(corpus):
    manifest = labelling.load_manifest(corpus)
    texts = labelling.load_texts(corpus, manifest)
    with pytest.raises(labelling.LabellingError, match="not a document"):
        labelling.make_span(manifest, texts, "doc-other", FACTS[1])


# ---------- the corpus gate ----------

def test_the_corpus_loads_and_is_verified(corpus):
    texts = labelling.load_texts(corpus, labelling.load_manifest(corpus))
    assert texts == {"doc-en": ENGLISH, "doc-hi": HINDI}


def test_a_changed_corpus_file_is_refused(corpus):
    (corpus / "doc-en.md").write_text(ENGLISH + "extra\n", encoding="utf-8")
    with pytest.raises(labelling.LabellingError, match="no corpus file matches"):
        labelling.load_texts(corpus, labelling.load_manifest(corpus))


def test_a_stale_manifest_text_hash_is_refused(corpus):
    path = corpus / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["documents"][0]["extracted_text_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(labelling.LabellingError, match="fails verification .*SOURCE_HASH_MISMATCH"):
        labelling.load_texts(corpus, labelling.load_manifest(corpus))


def test_a_draft_manifest_is_refused(corpus):
    path = corpus / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["stage"] = "draft"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((labelling.LabellingError, EvaluationDataError)):
        labelling.load_manifest(corpus)


# ---------- specs ----------

def test_a_spec_may_be_one_object_or_an_array():
    assert labelling.read_spec('{"a": 1}') == [{"a": 1}]
    assert labelling.read_spec('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]


@pytest.mark.parametrize("text", ['{"a": 1, "a": 2}', "not json", "[]", "[1]", '"x"'])
def test_a_bad_spec_is_refused(text):
    with pytest.raises(labelling.LabellingError):
        labelling.read_spec(text)


# ---------- adding ----------

def test_adding_an_answerable_question_anchors_and_records_it(dataset_dir, corpus):
    result = add(dataset_dir, corpus, [answerable(3)])
    assert result.written and result.added == ("q3",)
    questions = parse_questions((dataset_dir / "test.jsonl").read_text(encoding="utf-8"), split="test")
    span = questions[0].units[0].spans[0]
    assert ENGLISH[span.start:span.end] == span.quote == FACTS[3]
    assert span.source_sha256 == _sha(ENGLISH)
    assert questions[0].source_language == "en"  # derived from the manifest
    events = labelling.load_events(dataset_dir)
    assert events == [{"event": "label_review", "question_id": "q3", "reviewer": "Reviewer One",
                       "date": DATE, "author_type": "human",
                       "content_sha256": labelling.review_content_sha256(questions[0])}]
    assert labelling.check_all(dataset_dir, corpus) == []


def test_a_serialised_question_reads_back_equal(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    add(dataset_dir, corpus, [unanswerable(1)], review=SEMANTIC_REVIEW, is_answerable=False)
    dataset = labelling.load_dataset(dataset_dir)
    for name in ("dev", "test"):
        for question in dataset.split(name):
            assert parse_questions(labelling.serialise(question), split=name) == (question,)


def test_a_batch_is_all_or_nothing(dataset_dir, corpus):
    absent = [{"unit_id": "u1", "spans": [{"corpus_key": "doc-en", "quote": "absent text"}]}]
    items = [answerable(1), answerable(2, units=absent)]
    with pytest.raises(labelling.LabellingError, match=r"spec\[1\]\.units\[0\]\.spans\[0\]"):
        add(dataset_dir, corpus, items)
    assert not (dataset_dir / "test.jsonl").exists()
    assert not (dataset_dir / "audit.jsonl").exists()


def test_an_ambiguous_quote_is_refused_without_an_occurrence(dataset_dir, corpus):
    units = [{"unit_id": "u1", "spans": [{"corpus_key": "doc-en", "quote": "Repeat me."}]}]
    with pytest.raises(labelling.LabellingError, match="occurs 2 times"):
        add(dataset_dir, corpus, [answerable(1, units=units)])
    units[0]["spans"][0]["occurrence"] = 2
    assert add(dataset_dir, corpus, [answerable(1, units=units)]).written


def test_an_invalid_dataset_entry_writes_nothing(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    before = (dataset_dir / "test.jsonl").read_bytes()
    result = add(dataset_dir, corpus, [answerable(1)])  # duplicate id and duplicate text
    assert not result.written
    assert {"DATASET_DUPLICATE_QUESTION_ID", "DATASET_DUPLICATE_QUESTION_TEXT"} <= issue_codes(result.issues)
    assert (dataset_dir / "test.jsonl").read_bytes() == before
    assert len(labelling.load_events(dataset_dir)) == 1


def test_a_test_span_touching_a_dev_span_is_refused(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1, split="dev")])
    result = add(dataset_dir, corpus, [answerable(2, units=[{"unit_id": "u1", "spans": [
        {"corpus_key": "doc-en", "quote": FACTS[1]}]}])])
    assert "DATASET_SPLIT_COLLISION" in issue_codes(result.issues) and not result.written


def test_unknown_and_missing_spec_fields_are_refused(dataset_dir, corpus):
    with pytest.raises(labelling.LabellingError, match="unknown field"):
        add(dataset_dir, corpus, [answerable(1, bogus=True)])
    item = answerable(1)
    del item["difficulty"]
    with pytest.raises(labelling.LabellingError, match="missing: difficulty"):
        add(dataset_dir, corpus, [item])


@pytest.mark.parametrize("field", ["hindi_reviewed", "hindi_reviewer", "semantic_review"])
def test_a_spec_cannot_claim_a_review(dataset_dir, corpus, field):
    with pytest.raises(labelling.LabellingError, match="unknown field"):
        add(dataset_dir, corpus, [answerable(1, **{field: "x"})])


def test_a_hindi_question_needs_a_named_hindi_reviewer(dataset_dir, corpus):
    item = answerable(1, language="hi", question="डिवाइस नंबर एक का मान क्या है, प्रश्न एक?")
    with pytest.raises(labelling.LabellingError, match="--hindi-reviewer"):
        add(dataset_dir, corpus, [item])
    assert add(dataset_dir, corpus, [item], review=HINDI_REVIEW).written
    stored = labelling.load_dataset(dataset_dir).test[0]
    assert stored.hindi_reviewed is True and stored.hindi_reviewer == "Hindi Reviewer"


def test_an_english_question_never_carries_a_hindi_reviewer(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)], review=HINDI_REVIEW)
    stored = labelling.load_dataset(dataset_dir).test[0]
    assert stored.hindi_reviewed is False and stored.hindi_reviewer is None


def test_the_source_language_must_be_derivable_and_consistent(dataset_dir, corpus):
    mixed = [{"unit_id": "u1", "spans": [{"corpus_key": "doc-en", "quote": FACTS[1]}]},
             {"unit_id": "u2", "spans": [{"corpus_key": "doc-hi", "quote": "पहला वाक्य"}]}]
    with pytest.raises(labelling.LabellingError, match="cannot be derived"):
        add(dataset_dir, corpus, [answerable(1, units=mixed)])
    result = add(dataset_dir, corpus, [answerable(2, source_language="hi")])
    assert "DATASET_SOURCE_LANGUAGE_MISMATCH" in issue_codes(result.issues) and not result.written


def test_an_unanswerable_question_needs_a_semantic_reviewer(dataset_dir, corpus):
    with pytest.raises(labelling.LabellingError, match="--semantic-reviewer"):
        add(dataset_dir, corpus, [unanswerable(1)], is_answerable=False)
    result = add(dataset_dir, corpus, [unanswerable(1)], review=SEMANTIC_REVIEW, is_answerable=False)
    assert result.written
    stored = labelling.load_dataset(dataset_dir).dev[0]
    assert stored.semantic_review.reviewer == "Second Reader" and stored.units == ()


def test_an_unanswerable_question_cannot_go_in_the_test_split(dataset_dir, corpus):
    result = add(dataset_dir, corpus, [unanswerable(1, split="test")], review=SEMANTIC_REVIEW, is_answerable=False)
    assert "Q_UNANSWERABLE_SPLIT" in issue_codes(result.issues) and not result.written


def test_retrieval_assistance_is_logged_with_the_question(dataset_dir, corpus):
    assistance = {"date": DATE, "method": "dense top-5", "run_ref": None, "description": "checked the top 5 chunks"}
    add(dataset_dir, corpus, [unanswerable(1, retrieval_assistance=assistance)], review=SEMANTIC_REVIEW,
        is_answerable=False)
    assert labelling.load_dataset(dataset_dir).dev[0].retrieval_assistance.description == "checked the top 5 chunks"


def test_a_similar_question_is_a_warning_not_a_failure(dataset_dir, corpus):
    first = answerable(1, question="What is the stored value for widget number 1 in the table?")
    second = answerable(2, question="What is the stored value for widget number 2 in the table?")
    add(dataset_dir, corpus, [first])
    result = labelling.add_questions(dataset_dir, corpus, [second], answerable=True, review=REVIEW, threshold=0.5)
    assert result.written and "DATASET_SIMILAR_QUESTION" in issue_codes(result.issues)


@pytest.mark.parametrize("name", ["", "   "])
def test_a_blank_evidence_reviewer_is_refused(dataset_dir, corpus, name):
    with pytest.raises(labelling.LabellingError, match="non-blank"):
        add(dataset_dir, corpus, [answerable(1)], review=labelling.Review(name, DATE))


def test_an_invalid_review_date_is_refused(dataset_dir, corpus):
    with pytest.raises(labelling.LabellingError, match="real YYYY-MM-DD"):
        add(dataset_dir, corpus, [answerable(1)], review=labelling.Review("Reviewer", "2026-02-30"))


# ---------- check: stale and invalid evidence ----------

def _edit_test_file(dataset_dir: Path, change) -> None:
    path = dataset_dir / "test.jsonl"
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    change(record["units"][0]["spans"][0])
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def test_check_reports_a_stale_source_hash(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    _edit_test_file(dataset_dir, lambda span: span.update(source_sha256="1" * 64))
    assert {"DATASET_SPAN_HASH_MISMATCH"} <= issue_codes(labelling.check_all(dataset_dir, corpus))


def test_check_reports_wrong_offsets(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    _edit_test_file(dataset_dir, lambda span: span.update(start=span["start"] + 1, end=span["end"] + 1))
    assert "ANCHOR_QUOTE_MISMATCH" in issue_codes(labelling.check_all(dataset_dir, corpus))


def test_check_reports_an_edited_quote(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    _edit_test_file(dataset_dir, lambda span: span.update(quote=span["quote"][:-1] + "!"))
    codes = issue_codes(labelling.check_all(dataset_dir, corpus))
    assert {"ANCHOR_QUOTE_MISMATCH", "SPAN_QUOTE_SHA256_MISMATCH"} <= codes


def test_check_reports_a_question_without_a_recorded_review(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    (dataset_dir / "audit.jsonl").unlink()
    assert labelling.check_all(dataset_dir, corpus) == []  # structurally valid; the review is pending
    readiness = labelling.review_readiness(labelling.load_dataset(dataset_dir), labelling.load_events(dataset_dir))
    assert readiness.missing_evidence_review == ("q1",) and not readiness.ready


def test_check_validates_dataset_metadata_when_present(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    (dataset_dir / "dataset.json").write_text(json.dumps({
        "schema_version": "eval-dataset-v1", "dataset_version": "draft", "corpus_version": "wrong",
        "corpus_manifest_hash": "0" * 64, "created": DATE}), encoding="utf-8")
    assert {"META_CORPUS_VERSION_MISMATCH", "META_MANIFEST_HASH_MISMATCH"} <= issue_codes(
        labelling.check_all(dataset_dir, corpus))


# ---------- audit ----------

def _fill(dataset_dir: Path, corpus: Path, answerables: int, unanswerables: int = 0) -> None:
    add(dataset_dir, corpus, [answerable(n, split="test" if n % 2 else "dev") for n in range(answerables)])
    if unanswerables:
        add(dataset_dir, corpus, [unanswerable(n) for n in range(unanswerables)], review=SEMANTIC_REVIEW,
            is_answerable=False)


@pytest.mark.parametrize(("answerables", "expected"), [(5, 5), (30, 10), (50, 10), (60, 12)])
def test_the_audit_sample_is_twenty_percent_with_a_minimum_of_ten(dataset_dir, corpus, answerables, expected):
    _fill(dataset_dir, corpus, answerables)
    ids = labelling.draw_audit_sample(labelling.load_dataset(dataset_dir), seed=7)
    assert len(ids) == expected


def test_every_unanswerable_question_is_audited_and_the_draw_is_deterministic(dataset_dir, corpus):
    _fill(dataset_dir, corpus, 30, 3)
    dataset = labelling.load_dataset(dataset_dir)
    first = labelling.draw_audit_sample(dataset, seed=7)
    assert first == labelling.draw_audit_sample(dataset, seed=7)
    assert {"n0", "n1", "n2"} <= set(first) and len(first) == 13
    assert first != labelling.draw_audit_sample(dataset, seed=8)


@pytest.mark.parametrize("seed", [-1, True, "1", 1.5])
def test_an_invalid_seed_is_refused(dataset_dir, corpus, seed):
    _fill(dataset_dir, corpus, 2)
    with pytest.raises(labelling.LabellingError, match="seed"):
        labelling.draw_audit_sample(labelling.load_dataset(dataset_dir), seed)


def test_only_one_audit_sample_can_be_recorded(dataset_dir, corpus):
    _fill(dataset_dir, corpus, 3)
    dataset = labelling.load_dataset(dataset_dir)
    labelling.record_audit_sample(dataset_dir, dataset, 1, DATE)
    with pytest.raises(labelling.LabellingError, match="already recorded"):
        labelling.record_audit_sample(dataset_dir, dataset, 2, DATE)


def _audit(dataset_dir, corpus, items):
    return labelling.record_audit_results(dataset_dir, corpus, items, auditor="Auditor", date=DATE)


def test_an_audit_needs_a_sample_and_a_sampled_question(dataset_dir, corpus):
    _fill(dataset_dir, corpus, 3)
    with pytest.raises(labelling.LabellingError, match="no audit sample"):
        _audit(dataset_dir, corpus, [{"question_id": "q0", "spans": []}])
    labelling.record_audit_sample(dataset_dir, labelling.load_dataset(dataset_dir), 1, DATE)
    with pytest.raises(labelling.LabellingError, match="not in the audit sample"):
        _audit(dataset_dir, corpus, [{"question_id": "zz", "spans": []}])


def test_an_independent_span_that_overlaps_the_label_agrees(dataset_dir, corpus):
    _fill(dataset_dir, corpus, 3)
    labelling.record_audit_sample(dataset_dir, labelling.load_dataset(dataset_dir), 1, DATE)
    quote = "widget-1 is value-1"
    results = _audit(dataset_dir, corpus, [{"question_id": "q1", "spans": [{"corpus_key": "doc-en", "quote": quote}]}])
    assert results[0]["outcome"] == "agree" and results[0]["auditor"] == "Auditor"
    assert [issue for issue in labelling.check_all(dataset_dir, corpus) if issue.severity == "error"] == []


def test_a_different_span_is_a_disagreement_until_it_is_resolved(dataset_dir, corpus):
    _fill(dataset_dir, corpus, 3)
    labelling.record_audit_sample(dataset_dir, labelling.load_dataset(dataset_dir), 1, DATE)
    other = {"question_id": "q1", "spans": [{"corpus_key": "doc-en", "quote": FACTS[2]}]}
    assert _audit(dataset_dir, corpus, [other])[0]["outcome"] == "disagree"
    assert "AUDIT_DISAGREEMENT_UNRESOLVED" in issue_codes(labelling.check_all(dataset_dir, corpus))
    other["resolution"] = "the auditor misread the question; the label stands"
    _audit(dataset_dir, corpus, [other])
    assert "AUDIT_DISAGREEMENT_UNRESOLVED" not in issue_codes(labelling.check_all(dataset_dir, corpus))


def test_an_auditor_who_finds_evidence_for_an_unanswerable_question_disagrees(dataset_dir, corpus):
    _fill(dataset_dir, corpus, 3, 1)
    labelling.record_audit_sample(dataset_dir, labelling.load_dataset(dataset_dir), 1, DATE)
    found = {"question_id": "n0", "spans": [{"corpus_key": "doc-en", "quote": FACTS[0]}]}
    assert _audit(dataset_dir, corpus, [found])[0]["outcome"] == "disagree"
    assert _audit(dataset_dir, corpus, [{"question_id": "n0", "spans": []}])[0]["outcome"] == "agree"


def test_sampled_questions_without_a_result_are_warnings(dataset_dir, corpus):
    _fill(dataset_dir, corpus, 3)
    labelling.record_audit_sample(dataset_dir, labelling.load_dataset(dataset_dir), 1, DATE)
    issues = [i for i in labelling.check_all(dataset_dir, corpus) if i.code.startswith("AUDIT_")]
    assert {i.code for i in issues} == {"AUDIT_RESULT_MISSING"} and all(i.severity == "warning" for i in issues)


# ---------- command line ----------

def run(capsys, corpus: Path, dataset_dir: Path, *args: str) -> tuple[int, str, str]:
    code = labelling_cli.main([args[0], "--corpus-dir", str(corpus), "--dataset-dir", str(dataset_dir), *args[1:]])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def write_spec(tmp_path: Path, payload, name: str = "spec.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_cli_show_lists_documents_slices_text_and_finds_occurrences(capsys, corpus, dataset_dir):
    code, out, _ = run(capsys, corpus, dataset_dir, "show")
    assert code == 0 and "doc-en\ten" in out and "doc-hi\thi" in out
    code, out, _ = run(capsys, corpus, dataset_dir, "show", "--key", "doc-en", "--start", "0", "--end", "20")
    assert code == 0 and "offsets 0-20" in out and FACTS[0][:20] in out
    code, out, _ = run(capsys, corpus, dataset_dir, "show", "--key", "doc-en", "--find", "Repeat me.")
    assert code == 0 and "2 occurrence(s)" in out and "occurrence 2" in out


def test_cli_show_refuses_an_unknown_key(capsys, corpus, dataset_dir):
    code, _, err = run(capsys, corpus, dataset_dir, "show", "--key", "nope")
    assert code == 2 and "not a document" in err


def test_cli_add_check_and_audit_workflow(capsys, corpus, dataset_dir, tmp_path):
    spec = write_spec(tmp_path, [answerable(n, split="dev" if n < 3 else "test") for n in range(6)])
    code, out, _ = run(capsys, corpus, dataset_dir, "add-answerable", "--spec", spec, "--evidence-reviewer", "Rev",
                       "--date", DATE)
    assert code == 0 and "added 6 question(s)" in out

    spec = write_spec(tmp_path, [unanswerable(1)], "u.json")
    code, out, _ = run(capsys, corpus, dataset_dir, "add-unanswerable", "--spec", spec, "--evidence-reviewer", "Rev",
                       "--semantic-reviewer", "Second", "--date", DATE)
    assert code == 0

    code, out, _ = run(capsys, corpus, dataset_dir, "check")
    assert code == 0 and "3 dev and 3 test" not in out and "4 dev and 3 test question(s); 0 error(s)" in out

    questions, private = tmp_path / "public" / "questions.json", tmp_path / "private" / "alias_map.json"
    questions.parent.mkdir()
    code, out, err = run(capsys, corpus, dataset_dir, "audit-sample", "--seed", "5", "--out", str(questions),
                         "--private-dir", str(private.parent), "--alias-seed", "3", "--date", DATE)
    assert code == 0 and "sampled 7 question(s)" in out
    aliases = audit_bundle.read_private_map(private)
    assert "n1" in aliases.values() and not any(real in out + err for real in aliases.values())
    code, _, err = run(capsys, corpus, dataset_dir, "audit-sample", "--seed", "5", "--out", str(questions.parent / "q2.json"),
                       "--private-dir", str(tmp_path / "private2"))
    assert code == 2 and "already recorded" in err and not (tmp_path / "private2").exists()

    neutral = {key: doc for doc, key in audit_bundle.read_document_map(private).items()}
    audit = [{"question_id": alias, "spans": []} if real == "n1" else
             {"question_id": alias, "spans": [{"corpus_key": neutral["doc-en"], "quote": FACTS[int(real[1:])]}]}
             for alias, real in aliases.items()]
    code, out, _ = run(capsys, corpus, dataset_dir, "audit-record", "--spec", write_spec(tmp_path, audit, "a.json"),
                       "--auditor", "Auditor", "--date", DATE, "--alias-map", str(private))
    assert code == 0 and "0 disagreement(s)" in out
    code, out, _ = run(capsys, corpus, dataset_dir, "check")
    assert code == 0 and "AUDIT_" not in out and "0 error(s)" in out


def test_cli_add_refuses_and_writes_nothing_on_a_dataset_error(capsys, corpus, dataset_dir, tmp_path):
    spec = write_spec(tmp_path, [answerable(1), answerable(1)])
    code, out, _ = run(capsys, corpus, dataset_dir, "add-answerable", "--spec", spec, "--evidence-reviewer", "Rev")
    assert code == 1 and "DATASET_DUPLICATE_QUESTION_ID" in out and "nothing was written" in out
    assert not (dataset_dir / "test.jsonl").exists()


def test_cli_check_fails_on_stale_evidence(capsys, corpus, dataset_dir, tmp_path):
    run(capsys, corpus, dataset_dir, "add-answerable", "--spec", write_spec(tmp_path, [answerable(1)]),
        "--evidence-reviewer", "Rev", "--date", DATE)
    _edit_test_file(dataset_dir, lambda span: span.update(source_sha256="2" * 64))
    code, out, _ = run(capsys, corpus, dataset_dir, "check")
    assert code == 1 and "DATASET_SPAN_HASH_MISMATCH" in out


def test_cli_requires_the_reviewer_arguments(capsys, corpus, dataset_dir, tmp_path):
    spec = write_spec(tmp_path, [answerable(1)])
    with pytest.raises(SystemExit):
        labelling_cli.main(["add-answerable", "--spec", spec])
    capsys.readouterr()
    spec = write_spec(tmp_path, [unanswerable(1)])
    code, _, err = run(capsys, corpus, dataset_dir, "add-unanswerable", "--spec", spec,
                       "--evidence-reviewer", "Rev", "--semantic-reviewer", "  ")
    assert code == 2 and "--semantic-reviewer" in err


def test_cli_messages_never_contain_a_quote(capsys, corpus, dataset_dir, tmp_path):
    secret = "a quote that is not in any document"
    item = answerable(1, units=[{"unit_id": "u1", "spans": [{"corpus_key": "doc-en", "quote": secret}]}])
    code, out, err = run(capsys, corpus, dataset_dir, "add-answerable", "--spec", write_spec(tmp_path, [item]),
                         "--evidence-reviewer", "Rev")
    assert code == 2 and secret not in out + err


# ---------- the real published corpus ----------

REAL_CORPUS = Path(__file__).resolve().parent.parent / "eval" / "corpus"


@pytest.mark.skipif(not (REAL_CORPUS / "manifest.json").exists(), reason="the published corpus is not present")
def test_the_published_corpus_loads_and_verifies():
    manifest = labelling.load_manifest(REAL_CORPUS)
    recorded = {d.extraction.pypdf for d in manifest.documents if d.extraction and d.extraction.pypdf}
    try:
        installed = importlib.metadata.version("pypdf")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if recorded and installed not in recorded:
        pytest.skip("PDF text depends on the pypdf version recorded in the manifest")
    texts = labelling.load_texts(REAL_CORPUS, manifest)
    assert set(texts) == {d.corpus_key for d in manifest.documents}
    assert all(len(texts[d.corpus_key]) == d.extracted_text_length for d in manifest.documents)


# ---------- failure while writing: the two files stay consistent ----------

def _state(dataset_dir: Path) -> dict[str, bytes | None]:
    names = ("dev.jsonl", "test.jsonl", "audit.jsonl")
    return {name: (dataset_dir / name).read_bytes() if (dataset_dir / name).exists() else None for name in names}


def _no_temporary_files(dataset_dir: Path) -> bool:
    return not list(dataset_dir.glob("*.tmp"))


def _fail_append(*_args, **_kwargs):
    raise OSError("simulated audit log failure")


def test_a_failed_audit_write_restores_the_existing_question_files(dataset_dir, corpus, monkeypatch):
    add(dataset_dir, corpus, [answerable(1, split="dev"), answerable(2)])
    before = _state(dataset_dir)
    monkeypatch.setattr(labelling, "_append_events", _fail_append)
    with pytest.raises(labelling.LabellingError, match="restored and nothing was added"):
        add(dataset_dir, corpus, [answerable(3, split="dev"), answerable(4)])
    assert _state(dataset_dir) == before
    assert _no_temporary_files(dataset_dir)
    monkeypatch.undo()
    assert not [i for i in labelling.check_all(dataset_dir, corpus) if i.severity == "error"]


def test_a_failed_audit_write_leaves_no_question_file_behind(dataset_dir, corpus, monkeypatch):
    monkeypatch.setattr(labelling, "_append_events", _fail_append)
    with pytest.raises(labelling.LabellingError, match="restored and nothing was added"):
        add(dataset_dir, corpus, [answerable(1, split="dev"), answerable(2)])
    assert _state(dataset_dir) == {"dev.jsonl": None, "test.jsonl": None, "audit.jsonl": None}
    assert _no_temporary_files(dataset_dir)


def test_the_same_batch_can_be_added_after_a_failed_attempt(dataset_dir, corpus, monkeypatch):
    monkeypatch.setattr(labelling, "_append_events", _fail_append)
    with pytest.raises(labelling.LabellingError):
        add(dataset_dir, corpus, [answerable(1)])
    monkeypatch.undo()
    assert add(dataset_dir, corpus, [answerable(1)]).written
    assert [e["question_id"] for e in labelling.load_events(dataset_dir)] == ["q1"]


def test_a_failure_after_the_first_split_file_restores_both(dataset_dir, corpus, monkeypatch):
    add(dataset_dir, corpus, [answerable(1, split="dev")])
    before = _state(dataset_dir)
    real = labelling._write_split

    def fail_on_test(directory, name, questions):
        if name == "test":
            raise OSError("simulated failure")
        real(directory, name, questions)

    monkeypatch.setattr(labelling, "_write_split", fail_on_test)
    with pytest.raises(labelling.LabellingError, match="restored and nothing was added"):
        add(dataset_dir, corpus, [answerable(2, split="dev"), answerable(3)])
    assert _state(dataset_dir) == before


def test_a_partial_audit_append_is_cut_back(dataset_dir, corpus, monkeypatch):
    add(dataset_dir, corpus, [answerable(1, split="dev")])
    before = _state(dataset_dir)
    calls = []
    real = labelling.os.fsync

    def fsync_fails_on_the_audit_append(descriptor):
        calls.append(descriptor)
        if len(calls) == 2:  # call 1 syncs the question file, call 2 the audit log after it was written
            raise OSError("simulated disk failure")
        real(descriptor)

    monkeypatch.setattr(labelling.os, "fsync", fsync_fails_on_the_audit_append)
    with pytest.raises(labelling.LabellingError, match="restored and nothing was added"):
        add(dataset_dir, corpus, [answerable(2, split="dev")])
    monkeypatch.undo()
    assert len(calls) >= 2  # the third call is the restore of the question file
    assert _state(dataset_dir) == before


def test_a_failed_audit_sample_write_leaves_no_sample_recorded(dataset_dir, corpus, monkeypatch):
    _fill(dataset_dir, corpus, 3)
    before = _state(dataset_dir)
    dataset = labelling.load_dataset(dataset_dir)

    def fsync_fails(_descriptor):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(labelling.os, "fsync", fsync_fails)
    with pytest.raises(OSError, match="simulated"):
        labelling.record_audit_sample(dataset_dir, dataset, 1, DATE)
    monkeypatch.undo()
    assert _state(dataset_dir) == before
    assert labelling.record_audit_sample(dataset_dir, dataset, 1, DATE)  # not blocked by "already recorded"


def test_an_interrupt_still_restores_and_is_not_swallowed(dataset_dir, corpus, monkeypatch):
    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(labelling, "_append_events", interrupt)
    with pytest.raises(KeyboardInterrupt):
        add(dataset_dir, corpus, [answerable(1)])
    assert _state(dataset_dir) == {"dev.jsonl": None, "test.jsonl": None, "audit.jsonl": None}


def test_a_failed_restore_is_reported_so_check_can_be_run(dataset_dir, corpus, monkeypatch):
    monkeypatch.setattr(labelling, "_append_events", _fail_append)
    monkeypatch.setattr(labelling, "_restore", lambda *_args: False)
    with pytest.raises(labelling.LabellingError, match=r"test\.jsonl could not be restored; run check"):
        add(dataset_dir, corpus, [answerable(1)])
    monkeypatch.undo()
    ds = labelling.load_dataset(dataset_dir)  # the leftover is caught by readiness, not by check
    assert labelling.review_readiness(ds, labelling.load_events(dataset_dir)).missing_evidence_review == ("q1",)


def test_cli_reports_a_failed_write_and_changes_nothing(capsys, corpus, dataset_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(labelling, "_append_events", _fail_append)
    code, _, err = run(capsys, corpus, dataset_dir, "add-answerable", "--spec", write_spec(tmp_path, [answerable(1)]),
                       "--evidence-reviewer", "Rev", "--date", DATE)
    assert code == 2 and "restored and nothing was added" in err
    assert _state(dataset_dir) == {"dev.jsonl": None, "test.jsonl": None, "audit.jsonl": None}


# ---------- reviewer identity, Hindi review and audit events (E1.6.1) ----------

CORRECTION = {
    "event": "label_correction", "question_id": "q1", "date": DATE,
    "changes": {"difficulty": {"from": "direct", "to": "paraphrase"}, "evidence": "span extended"},
    "basis": "wording-overlap policy", "review_type": "ai_assisted_audit", "human_review": False,
    "note": "Not a human review.",
}


def _log(dataset_dir: Path, *events: dict) -> None:
    with open(dataset_dir / "audit.jsonl", "a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _events(dataset_dir: Path) -> list[dict]:
    return labelling.load_events(dataset_dir)


def _with_correction(dataset_dir, corpus, event=None, *, difficulty="paraphrase"):
    add(dataset_dir, corpus, [answerable(1, difficulty=difficulty)])
    _log(dataset_dir, CORRECTION if event is None else event)
    return labelling.check_all(dataset_dir, corpus)


@pytest.mark.parametrize("name", ["DRYRUN", "dry-run", "Placeholder", "TBD", "n/a", "None", "Claude Sonnet", "ChatGPT",
                                  " llm ", "AI Assistant", "Pending"])
def test_a_placeholder_or_ai_name_is_never_recorded_as_a_reviewer(dataset_dir, corpus, name):
    with pytest.raises(labelling.LabellingError, match="not a placeholder or an AI name"):
        add(dataset_dir, corpus, [answerable(1)], review=labelling.Review(name, DATE))
    hindi = answerable(2, language="hi", question="डिवाइस नंबर दो का मान क्या है, प्रश्न दो?")
    with pytest.raises(labelling.LabellingError, match="--hindi-reviewer"):
        add(dataset_dir, corpus, [hindi], review=labelling.Review("Reviewer One", DATE, hindi_reviewer=name))
    with pytest.raises(labelling.LabellingError, match="--semantic-reviewer"):
        add(dataset_dir, corpus, [unanswerable(1)], is_answerable=False,
            review=labelling.Review("Reviewer One", DATE, semantic_reviewer=name))
    assert not (dataset_dir / "test.jsonl").exists() and not (dataset_dir / "audit.jsonl").exists()


def test_a_real_name_that_contains_an_ai_word_is_accepted(dataset_dir, corpus):
    assert add(dataset_dir, corpus, [answerable(1)], review=labelling.Review("Claude Dupont", DATE)).written


def test_the_language_alone_never_marks_a_question_hindi_reviewed(dataset_dir, corpus):
    item = answerable(1, language="hi", question="डिवाइस नंबर एक का मान क्या है, प्रश्न एक?")
    with pytest.raises(labelling.LabellingError, match="--hindi-reviewer"):
        add(dataset_dir, corpus, [item], review=labelling.Review("Reviewer One", DATE))
    assert not (dataset_dir / "test.jsonl").exists()


def test_a_hindi_review_is_logged_separately_from_the_evidence_review(dataset_dir, corpus):
    hindi = answerable(1, language="hi", question="डिवाइस नंबर एक का मान क्या है, प्रश्न एक?")
    add(dataset_dir, corpus, [hindi, answerable(2)], review=HINDI_REVIEW)
    events = _events(dataset_dir)
    assert [(e["event"], e["question_id"], e["reviewer"]) for e in events] == [
        ("label_review", "q1", "Reviewer One"), ("label_review", "q2", "Reviewer One"),
        ("hindi_review", "q1", "Hindi Reviewer"),
    ]
    assert labelling.check_all(dataset_dir, corpus) == []


def test_a_hindi_review_event_must_match_the_question(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    _log(dataset_dir, {"event": "hindi_review", "question_id": "q1", "reviewer": "Hindi Reviewer", "date": DATE})
    issues = labelling.check_all(dataset_dir, corpus)
    assert issue_codes(issues) == {"AUDIT_EVENT_INVALID"}


def test_a_valid_label_correction_passes(dataset_dir, corpus):
    assert _with_correction(dataset_dir, corpus) == []
    assert _with_correction_minimal(dataset_dir, corpus) == []


def _with_correction_minimal(dataset_dir, corpus):
    event = {k: v for k, v in CORRECTION.items() if k != "note"}
    event["changes"] = {"evidence": "unit u2 added"}
    _log(dataset_dir, event)
    return labelling.check_all(dataset_dir, corpus)


def test_a_human_correction_needs_a_real_reviewer(dataset_dir, corpus):
    event = dict(CORRECTION, human_review=True, reviewer="Reviewer One")
    assert _with_correction(dataset_dir, corpus, event) == []


@pytest.mark.parametrize("change", [
    {"human_review": None},                     # removed below: the flag is required
    {"human_review": "no"},
    {"human_review": True},                     # a human correction names no reviewer
    {"human_review": True, "reviewer": "DRYRUN"},
    {"human_review": False, "reviewer": "Someone"},
    {"basis": ""}, {"basis": 5}, {"review_type": " "}, {"note": ""},
    {"changes": {}}, {"changes": []}, {"changes": {"text": "x"}},
    {"changes": {"difficulty": {"from": "direct", "to": "direct"}}},
    {"changes": {"difficulty": {"from": "direct", "to": "easy"}}},
    {"changes": {"difficulty": {"from": "direct"}}},
    {"changes": {"difficulty": {"from": "direct", "to": "paraphrase", "why": "x"}}},
    {"changes": {"evidence": "  "}},
    {"date": "2026-02-30"}, {"date": "13/11/2026"},
    {"question_id": "no-such-question"}, {"question_id": 7}, {"question_id": ["q1"]},
    {"unexpected": 1},
])
def test_a_malformed_label_correction_is_rejected(dataset_dir, corpus, change):
    event = dict(CORRECTION)
    for key, value in change.items():
        if value is None:
            event.pop(key)
        else:
            event[key] = value
    issues = _with_correction(dataset_dir, corpus, event)
    assert issues and all(i.severity == "error" for i in issues)
    assert issue_codes(issues) <= {"AUDIT_EVENT_INVALID", "AUDIT_UNKNOWN_QUESTION", "AUDIT_CORRECTION_MISMATCH"}


@pytest.mark.parametrize("field", ["question_id", "date", "changes", "basis", "review_type", "human_review"])
def test_every_required_correction_field_is_required(dataset_dir, corpus, field):
    event = {k: v for k, v in CORRECTION.items() if k != field}
    issues = _with_correction(dataset_dir, corpus, event)
    assert any(i.code == "AUDIT_EVENT_INVALID" and i.location.endswith(f".{field}") for i in issues)


def test_the_latest_correction_must_match_the_current_difficulty(dataset_dir, corpus):
    assert _with_correction(dataset_dir, corpus) == []            # the record is "paraphrase": the correction says so
    _log(dataset_dir, dict(CORRECTION, changes={"difficulty": {"from": "paraphrase", "to": "direct"}}))
    assert issue_codes(labelling.check_all(dataset_dir, corpus)) == {"AUDIT_CORRECTION_MISMATCH"}


def test_an_unknown_event_type_is_an_error_and_is_not_ignored(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    _log(dataset_dir, {"event": "totally_unknown"})
    issues = labelling.check_all(dataset_dir, corpus)
    assert [(i.code, i.severity) for i in issues] == [("AUDIT_EVENT_UNKNOWN", "error")]
    assert labelling_cli.main(["check", "--corpus-dir", str(corpus), "--dataset-dir", str(dataset_dir)]) == 1


@pytest.mark.parametrize("change", [{"reviewer": "DRYRUN"}, {"reviewer": ""}, {"reviewer": None}, {"date": "x"},
                                    {"author_type": "robot"}, {"extra": 1}, {"question_id": "nope"}])
def test_a_malformed_label_review_is_rejected(dataset_dir, corpus, change):
    add(dataset_dir, corpus, [answerable(1)])
    _log(dataset_dir, {"event": "label_review", "question_id": "q1", "reviewer": "Reviewer One", "date": DATE,
                       "author_type": "human", **change})
    assert any(i.severity == "error" for i in labelling.check_all(dataset_dir, corpus))


def test_the_existing_event_types_still_validate(dataset_dir, corpus):
    for k in range(1, 13):
        add(dataset_dir, corpus, [answerable(k)])
    assert [e["event"] for e in _events(dataset_dir)] == ["label_review"] * 12
    labelling.record_audit_sample(dataset_dir, labelling.load_dataset(dataset_dir), 1, DATE)
    issues = labelling.check_all(dataset_dir, corpus)
    assert "AUDIT_RESULT_MISSING" in issue_codes(issues) and not [i for i in issues if i.severity == "error"]


def test_the_repository_audit_log_validates():
    root = Path(__file__).resolve().parent.parent / "eval"
    if not (root / "dataset" / "audit.jsonl").exists():
        pytest.skip("no audit log in this checkout")
    issues = labelling.check_all(root / "dataset", root / "corpus")
    assert [i for i in issues if i.severity == "error"] == []


# ---------- review freshness and validation separation (E1.6.2) ----------

def _readiness(dataset_dir):
    return labelling.review_readiness(labelling.load_dataset(dataset_dir), labelling.load_events(dataset_dir))


def _change_test_question(dataset_dir, edit):
    path = dataset_dir / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    edit(rows[0])
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.mark.parametrize("name", ["Claude", "Ai", "Madonna", "Sam"])
def test_single_word_names_and_ambiguous_words_are_accepted(dataset_dir, corpus, name):
    assert add(dataset_dir, corpus, [answerable(1)], review=labelling.Review(name, DATE)).written


def test_a_current_review_is_not_stale(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    readiness = _readiness(dataset_dir)
    assert readiness.stale_evidence_review == () and readiness.missing_evidence_review == ()
    assert readiness.current_evidence_reviews == 1 and readiness.ready


@pytest.mark.parametrize("field", ["question", "expected_answer"])
def test_a_changed_question_or_answer_makes_the_review_stale(dataset_dir, corpus, field):
    add(dataset_dir, corpus, [answerable(1)])
    _change_test_question(dataset_dir, lambda row: row.update({field: row[field] + " (edited)"}))
    stale = _readiness(dataset_dir).stale_evidence_review
    assert [qid for qid, _ in stale] == ["q1"] and "changed after the review" in stale[0][1]


def test_changed_evidence_makes_the_review_stale(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    _change_test_question(dataset_dir, lambda row: row["units"][0]["spans"][0].update(start=0))
    assert [qid for qid, _ in _readiness(dataset_dir).stale_evidence_review] == ["q1"]


def test_relabelling_difficulty_does_not_make_the_review_stale(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    _change_test_question(dataset_dir, lambda row: row.update(difficulty="paraphrase"))
    assert _readiness(dataset_dir).stale_evidence_review == ()


def test_a_later_review_supersedes_an_earlier_one_and_keeps_history(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    _change_test_question(dataset_dir, lambda row: row.update(question=row["question"] + " (edited)"))
    before = (dataset_dir / "audit.jsonl").read_bytes()
    assert _readiness(dataset_dir).stale_evidence_review
    assert labelling.record_evidence_review(dataset_dir, ["q1"], "Second Reviewer", DATE) == 1
    after = (dataset_dir / "audit.jsonl").read_bytes()
    assert after.startswith(before) and len(after) > len(before)  # append-only
    assert _readiness(dataset_dir).stale_evidence_review == () and _readiness(dataset_dir).ready


def test_record_evidence_review_refuses_placeholders_and_unknown_ids(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    with pytest.raises(labelling.LabellingError):
        labelling.record_evidence_review(dataset_dir, ["q1"], "TBD", DATE)
    with pytest.raises(labelling.LabellingError):
        labelling.record_evidence_review(dataset_dir, ["nope"], "Real Person", DATE)
    with pytest.raises(labelling.LabellingError):
        labelling.record_evidence_review(dataset_dir, ["q1"], "Real Person", "2026-13-40")


def test_a_review_without_a_hash_is_stale_only_after_a_later_evidence_correction(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    (dataset_dir / "audit.jsonl").write_text(json.dumps(
        {"event": "label_review", "question_id": "q1", "reviewer": "Old Reviewer", "date": DATE,
         "author_type": "human"}) + "\n", encoding="utf-8")
    assert _readiness(dataset_dir).stale_evidence_review == ()
    _log(dataset_dir, dict(CORRECTION, changes={"evidence": "span extended"}))
    assert [qid for qid, _ in _readiness(dataset_dir).stale_evidence_review] == ["q1"]


def test_a_malformed_review_hash_is_a_structural_error(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    event = dict(_events(dataset_dir)[0], content_sha256="XYZ")
    (dataset_dir / "audit.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert "AUDIT_EVENT_INVALID" in issue_codes(labelling.check_all(dataset_dir, corpus))


def test_check_passes_for_a_working_set_that_is_not_yet_reviewed(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1), answerable(2)])
    (dataset_dir / "audit.jsonl").unlink()
    assert [i for i in labelling.check_all(dataset_dir, corpus) if i.severity == "error"] == []
    assert _readiness(dataset_dir).missing_evidence_review == ("q1", "q2")


def test_a_pending_hindi_review_is_reported_by_readiness_not_by_check(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1, language="hi", question="डिवाइस नंबर एक का मान क्या है, प्रश्न एक?")],
        review=labelling.Review("Reviewer One", DATE, hindi_reviewer="Hindi Reviewer"))
    _change_test_question(dataset_dir, lambda row: row.update(hindi_reviewed=False, hindi_reviewer=None))
    kept = [e for e in _events(dataset_dir) if e["event"] != "hindi_review"]  # no review was recorded
    (dataset_dir / "audit.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept),
                                             encoding="utf-8")
    assert labelling.check_all(dataset_dir, corpus) == []
    readiness = _readiness(dataset_dir)
    assert readiness.hindi_review_pending == ("q1",) and readiness.hindi_questions == 1 and not readiness.ready


def test_an_inconsistent_hindi_review_is_still_a_structural_error(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1, language="hi", question="डिवाइस नंबर एक का मान क्या है, प्रश्न एक?")],
        review=labelling.Review("Reviewer One", DATE, hindi_reviewer="Hindi Reviewer"))
    _change_test_question(dataset_dir, lambda row: row.update(hindi_reviewed=True, hindi_reviewer=None))
    assert "Q_HINDI_REVIEWER_REQUIRED" in issue_codes(labelling.check_all(dataset_dir, corpus))


# The ten working-set unanswerable questions: du-01..du-05, du-07..du-09 on_topic_not_covered; du-06, du-10 off_topic;
# du-07..du-10 Hindi. Their drafts carry no semantic_review; the independent AI review is documented in the packet.
DU_IDS = [f"du-{n:02d}" for n in range(1, 11)]


def _errors(dataset_dir, corpus):
    return [i for i in labelling.check_all(dataset_dir, corpus) if i.severity == "error"]


def _working_set_unanswerables(dataset_dir, corpus):
    """Faithful fixture: ten unanswerable dev questions whose records name no human second reader."""
    items = [unanswerable(n, question_id=DU_IDS[n - 1], question=f"Question {n}: which gadget colour applies in room {n}?",
                          author_type="llm_assisted",
                          unanswerable_type="off_topic" if DU_IDS[n - 1] in ("du-06", "du-10") else "on_topic_not_covered")
             for n in range(1, 11)]
    add(dataset_dir, corpus, items, is_answerable=False,
        review=labelling.Review("Reviewer One", DATE, semantic_reviewer="Human Reader"))
    path = dataset_dir / "dev.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["semantic_review"] = None  # the working-set record has no human reader; an AI review is an event
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_readiness_separates_ai_reviewed_unanswerables_from_unverified_ones(dataset_dir, corpus):
    _working_set_unanswerables(dataset_dir, corpus)
    assert [i for i in labelling.check_all(dataset_dir, corpus) if i.severity == "error"] == []
    assert _readiness(dataset_dir).semantic_review_unverified == tuple(DU_IDS)  # no provenance recorded yet
    assert _readiness(dataset_dir).ai_only_semantic_review == ()
    assert labelling.record_ai_semantic_review(dataset_dir, DU_IDS, "review_packet.md, review status section") == 10
    readiness = _readiness(dataset_dir)
    assert readiness.ai_only_semantic_review == tuple(DU_IDS) and readiness.semantic_review_unverified == ()
    assert _errors(dataset_dir, corpus) == []


def test_an_ai_review_event_has_no_reviewer_model_or_invented_date(dataset_dir, corpus):
    _working_set_unanswerables(dataset_dir, corpus)
    labelling.record_ai_semantic_review(dataset_dir, ["du-01"], "packet")
    event = _events(dataset_dir)[-1]
    assert event == {"event": "ai_semantic_review", "question_id": "du-01",
                     "review_type": "independent_ai_semantic_review", "human_review": False,
                     "result": "confirmed_unanswerable", "basis": "packet"}


@pytest.mark.parametrize("change", [{"human_review": True}, {"review_type": "human"}, {"reviewer": "Some Person"},
                                    {"basis": " "}, {"date": "2026-02-30"}, {"result": "other"}, {"model": "x"}])
def test_an_invalid_ai_review_event_is_a_structural_error(dataset_dir, corpus, change):
    _working_set_unanswerables(dataset_dir, corpus)
    labelling.record_ai_semantic_review(dataset_dir, ["du-01"], "packet")
    _log(dataset_dir, dict(_events(dataset_dir)[-1], **change))
    assert "AUDIT_EVENT_INVALID" in issue_codes(labelling.check_all(dataset_dir, corpus))


def test_an_ai_review_cannot_be_recorded_for_an_answerable_or_unknown_question(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    for ids in (["q1"], ["nope"]):
        with pytest.raises(labelling.LabellingError):
            labelling.record_ai_semantic_review(dataset_dir, ids, "packet")


def test_a_human_semantic_reader_is_not_reported_as_ai_only(dataset_dir, corpus):
    add(dataset_dir, corpus, [unanswerable(1)], is_answerable=False,
        review=labelling.Review("Reviewer One", DATE, semantic_reviewer="Human Reader"))
    labelling.record_ai_semantic_review(dataset_dir, ["n1"], "packet")
    readiness = _readiness(dataset_dir)
    assert readiness.ai_only_semantic_review == () and readiness.semantic_review_unverified == ()


@pytest.mark.parametrize("name", ["AI: independent model review", "DRYRUN", "ChatGPT", "Claude Sonnet"])
def test_an_ai_or_placeholder_name_in_the_human_field_is_a_structural_error(dataset_dir, corpus, name):
    """AI review incorrectly represented as human."""
    _working_set_unanswerables(dataset_dir, corpus)
    path = dataset_dir / "dev.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["semantic_review"] = {"reviewer": name, "date": DATE, "result": "confirmed_unanswerable"}
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    issues = labelling.check_all(dataset_dir, corpus)
    assert [i.location for i in issues if i.code == "SEMANTIC_REVIEWER_NOT_A_PERSON"] == [
        "dev.questions[0].semantic_review.reviewer"]
    assert "du-01" in _readiness(dataset_dir).semantic_review_unverified


def test_no_semantic_review_is_structurally_valid_but_not_marked_reviewed(dataset_dir, corpus):
    _working_set_unanswerables(dataset_dir, corpus)
    assert _errors(dataset_dir, corpus) == []
    readiness = _readiness(dataset_dir)
    assert readiness.semantic_review_unverified == tuple(DU_IDS) and not readiness.ready


def test_a_missing_or_blank_ai_provenance_is_refused(dataset_dir, corpus):
    for basis in (None, "", "  "):
        with pytest.raises(labelling.LabellingError, match="human --semantic-reviewer or an --ai-semantic-basis"):
            add(dataset_dir, corpus, [unanswerable(1)], is_answerable=False,
                review=labelling.Review("Reviewer One", DATE, ai_semantic_basis=basis))
    assert not (dataset_dir / "dev.jsonl").exists() and not (dataset_dir / "audit.jsonl").exists()


def test_promoting_ai_reviewed_unanswerables_keeps_provenance_and_names_no_human(dataset_dir, corpus):
    items = [unanswerable(n, question_id=DU_IDS[n - 1], author_type="llm_assisted",
                          question=f"Question {n}: which gadget colour applies in room {n}?") for n in (1, 2)]
    result = add(dataset_dir, corpus, items, is_answerable=False,
                 review=labelling.Review("Reviewer One", DATE, ai_semantic_basis="review_packet.md"))
    assert result.written
    rows = [json.loads(line) for line in (dataset_dir / "dev.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["semantic_review"] for r in rows] == [None, None]  # no human reviewer, date or model is recorded
    ai_events = [e for e in _events(dataset_dir) if e["event"] == "ai_semantic_review"]
    assert [e["question_id"] for e in ai_events] == ["du-01", "du-02"]
    assert all(e["human_review"] is False and "reviewer" not in e and "date" not in e and "model" not in e
               for e in ai_events)
    assert _errors(dataset_dir, corpus) == []
    readiness = _readiness(dataset_dir)
    assert readiness.ai_only_semantic_review == ("du-01", "du-02") and readiness.semantic_review_unverified == ()
    assert readiness.ready is True  # an AI-only semantic review is reported but does not block
    assert labelling.Readiness(1, (), (), (), 0, ("du-01",)).ready


def test_promotion_with_a_human_reader_keeps_the_human_structure_and_adds_no_ai_event(dataset_dir, corpus):
    add(dataset_dir, corpus, [unanswerable(1)], is_answerable=False,
        review=labelling.Review("Reviewer One", DATE, semantic_reviewer="Human Reader"))
    row = json.loads((dataset_dir / "dev.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["semantic_review"] == {"reviewer": "Human Reader", "date": DATE, "result": "confirmed_unanswerable"}
    assert not [e for e in _events(dataset_dir) if e["event"] == "ai_semantic_review"]
    readiness = _readiness(dataset_dir)
    assert readiness.ai_only_semantic_review == () and readiness.semantic_review_unverified == ()


def test_promotion_with_a_placeholder_human_reader_is_still_refused(dataset_dir, corpus):
    with pytest.raises(labelling.LabellingError, match="--semantic-reviewer"):
        add(dataset_dir, corpus, [unanswerable(1)], is_answerable=False,
            review=labelling.Review("Reviewer One", DATE, semantic_reviewer="AI Assistant",
                                    ai_semantic_basis="review_packet.md"))


def test_promotion_adds_ai_events_without_changing_earlier_events(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    before = (dataset_dir / "audit.jsonl").read_bytes()
    add(dataset_dir, corpus, [unanswerable(1)], is_answerable=False,
        review=labelling.Review("Reviewer One", DATE, ai_semantic_basis="review_packet.md"))
    assert (dataset_dir / "audit.jsonl").read_bytes().startswith(before)


def test_cli_promotes_an_ai_reviewed_unanswerable_without_a_human_reader(capsys, corpus, dataset_dir, tmp_path):
    code, _, _ = run(capsys, corpus, dataset_dir, "add-unanswerable", "--spec",
                     write_spec(tmp_path, [unanswerable(1)]), "--evidence-reviewer", "Real Person",
                     "--ai-semantic-basis", "review_packet.md", "--date", DATE)
    assert code == 0
    code, _, err = run(capsys, corpus, dataset_dir, "add-unanswerable", "--spec",
                       write_spec(tmp_path, [unanswerable(2)]), "--evidence-reviewer", "Real Person", "--date", DATE)
    assert code == 2 and "--ai-semantic-basis" in err


# ---------- stale-review edge cases ----------

def _hashless_review(dataset_dir, qid="q1"):
    (dataset_dir / "audit.jsonl").write_text(json.dumps(
        {"event": "label_review", "question_id": qid, "reviewer": "Old Reviewer", "date": DATE,
         "author_type": "human"}) + "\n", encoding="utf-8")


def test_a_hashless_review_is_not_stale_after_a_difficulty_only_correction(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1, difficulty="paraphrase")])
    _hashless_review(dataset_dir)
    _log(dataset_dir, CORRECTION)  # difficulty and a description of evidence ...
    assert _readiness(dataset_dir).stale_evidence_review  # ... is stale because it names an evidence change
    _hashless_review(dataset_dir)
    _log(dataset_dir, dict(CORRECTION, changes={"difficulty": {"from": "paraphrase", "to": "direct"}}))
    assert _readiness(dataset_dir).stale_evidence_review == ()


def test_a_correction_before_the_review_does_not_stale_a_later_hashless_review(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    (dataset_dir / "audit.jsonl").write_text(
        json.dumps(dict(CORRECTION, changes={"evidence": "span extended"})) + "\n"
        + json.dumps({"event": "label_review", "question_id": "q1", "reviewer": "Old Reviewer", "date": DATE,
                      "author_type": "human"}) + "\n", encoding="utf-8")
    assert _readiness(dataset_dir).stale_evidence_review == ()


def test_every_evidence_correction_after_the_latest_review_stales_it_until_rereview(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])
    _hashless_review(dataset_dir)
    _log(dataset_dir, dict(CORRECTION, changes={"evidence": "first change"}),
         dict(CORRECTION, changes={"evidence": "second change"}))
    assert [q for q, _ in _readiness(dataset_dir).stale_evidence_review] == ["q1"]
    before = (dataset_dir / "audit.jsonl").read_bytes()
    labelling.record_evidence_review(dataset_dir, ["q1"], "Real Person", DATE)
    assert (dataset_dir / "audit.jsonl").read_bytes().startswith(before)
    assert _readiness(dataset_dir).stale_evidence_review == ()


def test_a_hashed_review_ignores_correction_events_and_follows_the_content(dataset_dir, corpus):
    add(dataset_dir, corpus, [answerable(1)])  # review carries a hash
    _log(dataset_dir, dict(CORRECTION, changes={"evidence": "described but the record is unchanged"}))
    assert _readiness(dataset_dir).stale_evidence_review == ()  # the hash is authoritative


CANONICAL = Path(__file__).resolve().parents[1] / "eval" / "dataset"


_NEEDS_CANONICAL = pytest.mark.skipif(
    not (CANONICAL / "audit.jsonl").exists() or not (CANONICAL / "test.jsonl").exists(),
    reason="the canonical dataset is not present")
_RE_REVIEWED = ("t-asy-01", "t-dkr-01")  # evidence corrected on 2026-11-13, then re-read by a person


def _canonical_copy(tmp_path: Path) -> Path:
    copy = tmp_path / "canonical"
    copy.mkdir()
    for name in ("dev.jsonl", "test.jsonl", "audit.jsonl"):  # dev.jsonl does not exist until dev questions are promoted
        if (CANONICAL / name).exists():
            (copy / name).write_bytes((CANONICAL / name).read_bytes())
    return copy


@_NEEDS_CANONICAL
def test_the_canonical_corrected_questions_have_current_evidence_reviews(tmp_path):
    copy = _canonical_copy(tmp_path)
    ds = labelling.load_dataset(copy)
    events = labelling.load_events(copy)
    corrected = {e["question_id"] for e in events if e["event"] == "label_correction" and "evidence" in e["changes"]}
    assert set(_RE_REVIEWED) <= corrected  # these two really were corrected after their first review
    questions = {q.question_id: q for q in ds.questions()}
    for qid in _RE_REVIEWED:
        reviews = [(i, e) for i, e in enumerate(events) if e["event"] == "label_review" and e["question_id"] == qid]
        latest_index, latest = reviews[-1]
        assert len(reviews) >= 2  # the earlier hashless review is kept, never rewritten
        assert latest["content_sha256"] == labelling.review_content_sha256(questions[qid])
        assert reviews[0][1].get("content_sha256") is None
        assert all(i < latest_index for i, e in enumerate(events) if e["event"] == "label_correction"
                   and e["question_id"] == qid and "evidence" in e["changes"])
        assert labelling.reviewer_problem(latest["reviewer"]) is None
    readiness = labelling.review_readiness(ds, events)
    assert not {q for q, _ in readiness.stale_evidence_review} & set(_RE_REVIEWED)
    assert readiness.stale_evidence_review == () and readiness.missing_evidence_review == ()
    # Every question in the copied dataset has a current review; the count follows the data, not a constant.
    reviewed = {e["question_id"] for e in events if e["event"] == "label_review"} & set(questions)
    assert readiness.current_evidence_reviews == len(reviewed) == len(questions)


@_NEEDS_CANONICAL
@pytest.mark.parametrize("qid", _RE_REVIEWED)
def test_a_canonical_evidence_change_without_a_fresh_review_is_stale(tmp_path, qid):
    copy = _canonical_copy(tmp_path)
    before = labelling.review_readiness(labelling.load_dataset(copy), labelling.load_events(copy))
    assert before.stale_evidence_review == ()
    path = copy / "test.jsonl"
    lines = path.read_text(encoding="utf-8").split("\n")
    changed = 0
    for number, line in enumerate(lines):
        if line.strip() and json.loads(line)["question_id"] == qid:
            record = json.loads(line)
            record["units"][0]["spans"][0]["quote_sha256"] = labelling.sha256_text("a different quote")
            lines[number] = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            changed += 1
    assert changed == 1
    path.write_text("\n".join(lines), encoding="utf-8", newline="")
    after = labelling.review_readiness(labelling.load_dataset(copy), labelling.load_events(copy))
    assert [q for q, _ in after.stale_evidence_review] == [qid]  # no other question is affected
    assert "changed after the review" in after.stale_evidence_review[0][1]


def test_readiness_and_record_commands_via_the_cli(capsys, corpus, dataset_dir):
    add(dataset_dir, corpus, [answerable(1)])
    code, out, _ = run(capsys, corpus, dataset_dir, "readiness")
    assert code == 0 and "ready for promotion" in out
    _change_test_question(dataset_dir, lambda row: row.update(question=row["question"] + " (edited)"))
    code, out, _ = run(capsys, corpus, dataset_dir, "readiness")
    assert code == 1 and "STALE" in out and "q1" in out
    code, out, _ = run(capsys, corpus, dataset_dir, "record-evidence-review", "--questions", "q1",
                       "--evidence-reviewer", "Real Person", "--date", DATE)
    assert code == 0
    assert run(capsys, corpus, dataset_dir, "readiness")[0] == 0
    code, out, _ = run(capsys, corpus, dataset_dir, "check")
    assert "structural validation only" in out


# ---------- separate Hindi review date (T1) ----------

HINDI_QUESTION = "विजेट संख्या एक का मान क्या है, प्रश्न एक?"
HINDI_DATE = "2026-11-10"


def _hindi_item(n: int = 1, **extra) -> dict:
    return answerable(n, language="hi", question=f"विजेट संख्या {n} का मान क्या है, प्रश्न {n}?", **extra)


def _hindi_unanswerable(n: int = 1) -> dict:
    return unanswerable(n, question=f"गैजेट {n} फ़ैक्टरी में किस रंग में रंगा जाता है?", query_language="hi")


def _event_dates(dataset_dir: Path, kind: str) -> dict[str, str]:
    return {e["question_id"]: e["date"] for e in _events(dataset_dir) if e["event"] == kind}


def test_the_hindi_review_date_defaults_to_the_review_date(dataset_dir, corpus):
    add(dataset_dir, corpus, [_hindi_item(1), answerable(2)], review=HINDI_REVIEW)
    assert _event_dates(dataset_dir, "hindi_review") == {"q1": DATE}
    assert _event_dates(dataset_dir, "label_review") == {"q1": DATE, "q2": DATE}


def test_an_explicit_hindi_review_date_applies_only_to_the_hindi_review_event(dataset_dir, corpus):
    review = labelling.Review("Reviewer One", DATE, hindi_reviewer="Hindi Reviewer", hindi_review_date=HINDI_DATE)
    result = add(dataset_dir, corpus, [_hindi_item(1), answerable(2)], review=review)
    assert result.written
    assert _event_dates(dataset_dir, "hindi_review") == {"q1": HINDI_DATE}
    assert _event_dates(dataset_dir, "label_review") == {"q1": DATE, "q2": DATE}
    stored = {q.question_id: q for q in labelling.load_dataset(dataset_dir).questions()}
    assert stored["q1"].hindi_reviewed is True and stored["q1"].hindi_reviewer == "Hindi Reviewer"
    assert stored["q2"].hindi_reviewed is False and stored["q2"].hindi_reviewer is None
    assert [i for i in labelling.check_all(dataset_dir, corpus) if i.severity == "error"] == []


def test_a_hindi_review_date_does_not_change_the_semantic_review_date(dataset_dir, corpus):
    review = labelling.Review("Reviewer One", DATE, semantic_reviewer="Second Reader", hindi_reviewer="Hindi Reviewer",
                              hindi_review_date=HINDI_DATE)
    add(dataset_dir, corpus, [_hindi_unanswerable(1), unanswerable(2)], review=review, is_answerable=False)
    stored = {q.question_id: q for q in labelling.load_dataset(dataset_dir).questions()}
    assert stored["n1"].semantic_review.date == stored["n2"].semantic_review.date == DATE
    assert _event_dates(dataset_dir, "hindi_review") == {"n1": HINDI_DATE}
    assert _event_dates(dataset_dir, "label_review") == {"n1": DATE, "n2": DATE}
    assert [i for i in labelling.check_all(dataset_dir, corpus) if i.severity == "error"] == []


def test_an_earlier_hindi_review_date_than_the_evidence_review_is_allowed(dataset_dir, corpus):
    review = labelling.Review("Reviewer One", "2026-11-14", hindi_reviewer="Hindi Reviewer", hindi_review_date="2026-11-13")
    add(dataset_dir, corpus, [_hindi_item(1)], review=review)
    assert _event_dates(dataset_dir, "hindi_review") == {"q1": "2026-11-13"}
    assert _event_dates(dataset_dir, "label_review") == {"q1": "2026-11-14"}


@pytest.mark.parametrize("bad", ["2026-02-30", "2026-13-01", "10/11/2026", "", "2026-11-1"])
def test_an_invalid_hindi_review_date_is_refused_and_nothing_is_written(dataset_dir, corpus, bad):
    review = labelling.Review("Reviewer One", DATE, hindi_reviewer="Hindi Reviewer", hindi_review_date=bad)
    with pytest.raises(labelling.LabellingError, match="Hindi review date"):
        add(dataset_dir, corpus, [_hindi_item(1)], review=review)
    assert sorted(p.name for p in dataset_dir.iterdir()) == []


def test_a_hindi_review_date_without_a_hindi_question_is_refused_and_nothing_is_written(dataset_dir, corpus):
    review = labelling.Review("Reviewer One", DATE, hindi_review_date=HINDI_DATE)
    with pytest.raises(labelling.LabellingError, match="no question in the batch is a Hindi question"):
        add(dataset_dir, corpus, [answerable(1), answerable(2, split="dev")], review=review)
    review = labelling.Review("Reviewer One", DATE, semantic_reviewer="Second Reader", hindi_review_date=HINDI_DATE)
    with pytest.raises(labelling.LabellingError, match="no question in the batch is a Hindi question"):
        add(dataset_dir, corpus, [unanswerable(1)], review=review, is_answerable=False)
    assert sorted(p.name for p in dataset_dir.iterdir()) == []


def test_cli_hindi_review_date_is_recorded_and_refused_without_hindi(capsys, corpus, dataset_dir, tmp_path):
    spec = write_spec(tmp_path, [_hindi_item(1), answerable(2)])
    code, out, _ = run(capsys, corpus, dataset_dir, "add-answerable", "--spec", spec, "--evidence-reviewer", "Rev",
                       "--hindi-reviewer", "Hindi Reviewer", "--date", DATE, "--hindi-review-date", HINDI_DATE)
    assert code == 0 and "added 2 question(s)" in out
    assert _event_dates(dataset_dir, "hindi_review") == {"q1": HINDI_DATE}
    assert _event_dates(dataset_dir, "label_review") == {"q1": DATE, "q2": DATE}
    code, _, err = run(capsys, corpus, dataset_dir, "add-answerable", "--spec", write_spec(tmp_path, [answerable(3)], "e.json"),
                       "--evidence-reviewer", "Rev", "--date", DATE, "--hindi-review-date", HINDI_DATE)
    assert code == 2 and "no question in the batch is a Hindi question" in err
    code, _, err = run(capsys, corpus, dataset_dir, "add-unanswerable", "--spec", write_spec(tmp_path, [_hindi_unanswerable(1)], "u.json"),
                       "--evidence-reviewer", "Rev", "--semantic-reviewer", "Second Reader", "--hindi-reviewer", "Hindi Reviewer",
                       "--date", DATE, "--hindi-review-date", "2026-99-99")
    assert code == 2 and "Hindi review date" in err
    assert [e["question_id"] for e in _events(dataset_dir) if e["event"] == "label_review"] == ["q1", "q2"]


# ---------- hardened blind audit (T2) ----------

BLIND_REVIEW = labelling.Review("Reviewer One", DATE, semantic_reviewer="Second Reader", hindi_reviewer="Hindi Reviewer")
BLIND_PERSON = "Blind Person"


def _blind_dataset(dataset_dir: Path, corpus: Path, answerables: int = 30, unanswerables: int = 4) -> None:
    items = [_hindi_item(n, split="test" if n % 2 else "dev") if n % 5 == 0 else answerable(n, split="test" if n % 2 else "dev")
             for n in range(answerables)]
    add(dataset_dir, corpus, items, review=BLIND_REVIEW)
    extra = [_hindi_unanswerable(n) if n % 2 else unanswerable(n) for n in range(unanswerables)]
    add(dataset_dir, corpus, extra, review=BLIND_REVIEW, is_answerable=False)


def _draw(capsys, corpus, dataset_dir, tmp_path, *, seed: int = 11, alias_seed: int = 5):
    """Audit questions, and the private map in a new private folder (created by the tool)."""
    public, private = tmp_path / "public", tmp_path / "private"
    public.mkdir(exist_ok=True)
    code, out, err = run(capsys, corpus, dataset_dir, "audit-sample", "--seed", str(seed), "--alias-seed", str(alias_seed),
                         "--out", str(public / "questions.json"), "--private-dir", str(private), "--date", DATE)
    assert code == 0, err
    return public / "questions.json", private / audit_bundle.MAP_FILE, out + err


def _real_ids(dataset_dir: Path) -> list[str]:
    return [q.question_id for q in labelling.load_dataset(dataset_dir).questions()]


def test_aliases_are_a_shuffled_bijection_of_the_sample(dataset_dir, corpus):
    _blind_dataset(dataset_dir, corpus)
    dataset = labelling.load_dataset(dataset_dir)
    ids = labelling.draw_audit_sample(dataset, 11)
    aliases = audit_bundle.make_aliases(dataset, ids, 5)
    assert list(aliases) == [f"A{n:02d}" for n in range(1, len(ids) + 1)]
    assert sorted(aliases.values()) == sorted(ids) and len(set(aliases.values())) == len(ids)
    assert audit_bundle.make_aliases(dataset, ids, 5) == aliases  # reproducible with a seed
    assert audit_bundle.make_aliases(dataset, ids, 6) != aliases


@pytest.mark.parametrize("alias_seed", range(40))
def test_the_audit_order_never_follows_the_ids_the_split_or_the_answerability(dataset_dir, corpus, alias_seed):
    _blind_dataset(dataset_dir, corpus)
    dataset = labelling.load_dataset(dataset_dir)
    by_id = {q.question_id: q for q in dataset.questions()}
    aliases = audit_bundle.make_aliases(dataset, labelling.draw_audit_sample(dataset, 11), alias_seed)
    order = [aliases[a] for a in sorted(aliases)]
    assert order != sorted(order) and order != sorted(order, reverse=True)
    for key in (lambda q: q.answerable, lambda q: q.split):
        values = [key(by_id[i]) for i in order]
        assert len(set(values)) > 1 and sum(1 for a, b in itertools.pairwise(values) if a != b) + 1 > len(set(values))


def test_the_ordering_check_names_every_way_an_order_could_reveal_a_label(dataset_dir, corpus):
    _blind_dataset(dataset_dir, corpus)
    by_id = {q.question_id: q for q in labelling.load_dataset(dataset_dir).questions()}
    ids = ["q1", "q2", "q3", "q4", "n0", "n1"]
    problem = audit_bundle._ordering_problem
    assert problem(sorted(ids), by_id) == "the order follows the question ids"
    assert problem(sorted(ids, reverse=True), by_id) == "the order follows the question ids"
    assert problem(["q1", "q4", "q2", "q3", "n0", "n1"], by_id) == "the order groups the questions by answerability"
    assert problem(["q1", "q3", "q2", "q4"], by_id) == "the order groups the questions by split"  # q1/q3 test, q2/q4 dev
    assert problem(["q1", "n0", "q4", "n1", "q2", "q3"], by_id) is None
    assert problem(["q1", "n0"], by_id) is None  # two questions cannot be checked


def test_an_order_that_follows_the_ids_or_the_classification_is_drawn_again(dataset_dir, corpus, monkeypatch):
    _blind_dataset(dataset_dir, corpus)
    dataset = labelling.load_dataset(dataset_dir)
    ids = ["q1", "q2", "q3", "q4", "n0", "n1"]
    calls = []
    real_random = random.Random

    class Scripted:
        def __init__(self, seed=None):
            self.real = real_random(1)

        def shuffle(self, items):
            calls.append(1)
            if len(calls) == 1:
                items.sort()  # follows the ids
            elif len(calls) == 2:
                items.sort(reverse=True)
            elif len(calls) == 3:
                items.sort(key=lambda i: i[0], reverse=True)  # all answerable first
            else:
                self.real.shuffle(items)

    monkeypatch.setattr(audit_bundle.random, "Random", Scripted)
    aliases = audit_bundle.make_aliases(dataset, ids, 1)
    by_id = {q.question_id: q for q in dataset.questions()}
    assert len(calls) >= 4 and audit_bundle._ordering_problem([aliases[a] for a in sorted(aliases)], by_id) is None


def test_aliases_refuse_a_bad_sample_or_seed(dataset_dir, corpus):
    _blind_dataset(dataset_dir, corpus)
    dataset = labelling.load_dataset(dataset_dir)
    for ids in ([], ["q1", "q1"], ["q1", "missing"]):
        with pytest.raises(labelling.LabellingError, match="each question once"):
            audit_bundle.make_aliases(dataset, ids, 1)
    for seed in (-1, True, "1", 1.5):
        with pytest.raises(labelling.LabellingError, match="alias seed"):
            audit_bundle.make_aliases(dataset, ["q1", "q2", "q3"], seed)


def test_answerable_and_unanswerable_questions_have_identical_auditor_structures(dataset_dir, corpus):
    _blind_dataset(dataset_dir, corpus)
    dataset = labelling.load_dataset(dataset_dir)
    ids = labelling.draw_audit_sample(dataset, 11)
    entries = audit_bundle.auditor_questions(dataset, audit_bundle.make_aliases(dataset, ids, 5))
    assert {tuple(sorted(e)) for e in entries} == {("audit_id", "query_language", "question")}
    assert {e["query_language"] for e in entries} == {"en", "hi"}
    text = json.dumps(entries, ensure_ascii=False)
    assert not any(real in text for real in _real_ids(dataset_dir))
    assert not any(q.expected_answer in text for q in dataset.questions() if q.expected_answer)


@pytest.mark.parametrize("mutate", [
    lambda e: e.update(extra="x"), lambda e: e.pop("question"), lambda e: e.update(audit_id="A09"),
    lambda e: e.update(question="  "), lambda e: e.update(query_language="fr"), lambda e: e.update(question=5),
])
def test_the_auditor_question_schema_is_an_allowlist(mutate):
    entries = [{"audit_id": f"A{n:02d}", "question": "Q?", "query_language": "en"} for n in (1, 2)]
    assert audit_bundle.validate_questions(entries) == entries
    mutate(entries[0])
    with pytest.raises(labelling.LabellingError):
        audit_bundle.validate_questions(entries)
    for bad in ([], {}, "x"):
        with pytest.raises(labelling.LabellingError):
            audit_bundle.validate_questions(bad)


def test_the_private_map_round_trips_and_is_tamper_evident(tmp_path):
    aliases = {"A01": "q3", "A02": "q1", "A03": "n0"}
    documents = {"D01": "doc-hi", "D02": "doc-en"}
    path = tmp_path / "map.json"
    audit_bundle.write_private_map(path, aliases, documents, DATE)
    assert audit_bundle.read_private_map(path) == aliases and audit_bundle.read_document_map(path) == documents
    with pytest.raises(FileExistsError):
        audit_bundle.write_private_map(path, aliases, documents, DATE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for edit in (lambda p: p["aliases"].update(A02="q9"), lambda p: p.update(format="x"),
                 lambda p: p.update(extra=1), lambda p: p["aliases"].pop("A03"),
                 lambda p: p["aliases"].update(A02="q3"), lambda p: p.pop("documents"),
                 lambda p: p["documents"].pop("D01"), lambda p: p["documents"].update(D02="doc-hi"),
                 lambda p: p["documents"].update(D02=""), lambda p: p.update(documents={})):
        broken = json.loads(json.dumps(payload))
        edit(broken)
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(labelling.LabellingError, match="private alias map") as info:
            audit_bundle.read_private_map(bad)
        assert not any(real in str(info.value) for real in ("q1", "q3", "n0", "q9", "doc-en", "doc-hi"))
    with pytest.raises(labelling.LabellingError, match="cannot be read"):
        audit_bundle.read_private_map(tmp_path / "missing.json")


def test_cli_audit_sample_writes_two_separate_files_and_prints_no_question_id(capsys, corpus, dataset_dir, tmp_path):
    _blind_dataset(dataset_dir, corpus)
    questions, private, output = _draw(capsys, corpus, dataset_dir, tmp_path)
    aliases = audit_bundle.read_private_map(private)
    assert questions.parent != private.parent and len(aliases) == 14
    assert not any(real in output for real in aliases.values()) and "A01" not in output
    entries = json.loads(questions.read_text(encoding="utf-8"))
    assert [e["audit_id"] for e in entries] == sorted(aliases)
    assert {tuple(sorted(e)) for e in entries} == {("audit_id", "query_language", "question")}
    assert not any(real in questions.read_text(encoding="utf-8") for real in _real_ids(dataset_dir))
    sample = [e for e in _events(dataset_dir) if e["event"] == "audit_sample"]
    assert len(sample) == 1 and sample[0]["question_ids"] == sorted(aliases.values())  # real ids stay in the audit log
    assert [i for i in labelling.check_all(dataset_dir, corpus) if i.severity == "error"] == []


def test_cli_audit_sample_refuses_unsafe_output_locations_and_writes_nothing(capsys, corpus, dataset_dir, tmp_path):
    _blind_dataset(dataset_dir, corpus)
    audit_before = (dataset_dir / "audit.jsonl").read_bytes()
    other = tmp_path / "other"
    other.mkdir()
    cases = [
        (tmp_path / "pv" / "q.json", tmp_path / "pv", "different folder"),
        (other / "q.json", dataset_dir / "private", "dataset or corpus folder"),
        (other / "q.json", corpus / "private", "dataset or corpus folder"),
    ]
    for out, private, message in cases:
        code, _, err = run(capsys, corpus, dataset_dir, "audit-sample", "--seed", "1", "--out", str(out),
                           "--private-dir", str(private))
        assert code == 2 and message in err and not out.exists() and not private.exists()
    existing = other / "exists.json"
    existing.write_text("x", encoding="utf-8")
    code, _, err = run(capsys, corpus, dataset_dir, "audit-sample", "--seed", "1", "--out", str(existing),
                       "--private-dir", str(tmp_path / "p"))
    assert code == 2 and "already exists" in err and not (tmp_path / "p").exists()
    (tmp_path / "taken").mkdir()
    code, _, err = run(capsys, corpus, dataset_dir, "audit-sample", "--seed", "1", "--out", str(other / "q.json"),
                       "--private-dir", str(tmp_path / "taken"))
    assert code == 2 and "already exists" in err and not (other / "q.json").exists()
    code, _, err = run(capsys, corpus, dataset_dir, "audit-sample", "--seed", "1", "--out", str(other / "q.json"),
                       "--private-dir", str(tmp_path / "p"), "--date", "2026-02-30")
    assert code == 2 and "real YYYY-MM-DD" in err and not (other / "q.json").exists() and not (tmp_path / "p").exists()
    assert (dataset_dir / "audit.jsonl").read_bytes() == audit_before


def test_an_interrupted_sample_deletes_nothing_and_records_nothing(capsys, corpus, dataset_dir, tmp_path, monkeypatch):
    _blind_dataset(dataset_dir, corpus)
    audit_before = (dataset_dir / "audit.jsonl").read_bytes()

    def fail(*args, **kwargs):
        raise labelling.LabellingError("writing failed")

    monkeypatch.setattr(labelling, "record_audit_sample", fail)
    (tmp_path / "a").mkdir()
    questions, private = tmp_path / "a" / "q.json", tmp_path / "b"
    code, _, err = run(capsys, corpus, dataset_dir, "audit-sample", "--seed", "1", "--out", str(questions),
                       "--private-dir", str(private))
    assert code == 2 and "writing failed" in err and "nothing was deleted" in err and "Interrupted sampling" in err
    assert questions.is_file() and (private / audit_bundle.MAP_FILE).is_file()  # kept for manual recovery
    assert (dataset_dir / "audit.jsonl").read_bytes() == audit_before
    monkeypatch.undo()
    code, _, err = run(capsys, corpus, dataset_dir, "audit-sample", "--seed", "1", "--out", str(questions),
                       "--private-dir", str(private))
    assert code == 2 and "already exists" in err  # a rerun never overwrites what the interrupted run left


def test_a_map_file_that_cannot_be_verified_stops_sampling_before_the_questions_or_the_sample(
        capsys, corpus, dataset_dir, tmp_path, monkeypatch):
    _blind_dataset(dataset_dir, corpus)
    audit_before = (dataset_dir / "audit.jsonl").read_bytes()
    original = audit_bundle.verify_private

    def refuse_files(path, *, directory):
        if not directory:
            raise labelling.LabellingError("the private folder is not restricted to this account")
        original(path, directory=directory)

    monkeypatch.setattr(audit_bundle, "verify_private", refuse_files)
    (tmp_path / "a").mkdir()
    code, _, err = run(capsys, corpus, dataset_dir, "audit-sample", "--seed", "1", "--out", str(tmp_path / "a" / "q.json"),
                       "--private-dir", str(tmp_path / "b"))
    assert code == 2 and "not restricted" in err and "nothing was deleted" in err
    assert not (tmp_path / "a" / "q.json").exists()  # no questions are written next to an unprotected map
    assert (dataset_dir / "audit.jsonl").read_bytes() == audit_before


def _bundle(capsys, corpus, dataset_dir, tmp_path):
    questions, private, _ = _draw(capsys, corpus, dataset_dir, tmp_path)
    out = tmp_path / "bundle"
    code, output, err = run(capsys, corpus, dataset_dir, "audit-bundle", "--questions", str(questions),
                            "--private-map", str(private), "--out-dir", str(out))
    return code, output, err, out, questions, private


def _bundle_files(out: Path) -> dict[str, bytes]:
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def test_the_auditor_bundle_has_exactly_the_expected_files_and_leaks_nothing(capsys, corpus, dataset_dir, tmp_path):
    _blind_dataset(dataset_dir, corpus)
    code, output, err, out, questions, private = _bundle(capsys, corpus, dataset_dir, tmp_path)
    assert code == 0, err
    assert "14 question(s)" in output
    files = _bundle_files(out)
    assert set(files) == {"AUDITOR_INSTRUCTIONS.md", "questions.md", "document_list.md", "response_worksheet.md",
                          "declaration.md", "corpus/D01.txt", "corpus/D02.txt"}
    manifest = labelling.load_manifest(corpus)
    texts = labelling.load_texts(corpus, manifest)
    documents = audit_bundle.read_document_map(private)
    assert sorted(documents.values()) == ["doc-en", "doc-hi"]
    for alias, key in documents.items():
        assert files[f"corpus/{alias}.txt"] == texts[key].encode("utf-8")
    assert not any(key in name for name in files for key in documents.values())  # no real corpus name in any path
    aliases = audit_bundle.read_private_map(private)
    generated = {n: b.decode("utf-8") for n, b in files.items() if not n.startswith("corpus/")}
    joined = "\n".join(generated.values())
    assert not any(key in joined for key in documents.values())
    assert "| `D01` |" in generated["document_list.md"] and "Title" not in generated["document_list.md"]
    assert not any(real in joined for real in _real_ids(dataset_dir))
    assert not any(token in joined for token in audit_bundle._FORBIDDEN_TOKENS)
    assert not any(name in joined for name in ("Reviewer One", "Second Reader", "Hindi Reviewer", "Rev"))
    assert json.dumps(aliases) not in joined and not any("map" in n.lower() for n in files)
    assert not any(q.expected_answer in joined for q in labelling.load_dataset(dataset_dir).questions() if q.expected_answer)
    assert not (out / "map.json").exists() and not any(out in p.parents for p in [private])
    for entry in json.loads(questions.read_text(encoding="utf-8")):
        assert entry["question"] in generated["questions.md"]
    blocks = re.findall(r"## (A\d\d)\n\nLanguage: (English|Hindi)\n\n(.+)\n", generated["questions.md"])
    assert [b[0] for b in blocks] == sorted(aliases) and len(blocks) == 14
    assert "14 questions against 2 documents" in generated["AUDITOR_INSTRUCTIONS.md"]
    assert "the 2 documents as plain-text files" in generated["AUDITOR_INSTRUCTIONS.md"] and "six" not in generated["AUDITOR_INSTRUCTIONS.md"]
    assert "answerable" not in generated["AUDITOR_INSTRUCTIONS.md"].lower().replace("not answered", "")
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".audit-bundle-")] == []


def test_bundle_building_fails_closed(capsys, corpus, dataset_dir, tmp_path):
    _blind_dataset(dataset_dir, corpus)
    questions, private, _ = _draw(capsys, corpus, dataset_dir, tmp_path)
    out = tmp_path / "bundle"

    def build(**override):
        args = {"questions": questions, "private": private, "out": out, **override}
        return run(capsys, corpus, dataset_dir, "audit-bundle", "--questions", str(args["questions"]),
                   "--private-map", str(args["private"]), "--out-dir", str(args["out"]))

    entries = json.loads(questions.read_text(encoding="utf-8"))
    for name, edit, message in (
        ("extra", lambda e: e[0].update(difficulty="direct"), "exactly the fields"),
        ("text", lambda e: e[0].update(question=e[0]["question"] + " x"), "does not match"),
        ("language", lambda e: e[0].update(query_language="hi" if e[0]["query_language"] == "en" else "en"), "does not match"),
        ("drop", lambda e: e.pop(), "do not list the same"),
    ):
        broken = json.loads(json.dumps(entries, ensure_ascii=False))
        edit(broken)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
        code, _, err = build(questions=path)
        assert code == 2 and message in err and not out.exists()
    code, _, err = build(out=private.parent)
    assert code == 2 and "already exists" in err
    code, _, err = build(out=tmp_path / "missing" / "b")
    assert code == 2 and "does not exist" in err
    other_map = tmp_path / "private2.json"
    audit_bundle.write_private_map(other_map, {a: r for a, r in zip(sorted(audit_bundle.read_private_map(private)),
                                                                     reversed(list(audit_bundle.read_private_map(private).values())))},
                                   audit_bundle.read_document_map(private), DATE)
    code, _, err = build(private=other_map)
    assert code == 2 and not out.exists()
    wrong_documents = tmp_path / "private3.json"
    audit_bundle.write_private_map(wrong_documents, audit_bundle.read_private_map(private),
                                   {"D01": "doc-en", "D02": "other-doc"}, DATE)
    code, _, err = build(private=wrong_documents)
    assert code == 2 and "documents do not match the corpus manifest" in err and not out.exists()
    inside = tmp_path / "bundle2"
    code, _, err = build(private=inside / "map.json", out=inside)
    assert code == 2 and "must not be inside the bundle folder" in err and not inside.exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".audit-bundle-")] == []
    assert (build()[0], out.is_dir()) == (0, True)


def test_a_bundle_needs_a_recorded_sample(capsys, corpus, dataset_dir, tmp_path):
    _blind_dataset(dataset_dir, corpus)
    questions, private, _ = _draw(capsys, corpus, dataset_dir, tmp_path)
    lines = (dataset_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    (dataset_dir / "audit.jsonl").write_text("".join(l + "\n" for l in lines if '"audit_sample"' not in l), encoding="utf-8")
    code, _, err = run(capsys, corpus, dataset_dir, "audit-bundle", "--questions", str(questions), "--private-map", str(private),
                       "--out-dir", str(tmp_path / "b"))
    assert code == 2 and "exactly one recorded audit sample" in err and not (tmp_path / "b").exists()


def _validation_context(capsys, corpus, dataset_dir, tmp_path):
    code, _, err, out, questions, private = _bundle(capsys, corpus, dataset_dir, tmp_path)
    assert code == 0, err
    manifest = labelling.load_manifest(corpus)
    return out, {
        "questions": json.loads(questions.read_text(encoding="utf-8")), "aliases": audit_bundle.read_private_map(private),
        "documents": audit_bundle.read_document_map(private), "manifest": manifest, "texts": labelling.load_texts(corpus, manifest), "dataset": labelling.load_dataset(dataset_dir),
        "events": labelling.load_events(dataset_dir),
    }


def test_validate_bundle_accepts_a_good_bundle_and_rejects_every_leak(capsys, corpus, dataset_dir, tmp_path):
    _blind_dataset(dataset_dir, corpus)
    out, context = _validation_context(capsys, corpus, dataset_dir, tmp_path)
    audit_bundle.validate_bundle(out, **context)
    dataset = context["dataset"]
    real = next(q for q in dataset.questions() if not q.answerable)
    answer = next(q for q in dataset.questions() if q.answerable)
    leaks = {
        "a real question id": f"see {real.question_id} here",
        "an internal label name": "retrieval is en_to_en",
        "a reviewer name": "thanks Second Reader",
        "a hash": "0" * 64,
        "a corpus file name": "see doc-en for that",
        "an expected answer or an evidence quote": answer.units[0].spans[0].quote,
    }
    for expected, injected in leaks.items():
        path = out / "AUDITOR_INSTRUCTIONS.md"
        original = path.read_bytes()
        path.write_bytes(original + b"\n" + injected.encode("utf-8"))
        with pytest.raises(labelling.LabellingError, match="not the expected content"):
            audit_bundle.validate_bundle(out, **context)
        path.write_bytes(original)
        # The content check fires first; the scan is exercised by a context whose expected files contain the leak.
        poisoned = {**context, "questions": [dict(q) for q in context["questions"]]}
        poisoned["questions"][0]["question"] = context["questions"][0]["question"] + " " + injected
        poisoned["dataset"] = labelling.Dataset(
            dev=tuple(dataclasses.replace(q, question=poisoned["questions"][0]["question"]) if q.question_id == context["aliases"]["A01"] else q
                      for q in dataset.dev),
            test=tuple(dataclasses.replace(q, question=poisoned["questions"][0]["question"]) if q.question_id == context["aliases"]["A01"] else q
                       for q in dataset.test))
        (out / "questions.md").write_bytes(audit_bundle._questions_md(poisoned["questions"]).encode("utf-8"))
        if expected == "an expected answer or an evidence quote":
            (out / "questions.md").write_bytes((out / "questions.md").read_bytes())  # strict scan skips question text by design
            audit_bundle.validate_bundle(out, **poisoned)
        else:
            with pytest.raises(labelling.LabellingError, match=f"questions.md contains .*{expected}"):
                audit_bundle.validate_bundle(out, **poisoned)
        (out / "questions.md").write_bytes(_bundle_files(out)["questions.md"])
        (out / "questions.md").write_text(audit_bundle._questions_md(context["questions"]), encoding="utf-8", newline="")
    audit_bundle.validate_bundle(out, **context)


def test_validate_bundle_rejects_structural_tampering(capsys, corpus, dataset_dir, tmp_path):
    _blind_dataset(dataset_dir, corpus)
    out, context = _validation_context(capsys, corpus, dataset_dir, tmp_path)
    (out / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(labelling.LabellingError, match="exactly the expected files"):
        audit_bundle.validate_bundle(out, **context)
    (out / "extra.txt").unlink()
    (out / "map.json").write_text("x", encoding="utf-8")
    with pytest.raises(labelling.LabellingError, match="exactly the expected files"):
        audit_bundle.validate_bundle(out, **context)
    (out / "map.json").unlink()
    corpus_file = out / "corpus" / "D01.txt"
    original = corpus_file.read_bytes()
    corpus_file.write_bytes(original.replace(b"Fact", b"fact", 1).replace(b"\xe0\xa4", b"\xe0\xa5", 1))
    with pytest.raises(labelling.LabellingError, match="D01.txt is not the expected content"):
        audit_bundle.validate_bundle(out, **context)
    corpus_file.write_bytes(original)
    (out / "declaration.md").unlink()
    with pytest.raises(labelling.LabellingError, match="exactly the expected files"):
        audit_bundle.validate_bundle(out, **context)
    (out / "declaration.md").write_text(audit_bundle._declaration(), encoding="utf-8", newline="")
    audit_bundle.validate_bundle(out, **context)
    real_name = out / "corpus" / f"{context['documents']['D01']}.txt"  # a real corpus name is never an allowed file
    real_name.write_bytes(b"x")
    with pytest.raises(labelling.LabellingError, match="exactly the expected files"):
        audit_bundle.validate_bundle(out, **context)
    real_name.unlink()
    context["documents"] = {**context["documents"], "D01": "other-doc"}
    with pytest.raises(labelling.LabellingError, match="documents do not match the corpus manifest"):
        audit_bundle.validate_bundle(out, **context)


def test_validate_bundle_rejects_a_link_on_every_platform(capsys, corpus, dataset_dir, tmp_path, monkeypatch):
    """No real symbolic link is made (Windows may not allow it): the link test is simulated, so the branch runs everywhere."""
    _blind_dataset(dataset_dir, corpus)
    out, context = _validation_context(capsys, corpus, dataset_dir, tmp_path)
    (out / "link.txt").write_text("x", encoding="utf-8")
    (out / "linkdir").mkdir()
    with pytest.raises(labelling.LabellingError, match="exactly the expected files"):  # a plain extra file is another error
        audit_bundle.validate_bundle(out, **context)
    original = Path.is_symlink
    for pretend in ("link.txt", "linkdir"):
        monkeypatch.setattr(Path, "is_symlink", lambda self, pretend=pretend: self.name == pretend or original(self))
        with pytest.raises(labelling.LabellingError, match="the bundle contains a link"):
            audit_bundle.validate_bundle(out, **context)
    monkeypatch.undo()
    (out / "link.txt").unlink()
    (out / "linkdir").rmdir()
    audit_bundle.validate_bundle(out, **context)


def test_validate_bundle_rejects_an_order_that_reveals_a_classification(capsys, corpus, dataset_dir, tmp_path):
    _blind_dataset(dataset_dir, corpus)
    out, context = _validation_context(capsys, corpus, dataset_dir, tmp_path)
    by_id = {q.question_id: q for q in context["dataset"].questions()}
    grouped = sorted(context["aliases"].values(), key=lambda i: (by_id[i].answerable, i), reverse=True)
    context["aliases"] = {alias: real for alias, real in zip(sorted(context["aliases"]), grouped)}
    context["questions"] = audit_bundle.auditor_questions(context["dataset"], context["aliases"])
    with pytest.raises(labelling.LabellingError, match="not blind"):
        audit_bundle.validate_bundle(out, **context)


def test_audit_results_with_an_alias_map_record_real_question_ids(capsys, corpus, dataset_dir, tmp_path):
    _blind_dataset(dataset_dir, corpus)
    _, private, _ = _draw(capsys, corpus, dataset_dir, tmp_path)
    aliases = audit_bundle.read_private_map(private)
    neutral = {key: doc for doc, key in audit_bundle.read_document_map(private).items()}
    dataset = {q.question_id: q for q in labelling.load_dataset(dataset_dir).questions()}
    spec = [{"question_id": alias, "spans": [] if not dataset[real].answerable else
             [{"corpus_key": neutral["doc-en"], "quote": FACTS[int(real[1:])]}]} for alias, real in aliases.items()]
    code, out, _ = run(capsys, corpus, dataset_dir, "audit-record", "--spec", write_spec(tmp_path, spec),
                       "--auditor", BLIND_PERSON, "--date", DATE, "--alias-map", str(private))
    assert code == 0 and "14 result(s); 0 disagreement(s)" in out
    results = [e for e in _events(dataset_dir) if e["event"] == "audit_result"]
    assert sorted(e["question_id"] for e in results) == sorted(aliases.values())
    assert {e["auditor"] for e in results} == {BLIND_PERSON}
    assert not any(e["question_id"] in aliases for e in results)  # no audit id is ever stored
    spans = {d[0] for e in results for d in e["derived"]}
    assert spans == {"doc-en"}  # the neutral document id is translated back to the real corpus key
    assert not any(real in out for real in aliases.values())
    code, out, _ = run(capsys, corpus, dataset_dir, "check")
    assert code == 0 and "AUDIT_" not in out


def test_alias_mode_refuses_unknown_real_repeated_and_mismatched_ids(capsys, corpus, dataset_dir, tmp_path):
    _blind_dataset(dataset_dir, corpus)
    _, private, _ = _draw(capsys, corpus, dataset_dir, tmp_path)
    aliases = audit_bundle.read_private_map(private)
    real = next(iter(aliases.values()))
    audit_before = (dataset_dir / "audit.jsonl").read_bytes()

    def record(items, map_path=private):
        return run(capsys, corpus, dataset_dir, "audit-record", "--spec", write_spec(tmp_path, items, "s.json"),
                   "--auditor", BLIND_PERSON, "--date", DATE, "--alias-map", str(map_path))

    for items, message in (
        ([{"question_id": "A99", "spans": []}], "not in the alias map"),
        ([{"question_id": real, "spans": []}], "not in the alias map"),
        ([{"question_id": "A01", "spans": []}, {"question_id": "A01", "spans": []}], "repeated"),
    ):
        code, out, err = record(items)
        assert code == 2 and message in err and not any(r in out + err for r in aliases.values())
    shifted = tmp_path / "shifted.json"
    audit_bundle.write_private_map(shifted, {**aliases, "A01": "q9999"}, audit_bundle.read_document_map(private), DATE)
    code, _, err = record([{"question_id": "A01", "spans": []}], shifted)
    assert code == 2 and "does not match the recorded audit sample" in err
    by_id = {q.question_id: q for q in labelling.load_dataset(dataset_dir).questions()}
    answerable_alias = next(a for a, r in aliases.items() if by_id[r].answerable and by_id[r].query_language == "en")
    neutral = {key: doc for doc, key in audit_bundle.read_document_map(private).items()}
    quote = FACTS[int(aliases[answerable_alias][1:])]
    for key, message in (("doc-en", "not in the document map"), ("D99", "not in the document map"),
                         (neutral["doc-en"].lower(), "not in the document map")):
        code, out, err = record([{"question_id": answerable_alias, "spans": [{"corpus_key": key, "quote": quote}]}])
        assert code == 2 and message in err
    with pytest.raises(labelling.LabellingError, match="used together"):
        labelling.record_audit_results(dataset_dir, corpus, [], auditor=BLIND_PERSON, date=DATE, alias_map=aliases)
    with pytest.raises(labelling.LabellingError, match="used together"):
        labelling.record_audit_results(dataset_dir, corpus, [], auditor=BLIND_PERSON, date=DATE,
                                       document_map=audit_bundle.read_document_map(private))
    assert (dataset_dir / "audit.jsonl").read_bytes() == audit_before


def test_audit_results_without_an_alias_map_still_use_real_ids(dataset_dir, corpus):
    _fill(dataset_dir, corpus, 3)
    labelling.record_audit_sample(dataset_dir, labelling.load_dataset(dataset_dir), 1, DATE)
    assert _audit(dataset_dir, corpus, [{"question_id": "q1", "spans": [{"corpus_key": "doc-en", "quote": FACTS[1]}]}])[0]["question_id"] == "q1"


# ---------- T2.1: neutral document names and the private folder ----------

def test_document_aliases_are_neutral_shuffled_and_reproducible(corpus):
    manifest = labelling.load_manifest(corpus)
    documents = audit_bundle.make_document_aliases(manifest, 4)
    assert sorted(documents) == ["D01", "D02"] and sorted(documents.values()) == ["doc-en", "doc-hi"]
    assert audit_bundle.make_document_aliases(manifest, 4) == documents
    assert {audit_bundle.make_document_aliases(manifest, n)["D01"] for n in range(30)} == {"doc-en", "doc-hi"}
    assert {audit_bundle.make_document_aliases(manifest)["D01"] for _ in range(60)} == {"doc-en", "doc-hi"}
    manifest = dataclasses.replace(manifest, documents=tuple(
        dataclasses.replace(d, corpus_key=f"k{i:02d}") for i, d in enumerate(manifest.documents * 6)))
    assert list(audit_bundle.make_document_aliases(manifest, 1)) == [f"D{n:02d}" for n in range(1, 13)]
    twice = dataclasses.replace(manifest, documents=manifest.documents[:1] * 2)
    with pytest.raises(labelling.LabellingError, match="each document once"):
        audit_bundle.make_document_aliases(twice, 1)


class _FakeWindows:
    """Stands in for whoami, icacls and PowerShell so the Windows branch runs on any platform.

    It is stateful and follows what real icacls does: ``/inheritance:r`` drops only INHERITED entries (this model starts
    with explicit ones only, so it keeps them all), ``/grant:r`` replaces the account's own Allow entry, ``/remove``
    deletes every entry of the named SIDs. The starting state is what a real Windows run showed right after
    ``os.mkdir(path, 0o700)`` on Python 3.12.4: four explicit Allow entries and inheritance off. ``acl`` replaces the
    PowerShell output with fixed text; ``sticky`` names SIDs that ``/remove`` cannot delete.
    """

    ME = "S-1-5-21-111-222-333-1001"
    SYSTEM, ADMINS, OWNER_RIGHTS = "S-1-5-18", "S-1-5-32-544", "S-1-3-4"

    def __init__(self, monkeypatch, acl: str | None = None, fail: str | None = None, rules=None, sticky=()):
        self.calls: list[list[str]] = []
        self.acl, self.fail, self.sticky = acl, fail, set(sticky)
        self.protected = True
        self.rules = ([("Allow", s) for s in (self.ME, self.SYSTEM, self.ADMINS, self.OWNER_RIGHTS)]
                      if rules is None else list(rules))
        monkeypatch.setattr(audit_bundle, "_IS_WINDOWS", True)
        monkeypatch.setattr(audit_bundle, "_run", self)

    def _icacls(self, tokens):
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "/inheritance:r":
                self.protected = True
            elif token == "/grant:r":
                i += 1
                sid = tokens[i].split(":")[0].lstrip("*")
                self.rules = [r for r in self.rules if not (r[1] == sid and r[0] == "Allow")] + [("Allow", sid)]
            elif token == "/remove":
                sids = {t.lstrip("*") for t in tokens[i + 1:]}
                self.rules = [r for r in self.rules if r[1] not in sids or r[1] in self.sticky]
                return
            else:
                raise AssertionError(f"unexpected icacls argument {token}")
            i += 1

    def __call__(self, command):
        self.calls.append(list(command))
        if command[0] == self.fail:
            raise labelling.LabellingError(f"the private folder's permissions could not be set or checked ({command[0]} failed)")
        if command[0] == "whoami":
            return f'"HOST\\user","{self.ME}"\n'
        if command[0] == "icacls":
            self._icacls(command[2:])
            return ""
        if command[0] == "powershell.exe":
            if self.acl is not None:
                return self.acl
            return f"PROTECTED={self.protected}\n" + "".join(f"{kind};{sid}\n" for kind, sid in self.rules)
        return ""


def test_windows_private_folder_is_restricted_and_verified(tmp_path, monkeypatch):
    fake = _FakeWindows(monkeypatch)
    private = audit_bundle.create_private_dir(tmp_path / "private")
    assert private.is_dir()
    folder = str((tmp_path / "private").resolve())
    icacls = [c for c in fake.calls if c[0] == "icacls"]
    assert icacls[0][1:] == [folder, "/inheritance:r", "/grant:r", f"*{fake.ME}:(OI)(CI)F"]
    # os.mkdir(0o700) left SYSTEM, Administrators and OWNER RIGHTS behind; they are read back and removed by SID
    assert icacls[1][1:] == [folder, "/remove", "*S-1-3-4", "*S-1-5-18", "*S-1-5-32-544"] and len(icacls) == 2
    assert fake.rules == [("Allow", fake.ME)] and fake.protected
    assert any(c[0] == "powershell.exe" and "Get-Acl" in c[-1] and "-LiteralPath" in c[-1] for c in fake.calls)
    path = private / audit_bundle.MAP_FILE
    audit_bundle.write_private_map(path, {"A01": "q1"}, {"D01": "doc-en"}, DATE)
    fake.acl = f"PROTECTED=False\nAllow;{fake.ME}\n"  # a file inherits its folder's access: inheritance stays on
    audit_bundle.verify_private(path, directory=False)
    with pytest.raises(labelling.LabellingError, match="not restricted to this account"):
        audit_bundle.verify_private(private, directory=True)  # but a folder must have inheritance off


@pytest.mark.parametrize("acl,fail,message", [
    ("PROTECTED=True\nAllow;S-1-5-21-111-222-333-1001\nAllow;S-1-5-32-544\n", None, "not restricted"),  # Administrators too
    ("PROTECTED=True\nAllow;S-1-5-21-111-222-333-1001\nAllow;S-1-1-0\n", None, "not restricted"),  # Everyone
    ("PROTECTED=False\nAllow;S-1-5-21-111-222-333-1001\n", None, "not restricted"),  # still inheriting
    ("PROTECTED=True\nDeny;S-1-5-21-111-222-333-1001\n", None, "not restricted"),
    ("PROTECTED=True\n", None, "not restricted"),  # no rules at all
    ("", None, "could not be read"),
    ("garbage\n", None, "could not be read"),
    ("PROTECTED=True\nAllow;not-a-sid\n", None, "could not be read"),
    (None, "icacls", "icacls failed"),
    (None, "powershell.exe", "powershell.exe failed"),
    (None, "whoami", "whoami failed"),
])
def test_windows_private_folder_fails_closed_and_leaves_no_folder(tmp_path, monkeypatch, acl, fail, message):
    _FakeWindows(monkeypatch, acl=acl, fail=fail)
    with pytest.raises(labelling.LabellingError, match=message):
        audit_bundle.create_private_dir(tmp_path / "private")
    assert not (tmp_path / "private").exists()  # the empty folder that was just made is removed


@pytest.mark.parametrize("extras", [
    [_FakeWindows.SYSTEM], [_FakeWindows.ADMINS], [_FakeWindows.OWNER_RIGHTS],
    [_FakeWindows.SYSTEM, _FakeWindows.ADMINS, _FakeWindows.OWNER_RIGHTS], ["S-1-1-0"],
], ids=["SYSTEM", "Administrators", "OWNER-RIGHTS", "all-three", "Everyone"])
def test_windows_entries_left_by_mkdir_are_removed_never_accepted(tmp_path, monkeypatch, extras):
    fake = _FakeWindows(monkeypatch, rules=[("Allow", _FakeWindows.ME), *(("Allow", s) for s in extras)])
    with pytest.raises(labelling.LabellingError, match="not restricted to this account"):
        audit_bundle.verify_private(tmp_path, directory=True)  # the verifier alone still refuses every one of them
    audit_bundle.create_private_dir(tmp_path / "private")
    assert fake.rules == [("Allow", fake.ME)] and fake.protected
    removals = [c for c in fake.calls if c[0] == "icacls" and "/remove" in c]
    assert len(removals) == 1 and removals[0][removals[0].index("/remove") + 1:] == [f"*{s}" for s in sorted(extras)]
    audit_bundle.verify_private(tmp_path / "private", directory=True)


def test_windows_makes_no_removal_when_only_the_account_is_present(tmp_path, monkeypatch):
    fake = _FakeWindows(monkeypatch, rules=[("Allow", _FakeWindows.ME)])
    audit_bundle.create_private_dir(tmp_path / "private")
    assert not any("/remove" in c for c in fake.calls)


@pytest.mark.parametrize("sticky,rules", [
    ({_FakeWindows.SYSTEM}, None),  # /remove had no effect on SYSTEM
    ({_FakeWindows.OWNER_RIGHTS}, None),
    ({_FakeWindows.ADMINS}, None),
    (set(), [("Deny", _FakeWindows.ME), ("Allow", _FakeWindows.ME)]),  # a Deny for the account itself is never removed
], ids=["SYSTEM-stays", "OWNER-RIGHTS-stays", "Administrators-stay", "own-Deny"])
def test_windows_an_entry_that_cannot_be_removed_fails_closed_and_leaves_no_folder(tmp_path, monkeypatch, sticky, rules):
    _FakeWindows(monkeypatch, sticky=sticky, rules=rules)
    with pytest.raises(labelling.LabellingError, match="not restricted to this account"):
        audit_bundle.create_private_dir(tmp_path / "private")
    assert not (tmp_path / "private").exists()


def test_the_account_running_the_process_must_be_identifiable(tmp_path, monkeypatch):
    fake = _FakeWindows(monkeypatch)
    for output in ("", "garbage", '"a","b","c"', '"HOST\\user","not-a-sid"'):
        monkeypatch.setattr(audit_bundle, "_run", lambda command, output=output: output)
        with pytest.raises(labelling.LabellingError, match="could not be identified"):
            audit_bundle._windows_sid()
    assert fake.ME


def test_a_windows_tool_that_cannot_run_or_fails_is_a_refusal(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("icacls")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(labelling.LabellingError, match="did not run"):
        audit_bundle._run(["icacls", "x"])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 5, "", ""))
    with pytest.raises(labelling.LabellingError, match="failed"):
        audit_bundle._run(["icacls", "x"])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "out", ""))
    assert audit_bundle._run(["icacls", "x"]) == "out"


def test_the_private_folder_must_be_new_and_have_an_existing_parent(tmp_path):
    (tmp_path / "taken").mkdir()
    with pytest.raises(labelling.LabellingError, match="already exists"):
        audit_bundle.create_private_dir(tmp_path / "taken")
    with pytest.raises(labelling.LabellingError, match="parent folder must exist"):
        audit_bundle.create_private_dir(tmp_path / "missing" / "private")


@pytest.mark.skipif(os.name == "nt", reason="owner-only mode bits are a POSIX mechanism; Windows is covered separately")
def test_posix_private_folder_and_map_are_owner_only(tmp_path):
    private = audit_bundle.create_private_dir(tmp_path / "private")
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    path = private / audit_bundle.MAP_FILE
    audit_bundle.write_private_map(path, {"A01": "q1"}, {"D01": "doc-en"}, DATE)
    assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
    audit_bundle.verify_private(path, directory=False)
    audit_bundle.verify_private(private, directory=True)
    path.chmod(0o644)
    with pytest.raises(labelling.LabellingError, match="not restricted"):
        audit_bundle.verify_private(path, directory=False)
    private.chmod(0o755)
    with pytest.raises(labelling.LabellingError, match="not restricted"):
        audit_bundle.verify_private(private, directory=True)
    with pytest.raises(labelling.LabellingError, match="not restricted"):
        audit_bundle.verify_private(path, directory=True)  # a file is not a folder
    with pytest.raises(labelling.LabellingError, match="cannot be inspected"):
        audit_bundle.verify_private(tmp_path / "missing", directory=True)


@pytest.mark.skipif(os.name != "nt", reason="needs the real Windows tools (icacls, PowerShell) and NTFS")
def test_windows_private_folder_with_the_real_tools(tmp_path):
    private = audit_bundle.create_private_dir(tmp_path / "private")
    path = private / audit_bundle.MAP_FILE
    audit_bundle.write_private_map(path, {"A01": "q1"}, {"D01": "doc-en"}, DATE)
    audit_bundle.verify_private(path, directory=False)
    protected, rules = audit_bundle._windows_acl(private.resolve())
    assert protected and {sid for _, sid in rules} == {audit_bundle._windows_sid()}
    audit_bundle._run(["icacls", str(private.resolve()), "/grant", "*S-1-1-0:(R)"])  # grant Everyone read
    with pytest.raises(labelling.LabellingError, match="not restricted"):
        audit_bundle.verify_private(private, directory=True)


# ---------- second-pass wording (E1.8): the bundle must not claim independence or blindness ----------

_ALLOWED_PHRASES = ("non-independent", "not an independent or blind validation", "independent or blind validation")


def _flat(text: str) -> str:
    """The text with every run of whitespace (including the line wrapping) as one space."""
    return " ".join(text.split())


def _wording_files(capsys, corpus, dataset_dir, tmp_path):
    _blind_dataset(dataset_dir, corpus)
    code, _, err, out, _, _ = _bundle(capsys, corpus, dataset_dir, tmp_path)
    assert code == 0, err
    return out, {n: b.decode("utf-8") for n, b in _bundle_files(out).items() if not n.startswith("corpus/")}


def _flat_files(files):
    return {n: _flat(text) for n, text in files.items()}


def test_the_instructions_disclose_a_non_independent_second_pass(capsys, corpus, dataset_dir, tmp_path):
    _, files = _wording_files(capsys, corpus, dataset_dir, tmp_path)
    text = _flat(files["AUDITOR_INSTRUCTIONS.md"])
    assert text.startswith("# Evidence audit: instructions (non-independent second pass)")
    assert "non-independent second pass" in text
    assert "has already seen the dataset labels, expected answers" in text
    assert "not an independent or blind validation" in text
    assert "Do not look up earlier labels" in text and "reminds you of an earlier decision" in text
    assert "they do not hide anything from the auditor" in text


def test_no_generated_file_claims_independence_blindness_or_that_the_auditor_saw_nothing(capsys, corpus, dataset_dir, tmp_path):
    _, files = _wording_files(capsys, corpus, dataset_dir, tmp_path)
    for name in ("AUDITOR_INSTRUCTIONS.md", "declaration.md"):
        text = _flat(files[name])
        for false_claim in ("have not seen", "not been given", "Independent evidence audit", "I have not",
                            "independent audit", "blind audit", "anyone who has seen"):
            assert false_claim not in text, (name, false_claim)
        remaining = text
        for phrase in _ALLOWED_PHRASES:
            remaining = remaining.replace(phrase, "")
        assert "independent" not in remaining.lower() and "blind" not in remaining.lower(), name
        assert "signed" not in text.lower()


def test_the_declaration_states_what_is_true_for_a_reviewer_who_has_seen_the_labels(capsys, corpus, dataset_dir, tmp_path):
    _, files = _wording_files(capsys, corpus, dataset_dir, tmp_path)
    raw = files["declaration.md"]
    text = _flat(raw)
    assert text.startswith("# Declaration (non-independent second pass)")
    assert "before this audit I have seen the labels, expected answers, evidence passages and earlier decisions" in text
    assert "no internet, search engine, AI assistant or other document" in text
    assert "I followed the rules in the instructions." in text
    assert raw.rstrip().endswith("Name:\n\nDate:\n\nSignature:")


def test_the_corrected_wording_keeps_the_privacy_protections(capsys, corpus, dataset_dir, tmp_path):
    out, files = _wording_files(capsys, corpus, dataset_dir, tmp_path)
    context = _validation_context_of(out, capsys, corpus, dataset_dir, tmp_path)
    audit_bundle.validate_bundle(out, **context)  # the unchanged leak scan accepts the new text
    joined = "\n".join(files.values())
    assert not any(token in joined for token in audit_bundle._FORBIDDEN_TOKENS)
    assert not any(real in joined for real in _real_ids(dataset_dir))
    assert not any(key in joined for key in context["documents"].values())
    assert not any(name in joined for name in ("Reviewer One", "Second Reader", "Hindi Reviewer", "Rev"))
    assert not re.search(r"[0-9a-f]{64}", joined) and json.dumps(context["aliases"]) not in joined
    assert "answerable" not in joined.lower().replace("not answered", "")


def _validation_context_of(out, capsys, corpus, dataset_dir, tmp_path):
    manifest = labelling.load_manifest(corpus)
    private = tmp_path / "private" / audit_bundle.MAP_FILE
    return {
        "questions": json.loads((tmp_path / "public" / "questions.json").read_text(encoding="utf-8")),
        "aliases": audit_bundle.read_private_map(private), "documents": audit_bundle.read_document_map(private),
        "manifest": manifest, "texts": labelling.load_texts(corpus, manifest),
        "dataset": labelling.load_dataset(dataset_dir), "events": labelling.load_events(dataset_dir),
    }


def test_the_evidence_procedure_is_unchanged(capsys, corpus, dataset_dir, tmp_path):
    _, files = _wording_files(capsys, corpus, dataset_dir, tmp_path)
    text = _flat(files["AUDITOR_INSTRUCTIONS.md"])
    for required in (
        "Use only the files in `corpus/`", "Do not use the internet, search engines, AI assistants or any other",
        "do not use outside knowledge", "Some questions may have no answer in the documents",
        "**Answered**", "**Not answered**", "**Unsure**", "record Unsure",
        "Copy each quote exactly from the text file", "Quote the smallest passage that states the fact",
        "give a separate quote for each fact", "give each place as its own quote", "Give the document key with every quote",
        "Your confidence: High, Medium or Low", "Do not guess.", "Judge each question on its own.",
        "Do not ask any other person or tool to find the evidence for you, and do not share these files.",
    ):
        assert required in text, required
    assert "completed `response_worksheet.md` and the completed `declaration.md`" in text


def test_a_bundle_with_the_old_claims_is_rejected_by_validation(capsys, corpus, dataset_dir, tmp_path):
    out, files = _wording_files(capsys, corpus, dataset_dir, tmp_path)
    context = _validation_context_of(out, capsys, corpus, dataset_dir, tmp_path)
    old_declaration = ("# Declaration\n\nI confirm that, before and during this audit, I have not seen any answer, label, "
                       "evidence passage, review decision\nor earlier audit material for these questions, that I used only "
                       "the files I was given, and that I followed the\nrules in the instructions.\n\nName:\n\nDate:\n\n"
                       "Signature:\n")
    (out / "declaration.md").write_text(old_declaration, encoding="utf-8", newline="")
    with pytest.raises(labelling.LabellingError, match="declaration.md is not the expected content"):
        audit_bundle.validate_bundle(out, **context)
    (out / "declaration.md").write_text(files["declaration.md"], encoding="utf-8", newline="")
    old_instructions = files["AUDITOR_INSTRUCTIONS.md"].replace(
        "(non-independent second pass)", "").replace("Evidence audit", "Independent evidence audit", 1)
    (out / "AUDITOR_INSTRUCTIONS.md").write_text(old_instructions, encoding="utf-8", newline="")
    with pytest.raises(labelling.LabellingError, match="AUDITOR_INSTRUCTIONS.md is not the expected content"):
        audit_bundle.validate_bundle(out, **context)
    (out / "AUDITOR_INSTRUCTIONS.md").write_text(files["AUDITOR_INSTRUCTIONS.md"], encoding="utf-8", newline="")
    audit_bundle.validate_bundle(out, **context)
