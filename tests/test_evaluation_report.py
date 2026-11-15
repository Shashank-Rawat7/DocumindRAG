"""E1.7.4: deterministic report serialization and run metadata (no database, no dataset files).

A  determinism, canonical form and JSON round trip
B  label and metadata completeness
C  hash and identity mismatch refusal (serialization)
D  comparison guard
E  privacy
F  oracle validity and unanswerable reporting
G  statistical suppression, truncation, warnings
H  non-finite and unserialisable values, metadata validation
I  code identity
J  purity
K  E2.2: the retrieval_baseline kind (pilot contract, code identity, freeze binding, loading, comparison)

Run focused: pytest tests/test_evaluation_report.py --noconftest -p no:cacheprovider
"""
import ast
import copy
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import test_evaluation_engine as fx  # the small hand-computed fixtures of E1.7.3

from app.rag.evaluation import baselines
from app.rag.evaluation import engine as e
from app.rag.evaluation import freeze_binding as fb
from app.rag.evaluation import pool as p
from app.rag.evaluation import ranking_hash as rh
from app.rag.evaluation import report as rep
from app.rag.evaluation import schema as s
from app.rag.evaluation import score_diagnostics as sd
from app.rag.evaluation.metrics import RankedChunk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OWNER = 481516          # unusual on purpose: it must never appear in a report
ID_A, ID_B = 7351, 6029
HALF = "half_coverage"

REMAP = {1: ID_A, 2: ID_A, 3: ID_A, 4: ID_B, 5: ID_B, 6: ID_B}
CHUNKS = {i: dataclasses.replace(c, document_id=REMAP[i]) for i, c in fx.C.items()}


def make_pool():
    rows = [
        p.PoolDocument(ID_A, OWNER, "doc-a.txt", "c1:doc-a", 1, 1, fx.EMB, fx.sha256_text(fx.TEXT_A)),
        p.PoolDocument(ID_B, OWNER, "doc-b.txt", "c1:doc-b", 1, 1, fx.EMB, fx.sha256_text(fx.TEXT_B)),
    ]
    dmap = p.map_documents(fx.MANIFEST, rows, owner_id=OWNER, embedding_id=fx.EMB)
    return p.build_pool(fx.MANIFEST, dmap, list(CHUNKS.values()), fx.TEXTS)


def retriever(query):
    return [CHUNKS[i] for i in fx.DENSE[query]]


IDENTITY = rep.combine_identity({"mod.b": "1" * 64, "mod.a": "0" * 64})
STAMP = "2026-11-14T08:30:00Z"


def meta(**over):
    base = {
        "run_timestamp_utc": STAMP, "python_version": "3.12.4", "embedder_id": fx.EMB,
        "embedder_signature": "sig-abc123", "code_identity": IDENTITY,
        "git_commit": "a" * 40, "working_tree_clean": False,
    }
    base.update(over)
    return rep.RunMetadata(**base)


@pytest.fixture(scope="module")
def bundle():
    inputs, pool = fx.inputs_of(fx.Q1, fx.Q2, fx.Q3, fx.Q4, fx.U1, fx.U2), make_pool()
    run = e.run_pilot(inputs, pool, retriever, seeds=fx.SEEDS, bootstrap_seed=7, resamples=fx.RESAMPLES,
                      adjudicated_unreachable={"q4": ["u1"]})
    return run, inputs, pool


@pytest.fixture(scope="module")
def rendered(bundle):
    return rep.render_report(*bundle, meta())


def refused(code, fn, *args, **kwargs):
    with pytest.raises(s.EvaluationDataError) as info:
        fn(*args, **kwargs)
    assert {i.code for i in info.value.issues} == {code}


# ---------- A: determinism, canonical form, round trip ----------

def test_identical_inputs_and_metadata_give_identical_bytes_and_hash(bundle, rendered):
    again = rep.render_report(*bundle, meta())
    assert again.json_bytes == rendered.json_bytes and again.sha256 == rendered.sha256
    assert len(rendered.sha256) == 64


def test_different_metadata_changes_the_bytes(bundle, rendered):
    other = rep.render_report(*bundle, meta(run_timestamp_utc="2026-11-14T08:31:00Z"))
    assert other.json_bytes != rendered.json_bytes and other.sha256 != rendered.sha256


def test_the_bytes_are_canonical_utf8_json(rendered):
    raw = rendered.json_bytes
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    text = raw.decode("utf-8")
    assert json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" == text
    assert "NaN" not in text and "Infinity" not in text
    assert text.startswith('{"code_identity":')  # keys are sorted
    assert rep.REPORT_LABEL in text  # the em dash is written as is, not escaped


def test_the_report_round_trips_through_json(rendered):
    loaded = rep.load_report(rendered.json_bytes)
    assert loaded == rendered.document
    assert rep.serialize_report(loaded) == rendered.json_bytes
    assert json.loads(rendered.json_bytes) == rendered.document


def test_the_per_question_order_is_the_dev_file_order(rendered):
    for section in ("dense", "oracle", "random"):
        assert [q["question_id"] for q in rendered.document[section]["questions"]] == ["q1", "q2", "q3", "q4"]


# ---------- B: label and metadata ----------

def test_the_report_is_labelled_a_validity_pilot_not_the_v1_baseline(rendered):
    doc = rendered.document
    assert doc["label"] == "Validity pilot — not the V1 baseline"
    assert doc["kind"] == "validity_pilot" and doc["is_v1_baseline"] is False
    assert doc["report_format_version"] == rep.REPORT_FORMAT_VERSION


def test_hashes_and_corpus_identity_are_recorded(bundle, rendered):
    run, inputs, pool = bundle
    i = rendered.document["inputs"]
    assert i["manifest"]["sha256"] == inputs.manifest_sha256 == run.manifest_sha256
    assert i["dev_dataset"]["sha256"] == inputs.dev_dataset_sha256
    assert i["candidate_pool"] == {"sha256": pool.pool_sha256, "size": 6}
    assert i["manifest"]["corpus_version"] == "c1" and i["manifest"]["stage"] == "ingested"
    assert i["manifest"]["document_count"] == 2 and i["corpus_keys"] == ["doc-a", "doc-b"]
    assert i["dev_dataset"]["split"] == "dev" and i["dev_dataset"]["test_split_read"] is False
    assert (i["dev_dataset"]["answerable_count"], i["dev_dataset"]["unanswerable_count"]) == (4, 2)


def test_run_embedder_and_code_metadata_are_recorded(rendered):
    doc = rendered.document
    assert doc["run"] == {"timestamp_utc": STAMP, "python_version": "3.12.4", "git_commit": "a" * 40,
                          "working_tree_clean": False}
    assert doc["embedder"] == {"id": fx.EMB, "signature": "sig-abc123"}
    assert doc["code_identity"]["combined_sha256"] == IDENTITY.combined_sha256
    assert doc["code_identity"]["files"] == [["mod.a", "0" * 64], ["mod.b", "1" * 64]]
    assert doc["code_identity"]["method"] == rep.CODE_IDENTITY_METHOD


def test_settings_record_cutoffs_rules_seeds_and_statistics(bundle, rendered):
    st = rendered.document["settings"]
    assert st["cutoffs"] == [1, 3, 5, 10, 20] and st["max_rank"] == 20
    assert st["relevance_rules"] == ["any_overlap", "half_coverage", "full_coverage"]
    assert st["primary_rule"] == HALF and st["random_seeds"] == list(range(1000)) and st["random_seed_count"] == 1000
    assert st["bootstrap_seed"] == 7 and st["resamples"] == fx.RESAMPLES
    assert st["confidence_level"] == 0.95 and st["min_slice_questions"] == 10
    assert st["oracle_method"] == baselines.ORACLE_METHOD and st["percentile_convention"] == "nearest_rank"
    assert st["random_reproducibility"]["algorithm"] == baselines.RANDOM_ALGORITHM
    assert st["metric_count"] == 76 and len(st["series"]["binary_hit"]) == 76
    assert {x["kind"] for x in st["series"]["all_continuous"]} == {"continuous"}


def test_per_question_results_carry_metrics_rankings_and_random_percentiles(bundle, rendered):
    run = bundle[0]
    dense = rendered.document["dense"]["questions"][2]
    assert dense["metrics"] == dict(run.dense_observations[2].values)
    assert dense["metrics"]["raw_unit_recall_at_5_half_coverage"] == 0.5
    assert set(dense["ranking"][0]) == {"corpus_key", "start", "end", "text_sha256"} and len(dense["ranking"]) == 6
    assert dense["retrieval_type"] == "en_to_en" and dense["unit_structure"] == "multi_unit"
    rnd = rendered.document["random"]["questions"][0]
    assert rnd["n_seeds"] == 1000 and set(rnd) >= {"metrics_mean", "metrics_p2_5", "metrics_p95", "metrics_p97_5"}
    assert rnd["metrics_p2_5"]["raw_hit_at_1_half_coverage"] <= rnd["metrics_p97_5"]["raw_hit_at_1_half_coverage"]
    macro = rendered.document["random"]["macro"]
    assert len(macro) == 76 and set(macro[0]) >= {"metric", "n_seeds", "mean", "p2_5", "p95", "p97_5"}


def test_the_dense_versus_random_comparison_is_reported_without_a_threshold(rendered):
    block = rendered.document["dense_vs_random"]
    assert block["baseline"] == "random" and block["candidate"] == "dense" and "no dense-versus-random" in block["note"]
    series = next(x for x in block["report"]["overall"]["series"] if x["name"] == "raw_hit_at_1_half_coverage")
    assert series["status"] == "ok" and 0.28 < series["mean_difference"] < 0.39
    flat = json.dumps(rendered.document).lower()
    assert "threshold_passed" not in flat and '"passed"' not in flat and '"pass_fail"' not in flat


# ---------- C: hash and identity mismatch ----------

@pytest.mark.parametrize("field", ["manifest_sha256", "dev_dataset_sha256", "pool_sha256"])
def test_a_run_hash_that_disagrees_with_the_inputs_or_pool_is_refused(bundle, field):
    run, inputs, pool = bundle
    refused(rep.REPORT_HASH_MISMATCH, rep.build_report, dataclasses.replace(run, **{field: "0" * 64}), inputs, pool, meta())


def test_a_pool_size_mismatch_is_refused(bundle):
    run, inputs, pool = bundle
    refused(rep.REPORT_HASH_MISMATCH, rep.build_report, dataclasses.replace(run, pool_size=7), inputs, pool, meta())


def test_inputs_whose_recorded_hash_does_not_describe_their_content_are_refused(bundle):
    run, inputs, pool = bundle
    forged = dataclasses.replace(inputs, dev_dataset_sha256="0" * 64)
    refused(rep.REPORT_HASH_MISMATCH, rep.build_report, dataclasses.replace(run, dev_dataset_sha256="0" * 64),
            forged, pool, meta())
    forged_manifest = dataclasses.replace(inputs, manifest_sha256="1" * 64)
    refused(rep.REPORT_HASH_MISMATCH, rep.build_report, dataclasses.replace(run, manifest_sha256="1" * 64),
            forged_manifest, pool, meta())


def test_the_embedder_must_be_the_one_of_the_pool(bundle):
    run, inputs, pool = bundle
    refused(rep.REPORT_EMBEDDER_MISMATCH, rep.build_report, run, inputs, pool, meta(embedder_id="other-embedder"))


def test_question_ids_of_every_version_must_match_the_dev_split(bundle):
    run, inputs, pool = bundle
    reordered = dataclasses.replace(run, dense_observations=tuple(reversed(run.dense_observations)))
    refused(rep.REPORT_QUESTION_MISMATCH, rep.build_report, reordered, inputs, pool, meta())
    dropped = dataclasses.replace(run, random_observations=run.random_observations[:-1])
    refused(rep.REPORT_QUESTION_MISMATCH, rep.build_report, dropped, inputs, pool, meta())
    swapped = dataclasses.replace(run, unanswerable=run.unanswerable[:1])
    refused(rep.REPORT_QUESTION_MISMATCH, rep.build_report, swapped, inputs, pool, meta())


def test_argument_types(bundle):
    run, inputs, pool = bundle
    for args in (("x", inputs, pool), (run, "x", pool), (run, inputs, "x")):
        with pytest.raises(TypeError):
            rep.build_report(*args, meta())


# ---------- D: comparison guard ----------

def clone(doc):
    return copy.deepcopy(doc)


def test_identical_labels_are_comparable(rendered):
    rep.check_comparable(rendered.document, clone(rendered.document))
    rep.check_comparable(rendered.document, rep.load_report(rendered.json_bytes))


@pytest.mark.parametrize("path", [
    ("inputs", "manifest", "sha256"), ("inputs", "dev_dataset", "sha256"),
])
def test_reports_with_different_label_hashes_are_never_comparable(rendered, path):
    other = clone(rendered.document)
    target = other
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = "f" * 64
    refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, rendered.document, other)
    refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, other, rendered.document)


def test_different_questions_series_or_cutoffs_are_not_comparable(rendered):
    for mutate in (
        lambda d: d["dense"]["questions"].pop(),
        lambda d: d["settings"]["series"]["binary_hit"].pop(),
        lambda d: d["settings"]["cutoffs"].pop(),
        lambda d: d["settings"]["relevance_rules"].pop(),
    ):
        other = clone(rendered.document)
        mutate(other)
        refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, rendered.document, other)


def test_a_different_pool_is_refused_unless_explicitly_allowed(rendered):
    other = clone(rendered.document)
    other["inputs"]["candidate_pool"]["sha256"] = "e" * 64
    refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, rendered.document, other)
    rep.check_comparable(rendered.document, other, require_same_pool=False)


def test_a_different_embedder_or_time_does_not_block_comparison(rendered):
    other = clone(rendered.document)
    other["embedder"]["id"] = "another"
    other["run"]["timestamp_utc"] = "2027-01-01T00:00:00Z"
    rep.check_comparable(rendered.document, other)


def test_a_document_that_is_not_a_report_is_refused(rendered):
    refused(rep.REPORT_INVALID, rep.check_comparable, {}, rendered.document)
    refused(rep.REPORT_INVALID, rep.check_comparable, rendered.document, {"kind": "validity_pilot"})
    broken = clone(rendered.document)
    del broken["inputs"]
    refused(rep.REPORT_INVALID, rep.check_comparable, rendered.document, broken)


def test_load_report_rejects_other_labels_and_garbage(rendered):
    doc = clone(rendered.document)
    doc["is_v1_baseline"] = True
    refused(rep.REPORT_INVALID, rep.load_report, rep.serialize_report(doc))
    doc = clone(rendered.document)
    doc["label"] = "V1 baseline"
    refused(rep.REPORT_INVALID, rep.load_report, rep.serialize_report(doc))
    refused(rep.REPORT_INVALID, rep.load_report, b"not json")
    refused(rep.REPORT_INVALID, rep.load_report, b"[1]")
    refused(rep.REPORT_INVALID, rep.load_report, b"\xff\xfe")
    with pytest.raises(TypeError):
        rep.load_report("{}")


# ---------- E: privacy ----------

def test_no_question_answer_quote_source_text_or_database_id_is_in_the_report(bundle, rendered):
    _, inputs, _ = bundle
    text = rendered.json_bytes.decode("utf-8")
    for question in inputs.questions:
        assert question.question not in text
        for unit in question.units or ():
            for span in unit.spans:
                assert span.quote not in text
    for source in fx.TEXTS.values():
        assert source[:12] not in text and source not in text
    for word in ("Alpha", "Zeta", "clause", "section"):
        assert word not in text
    for secret in ("owner", "password", "DATABASE_URL", "SECRET_KEY", "postgresql"):
        assert secret not in text
    # digits can occur inside hashes, so the numeric ids are looked for as numbers
    assert not {OWNER, ID_A, ID_B} & set(_numbers(rendered.document))


def _numbers(value):
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)


def test_a_leak_through_any_metadata_string_is_refused(bundle):
    run, inputs, pool = bundle
    for leak in (fx.Q1.question, "x " + fx.Q2.question + " y", "Alpha clause one."):
        refused(rep.REPORT_LEAK, rep.build_report, run, inputs, pool, meta(embedder_signature=leak))


def test_an_expected_answer_is_also_forbidden():
    answer = "A distinctive expected answer"
    question = dataclasses.replace(fx.Q1, expected_answer=answer)
    inputs = dataclasses.replace(fx.inputs_of(fx.Q1, fx.U1), questions=(question,))
    refused(rep.REPORT_LEAK, rep._check_no_leak, {"a": ["see " + answer]}, inputs)
    rep._check_no_leak({"a": ["clean"]}, inputs)


def test_short_strings_are_not_mistaken_for_leaks(bundle):
    run, inputs, pool = bundle
    assert rep.build_report(run, inputs, pool, meta(embedder_signature="v1"))


# ---------- F: oracle validity and unanswerable ----------

def test_oracle_validity_unreachable_units_and_adjudications_are_reported(rendered):
    validity = rendered.document["oracle"]["validity"]
    assert validity["ok"] is True and validity["rule"] == HALF and validity["method"] == baselines.ORACLE_METHOD
    checks = {c["question_id"]: c for c in validity["checks"]}
    assert set(checks) == {"q1", "q2", "q3", "q4"}
    q4 = checks["q4"]
    assert q4["adjudicated"] == ["u1"] and q4["unadjudicated"] == [] and q4["reachable_units"] == 0
    assert q4["hit_at_1"] is None and q4["unit_recall_at_20"] is None and q4["ok"] is True
    assert dict(map(tuple, q4["unreachable_by_rule"]))[HALF] == ["u1"]
    assert checks["q3"]["hit_at_1"] == 1.0 and checks["q3"]["unit_recall_at_20"] == 1.0
    per_question = {q["question_id"]: q for q in rendered.document["oracle"]["questions"]}
    assert per_question["q4"]["unreachable_by_rule"][HALF] == ["u1"]
    assert per_question["q4"]["reachable_only"][HALF] == {"unit_count": 0, "metrics": None}
    assert per_question["q3"]["chunks_needed"] == 2 and per_question["q3"]["method"] == baselines.ORACLE_METHOD
    assert per_question["q3"]["reachable_only"][HALF]["metrics"]["raw_unit_recall_at_1_half_coverage"] == 0.5


def test_an_unadjudicated_unreachable_unit_is_reported_as_not_ok():
    inputs, pool = fx.inputs_of(fx.Q1, fx.Q4, fx.U1), make_pool()
    run = e.run_pilot(inputs, pool, retriever, seeds=fx.SEEDS, bootstrap_seed=1, resamples=fx.RESAMPLES)
    doc = rep.build_report(run, inputs, pool, meta())
    assert doc["oracle"]["validity"]["ok"] is False
    q4 = next(c for c in doc["oracle"]["validity"]["checks"] if c["question_id"] == "q4")
    assert q4["unadjudicated"] == ["u1"] and q4["adjudicated"] == [] and q4["ok"] is False


def test_unanswerable_questions_are_reported_apart_without_retrieval_metrics(rendered):
    block = rendered.document["unanswerable"]
    assert block["count"] == 2 and block["retrieval_metrics"] is None and "no answerable retrieval metric" in block["note"]
    assert block["questions"] == [
        {"question_id": "du1", "query_language": "en", "unanswerable_type": "off_topic"},
        {"question_id": "du2", "query_language": "en", "unanswerable_type": "off_topic"},
    ]
    for section in ("dense", "oracle", "random"):
        assert all(q["question_id"] not in ("du1", "du2") for q in rendered.document[section]["questions"])
        outcomes = [o for r in rendered.document[section]["report"]["overall"]["series"] for o in r["outcomes"]]
        assert all(o[0] not in ("du1", "du2") for o in outcomes)


# ---------- G: suppression, truncation, warnings ----------

def test_small_slices_are_marked_suppressed_and_carry_no_statistics(rendered):
    doc = rendered.document
    entries = doc["suppressed_slices"]
    assert len(entries) == 12  # four retrieval types, each suppressed, in three reports
    assert {(x["report"], x["retrieval_type"]) for x in entries} == {
        (r, t) for r in ("dense", "oracle", "random") for t in ("en_to_en", "hi_to_hi", "en_to_hi", "hi_to_en")}
    assert {x["n"] for x in entries if x["retrieval_type"] == "en_to_en"} == {4}
    assert all("fewer than 10" in x["reason"] for x in entries)
    en = next(sl for sl in doc["dense"]["report"]["slices"] if sl["retrieval_type"] == "en_to_en")
    for result in en["series"]:
        assert result["status"] == "suppressed" and result["rate_or_mean"] is None and result["interval"] is None
        assert len(result["outcomes"]) == 4
    overall = doc["dense"]["report"]["overall"]["series"][0]
    assert overall["status"] == "ok" and overall["interval"] is not None


def test_truncation_counts_are_reported_for_every_cutoff(rendered):
    counts = rendered.document["truncation_counts"]
    assert counts["dense"] == [[1, 0], [3, 0], [5, 0], [10, 4], [20, 4]]  # six chunks cannot fill k = 10 or 20
    assert counts["oracle"] == counts["dense"] and counts["random"] == counts["dense"]


def test_dev_gate_warnings_are_listed_without_their_messages_carrying_text(bundle, rendered):
    _, inputs, _ = bundle
    assert len(rendered.document["warnings"]) == len(inputs.warnings)
    assert all(set(w) == {"code", "location", "severity"} for w in rendered.document["warnings"])


# ---------- H: values and metadata validation ----------

def test_non_finite_numbers_and_unknown_types_are_refused():
    for bad in (math.nan, math.inf, -math.inf):
        refused(rep.REPORT_NONFINITE, rep._plain, {"x": [bad]})
    for bad in ({1, 2}, object(), b"x", {1: 2}):
        refused(rep.REPORT_UNSERIALIZABLE, rep._plain, {"x": bad})
    refused(rep.REPORT_UNSERIALIZABLE, rep.serialize_report, {"x": math.nan})
    refused(rep.REPORT_UNSERIALIZABLE, rep.serialize_report, {"x": object()})


def test_the_plain_conversion_turns_dataclasses_and_tuples_into_json_types():
    assert rep._plain({"a": (1, 2.5, None, True, "s")}) == {"a": [1, 2.5, None, True, "s"]}


@pytest.mark.parametrize("over", [
    {"run_timestamp_utc": "2026-11-14 08:30:00"}, {"run_timestamp_utc": "2026-11-14T08:30:00+05:30"},
    {"run_timestamp_utc": None}, {"python_version": "3.12"}, {"python_version": 3.12},
    {"embedder_id": ""}, {"embedder_id": None}, {"embedder_signature": ""}, {"embedder_signature": 5},
    {"git_commit": "abc"}, {"git_commit": "A" * 40}, {"working_tree_clean": "yes"},
    {"code_identity": None}, {"code_identity": rep.CodeIdentity("m", (), "0" * 64)},
])
def test_invalid_metadata_is_refused(bundle, over):
    run, inputs, pool = bundle
    refused(rep.REPORT_METADATA_INVALID, rep.build_report, run, inputs, pool, meta(**over))


def test_optional_git_fields_may_be_absent(bundle):
    run, inputs, pool = bundle
    doc = rep.build_report(run, inputs, pool, meta(git_commit=None, working_tree_clean=None))
    assert doc["run"]["git_commit"] is None and doc["run"]["working_tree_clean"] is None
    refused(rep.REPORT_METADATA_INVALID, rep.build_report, run, inputs, pool, "metadata")


def test_the_timestamp_and_python_version_helpers():
    assert rep.utc_timestamp(datetime(2026, 11, 14, 14, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))) == STAMP
    with pytest.raises(ValueError):
        rep.utc_timestamp(datetime(2026, 11, 14))
    assert rep.current_python_version() == "{}.{}.{}".format(*sys.version_info[:3])
    assert rep._TIMESTAMP.fullmatch(rep.utc_timestamp())


# ---------- I: code identity ----------

def test_source_hashing_ignores_line_endings():
    lf, crlf, cr = b"a = 1\nb = 2\n", b"a = 1\r\nb = 2\r\n", b"a = 1\rb = 2\r"
    assert rep.hash_source_bytes(lf) == rep.hash_source_bytes(crlf) == rep.hash_source_bytes(cr)
    assert rep.hash_source_bytes(lf) != rep.hash_source_bytes(b"a = 1\nb = 3\n")
    with pytest.raises(TypeError):
        rep.hash_source_bytes("text")


def test_the_combined_identity_depends_on_names_and_hashes_not_on_input_order():
    one = rep.combine_identity({"b": "1" * 64, "a": "0" * 64})
    two = rep.combine_identity({"a": "0" * 64, "b": "1" * 64})
    assert one == two and one.files == (("a", "0" * 64), ("b", "1" * 64))
    assert rep.combine_identity({"a": "0" * 64, "b": "2" * 64}).combined_sha256 != one.combined_sha256
    assert rep.combine_identity({"a2": "0" * 64, "b": "1" * 64}).combined_sha256 != one.combined_sha256


def test_the_real_code_identity_is_deterministic_and_covers_every_listed_module():
    first = rep.compute_code_identity()
    assert first == rep.compute_code_identity()
    assert [m for m, _ in first.files] == sorted(rep.CODE_IDENTITY_MODULES)
    assert all(len(h) == 64 for _, h in first.files) and first.method == rep.CODE_IDENTITY_METHOD
    assert first == rep.combine_identity(dict(first.files))
    assert "app.rag.evaluation.engine" in dict(first.files) and "app.rag.retrieval" in dict(first.files)


def test_the_identity_changes_with_the_source_but_not_with_line_endings(tmp_path, monkeypatch):
    path = tmp_path / "fake_mod.py"
    path.write_bytes(b"x = 1\n")
    spec = type("Spec", (), {"origin": str(path)})()
    monkeypatch.setattr(rep.importlib.util, "find_spec", lambda name: spec)
    base = rep.compute_code_identity(["fake_mod"])
    path.write_bytes(b"x = 1\r\n")
    assert rep.compute_code_identity(["fake_mod"]) == base
    path.write_bytes(b"x = 2\n")
    assert rep.compute_code_identity(["fake_mod"]).combined_sha256 != base.combined_sha256


def test_an_unreadable_or_unknown_module_is_refused(monkeypatch):
    refused(rep.REPORT_IDENTITY_UNAVAILABLE, rep.compute_code_identity, ["app.no_such_module_e174"])
    monkeypatch.setattr(rep.importlib.util, "find_spec", lambda name: type("Spec", (), {"origin": "/no/such/file.py"})())
    refused(rep.REPORT_IDENTITY_UNAVAILABLE, rep.compute_code_identity, ["anything"])
    monkeypatch.setattr(rep.importlib.util, "find_spec", lambda name: type("Spec", (), {"origin": "built-in"})())
    refused(rep.REPORT_IDENTITY_UNAVAILABLE, rep.compute_code_identity, ["anything"])
    with pytest.raises(TypeError):
        rep.compute_code_identity("app.x")
    with pytest.raises(TypeError):
        rep.compute_code_identity([1])


# ---------- J: purity and the test split ----------

def test_the_report_module_imports_no_database_or_retrieval_code_and_names_no_test_file():
    code = (
        "import sys\nimport app.rag.evaluation.report\n"
        "bad = [m for m in ('sqlalchemy','app.core.database','app.core.config','app.models','app.rag.retrieval',"
        "'app.rag.embedder','numpy','torch') if m in sys.modules]\nprint('bad=' + ','.join(bad))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "SECRET_KEY")}
    env.update({"PYTHONPATH": str(PROJECT_ROOT), "PYTHONDONTWRITEBYTECODE": "1"})
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-600:]
    assert done.stdout.strip() == "bad="
    source = (PROJECT_ROOT / "app" / "rag" / "evaluation" / "report.py").read_text(encoding="utf-8")
    assert "test.jsonl" not in source


# ---------- database snapshot record (E1.7.5) ----------

def test_the_snapshot_record_is_optional_and_appears_in_the_document(bundle):
    run, inputs, pool = bundle
    assert rep.build_report(run, inputs, pool, meta())["database_snapshots"] is None
    record = rep.DatabaseSnapshotRecord("a" * 64, "a" * 64, True)
    document = rep.build_report(run, inputs, pool, meta(database_snapshots=record))
    assert document["database_snapshots"] == {"before_fingerprint": "a" * 64, "after_fingerprint": "a" * 64,
                                              "unchanged": True}
    first = rep.render_report(run, inputs, pool, meta(database_snapshots=record))
    assert first.sha256 == rep.render_report(run, inputs, pool, meta(database_snapshots=record)).sha256


@pytest.mark.parametrize("record", [
    rep.DatabaseSnapshotRecord("A" * 64, "a" * 64, True), rep.DatabaseSnapshotRecord("a" * 63, "a" * 64, True),
    rep.DatabaseSnapshotRecord("a" * 64, 5, True), rep.DatabaseSnapshotRecord("a" * 64, "a" * 64, "yes"),
    {"before_fingerprint": "a" * 64},
])
def test_an_invalid_snapshot_record_is_refused(bundle, record):
    run, inputs, pool = bundle
    refused(rep.REPORT_METADATA_INVALID, rep.build_report, run, inputs, pool, meta(database_snapshots=record))


# ---------- K: E2.2 the retrieval_baseline kind ----------

# The pilot contract at the commit before E2.2, rendered with the FIXED test identity of this file. This pins the
# document structure, fields and serialisation of the validity_pilot kind. It says nothing about REAL pilot runs:
# their code identity is computed from the source files and changes whenever a listed module changes (see
# test_the_real_pilot_code_identity_changes_by_design_while_the_document_contract_does_not).
PILOT_GOLDEN_BYTES = 431541
PILOT_GOLDEN_SHA256 = "17178c03a6f4e1e9def787398bcaf4390901bd23045c6299494fe58f19d00f6a"

# CODE_IDENTITY_MODULES before E2.2.
CODE_IDENTITY_BEFORE_E2_2 = (
    "app.core.text_extraction", "app.rag.chunking", "app.rag.embedder", "app.rag.evaluation.aggregation",
    "app.rag.evaluation.anchoring", "app.rag.evaluation.baselines", "app.rag.evaluation.dataset",
    "app.rag.evaluation.engine", "app.rag.evaluation.intervals", "app.rag.evaluation.metrics",
    "app.rag.evaluation.parsing", "app.rag.evaluation.passages", "app.rag.evaluation.pool",
    "app.rag.evaluation.report", "app.rag.evaluation.runner_inputs", "app.rag.evaluation.schema",
    "app.rag.evaluation.statistics", "app.rag.evaluation.structure", "app.rag.evaluation.validation",
    "app.rag.indexing_service", "app.rag.retrieval", "app.rag.text_boundaries",
)
FREEZE_FIELDS = tuple(f.name for f in dataclasses.fields(fb.FreezeBinding))
HASH_FIELDS = tuple(name for name in FREEZE_FIELDS if name.endswith("_sha256"))
# CODE_IDENTITY_MODULES at E2.2 (E2.3 then added the score-diagnostics module).
CODE_IDENTITY_BEFORE_E2_3 = tuple(sorted((*CODE_IDENTITY_BEFORE_E2_2, "app.rag.evaluation.freeze_binding")))
# CODE_IDENTITY_MODULES at E2.3 (E2.4b then added the ranking-hash module).
CODE_IDENTITY_BEFORE_E2_4B = tuple(sorted((*CODE_IDENTITY_BEFORE_E2_3, "app.rag.evaluation.score_diagnostics")))
BASELINE_ONLY_KEYS = ("kind", "label", "report_format_version", "scope", "freeze_binding", "score_diagnostics",
                      "ranking_determinism", "unanswerable")


_SECTIONS = {}


def unanswerable_retriever(query):
    """A deterministic ranking for the unanswerable dev questions, with strictly decreasing scores."""
    return [dataclasses.replace(CHUNKS[i], score=0.5 - 0.0625 * n) for n, i in enumerate(range(1, 7))]


def diagnostics_of(bundle):
    """The score_diagnostics section for the bundle's run (built once per run, returned as a fresh copy)."""
    run, inputs, pool = bundle
    if id(run) not in _SECTIONS:
        recorder = sd.ScoreRecorder(retriever)
        for question in inputs.answerable:  # the engine's calls, in dev-file order
            recorder(question.question)
        _SECTIONS[id(run)] = sd.build_score_diagnostics(
            run, inputs, pool, recorder.take(), sd.collect_unanswerable_scores(inputs, pool, unanswerable_retriever))
    return copy.deepcopy(_SECTIONS[id(run)])


def determinism_of(bundle):
    """The ranking_determinism section for the bundle's run (its D7 ranking hash, two identical passes)."""
    return rep.ranking_determinism_section(rh.ranking_sha256(*bundle))


def freeze_of(run, **over):
    values = {
        "dataset_version": "eval-dataset-v1.0.0", "frozen_on": "2026-11-14", "corpus_version": "eval-corpus-v1",
        "record_sha256": "a" * 64, "dev_file_sha256": "b" * 64, "dataset_json_sha256": "c" * 64,
        "manifest_file_sha256": "d" * 64, "dev_label_binding_sha256": run.dev_dataset_sha256,
        "canonical_manifest_sha256": run.manifest_sha256,
    }
    values.update(over)
    return fb.FreezeBinding(**values)


@pytest.fixture(scope="module")
def baseline(bundle):
    return rep.render_report(
        *bundle, meta(), kind=rep.BASELINE_KIND, score_diagnostics=diagnostics_of(bundle),
        ranking_determinism=determinism_of(bundle), freeze=freeze_of(bundle[0]))


def tampered(document, change):
    clone = copy.deepcopy(document)
    change(clone)
    return clone


# --- K1: the pilot contract is unchanged ---

def test_the_pilot_contract_golden_bytes_are_unchanged(rendered):
    assert len(rendered.json_bytes) == PILOT_GOLDEN_BYTES
    assert rendered.sha256 == PILOT_GOLDEN_SHA256


def test_the_pilot_constants_and_label_are_unchanged():
    assert rep.REPORT_KIND == "validity_pilot" and rep.REPORT_FORMAT_VERSION == "e1.7-report-1"
    assert rep.REPORT_LABEL == "Validity pilot — not the V1 baseline"


def test_a_pilot_report_has_none_of_the_baseline_keys_and_the_default_kind_is_the_pilot(bundle, rendered):
    assert "scope" not in rendered.document and "freeze_binding" not in rendered.document
    assert rep.render_report(*bundle, meta(), kind=rep.REPORT_KIND).json_bytes == rendered.json_bytes


def test_a_pilot_report_takes_no_freeze_binding(bundle):
    with pytest.raises(ValueError):
        rep.build_report(*bundle, meta(), freeze=freeze_of(bundle[0]))
    with pytest.raises(ValueError):
        rep.build_report(*bundle, meta(), kind=rep.REPORT_KIND, freeze=freeze_of(bundle[0]))


# --- K2: the code identity changes by design; the document contract does not ---

def test_the_new_identity_modules_are_the_freeze_gate_the_score_diagnostics_and_the_ranking_hash():
    added = {"app.rag.evaluation.freeze_binding", "app.rag.evaluation.score_diagnostics",
             "app.rag.evaluation.ranking_hash"}
    assert list(rep.CODE_IDENTITY_MODULES) == sorted(rep.CODE_IDENTITY_MODULES)
    assert set(rep.CODE_IDENTITY_MODULES) - set(CODE_IDENTITY_BEFORE_E2_2) == added
    assert set(CODE_IDENTITY_BEFORE_E2_2) <= set(rep.CODE_IDENTITY_MODULES)
    assert set(rep.CODE_IDENTITY_MODULES) - set(CODE_IDENTITY_BEFORE_E2_3) == (
        added - {"app.rag.evaluation.freeze_binding"})
    assert set(rep.CODE_IDENTITY_MODULES) - set(CODE_IDENTITY_BEFORE_E2_4B) == {"app.rag.evaluation.ranking_hash"}
    assert len(rep.CODE_IDENTITY_MODULES) == len(CODE_IDENTITY_BEFORE_E2_4B) + 1 == 25
    assert added <= set(dict(rep.compute_code_identity().files))


def test_the_real_pilot_code_identity_changes_by_design_while_the_document_contract_does_not(bundle):
    """The code identity hashes report.py itself and now freeze_binding.py, so a real pilot run's identity and
    bytes change with this checkpoint. Only the code_identity section differs; every other part is the same."""
    def stable(name):  # one hash per module NAME, the same in both identities
        return rep.hash_source_bytes(name.encode("utf-8"))

    added = "app.rag.evaluation.ranking_hash"  # E2.4b added exactly one module to the E2.3 list
    old = rep.combine_identity({m: stable(m) for m in CODE_IDENTITY_BEFORE_E2_4B})
    new = rep.combine_identity({m: stable(m) for m in rep.CODE_IDENTITY_MODULES})
    old_files, new_files = dict(old.files), dict(new.files)
    assert len(old_files) == 24 and len(new_files) == 25
    assert set(new_files) - set(old_files) == {added} and set(old_files) <= set(new_files)
    assert all(new_files[m] == old_files[m] for m in old_files)  # the 24 existing hashes are unchanged
    assert new_files[added] == stable(added)
    assert old.combined_sha256 != new.combined_sha256
    before = rep.render_report(*bundle, meta(code_identity=old))
    after = rep.render_report(*bundle, meta(code_identity=new))
    assert before.json_bytes != after.json_bytes and before.sha256 != after.sha256
    assert set(before.document) == set(after.document)
    assert {k: v for k, v in before.document.items() if k != "code_identity"} == {
        k: v for k, v in after.document.items() if k != "code_identity"}
    assert len(after.document["code_identity"]["files"]) == len(before.document["code_identity"]["files"]) + 1
    for rendered_report in (before, after):
        assert rep.load_report(rendered_report.json_bytes) == rendered_report.document
    rep.check_comparable(before.document, after.document)


# --- K3: building a baseline ---

def test_the_baseline_identifies_itself_as_retrieval_only_dense_dev_and_not_a_v1_baseline(baseline):
    doc = baseline.document
    assert doc["kind"] == "retrieval_baseline" and doc["is_v1_baseline"] is False
    assert doc["report_format_version"] == "e2.4-report-1"
    assert doc["label"] == "Retrieval-only dense V1 baseline (dev split) — not the full V1 RAG baseline"
    assert doc["scope"] == {
        "stage": "retrieval_only", "retrieval": "dense", "retrieval_version": "dense-v1", "split": "dev",
        "generation_evaluated": False, "answer_quality_evaluated": False, "full_rag_baseline": False,
    }
    assert doc["inputs"]["dev_dataset"]["split"] == "dev" and doc["inputs"]["dev_dataset"]["test_split_read"] is False
    assert doc["scope"] is not rep.BASELINE_SCOPE  # a copy: the constant cannot be changed through a document


def test_the_baseline_holds_the_supplied_binding_and_the_same_sections_as_a_pilot(bundle, baseline, rendered):
    assert baseline.document["freeze_binding"] == dataclasses.asdict(freeze_of(bundle[0]))
    assert set(baseline.document["freeze_binding"]) == set(FREEZE_FIELDS) and len(FREEZE_FIELDS) == 9
    assert set(baseline.document) == set(rendered.document) | {
        "scope", "freeze_binding", "score_diagnostics", "ranking_determinism"}
    for key in set(rendered.document) - set(BASELINE_ONLY_KEYS):
        assert baseline.document[key] == rendered.document[key], key
    # the unanswerable section differs from the pilot's only in its note (the questions are retrieved for scores)
    pilot_part, baseline_part = dict(rendered.document["unanswerable"]), dict(baseline.document["unanswerable"])
    assert pilot_part.pop("note") == rep.UNANSWERABLE_NOTE
    assert baseline_part.pop("note") == rep.UNANSWERABLE_DIAGNOSTIC_NOTE
    assert pilot_part == baseline_part and baseline_part["retrieval_metrics"] is None
    assert baseline.document["oracle"]["questions"] and baseline.document["random"]["questions"]  # reference rows


def test_the_baseline_bytes_are_deterministic_canonical_and_round_trip(bundle, baseline):
    again = rep.render_report(*bundle, meta(), kind=rep.BASELINE_KIND,
        score_diagnostics=diagnostics_of(bundle),
        ranking_determinism=determinism_of(bundle), freeze=freeze_of(bundle[0]))
    assert again.json_bytes == baseline.json_bytes and again.sha256 == baseline.sha256
    assert rep.load_report(baseline.json_bytes) == baseline.document
    assert rep.serialize_report(rep.load_report(baseline.json_bytes)) == baseline.json_bytes
    assert baseline.sha256 != PILOT_GOLDEN_SHA256


def test_the_baseline_run_hashes_are_the_canonical_hashes_not_the_raw_file_hash(bundle, baseline):
    run = bundle[0]
    freeze = baseline.document["freeze_binding"]
    document_inputs = baseline.document["inputs"]
    assert document_inputs["manifest"]["sha256"] == run.manifest_sha256 == freeze["canonical_manifest_sha256"]
    assert document_inputs["dev_dataset"]["sha256"] == run.dev_dataset_sha256 == freeze["dev_label_binding_sha256"]
    assert freeze["manifest_file_sha256"] not in (run.manifest_sha256, run.dev_dataset_sha256)
    # the raw file hash is recorded, not compared with the run
    other = freeze_of(run, manifest_file_sha256="e" * 64)
    built = rep.build_report(*bundle, meta(), kind=rep.BASELINE_KIND,
        score_diagnostics=diagnostics_of(bundle), ranking_determinism=determinism_of(bundle), freeze=other)
    assert built["freeze_binding"]["manifest_file_sha256"] == "e" * 64


def test_the_baseline_report_still_refuses_dev_text(bundle):
    leak = fx.Q1.question
    refused(rep.REPORT_LEAK, rep.build_report, *bundle, meta(embedder_signature=leak),
            kind=rep.BASELINE_KIND,
                score_diagnostics=diagnostics_of(bundle),
                ranking_determinism=determinism_of(bundle), freeze=freeze_of(bundle[0]))


# --- K4: build refusals ---

@pytest.mark.parametrize("value", [None, {}, "x", dataclasses.asdict, 1])
def test_a_baseline_requires_a_freeze_binding_object(bundle, value):
    with pytest.raises(TypeError):
        rep.build_report(*bundle, meta(), kind=rep.BASELINE_KIND,
            score_diagnostics=diagnostics_of(bundle), ranking_determinism=determinism_of(bundle), freeze=value)


def test_a_dict_that_looks_like_a_binding_is_not_accepted(bundle):
    with pytest.raises(TypeError):
        rep.build_report(*bundle, meta(), kind=rep.BASELINE_KIND,
            score_diagnostics=diagnostics_of(bundle),
            ranking_determinism=determinism_of(bundle), freeze=dataclasses.asdict(freeze_of(bundle[0])))


@pytest.mark.parametrize("kind", ["", "pilot", "Retrieval_Baseline", "v1_baseline", None, 1, b"retrieval_baseline"])
def test_an_unknown_report_kind_is_refused(bundle, kind):
    with pytest.raises(ValueError):
        rep.build_report(*bundle, meta(), kind=kind)


@pytest.mark.parametrize("over", [
    {"dataset_version": "v1"}, {"dataset_version": "eval-dataset-v1.0"}, {"frozen_on": "2026-13-45"},
    {"frozen_on": "2026-1-1"}, {"corpus_version": ""}, {"record_sha256": "A" * 64}, {"dev_file_sha256": "a" * 63},
    {"dataset_json_sha256": ""}, {"manifest_file_sha256": 7}, {"frozen_on": None},
])
def test_a_malformed_binding_is_refused_with_a_fixed_message(bundle, over):
    freeze = freeze_of(bundle[0], **over)
    with pytest.raises(s.EvaluationDataError) as info:
        rep.build_report(*bundle, meta(), kind=rep.BASELINE_KIND,
            score_diagnostics=diagnostics_of(bundle), ranking_determinism=determinism_of(bundle), freeze=freeze)
    assert {i.code for i in info.value.issues} == {rep.REPORT_FREEZE_INVALID}
    assert all(i.message == rep._MESSAGES[rep.REPORT_FREEZE_INVALID] for i in info.value.issues)
    assert "2026-13-45" not in repr(info.value.issues) and "eval-dataset-v1.0" not in repr(info.value.issues)


@pytest.mark.parametrize("field", ["dev_label_binding_sha256", "canonical_manifest_sha256"])
def test_a_binding_that_disagrees_with_the_run_is_refused(bundle, field):
    freeze = freeze_of(bundle[0], **{field: "f" * 64})
    refused(rep.REPORT_FREEZE_MISMATCH, rep.build_report, *bundle, meta(), kind=rep.BASELINE_KIND,
        score_diagnostics=diagnostics_of(bundle), ranking_determinism=determinism_of(bundle), freeze=freeze)
    assert rep._MESSAGES[rep.REPORT_FREEZE_MISMATCH] and "f" * 64 not in repr(rep._MESSAGES)


def test_the_raw_manifest_hash_is_not_accepted_in_place_of_the_canonical_one(bundle):
    run = bundle[0]
    # a raw-file-style hash, not the run's canonical one
    freeze = freeze_of(run, canonical_manifest_sha256="d" * 64)
    refused(rep.REPORT_FREEZE_MISMATCH, rep.build_report, *bundle, meta(), kind=rep.BASELINE_KIND,
        score_diagnostics=diagnostics_of(bundle), ranking_determinism=determinism_of(bundle), freeze=freeze)
    swapped = freeze_of(run, dev_label_binding_sha256=run.manifest_sha256,
                        canonical_manifest_sha256=run.dev_dataset_sha256)
    refused(rep.REPORT_FREEZE_MISMATCH, rep.build_report, *bundle, meta(), kind=rep.BASELINE_KIND,
        score_diagnostics=diagnostics_of(bundle), ranking_determinism=determinism_of(bundle), freeze=swapped)


def test_the_baseline_keeps_every_existing_verification(bundle):
    run, inputs, pool = bundle
    freeze = freeze_of(run)
    refused(rep.REPORT_HASH_MISMATCH, rep.build_report, dataclasses.replace(run, pool_size=7), inputs, pool, meta(),
            kind=rep.BASELINE_KIND,
                score_diagnostics=diagnostics_of(bundle), ranking_determinism=determinism_of(bundle), freeze=freeze)
    refused(rep.REPORT_EMBEDDER_MISMATCH, rep.build_report, run, inputs, pool, meta(embedder_id="other"),
            kind=rep.BASELINE_KIND,
                score_diagnostics=diagnostics_of(bundle), ranking_determinism=determinism_of(bundle), freeze=freeze)
    refused(rep.REPORT_METADATA_INVALID, rep.build_report, run, inputs, pool, meta(git_commit="xyz"),
            kind=rep.BASELINE_KIND,
                score_diagnostics=diagnostics_of(bundle), ranking_determinism=determinism_of(bundle), freeze=freeze)


# --- K5: strict loading ---

BAD_HASHES = ["A" * 64, "a" * 63, "a" * 65, "", 7, None, ["a" * 64], "g" * 64]


def loads_invalid(document):
    refused(rep.REPORT_INVALID, rep.load_report, rep.serialize_report(document))


def test_a_valid_baseline_loads_and_a_pilot_still_loads(baseline, rendered):
    assert rep.load_report(baseline.json_bytes) == baseline.document
    assert rep.load_report(rendered.json_bytes) == rendered.document


@pytest.mark.parametrize("change", [
    lambda d: d.update(label=rep.REPORT_LABEL),
    lambda d: d.update(label=rep.BASELINE_LABEL + "."),
    lambda d: d.update(label="Retrieval-only dense V1 baseline (dev split)"),
    lambda d: d.update(report_format_version=rep.REPORT_FORMAT_VERSION),
    lambda d: d.update(report_format_version="e2.2-report-2"),
    lambda d: d.update(report_format_version="e2.3-report-1"),
    lambda d: d.update(is_v1_baseline=True),
    lambda d: d.update(is_v1_baseline=0),
    lambda d: d.update(is_v1_baseline=None),
    lambda d: d.update(kind="validity_pilot"),
    lambda d: d.update(kind="retrieval_baseline "),
    lambda d: d.update(kind=["retrieval_baseline"]),
    lambda d: d.pop("scope"),
    lambda d: d.pop("freeze_binding"),
    lambda d: d.update(scope=None),
    lambda d: d.update(scope=[]),
    lambda d: d.update(freeze_binding=None),
    lambda d: d.update(freeze_binding=list(d["freeze_binding"].values())),
])
def test_a_tampered_baseline_header_is_refused(baseline, change):
    loads_invalid(tampered(baseline.document, change))


@pytest.mark.parametrize("key", sorted(rep.BASELINE_SCOPE))
def test_every_scope_field_is_checked_exactly(baseline, key):
    value = rep.BASELINE_SCOPE[key]
    flipped = (not value) if isinstance(value, bool) else value + "-x"
    loads_invalid(tampered(baseline.document, lambda d: d["scope"].update({key: flipped})))
    loads_invalid(tampered(baseline.document, lambda d: d["scope"].pop(key)))
    if isinstance(value, bool):
        loads_invalid(tampered(baseline.document, lambda d: d["scope"].update({key: int(value)})))  # 0 is not false


def test_an_extra_scope_key_is_refused(baseline):
    loads_invalid(tampered(baseline.document, lambda d: d["scope"].update(extra=True)))


def test_a_baseline_claiming_to_be_a_full_rag_baseline_is_refused(baseline):
    loads_invalid(tampered(baseline.document, lambda d: d["scope"].update(full_rag_baseline=True)))
    loads_invalid(tampered(baseline.document, lambda d: d["scope"].update(generation_evaluated=True)))


@pytest.mark.parametrize("field", FREEZE_FIELDS)
def test_a_missing_freeze_field_is_refused(baseline, field):
    loads_invalid(tampered(baseline.document, lambda d: d["freeze_binding"].pop(field)))


def test_an_extra_freeze_field_is_refused(baseline):
    loads_invalid(tampered(baseline.document, lambda d: d["freeze_binding"].update(test_label_binding_sha256="a" * 64)))


@pytest.mark.parametrize("value", BAD_HASHES)
@pytest.mark.parametrize("field", HASH_FIELDS)
def test_every_freeze_hash_must_be_64_lowercase_hex(baseline, field, value):
    loads_invalid(tampered(baseline.document, lambda d: d["freeze_binding"].update({field: value})))


@pytest.mark.parametrize("field,value", [
    ("dataset_version", "eval-dataset-v1.0"), ("dataset_version", "eval-dataset-v1.0.0-rc1"),
    ("dataset_version", "EVAL-DATASET-V1.0.0"), ("dataset_version", ""), ("dataset_version", 1),
    ("dataset_version", "eval-dataset-v\u0661.0.0"),
    ("frozen_on", "2026-13-45"), ("frozen_on", "2026-02-30"), ("frozen_on", "2026-1-1"), ("frozen_on", ""),
    ("frozen_on", "20261114"), ("frozen_on", "2026-11-14T00:00:00"), ("frozen_on", "\u0662026-11-14"),
    ("frozen_on", 20261114), ("corpus_version", ""), ("corpus_version", None), ("corpus_version", 1),
])
def test_the_freeze_version_date_and_corpus_are_checked(baseline, field, value):
    loads_invalid(tampered(baseline.document, lambda d: d["freeze_binding"].update({field: value})))


@pytest.mark.parametrize("field,path", [
    ("dev_label_binding_sha256", ("dev_dataset", "sha256")), ("canonical_manifest_sha256", ("manifest", "sha256")),
])
def test_the_binding_must_agree_with_the_documents_own_input_hashes(baseline, field, path):
    loads_invalid(tampered(baseline.document, lambda d: d["freeze_binding"].update({field: "e" * 64})))
    loads_invalid(tampered(baseline.document, lambda d: d["inputs"][path[0]].update({path[1]: "e" * 64})))


def test_the_raw_file_hashes_may_differ_from_the_run_hashes_when_loading(baseline):
    document = tampered(baseline.document, lambda d: d["freeze_binding"].update(manifest_file_sha256="9" * 64))
    assert rep.load_report(rep.serialize_report(document)) == document


@pytest.mark.parametrize("data", [b"not json", b"[1]", b"\xff\xfe", b"{}", b'{"kind": "retrieval_baseline"}', b"null"])
def test_garbage_is_still_refused(data):
    refused(rep.REPORT_INVALID, rep.load_report, data)


# --- K6: comparison ---

def retargeted(document, field, value):
    """A baseline whose freeze field AND the matching own input hash both hold ``value``: still internally valid."""
    own = {"dev_label_binding_sha256": ("dev_dataset", "sha256"), "canonical_manifest_sha256": ("manifest", "sha256")}

    def change(d):
        d["freeze_binding"][field] = value
        if field in own:
            d["inputs"][own[field][0]][own[field][1]] = value

    return tampered(document, change)


def redetermine(document):
    """Recompute ``ranking_determinism`` from the document's own dense rankings (as the validator does)."""
    rows = [(q["question_id"], [RankedChunk(c["corpus_key"], c["start"], c["end"], c["text_sha256"])
                                for c in q["ranking"]]) for q in document["dense"]["questions"]]
    length = rh.expected_length(document["inputs"]["candidate_pool"]["size"])
    document["ranking_determinism"] = rep.ranking_determinism_section(rh.ranking_digest(length, rows))


def with_answerable(document, indices):
    """Keep the answerable questions ``indices`` (in that order) in the dense section AND the diagnostics, so
    the document stays internally valid and only the compared identity changes."""
    section = document["score_diagnostics"]
    rows = section["questions"]["answerable"]
    document["dense"]["questions"] = [document["dense"]["questions"][i] for i in indices]
    answerable = [sd.QuestionScores(rows[i]["question_id"], tuple(rows[i]["scores"])) for i in indices]
    unanswerable = [sd.QuestionScores(r["question_id"], tuple(r["scores"])) for r in section["questions"]["unanswerable"]]
    document["score_diagnostics"] = sd._section(
        section["ranking_length"], answerable, unanswerable, section["second_pass_scores_sha256"])
    redetermine(document)


def other_baseline(bundle, **over):
    return rep.render_report(*bundle, meta(), kind=rep.BASELINE_KIND,
        score_diagnostics=diagnostics_of(bundle),
        ranking_determinism=determinism_of(bundle), freeze=freeze_of(bundle[0], **over))


def test_two_baselines_of_the_same_run_are_comparable(bundle, baseline):
    rep.check_comparable(baseline.document, tampered(baseline.document, lambda d: None))
    rep.check_comparable(baseline.document, rep.load_report(baseline.json_bytes))
    rep.check_comparable(baseline.document, other_baseline(bundle, record_sha256="a" * 64).document)


def test_a_pilot_and_a_baseline_are_never_comparable(baseline, rendered):
    refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, rendered.document, baseline.document)
    refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, baseline.document, rendered.document)
    refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, rendered.document, baseline.document,
            require_same_pool=False)


@pytest.mark.parametrize("field,value", [
    ("dataset_version", "eval-dataset-v9.9.9"), ("frozen_on", "2026-01-01"), ("corpus_version", "other"),
    ("record_sha256", "e" * 64), ("dev_file_sha256", "e" * 64), ("dataset_json_sha256", "e" * 64),
    ("manifest_file_sha256", "e" * 64), ("dev_label_binding_sha256", "e" * 64),
    ("canonical_manifest_sha256", "e" * 64),
])
def test_baselines_with_different_freeze_bindings_are_not_comparable(baseline, field, value):
    other = retargeted(baseline.document, field, value)
    refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, baseline.document, other)
    refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, other, baseline.document)
    refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, baseline.document, other, require_same_pool=False)


@pytest.mark.parametrize("change", [
    lambda d: d.pop("freeze_binding"), lambda d: d.update(freeze_binding=None),
    lambda d: d["freeze_binding"].pop("record_sha256"), lambda d: d["freeze_binding"].update(record_sha256="A" * 64),
])
def test_a_baseline_without_a_valid_freeze_binding_cannot_be_compared(baseline, change):
    broken = tampered(baseline.document, change)
    refused(rep.REPORT_INVALID, rep.check_comparable, baseline.document, broken)
    refused(rep.REPORT_INVALID, rep.check_comparable, broken, baseline.document)
    refused(rep.REPORT_INVALID, rep.check_comparable, broken, baseline.document, require_same_pool=False)


@pytest.mark.parametrize("change", [
    lambda d: (d["inputs"]["manifest"].update(sha256="e" * 64),
               d["freeze_binding"].update(canonical_manifest_sha256="e" * 64)),
    lambda d: (d["inputs"]["dev_dataset"].update(sha256="e" * 64),
               d["freeze_binding"].update(dev_label_binding_sha256="e" * 64)),
    lambda d: with_answerable(d, [3, 2, 1, 0]),
    lambda d: with_answerable(d, [0, 1, 2]),
    lambda d: d["settings"].update(cutoffs=d["settings"]["cutoffs"] + [99]),
    lambda d: d["settings"].update(relevance_rules=list(reversed(d["settings"]["relevance_rules"]))),
    lambda d: d["settings"]["series"]["binary_hit"][0].update(name="renamed"),
])
def test_every_existing_identity_check_still_applies_to_baselines(baseline, change):
    other = tampered(baseline.document, change)
    refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, baseline.document, other)
    refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, baseline.document, other, require_same_pool=False)


def test_the_pool_check_applies_to_baselines_unless_relaxed(baseline):
    other = tampered(baseline.document, lambda d: d["inputs"]["candidate_pool"].update(sha256="e" * 64))
    refused(rep.REPORT_NOT_COMPARABLE, rep.check_comparable, baseline.document, other)
    rep.check_comparable(baseline.document, other, require_same_pool=False)


@pytest.mark.parametrize("change", [
    lambda d: d["scope"].update(full_rag_baseline=True),
    lambda d: d["scope"].update(generation_evaluated=1),
    lambda d: d["scope"].update(retrieval_version="dense-v2"),
    lambda d: d["scope"].update(extra=True),
    lambda d: d["scope"].pop("split"),
    lambda d: d.pop("scope"),
    lambda d: d.update(scope=None),
])
@pytest.mark.parametrize("require_same_pool", [True, False])
def test_an_invalid_baseline_scope_is_refused_even_against_itself(baseline, change, require_same_pool):
    broken = tampered(baseline.document, change)
    refused(rep.REPORT_INVALID, rep.check_comparable, broken, broken, require_same_pool=require_same_pool)
    refused(rep.REPORT_INVALID, rep.check_comparable, baseline.document, broken, require_same_pool=require_same_pool)
    refused(rep.REPORT_INVALID, rep.check_comparable, broken, baseline.document, require_same_pool=require_same_pool)


@pytest.mark.parametrize("change", [
    lambda d: d["freeze_binding"].update(dev_label_binding_sha256="e" * 64),
    lambda d: d["freeze_binding"].update(canonical_manifest_sha256="e" * 64),
    lambda d: d["inputs"]["dev_dataset"].update(sha256="e" * 64),
    lambda d: d["inputs"]["manifest"].update(sha256="e" * 64),
])
@pytest.mark.parametrize("require_same_pool", [True, False])
def test_a_baseline_whose_freeze_hash_disagrees_with_its_own_inputs_is_refused_even_against_itself(
    baseline, change, require_same_pool,
):
    broken = tampered(baseline.document, change)
    refused(rep.REPORT_INVALID, rep.check_comparable, broken, broken, require_same_pool=require_same_pool)
    refused(rep.REPORT_INVALID, rep.check_comparable, baseline.document, broken, require_same_pool=require_same_pool)
    refused(rep.REPORT_INVALID, rep.check_comparable, broken, baseline.document, require_same_pool=require_same_pool)


@pytest.mark.parametrize("change", [
    lambda d: d.update(label="other"), lambda d: d.update(is_v1_baseline=True), lambda d: d.update(is_v1_baseline=0),
    lambda d: d.update(report_format_version="x"), lambda d: d.pop("label"),
])
@pytest.mark.parametrize("which", ["pilot", "baseline"])
def test_label_version_and_v1_flag_are_checked_before_any_comparison(baseline, rendered, change, which):
    document = (rendered if which == "pilot" else baseline).document
    broken = tampered(document, change)
    refused(rep.REPORT_INVALID, rep.check_comparable, broken, broken, require_same_pool=False)
    refused(rep.REPORT_INVALID, rep.check_comparable, document, broken)


def test_valid_comparisons_are_unchanged_by_the_complete_contract_check(bundle, baseline, rendered):
    rep.check_comparable(rendered.document, rendered.document)
    rep.check_comparable(baseline.document, baseline.document, require_same_pool=False)
    rep.check_comparable(rendered.document, rep.load_report(rendered.json_bytes))
    rep.check_comparable(baseline.document, rep.load_report(baseline.json_bytes))


@pytest.mark.parametrize("kind", [None, "other", ["retrieval_baseline"], 1])
def test_comparison_still_refuses_unknown_kinds_and_versions(baseline, kind):
    broken = tampered(baseline.document, lambda d: d.update(kind=kind))
    refused(rep.REPORT_INVALID, rep.check_comparable, baseline.document, broken)
    wrong = tampered(baseline.document, lambda d: d.update(report_format_version=rep.REPORT_FORMAT_VERSION))
    refused(rep.REPORT_INVALID, rep.check_comparable, baseline.document, wrong)
    refused(rep.REPORT_INVALID, rep.check_comparable, {}, baseline.document)
    refused(rep.REPORT_INVALID, rep.check_comparable, baseline.document, "text")


# --- K7: purity and privacy ---

def test_the_report_module_reads_no_file_except_for_the_code_identity():
    tree = ast.parse(Path(rep.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "compute_code_identity":
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            reads = {"read_bytes", "read_text", "open", "stat", "lstat", "exists"}
            assert "open" not in names and not attrs & reads, node.name


def test_the_baseline_document_holds_no_path_credential_or_test_split_content(baseline):
    text = baseline.json_bytes.decode("utf-8")
    assert "test.jsonl" not in text and "eval/dataset" not in text
    assert baseline.document["inputs"]["dev_dataset"]["test_split_read"] is False


# ---------- L: E2.3 the score_diagnostics section of the baseline ----------

def _scores_tampered(baseline, change):
    return tampered(baseline.document, change)


def _recomputed(document, answerable_vectors=None, second=None):
    """The section rebuilt from (changed) vectors, so only the NUMBERS differ and the section stays valid."""
    section = document["score_diagnostics"]
    rows = section["questions"]
    a = [sd.QuestionScores(r["question_id"], tuple(answerable_vectors[i] if answerable_vectors else r["scores"]))
         for i, r in enumerate(rows["answerable"])]
    u = [sd.QuestionScores(r["question_id"], tuple(r["scores"])) for r in rows["unanswerable"]]
    return sd._section(section["ranking_length"], a, u, second)


def test_a_baseline_holds_the_section_and_the_diagnostic_note(bundle, baseline):
    document = baseline.document
    assert document["score_diagnostics"] == diagnostics_of(bundle)
    assert document["unanswerable"]["note"] == rep.UNANSWERABLE_DIAGNOSTIC_NOTE
    assert document["unanswerable"]["retrieval_metrics"] is None and document["unanswerable"]["count"] == 2
    assert rep.UNANSWERABLE_DIAGNOSTIC_NOTE != rep.UNANSWERABLE_NOTE
    assert rep.load_report(baseline.json_bytes) == document


def test_the_pilot_document_has_no_score_diagnostics_and_keeps_its_note(rendered):
    assert "score_diagnostics" not in rendered.document
    assert rendered.document["unanswerable"]["note"] == rep.UNANSWERABLE_NOTE


def test_a_pilot_report_takes_no_score_diagnostics(bundle):
    with pytest.raises(ValueError):
        rep.build_report(*bundle, meta(), score_diagnostics=diagnostics_of(bundle))
    with pytest.raises(ValueError):
        rep.build_report(*bundle, meta(), kind=rep.REPORT_KIND, score_diagnostics={})
    with pytest.raises(ValueError):
        rep.build_report(*bundle, meta(), ranking_determinism=determinism_of(bundle))
    with pytest.raises(ValueError):
        rep.build_report(*bundle, meta(), kind=rep.REPORT_KIND, ranking_determinism={})


@pytest.mark.parametrize("value", [None, "x", [], 1, b"x"])
def test_a_baseline_requires_a_mapping_for_the_diagnostics(bundle, value):
    with pytest.raises(TypeError):
        rep.build_report(*bundle, meta(), kind=rep.BASELINE_KIND, freeze=freeze_of(bundle[0]), score_diagnostics=value,
                         ranking_determinism=determinism_of(bundle))


@pytest.mark.parametrize("change", [
    lambda d: d["groups"]["answerable"]["top1"].update(mean=0.0),
    lambda d: d["groups"]["unanswerable"].update(tied_adjacent_pairs=99),
    lambda d: d["groups"]["answerable"]["per_rank"][0].update(median=0.0),
    lambda d: d["questions"]["answerable"][0]["scores"].__setitem__(0, 0.75),
    lambda d: d["questions"]["answerable"].reverse(),
    lambda d: d["questions"]["unanswerable"].pop(),
    lambda d: d.update(scores_sha256="0" * 64),
    lambda d: d.update(ranking_length=5),
    lambda d: d.update(format="e2.3-score-diagnostics-2"),
    lambda d: d.update(extra=1),
    lambda d: d.pop("groups"),
    lambda d: d.update(second_pass_scores_sha256="0" * 64),  # without the matching identical flag
    lambda d: d.update(second_pass_scores_sha256="A" * 64, scores_identical_across_passes=False),
])
def test_a_section_that_does_not_recompute_is_refused_when_building(bundle, change):
    section = diagnostics_of(bundle)
    change(section)
    refused(rep.REPORT_SCORES_INVALID, rep.build_report, *bundle, meta(), kind=rep.BASELINE_KIND,
            score_diagnostics=section, ranking_determinism=determinism_of(bundle), freeze=freeze_of(bundle[0]))


def test_a_section_for_other_questions_is_refused_when_building(bundle):
    section = diagnostics_of(bundle)
    section["questions"]["answerable"][0]["question_id"] = "other"
    refused(rep.REPORT_SCORES_INVALID, rep.build_report, *bundle, meta(), kind=rep.BASELINE_KIND,
            score_diagnostics=section, ranking_determinism=determinism_of(bundle), freeze=freeze_of(bundle[0]))


@pytest.mark.parametrize("change", [
    lambda d: d["groups"]["answerable"]["spread"].update(max=9.0),
    lambda d: d["groups"]["answerable"].update(question_count=9),
    lambda d: d["groups"]["unanswerable"]["per_rank"].pop(),
    lambda d: d["questions"]["unanswerable"][0]["scores"].reverse(),         # rising: not non-increasing
    lambda d: d["questions"]["answerable"][0]["scores"].pop(),               # shorter than its ranking
    lambda d: d["questions"]["answerable"][1].update(scores=[1] * 6),        # ints, not floats
    lambda d: d["questions"]["answerable"][1]["scores"].__setitem__(0, None),
    lambda d: d.update(scores_sha256="1" * 64),
    lambda d: d.update(ranking_length=6.0),
    lambda d: d.update(scores_identical_across_passes=True),
    lambda d: d.pop("scores_sha256"),
    lambda d: d.update(extra=1),
])
def test_load_report_and_check_comparable_refuse_an_invalid_section(baseline, change):
    broken = tampered(baseline.document, lambda document: change(document["score_diagnostics"]))
    refused(rep.REPORT_INVALID, rep.load_report, rep.serialize_report(broken))
    for flag in (True, False):
        refused(rep.REPORT_INVALID, rep.check_comparable, broken, broken, require_same_pool=flag)
        refused(rep.REPORT_INVALID, rep.check_comparable, baseline.document, broken, require_same_pool=flag)
        refused(rep.REPORT_INVALID, rep.check_comparable, broken, baseline.document, require_same_pool=flag)


@pytest.mark.parametrize("change", [
    lambda d: d.pop("score_diagnostics"),
    lambda d: d.update(score_diagnostics=None),
    lambda d: d["unanswerable"].update(note=rep.UNANSWERABLE_NOTE),
    lambda d: d["unanswerable"].update(note=""),
    lambda d: d["unanswerable"]["questions"].reverse(),
    lambda d: d["dense"]["questions"].reverse(),
    lambda d: d["inputs"]["candidate_pool"].update(size=3),
])
def test_load_report_and_check_comparable_refuse_a_baseline_outside_its_diagnostics(baseline, change):
    broken = tampered(baseline.document, change)
    refused(rep.REPORT_INVALID, rep.load_report, rep.serialize_report(broken))
    refused(rep.REPORT_INVALID, rep.check_comparable, broken, broken, require_same_pool=False)


def test_a_valid_section_with_a_second_pass_hash_round_trips(bundle):
    document = tampered(rep.render_report(
        *bundle, meta(), kind=rep.BASELINE_KIND, freeze=freeze_of(bundle[0]),
        score_diagnostics=diagnostics_of(bundle), ranking_determinism=determinism_of(bundle)).document,
        lambda d: None)
    document["score_diagnostics"] = _recomputed(document, second=document["score_diagnostics"]["scores_sha256"])
    assert document["score_diagnostics"]["scores_identical_across_passes"] is True
    assert rep.load_report(rep.serialize_report(document)) == document
    other = _recomputed(document, second="a" * 64)
    assert other["scores_identical_across_passes"] is False  # a mismatch is recorded, not refused
    document["score_diagnostics"] = other
    assert rep.load_report(rep.serialize_report(document)) == document


def test_baselines_with_different_but_valid_scores_remain_comparable(baseline):
    vectors = [[0.75, 0.5, 0.5, 0.25, 0.25, 0.0]] * 4
    other = tampered(baseline.document, lambda d: d.update(score_diagnostics=_recomputed(d, vectors)))
    assert other["score_diagnostics"]["scores_sha256"] != baseline.document["score_diagnostics"]["scores_sha256"]
    rep.check_comparable(baseline.document, other)
    assert rep.load_report(rep.serialize_report(other)) == other


def test_the_baseline_report_holds_no_text_database_id_or_path(bundle, baseline):
    text = baseline.json_bytes.decode("utf-8")
    for question in bundle[1].questions:
        assert question.question not in text
    assert str(OWNER) not in text and "document_id" not in text and "query_sha256" not in text
    assert "test.jsonl" not in text and "audit.jsonl" not in text


# --- M: the ranking-determinism section (E2.4b) ---

DETERMINISM_KEYS = ("format", "passes", "ranking_sha256", "rankings_identical_across_passes")


def both_entry_points_refuse(baseline, broken):
    """Tampering must be caught by load_report AND by check_comparable (against itself and a valid report)."""
    refused(rep.REPORT_INVALID, rep.load_report, rep.serialize_report(broken))
    for flag in (True, False):
        refused(rep.REPORT_INVALID, rep.check_comparable, broken, broken, require_same_pool=flag)
        refused(rep.REPORT_INVALID, rep.check_comparable, baseline.document, broken, require_same_pool=flag)
        refused(rep.REPORT_INVALID, rep.check_comparable, broken, baseline.document, require_same_pool=flag)


def test_a_baseline_holds_the_exact_determinism_section_and_it_loads(bundle, baseline):
    section = baseline.document["ranking_determinism"]
    assert set(section) == set(DETERMINISM_KEYS)
    assert section["format"] == "e2.4-determinism-1" and section["passes"] == 2 and type(section["passes"]) is int
    assert section["rankings_identical_across_passes"] is True
    assert section == determinism_of(bundle)
    assert rep.load_report(baseline.json_bytes) == baseline.document
    rep.check_comparable(baseline.document, baseline.document)


def test_the_hash_is_recomputed_from_the_serialized_dense_rankings(bundle, baseline):
    run = bundle[0]
    rows = [[q["question_id"], [[c["corpus_key"], c["start"], c["end"], c["text_sha256"]] for c in q["ranking"]]]
            for q in baseline.document["dense"]["questions"]]
    text = json.dumps(["e2.4-ranking-digest-1", run.pool_size, rows], separators=(",", ":"), ensure_ascii=False)
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert baseline.document["ranking_determinism"]["ranking_sha256"] == expected == rh.ranking_sha256(*bundle)
    assert [r[0] for r in rows] == ["q1", "q2", "q3", "q4"]  # the answerable questions only


def test_the_section_builder_accepts_only_a_lowercase_sha256():
    assert rep.ranking_determinism_section("a" * 64) == {
        "format": "e2.4-determinism-1", "passes": 2, "ranking_sha256": "a" * 64,
        "rankings_identical_across_passes": True}
    for bad in ("", "A" * 64, "a" * 63, "a" * 65, "g" * 64, None, 5, b"a" * 64):
        with pytest.raises(ValueError):
            rep.ranking_determinism_section(bad)


@pytest.mark.parametrize("change", [
    lambda d: d.pop("ranking_determinism"),
    lambda d: d.update(ranking_determinism=None),
    lambda d: d.update(ranking_determinism=[]),
    lambda d: d.update(ranking_determinism="x"),
    lambda d: d.update(ranking_determinism={}),
    lambda d: d["ranking_determinism"].update(extra=1),
    lambda d: d["ranking_determinism"].update(second_pass_ranking_sha256="a" * 64),
    lambda d: d["ranking_determinism"].pop("format"),
    lambda d: d["ranking_determinism"].pop("passes"),
    lambda d: d["ranking_determinism"].pop("ranking_sha256"),
    lambda d: d["ranking_determinism"].pop("rankings_identical_across_passes"),
    lambda d: d["ranking_determinism"].update(format="e2.4-determinism-2"),
    lambda d: d["ranking_determinism"].update(format=""),
    lambda d: d["ranking_determinism"].update(format=None),
    lambda d: d["ranking_determinism"].update(passes=1),
    lambda d: d["ranking_determinism"].update(passes=3),
    lambda d: d["ranking_determinism"].update(passes=0),
    lambda d: d["ranking_determinism"].update(passes=True),
    lambda d: d["ranking_determinism"].update(passes=2.0),
    lambda d: d["ranking_determinism"].update(passes="2"),
    lambda d: d["ranking_determinism"].update(passes=None),
    lambda d: d["ranking_determinism"].update(ranking_sha256=d["ranking_determinism"]["ranking_sha256"][:63]),
    lambda d: d["ranking_determinism"].update(ranking_sha256=d["ranking_determinism"]["ranking_sha256"] + "0"),
    lambda d: d["ranking_determinism"].update(ranking_sha256=d["ranking_determinism"]["ranking_sha256"].upper()),
    lambda d: d["ranking_determinism"].update(ranking_sha256="g" * 64),
    lambda d: d["ranking_determinism"].update(ranking_sha256=""),
    lambda d: d["ranking_determinism"].update(ranking_sha256=None),
    lambda d: d["ranking_determinism"].update(ranking_sha256=5),
    lambda d: d["ranking_determinism"].update(rankings_identical_across_passes=False),
    lambda d: d["ranking_determinism"].update(rankings_identical_across_passes=1),
    lambda d: d["ranking_determinism"].update(rankings_identical_across_passes=0),
    lambda d: d["ranking_determinism"].update(rankings_identical_across_passes="true"),
    lambda d: d["ranking_determinism"].update(rankings_identical_across_passes=None),
    lambda d: d["ranking_determinism"].update(rankings_identical_across_passes=1.0),
])
def test_a_malformed_determinism_section_is_refused_by_both_entry_points(baseline, change):
    both_entry_points_refuse(baseline, tampered(baseline.document, change))


def swap_first_two(d):
    ranking = d["dense"]["questions"][0]["ranking"]
    ranking[0], ranking[1] = ranking[1], ranking[0]


@pytest.mark.parametrize("change", [
    lambda d: d["ranking_determinism"].update(ranking_sha256="1" * 64),   # well formed, wrong
    lambda d: d["ranking_determinism"].update(ranking_sha256="0" * 64),
    swap_first_two,                                                       # a ranking order changed
    lambda d: d["dense"]["questions"][1]["ranking"].reverse(),
    lambda d: d["dense"]["questions"][0]["ranking"][2].update(
        start=d["dense"]["questions"][0]["ranking"][2]["start"] + 1),
    lambda d: d["dense"]["questions"][0]["ranking"][0].update(end=d["dense"]["questions"][0]["ranking"][0]["end"] + 1),
    lambda d: d["dense"]["questions"][0]["ranking"][0].update(corpus_key="doc-other"),
    lambda d: d["dense"]["questions"][0]["ranking"][0].update(text_sha256="0" * 64),
    lambda d: d["dense"]["questions"][0].update(question_id="q9"),        # a question id changed
    lambda d: d["dense"]["questions"].reverse(),                          # the question order changed
    lambda d: d["dense"]["questions"][0]["ranking"].pop(),                # an invalid ranking length
    lambda d: d["dense"]["questions"][0]["ranking"].append(dict(d["dense"]["questions"][0]["ranking"][0])),
    lambda d: d["dense"]["questions"][0].update(ranking=[]),
    lambda d: d["dense"]["questions"][0]["ranking"][0].update(extra=1),   # a malformed serialized chunk
    lambda d: d["dense"]["questions"][0]["ranking"][0].pop("start"),
    lambda d: d["dense"]["questions"][0]["ranking"][0].update(start=True),
    lambda d: d["dense"]["questions"][0]["ranking"][0].update(start="0"),
    lambda d: d["dense"]["questions"][0].pop("ranking"),
    lambda d: d["dense"]["questions"][0].update(ranking=None),
    lambda d: d["dense"].update(questions=None),
])
def test_a_hash_that_does_not_recompute_from_the_dense_rankings_is_refused(baseline, change):
    both_entry_points_refuse(baseline, tampered(baseline.document, change))


def test_the_ranking_hash_check_alone_catches_a_changed_ranking(baseline):
    """The other sections stay valid, so only the recomputation can refuse a re-ordered ranking whose supplied
    hash is still a well-formed 64-hex string."""
    broken = tampered(baseline.document, swap_first_two)
    assert rep._scores_ok(broken) and rep._freeze_section_ok(broken["freeze_binding"])
    assert rep._determinism_ok(baseline.document) is True and rep._determinism_ok(broken) is False
    assert broken["ranking_determinism"] == baseline.document["ranking_determinism"]  # the hash was not touched
    both_entry_points_refuse(baseline, broken)


def test_a_consistently_rehashed_ranking_is_accepted_so_the_check_is_the_recomputation_itself(baseline):
    changed = tampered(baseline.document, swap_first_two)
    redetermine(changed)
    assert (changed["ranking_determinism"]["ranking_sha256"]
            != baseline.document["ranking_determinism"]["ranking_sha256"])
    assert rep.load_report(rep.serialize_report(changed)) == changed
    rep.check_comparable(baseline.document, changed)  # comparable reports need not share a ranking hash
    rep.check_comparable(changed, baseline.document, require_same_pool=False)


def test_the_pool_size_used_for_L_is_the_documents_own(baseline):
    broken = tampered(baseline.document, lambda d: d["inputs"]["candidate_pool"].update(size=7))
    both_entry_points_refuse(baseline, broken)


def test_a_baseline_requires_a_mapping_for_the_determinism(bundle):
    for value in (None, "x", [], 1, b"x"):
        with pytest.raises(TypeError):
            rep.build_report(*bundle, meta(), kind=rep.BASELINE_KIND, freeze=freeze_of(bundle[0]),
                             score_diagnostics=diagnostics_of(bundle), ranking_determinism=value)


@pytest.mark.parametrize("change", [
    lambda d: d.update(ranking_sha256="1" * 64),
    lambda d: d.update(extra=1),
    lambda d: d.pop("passes"),
    lambda d: d.update(passes=True),
    lambda d: d.update(rankings_identical_across_passes=False),
    lambda d: d.update(format="x"),
    lambda d: d.clear(),
])
def test_a_determinism_section_that_does_not_match_the_run_is_refused_when_building(bundle, change):
    section = determinism_of(bundle)
    change(section)
    refused(rep.REPORT_RANKING_INVALID, rep.build_report, *bundle, meta(), kind=rep.BASELINE_KIND,
            freeze=freeze_of(bundle[0]), score_diagnostics=diagnostics_of(bundle), ranking_determinism=section)


def test_the_determinism_message_is_fixed_and_holds_no_data(bundle):
    section = determinism_of(bundle)
    real = section["ranking_sha256"]
    section["ranking_sha256"] = "1" * 64
    with pytest.raises(s.EvaluationDataError) as info:
        rep.build_report(*bundle, meta(), kind=rep.BASELINE_KIND, freeze=freeze_of(bundle[0]),
                         score_diagnostics=diagnostics_of(bundle), ranking_determinism=section)
    assert [i.code for i in info.value.issues] == [rep.REPORT_RANKING_INVALID]
    assert [i.message for i in info.value.issues] == [rep._MESSAGES[rep.REPORT_RANKING_INVALID]]
    assert "1" * 64 not in repr(info.value.issues) and real not in repr(info.value.issues)


def test_the_pilot_has_no_determinism_section_and_its_version_is_unchanged(bundle, rendered):
    assert "ranking_determinism" not in rendered.document
    assert rep.REPORT_FORMAT_VERSION == "e1.7-report-1"
    assert rendered.document["report_format_version"] == "e1.7-report-1"
    assert rep.BASELINE_FORMAT_VERSION == "e2.4-report-1"
    assert rendered.sha256 == PILOT_GOLDEN_SHA256 and len(rendered.json_bytes) == PILOT_GOLDEN_BYTES


def test_a_pilot_document_carrying_the_section_is_refused_by_both_entry_points(rendered):
    with_section = tampered(rendered.document, lambda d: d.update(ranking_determinism={
        "format": "e2.4-determinism-1", "passes": 2, "ranking_sha256": "a" * 64,
        "rankings_identical_across_passes": True}))
    refused(rep.REPORT_INVALID, rep.load_report, rep.serialize_report(with_section))
    refused(rep.REPORT_INVALID, rep.check_comparable, with_section, with_section)
    refused(rep.REPORT_INVALID, rep.check_comparable, rendered.document, with_section, require_same_pool=False)
    assert rep.load_report(rendered.json_bytes) == rendered.document  # the pilot itself still loads


def test_the_baseline_with_a_changed_ranking_hash_still_keeps_freeze_and_score_validation(baseline):
    both_entry_points_refuse(baseline, tampered(baseline.document, lambda d: d["freeze_binding"].update(
        canonical_manifest_sha256="e" * 64)))
    both_entry_points_refuse(baseline, tampered(baseline.document, lambda d: d["score_diagnostics"].update(
        scores_sha256="1" * 64)))


def test_the_report_module_imports_the_ranking_hash_but_has_no_second_serializer():
    source = Path(rep.__file__).read_text(encoding="utf-8")
    assert "ranking_hash as rh" in source and "rh.ranking_digest" in source
    assert "e2.4-ranking-digest-1" not in source  # the payload format lives only in ranking_hash.py
