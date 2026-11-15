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

Run focused: pytest tests/test_evaluation_report.py --noconftest -p no:cacheprovider
"""
import copy
import dataclasses
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
from app.rag.evaluation import pool as p
from app.rag.evaluation import report as rep
from app.rag.evaluation import schema as s

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
