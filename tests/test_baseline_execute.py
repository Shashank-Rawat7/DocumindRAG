"""E2.4d: the two-pass dense run, the D7 determinism gate and the baseline run flow.

Everything runs on fakes: no database, no embedder, no real corpus, no held-out data. ``baseline_run`` is tested
directly; the command line is tested end to end with the fake session of the pilot tests (read-only methods only).

Run focused: pytest tests/test_baseline_execute.py --noconftest -p no:cacheprovider -W error
"""
import ast
import builtins
import io
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_evaluation_engine as fx
import test_evaluation_score_diagnostics as sdx
from test_pilot_cli import DIGEST, FINGERPRINT, IDENTITY, OWNER_ID, QUESTIONS
from test_pilot_cli import Ctx as PilotCtx

from app.rag import baseline_cli as bc
from app.rag import baseline_run as br
from app.rag import pilot_cli
from app.rag.evaluation import baselines, engine
from app.rag.evaluation import ranking_hash as rh
from app.rag.evaluation import report as rep
from app.rag.evaluation import schema as s
from app.rag.evaluation import score_diagnostics as sd
from app.rag.evaluation import statistics as stats
from app.rag.evaluation.freeze_binding import FreezeBinding

REPO = Path(__file__).resolve().parents[1]
RECORD = "a" * 64
CLEAN = ("c" * 40, bc.GIT_CLEAN)
ADJ = ["--adjudicate", "q4:u1"]
READ_ONLY_METHODS = {"check_live", "assert_read_only", "take_snapshot", "verify_corpus", "document_rows",
                     "pool_chunks", "dense_retriever"}


SMALL_BOOTSTRAP = 10


_RANDOM_RESULTS: dict = {}


@pytest.fixture(autouse=True)
def fast_engine(monkeypatch):
    """Tests only: a smaller bootstrap minimum, and the random baseline (a pure function of the question, the
    ranked pool and the seeds, 1000 seeds each) computed once per question. ``test_a_full_size_run`` undoes both."""
    monkeypatch.setattr(stats, "MIN_RESAMPLES", SMALL_BOOTSTRAP)
    real = baselines.random_baseline

    def cached(question, ranked, seeds):
        key = (question.question_id, len(ranked), len(seeds), seeds[0], seeds[-1])
        if key not in _RANDOM_RESULTS:
            _RANDOM_RESULTS[key] = real(question, ranked, seeds)
        return _RANDOM_RESULTS[key]

    monkeypatch.setattr(baselines, "random_baseline", cached)


# ---------- direct tests of the two-pass module ----------

def pool_and_inputs():
    return fx.inputs_of(*QUESTIONS), fx.make_pool()


class Spy:
    """Records the calls of the engine, the recorder and the unanswerable collection."""

    def __init__(self, monkeypatch, *, pass_order=None, pass_vectors=None):
        self.log, self.retrievers, self.recorders, self.engine_calls, self.collect_calls = [], [], [], [], []
        self.recorder_queries = []
        self.pass_order, self.pass_vectors = pass_order or {}, pass_vectors or {}
        spy = self
        real_run, real_collect = engine.run_pilot, sd.collect_unanswerable_scores

        def run_pilot(inputs, pool, retriever, **kwargs):
            spy.log.append("run_pilot")
            spy.engine_calls.append((inputs, pool, retriever, kwargs))
            return real_run(inputs, pool, retriever, **kwargs)

        def collect(inputs, pool, retriever):
            spy.log.append("collect_unanswerable")
            spy.collect_calls.append((inputs, pool, retriever))
            return real_collect(inputs, pool, retriever)

        class Recorder(sd.ScoreRecorder):
            def __init__(self, retriever):
                super().__init__(retriever)
                spy.recorders.append(self)

            def __call__(self, query):
                spy.recorder_queries.append(query)
                return super().__call__(query)

        monkeypatch.setattr(engine, "run_pilot", run_pilot)
        monkeypatch.setattr(sd, "collect_unanswerable_scores", collect)
        monkeypatch.setattr(sd, "ScoreRecorder", Recorder)

    def new_retriever(self):
        number = len(self.retrievers) // 2 + 1  # two retrievers per pass
        self.log.append("new_retriever")
        base = sdx.scored(self.pass_vectors.get(number), self.pass_order.get(number))

        def retrieve(query):
            return base(query)

        self.retrievers.append(retrieve)
        return retrieve


def two_passes(spy, adjudicated=None, **over):
    inputs, pool = pool_and_inputs()
    kwargs = {"seeds": fx.SEEDS, "bootstrap_seed": 7, "resamples": SMALL_BOOTSTRAP, "adjudicated": adjudicated}
    kwargs.update(over)
    return br.run_two_passes(spy.new_retriever, inputs, pool, **kwargs), inputs, pool


def test_both_passes_run_completely_and_in_order(monkeypatch):
    spy = Spy(monkeypatch)
    result, _, _ = two_passes(spy, {"q4": ["u1"]})
    assert spy.log == ["new_retriever", "run_pilot", "new_retriever", "collect_unanswerable"] * 2
    assert isinstance(result.first.run, engine.PilotRun) and isinstance(result.second.run, engine.PilotRun)
    assert len(result.first.run.questions) == len(result.second.run.questions) == 4
    assert len(result.first.unanswerable) == len(result.second.unanswerable) == 2


def test_every_pass_gets_fresh_retrievers_and_a_fresh_recorder(monkeypatch):
    spy = Spy(monkeypatch)
    two_passes(spy)
    assert len(spy.retrievers) == 4 and len({id(r) for r in spy.retrievers}) == 4
    assert len(spy.recorders) == 2 and spy.recorders[0] is not spy.recorders[1]
    for number, (_, _, retriever, _) in enumerate(spy.engine_calls):
        assert retriever is spy.recorders[number]  # the engine sees the recorder, never a bare retriever
    # each recorder wraps the first fresh retriever of its pass; the unanswerable path gets the second, a plain one
    assert [spy.collect_calls[i][2] for i in (0, 1)] == [spy.retrievers[1], spy.retrievers[3]]
    assert not any(isinstance(call[2], sd.ScoreRecorder) for call in spy.collect_calls)
    for recorder in spy.recorders:  # sealed: a recorder can never serve a second pass
        with pytest.raises(s.EvaluationDataError):
            recorder("anything")


def test_the_unanswerable_questions_never_pass_through_a_recorder(monkeypatch):
    spy = Spy(monkeypatch)
    two_passes(spy)
    assert len(spy.recorder_queries) == 8  # four answerable questions per pass
    assert fx.U1.question not in spy.recorder_queries and fx.U2.question not in spy.recorder_queries


def test_both_passes_use_the_same_immutable_inputs_pool_seeds_and_configuration(monkeypatch):
    spy = Spy(monkeypatch)
    adjudicated = {"q4": ("u1",)}
    _, inputs, pool = two_passes(spy, adjudicated)
    (i1, p1, _, k1), (i2, p2, _, k2) = spy.engine_calls
    assert i1 is i2 is inputs and p1 is p2 is pool
    assert k1["seeds"] is k2["seeds"] and isinstance(k1["seeds"], tuple)
    assert (k1["bootstrap_seed"], k1["resamples"]) == (k2["bootstrap_seed"], k2["resamples"]) == (7, SMALL_BOOTSTRAP)
    assert k1["adjudicated_unreachable"] == k2["adjudicated_unreachable"] == {"q4": ["u1"]}
    assert k1["adjudicated_unreachable"] is not k2["adjudicated_unreachable"]  # a fresh copy per pass
    assert adjudicated == {"q4": ("u1",)}  # the caller's mapping is untouched
    assert all(call[0] is inputs and call[1] is pool for call in spy.collect_calls)


def test_matching_ranking_hashes_return_the_verified_hash_and_the_section(monkeypatch):
    spy = Spy(monkeypatch)
    result, inputs, pool = two_passes(spy)
    expected = rh.ranking_sha256(result.first.run, inputs, pool)
    assert result.ranking_sha256 == result.first.ranking_sha256 == result.second.ranking_sha256 == expected
    section = result.score_diagnostics
    assert section["scores_identical_across_passes"] is True
    assert section["second_pass_scores_sha256"] == section["scores_sha256"]


REVERSED_Q1 = {1: {**sdx.ORDER}, 2: {**sdx.ORDER, fx.Q1.question: [2, 1, 3, 4, 5, 6]}}


def test_different_ranking_hashes_refuse_before_anything_is_built(monkeypatch):
    spy = Spy(monkeypatch, pass_order=REVERSED_Q1)
    built = []
    real = sd.build_score_diagnostics
    monkeypatch.setattr(sd, "build_score_diagnostics", lambda *a, **k: built.append(1) or real(*a, **k))
    with pytest.raises(s.EvaluationDataError) as info:
        two_passes(spy)
    assert [i.code for i in info.value.issues] == [br.BASELINE_RANKINGS_DIFFER]
    assert built == []  # no diagnostics, so nothing a report could be made from
    assert spy.log.count("run_pilot") == 2  # both passes completed before the gate


def test_a_tie_reordering_is_a_mismatch_not_something_to_repair(monkeypatch):
    tied = {fx.Q3.question: [0.5] * 6}
    vectors = {1: {**sdx.VECTORS, **tied}, 2: {**sdx.VECTORS, **tied}}
    order = {1: {**sdx.ORDER}, 2: {**sdx.ORDER, fx.Q3.question: [5, 2, 3, 1, 4, 6]}}  # same scores, other order
    with pytest.raises(s.EvaluationDataError):
        two_passes(Spy(monkeypatch, pass_order=order, pass_vectors=vectors))


def test_a_score_digest_mismatch_is_only_a_diagnostic(monkeypatch):
    changed = {**sdx.VECTORS, fx.Q2.question: [0.875, 0.75, 0.5, 0.5, 0.25, 0.0625]}  # same order, other last score
    spy = Spy(monkeypatch, pass_vectors={2: changed})
    result, _, _ = two_passes(spy)
    assert result.first.ranking_sha256 == result.second.ranking_sha256  # D7 holds
    section = result.score_diagnostics
    assert section["scores_identical_across_passes"] is False
    assert section["second_pass_scores_sha256"] != section["scores_sha256"]
    assert sd.SHA256_HEX_PATTERN.fullmatch(section["second_pass_scores_sha256"])


def test_unanswerable_scores_are_collected_independently_in_each_pass(monkeypatch):
    changed = {**sdx.VECTORS, fx.U1.question: [0.5, 0.5, 0.25, 0.25, 0.125, 0.0625]}
    spy = Spy(monkeypatch, pass_vectors={2: changed})
    result, _, _ = two_passes(spy)
    assert result.first.unanswerable != result.second.unanswerable
    assert result.first.unanswerable[1] == result.second.unanswerable[1]  # U2 unchanged
    assert result.score_diagnostics["scores_identical_across_passes"] is False
    assert result.first.ranking_sha256 == result.second.ranking_sha256  # answerable rankings are unaffected


def test_the_section_reports_the_first_pass_and_validates_against_its_run(monkeypatch):
    result, inputs, pool = two_passes(Spy(monkeypatch))
    assert sd.section_ok(
        result.score_diagnostics,
        answerable_ids=[q.question_id for q in result.first.run.questions],
        unanswerable_ids=[q.question_id for q in result.first.run.unanswerable],
        pool_size=len(pool), answerable_ranking_lengths=[len(q.dense) for q in result.first.run.questions],
    )


def test_a_failing_second_pass_returns_nothing(monkeypatch):
    spy = Spy(monkeypatch)
    inputs, pool = pool_and_inputs()
    calls = []

    def new_retriever():
        calls.append(1)
        if len(calls) > 2:
            raise RuntimeError("database went away")
        return spy.new_retriever()

    with pytest.raises(RuntimeError):
        br.run_two_passes(new_retriever, inputs, pool, seeds=fx.SEEDS, bootstrap_seed=7, resamples=SMALL_BOOTSTRAP)


def test_the_new_retriever_must_be_callable():
    inputs, pool = pool_and_inputs()
    with pytest.raises(TypeError):
        br.run_two_passes("x", inputs, pool, seeds=fx.SEEDS, bootstrap_seed=7, resamples=1)  # type: ignore[arg-type]


def test_the_module_is_pure_orchestration():
    tree = ast.parse((REPO / "app" / "rag" / "baseline_run.py").read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree)
                                                                          if isinstance(n, ast.Attribute)}
    assert not {"open", "print", "environ", "sorted", "sort", "subprocess", "commit", "execute", "write"} & names


# ---------- the command line, end to end ----------

class Session:
    def __init__(self, ctx, number):
        self.ctx, self.number = ctx, number

    def _note(self, name):
        self.ctx.session_calls.append((self.number, name))

    def check_live(self, name):
        self._note("check_live")
        return self.ctx.live_problem

    def assert_read_only(self):
        self._note("assert_read_only")
        return self.ctx.readonly_problem

    def take_snapshot(self, clock):
        self._note("take_snapshot")
        return {"fingerprint": FINGERPRINT}

    def verify_corpus(self, files, *, owner_id, owner_sha256, embedding_id):
        self._note("verify_corpus")
        return self.ctx.verification

    def document_rows(self, files):
        self._note("document_rows")
        return self.ctx.rows

    def pool_chunks(self, *, owner_id, embedding_id, document_ids):
        self._note("pool_chunks")
        return self.ctx.chunks

    def dense_retriever(self, embedder, *, owner_id, document_ids, top_k):
        self._note("dense_retriever")
        self.ctx.top_k = top_k
        number = len(self.ctx.retrievers) // 2 + 1
        base = sdx.scored(self.ctx.pass_vectors.get(number), self.ctx.pass_order.get(number))
        queries = []

        def retrieve(query):
            queries.append(query)
            if self.ctx.retrieve_hook:
                self.ctx.retrieve_hook(number, query)
            return base(query)

        retrieve.queries, retrieve.session = queries, self.number
        self.ctx.retrievers.append(retrieve)
        return retrieve


class Ctx(PilotCtx):
    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.out = self.outside / "baseline.json"
        self.events, self.session_calls, self.retrievers, self.session_log = [], [], [], []
        self.pass_vectors, self.pass_order, self.retrieve_hook = {}, {}, None
        self.freeze_calls = self.git_calls = 0
        self.freeze_final = None  # a replacement binding (or exception) for the final load
        self.git_script = [CLEAN, CLEAN]
        self.git_hook = None
        self.resamples = SMALL_BOOTSTRAP

    def binding(self, **over):
        values = {
            "dataset_version": "eval-dataset-v1.0.0", "frozen_on": "2026-11-14", "corpus_version": "eval-corpus-v1",
            "record_sha256": RECORD, "dev_file_sha256": "b" * 64, "dataset_json_sha256": "c" * 64,
            "manifest_file_sha256": "d" * 64, "dev_label_binding_sha256": self.inputs.dev_dataset_sha256,
            "canonical_manifest_sha256": self.inputs.manifest_sha256,
        }
        values.update(over)
        return FreezeBinding(**values)

    def freeze_loader(self, root):
        self.freeze_calls += 1
        self.events.append("freeze")
        if self.freeze_calls > 1 and self.freeze_final is not None:
            if isinstance(self.freeze_final, BaseException):
                raise self.freeze_final
            return self.freeze_final
        return self.binding()

    def git_reader(self, repo):
        self.git_calls += 1
        self.events.append("git")
        if self.git_hook:
            self.git_hook(self.git_calls)
        index = min(self.git_calls, len(self.git_script)) - 1
        return self.git_script[index]

    def tools(self, runtime):
        @contextmanager
        def open_session():
            number = len(self.session_log) // 2 + 1
            self.session_log.append(("open", number))
            self.events.append("open")
            try:
                yield Session(self, number)
            finally:
                self.session_log.append(("close", number))

        return bc_tools(self.tools_embedding_id, open_session, lambda: SimpleNamespace(embedding_id=self.embedder_id))

    def run(self, *extra, out=None):
        argv = ["--database", "documind_db", "--owner-id", str(OWNER_ID), "--evaluation-email-sha256", DIGEST,
                "--expected-freeze-record-sha256", RECORD, "--out", str(self.out if out is None else out),
                "--resamples", str(self.resamples), *extra]
        stdout, stderr = io.StringIO(), io.StringIO()
        code = bc.main(
            argv, freeze_loader=self.freeze_loader, git_reader=self.git_reader, runtime_loader=self.runtime,
            tools_loader=self.tools, inputs_loader=self.load_inputs, corpus_loader=self.corpus,
            code_identity=lambda: IDENTITY, environ={}, cwd=self.cwd, repo_root=self.repo, stdout=stdout,
            stderr=stderr,
        )
        return code, stdout.getvalue(), stderr.getvalue()


def bc_tools(embedding_id, open_session, load_embedder):
    return pilot_cli.Tools(embedding_id, open_session, load_embedder)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    c = Ctx(tmp_path)
    monkeypatch.setattr("app.rag.evaluation.snapshot.compare_snapshots",
                        lambda before, after: SimpleNamespace(identical=c.identical))
    return c


def written(ctx):
    return json.loads(ctx.out.read_text(encoding="utf-8"))


def test_a_full_size_run_writes_one_verified_baseline_report(ctx, monkeypatch):
    monkeypatch.undo()  # the real bootstrap minimum (1000) and the real random baseline, as in production
    monkeypatch.setattr("app.rag.evaluation.snapshot.compare_snapshots",
                        lambda before, after: SimpleNamespace(identical=True))
    ctx.resamples = 1000
    code, out, err = ctx.run(*ADJ)
    assert (code, err) == (0, "")
    document = written(ctx)
    assert rep.load_report(ctx.out.read_bytes())["kind"] == rep.BASELINE_KIND
    determinism = document["ranking_determinism"]
    assert determinism["passes"] == 2 and determinism["rankings_identical_across_passes"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", determinism["ranking_sha256"])
    assert document["score_diagnostics"]["scores_identical_across_passes"] is True
    assert document["freeze_binding"]["record_sha256"] == RECORD
    assert document["database_snapshots"] == {
        "before_fingerprint": FINGERPRINT, "after_fingerprint": FINGERPRINT, "unchanged": True}
    assert rep.BASELINE_LABEL in out and "2 passes identical" in out


def test_both_passes_share_one_read_only_transaction_and_the_snapshot_session_is_separate(ctx):
    assert ctx.run(*ADJ)[0] == 0
    assert ctx.session_log == [("open", 1), ("close", 1), ("open", 2), ("close", 2)]
    assert len(ctx.retrievers) == 4 and {r.session for r in ctx.retrievers} == {1}  # all four in the first session
    first = [name for number, name in ctx.session_calls if number == 1]
    assert first.count("dense_retriever") == 4
    assert [name for number, name in ctx.session_calls if number == 2] == ["check_live", "assert_read_only",
                                                                         "take_snapshot"]


def test_the_corpus_and_the_pool_are_loaded_once_for_both_passes(ctx):
    ctx.run(*ADJ)
    names = [name for number, name in ctx.session_calls if number == 1]
    assert (names.count("verify_corpus"), names.count("document_rows"), names.count("pool_chunks")) == (1, 1, 1)
    assert ctx.inputs_loaded == 1 and ctx.top_k == 20


def test_each_pass_uses_fresh_retrievers_and_only_answerable_text_reaches_the_recorder_path(ctx):
    ctx.run(*ADJ)
    a1, u1, a2, u2 = ctx.retrievers
    assert len({id(r) for r in ctx.retrievers}) == 4
    answerable = [q.question for q in (fx.Q1, fx.Q2, fx.Q3, fx.Q4)]
    unanswerable = [fx.U1.question, fx.U2.question]
    assert a1.queries == a2.queries == answerable and u1.queries == u2.queries == unanswerable


def test_the_final_steps_run_in_the_required_order(ctx, monkeypatch):
    real_render, real_write = rep.render_report, bc._write_report
    monkeypatch.setattr(rep, "render_report", lambda *a, **k: ctx.events.append("render") or real_render(*a, **k))
    monkeypatch.setattr(bc, "_write_report", lambda *a, **k: ctx.events.append("write") or real_write(*a, **k))
    assert ctx.run(*ADJ)[0] == 0
    assert ctx.events[:2] == ["freeze", "git"]  # the preflight
    assert ctx.events[-4:] == ["freeze", "render", "git", "write"]  # recheck, render, final git, then the file


def test_the_exit_code_summary_and_report_hold_no_text_ids_or_credentials(ctx):
    _, out, err = ctx.run(*ADJ)
    text = out + err + ctx.out.read_text(encoding="utf-8")
    for question in QUESTIONS:
        assert question.question not in text
    for secret in ("hunter2", "user:", str(OWNER_ID), DIGEST):
        assert secret not in out + err
    assert str(OWNER_ID) not in ctx.out.read_text(encoding="utf-8")
    for doc_id in (fx.ID_A, fx.ID_B):
        assert not re.search(rf"(?<![0-9A-Za-z.]){doc_id}(?![0-9A-Za-z.])", out + err)


# ---------- the determinism gate ----------

def test_different_d7_hashes_write_no_report_and_stop_before_the_final_checks(ctx):
    ctx.pass_order = REVERSED_Q1
    code, out, err = ctx.run(*ADJ)
    assert code == bc.EXIT_FAILURE and not ctx.out.exists() and list(ctx.outside.iterdir()) == []
    assert "No report was written" in err and br.BASELINE_RANKINGS_DIFFER in err
    assert ctx.freeze_calls == 1 and ctx.git_calls == 1  # never reached the final rechecks
    assert len(ctx.session_log) == 2  # no after-snapshot session either


def test_a_score_digest_mismatch_still_writes_the_report_and_says_so(ctx):
    ctx.pass_vectors = {2: {**sdx.VECTORS, fx.Q2.question: [0.875, 0.75, 0.5, 0.5, 0.25, 0.0625]}}
    code, out, err = ctx.run(*ADJ)
    assert (code, err) == (0, "")
    document = written(ctx)
    assert document["ranking_determinism"]["rankings_identical_across_passes"] is True
    assert document["score_diagnostics"]["scores_identical_across_passes"] is False
    assert "score digests identical across passes: no (diagnostic only)" in out


def test_only_the_first_pass_is_reported(ctx):
    ctx.pass_vectors = {2: {**sdx.VECTORS, fx.U1.question: [0.5, 0.5, 0.25, 0.25, 0.125, 0.0625]}}
    ctx.run(*ADJ)
    document = written(ctx)
    expected = [0.5, 0.5, 0.25, 0.25, 0.125, 0.0]  # the first pass's U1 vector (sdx.VECTORS)
    rows = {r["question_id"]: r["scores"] for r in document["score_diagnostics"]["questions"]["unanswerable"]}
    assert rows[fx.U1.question_id] == expected


# ---------- the final rechecks ----------

def test_the_freeze_binding_is_checked_again_and_a_changed_binding_blocks_the_report(ctx, monkeypatch):
    ctx.freeze_final = ctx.binding(dev_file_sha256="e" * 64)
    rendered = []
    real = rep.render_report
    monkeypatch.setattr(rep, "render_report", lambda *a, **k: rendered.append(1) or real(*a, **k))
    code, _, err = ctx.run(*ADJ)
    assert code == bc.EXIT_REFUSED and ctx.freeze_calls == 2 and rendered == [] and not ctx.out.exists()
    assert "changed during the run" in err and ctx.git_calls == 1


def test_a_final_binding_that_no_longer_matches_the_anchor_blocks_the_report(ctx):
    ctx.freeze_final = ctx.binding(record_sha256="9" * 64)
    code, _, err = ctx.run(*ADJ)
    assert code == bc.EXIT_REFUSED and not ctx.out.exists() and "9" * 64 not in err


@pytest.mark.parametrize("failure", [OSError("C:\\secret\\freeze_record.json"), RuntimeError("x " + RECORD),
                                     s.EvaluationDataError([s.Issue("FREEZE_X", "somewhere", "m")])])
def test_a_failing_final_freeze_load_prints_only_a_fixed_message(ctx, failure):
    ctx.freeze_final = failure
    code, out, err = ctx.run(*ADJ)
    assert code == bc.EXIT_REFUSED and not ctx.out.exists()
    assert err.strip().splitlines()[-1] == ("REFUSED: the frozen dataset record could not be verified again. "
                                            "No report was written.")
    for needle in ("secret", "FREEZE_X", "somewhere", RECORD, "OSError", "RuntimeError"):
        assert needle not in err


@pytest.mark.parametrize("final", [
    ("c" * 40, bc.GIT_STAGED), ("c" * 40, bc.GIT_UNSTAGED), (None, bc.GIT_UNKNOWN), ("c" * 40, bc.GIT_UNKNOWN),
    ("d" * 40, bc.GIT_CLEAN), (None, bc.GIT_CLEAN),
])
def test_a_dirty_unknown_or_moved_final_git_state_prevents_any_output(ctx, final):
    ctx.git_script = [CLEAN, final]
    code, _, err = ctx.run(*ADJ)
    assert code == bc.EXIT_REFUSED and ctx.git_calls == 2
    assert not ctx.out.exists() and list(ctx.outside.iterdir()) == []
    assert "No report was written" in err


def test_the_final_git_state_is_read_after_rendering_and_just_before_the_file(ctx):
    seen = []
    ctx.git_hook = lambda call: seen.append((call, ctx.out.exists()))
    ctx.run(*ADJ)
    assert seen == [(1, False), (2, False)] and ctx.out.exists()


# ---------- output file ----------

def test_an_existing_output_file_is_never_overwritten(ctx):
    ctx.out.write_text("keep", encoding="utf-8")
    code, _, _ = ctx.run(*ADJ)
    assert code == bc.EXIT_REFUSED and ctx.out.read_text(encoding="utf-8") == "keep"
    assert ctx.session_log == []  # refused in the preflight, before anything was read


def test_a_file_created_during_the_run_is_never_overwritten(ctx):
    ctx.git_hook = lambda call: ctx.out.write_text("someone else", encoding="utf-8") if call == 2 else None
    code, _, err = ctx.run(*ADJ)
    assert code == bc.EXIT_REFUSED and ctx.out.read_text(encoding="utf-8") == "someone else"


def test_a_file_that_appears_at_the_last_moment_is_refused_by_the_exclusive_create(ctx, monkeypatch):
    def racy(raw, root, err):
        ctx.out.write_text("raced", encoding="utf-8")  # appears after the final check
        return Path(raw)

    monkeypatch.setattr(pilot_cli, "_check_report_path", lambda raw, root, err: (
        Path(raw) if ctx.git_calls < 2 else racy(raw, root, err)))
    code, _, err = ctx.run(*ADJ)
    assert code == bc.EXIT_REFUSED and ctx.out.read_text(encoding="utf-8") == "raced" and "appeared" in err


@pytest.fixture
def no_deletion(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a file must never be deleted")

    for name in ("remove", "unlink", "rmdir"):
        monkeypatch.setattr(os, name, refuse)
    monkeypatch.setattr(Path, "unlink", refuse)
    monkeypatch.setattr(Path, "rmdir", refuse)


def test_a_write_failure_leaves_the_partial_file_in_place(ctx, monkeypatch, no_deletion):
    real_open = builtins.open

    class Failing:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()

        def write(self, text):
            self.handle.write(text[:100])
            self.handle.flush()
            raise OSError(28, "No space left on device: C:\\secret")

    def fake_open(path, mode="r", *a, **k):
        handle = real_open(path, mode, *a, **k)
        return Failing(handle) if mode == "x" else handle

    monkeypatch.setattr(bc, "open", fake_open, raising=False)
    code, _, err = ctx.run(*ADJ)
    assert code == bc.EXIT_FAILURE
    assert ctx.out.exists() and ctx.out.stat().st_size == 100  # left for inspection
    assert "partial" in err and "left in place" in err and "OSError" in err and "secret" not in err


def test_a_readback_difference_leaves_the_file_in_place(ctx, monkeypatch, no_deletion):
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"different")
    code, _, err = ctx.run(*ADJ)
    assert code == bc.EXIT_FAILURE and ctx.out.exists() and "differs" in err and "left in place" in err


def test_a_readback_error_leaves_the_file_in_place(ctx, monkeypatch, no_deletion):
    def boom(self):
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "read_bytes", boom)
    code, _, err = ctx.run(*ADJ)
    assert code == bc.EXIT_FAILURE and ctx.out.exists() and "read back" in err


def test_an_interrupt_while_writing_is_130_and_keeps_the_partial_file(ctx, monkeypatch, no_deletion):
    real_open = builtins.open

    def fake_open(path, mode="r", *a, **k):
        handle = real_open(path, mode, *a, **k)
        if mode == "x":
            handle.write("{")
            handle.flush()
            handle.close()
            raise KeyboardInterrupt
        return handle

    monkeypatch.setattr(bc, "open", fake_open, raising=False)
    code, _, err = ctx.run(*ADJ)
    assert code == 130 and ctx.out.read_text(encoding="utf-8") == "{" and "left for inspection" in err


# ---------- no database writes ----------

def test_only_read_only_session_methods_are_used(ctx):
    ctx.run(*ADJ)
    assert {name for _, name in ctx.session_calls} <= READ_ONLY_METHODS


def test_the_new_code_contains_no_database_write_or_transaction_control():
    banned_attributes = {"commit", "rollback", "execute", "executemany", "add", "add_all", "flush", "delete", "begin",
                         "merge", "bulk_save_objects", "bulk_insert_mappings"}
    for name in ("baseline_cli.py", "baseline_run.py"):
        source = (REPO / "app" / "rag" / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not banned_attributes & attributes, name
        assert not re.search(r"\b(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE)\b", source), name


# ---------- failures, interrupts and refusals ----------

def test_a_retriever_error_is_exit_1_with_only_the_class_name(ctx):
    def boom(number, query):
        raise RuntimeError("password=hunter2 host=db.example.com")

    ctx.retrieve_hook = boom
    code, out, err = ctx.run(*ADJ)
    assert code == bc.EXIT_FAILURE and not ctx.out.exists()
    assert "RuntimeError" in err and "hunter2" not in out + err and "example.com" not in out + err


def test_an_error_in_the_second_pass_is_exit_1_and_writes_nothing(ctx):
    def boom(number, query):
        if number == 2:
            raise RuntimeError("late")

    ctx.retrieve_hook = boom
    code, _, err = ctx.run(*ADJ)
    assert code == bc.EXIT_FAILURE and not ctx.out.exists() and ctx.freeze_calls == 1


def test_an_interrupt_during_a_pass_is_130_and_writes_nothing(ctx):
    def interrupt(number, query):
        raise KeyboardInterrupt

    ctx.retrieve_hook = interrupt
    code, _, err = ctx.run(*ADJ)
    assert code == 130 and not ctx.out.exists() and "no report was written" in err
    assert ctx.session_log[-1] == ("close", 1)  # the transaction was released


def test_a_changed_database_blocks_the_report(ctx):
    ctx.identical = False
    code, _, err = ctx.run(*ADJ)
    assert code == bc.EXIT_FAILURE and not ctx.out.exists() and "changed between the two snapshots" in err
    assert ctx.freeze_calls == 1


def test_a_failed_corpus_verification_is_exit_1(ctx):
    ctx.verification = SimpleNamespace(ok=False, problems=["bad"], documents=[])
    code, _, _ = ctx.run(*ADJ)
    assert code == bc.EXIT_FAILURE and not ctx.out.exists() and not ctx.retrievers


@pytest.mark.parametrize("change", [{"embedder_id": "other@1"}, {"live_problem": "wrong database"},
                                    {"readonly_problem": "not read only"}])
def test_refusals_before_or_at_the_start_of_reading_write_nothing(ctx, change):
    for name, value in change.items():
        setattr(ctx, name, value)
    code, _, _ = ctx.run(*ADJ)
    assert code == bc.EXIT_REFUSED and not ctx.out.exists() and not ctx.retrievers


def test_too_few_random_seeds_are_refused_before_the_database(ctx):
    code, _, err = ctx.run(*ADJ, "--random-seeds", "999")
    assert code == bc.EXIT_REFUSED and ctx.session_log == [] and "at least 1000" in err


def test_the_static_safety_gates_still_refuse_before_any_read(ctx):
    ctx.url = "postgresql://u:p@db.example.com:5432/documind_db"
    code, _, _ = ctx.run(*ADJ)
    assert code == bc.EXIT_REFUSED and ctx.session_log == [] and ctx.inputs_loaded == 0


def test_an_oracle_validity_failure_exits_1_after_the_report_is_written(ctx):
    code, _, err = ctx.run()  # no adjudication for q4:u1
    assert code == bc.EXIT_FAILURE and ctx.out.exists() and "oracle validity" in err
    assert rep.load_report(ctx.out.read_bytes())["ranking_determinism"]["passes"] == 2


def test_the_default_seeds_and_the_pilots_bootstrap_seed_are_used_when_none_is_given(ctx):
    ctx.run(*ADJ)
    settings = written(ctx)["settings"]
    assert settings["random_seeds"] == list(range(1000)) and settings["random_seed_count"] == 1000
    assert settings["bootstrap_seed"] == pilot_cli.DEFAULT_BOOTSTRAP_SEED
    assert settings["resamples"] == SMALL_BOOTSTRAP


def test_explicit_seed_arguments_reach_the_run(ctx):
    ctx.run(*ADJ, "--random-seeds", "1001", "--bootstrap-seed", "5")
    settings = written(ctx)["settings"]
    assert settings["random_seed_count"] == 1001 and settings["bootstrap_seed"] == 5


def test_the_pilot_report_kind_is_not_produced_by_the_baseline_command(ctx):
    ctx.run(*ADJ)
    assert written(ctx)["kind"] == rep.BASELINE_KIND != rep.REPORT_KIND
