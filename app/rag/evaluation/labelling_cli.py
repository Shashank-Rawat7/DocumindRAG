"""Command line for labelling the evaluation dataset (E1.6). All logic is in ``labelling``.

Run from the project root (see ``scripts/e16_label.py``):

    show            list the corpus documents, or print a slice of one (read-only)
    add-answerable  add answerable questions from a JSON spec; quotes are anchored by the tool
    add-unanswerable
    check           structural validation of eval/dataset (read-only); does not need human reviews
    readiness       which human reviews are missing, stale or pending (read-only; exit 1 if not ready)
    record-evidence-review
                    append a human evidence review of named questions as they are now (supersedes earlier ones)
    record-ai-semantic-review
                    append provenance of an independent AI review of unanswerable questions (not a human review)
    audit-sample    draw the blind audit sample; write the auditor questions (opaque audit ids, shuffled) and,
                    in a new private folder whose owner-only access is verified, the PRIVATE alias map (audit
                    ids and neutral document ids D01.. back to real ones)
    audit-bundle    build and validate the bundle for the independent auditor (never contains the alias map)
    audit-record    record the auditor's independently found evidence and compare it (--alias-map translates
                    audit ids and document ids back to the real question ids and corpus keys)

The tool reads only repository files under eval/. It never touches the database, and it writes
only the files in the dataset directory (and the audit files at --out, --private-dir and --out-dir).
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date as _date
from pathlib import Path

from app.rag.evaluation import audit_bundle, labelling
from app.rag.evaluation.schema import SEVERITY_ERROR, EvaluationDataError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = PROJECT_ROOT / "eval" / "corpus"
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "dataset"


def _threshold(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number from 0 to 1") from None
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be a number from 0 to 1")
    return number


def _seed(value: str) -> int:
    if not value.isdigit():
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="e16_label", description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser, *, threshold: bool = False) -> None:
        sub.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
        sub.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
        if threshold:
            sub.add_argument("--near-duplicate-threshold", type=_threshold,
                             default=labelling.PROVISIONAL_NEAR_DUPLICATE_THRESHOLD,
                             help="provisional default; matches are warnings (calibrate on real dev data)")

    show = commands.add_parser("show", help="read-only: list documents or print a slice", allow_abbrev=False)
    common(show)
    show.add_argument("--key", help="corpus key; omit to list the documents")
    show.add_argument("--start", type=int, default=0)
    show.add_argument("--end", type=int, help="default: start + 2000")
    show.add_argument("--find", help="print the offsets of every exact occurrence of this text")

    for name, help_text in (("add-answerable", "add answerable questions from a JSON spec"),
                            ("add-unanswerable", "add unanswerable dev questions from a JSON spec")):
        add = commands.add_parser(name, help=help_text, allow_abbrev=False)
        common(add, threshold=True)
        add.add_argument("--spec", type=Path, required=True)
        add.add_argument("--evidence-reviewer", required=True, help="the person who reviewed the evidence labels (a real name, never a placeholder)")
        add.add_argument("--hindi-reviewer", help="the native speaker who reviewed the Hindi wording; required when any question has query_language hi. Name no one for a review that has not happened: a Hindi question is added only once it is reviewed")
        if name == "add-unanswerable":
            add.add_argument("--semantic-reviewer",
                             help="a person who read the unanswerable questions as a second reader (a real name); "
                                  "optional if --ai-semantic-basis is given")
            add.add_argument("--ai-semantic-basis",
                             help="where an independent AI semantic review is documented; records an ai_semantic_review "
                                  "event and no human reviewer, model or date")
        add.add_argument("--date", default=None, help="review date YYYY-MM-DD (default: today)")
        add.add_argument("--hindi-review-date", default=None,
                         help="date of the native-speaker Hindi review YYYY-MM-DD, when it differs from --date; "
                              "used only for the hindi_review event (default: --date); refused if no question is Hindi")

    check = commands.add_parser("check", help="read-only: structural validation (not review state)", allow_abbrev=False)
    common(check, threshold=True)

    ready = commands.add_parser("readiness", help="read-only: missing, stale and pending human reviews",
                                allow_abbrev=False)
    common(ready)

    review = commands.add_parser("record-evidence-review", allow_abbrev=False,
                                 help="record a human evidence review of the questions as they are now")
    common(review)
    review.add_argument("--questions", nargs="+", required=True, help="question ids that were actually reviewed")
    review.add_argument("--evidence-reviewer", required=True,
                        help="the reviewer's name, recorded as attribution only (the tool does not verify identity)")
    review.add_argument("--date", default=None, help="review date YYYY-MM-DD (default: today)")

    ai = commands.add_parser("record-ai-semantic-review", allow_abbrev=False,
                             help="record that an independent AI review confirmed unanswerable questions")
    common(ai)
    ai.add_argument("--questions", nargs="+", required=True)
    ai.add_argument("--basis", required=True, help="where the review is documented (no reviewer or model is invented)")
    ai.add_argument("--date", default=None, help="only if the review date is known")

    sample = commands.add_parser("audit-sample", help="draw and record the blind audit sample", allow_abbrev=False)
    common(sample)
    sample.add_argument("--seed", type=_seed, required=True)
    sample.add_argument("--out", type=Path, required=True,
                        help="where to write the auditor questions (audit id, question, language only; must not exist)")
    sample.add_argument("--private-dir", type=Path, required=True,
                        help="a NEW folder for the PRIVATE alias map; created with owner-only access that is verified "
                             "(refused if it cannot be). Must not exist, must not contain --out, and must be outside "
                             "the dataset and corpus folders")
    sample.add_argument("--alias-seed", type=_seed, default=None,
                        help="makes the audit-id shuffle reproducible (rehearsals only; default: system randomness)")
    sample.add_argument("--date", default=None)

    bundle = commands.add_parser("audit-bundle", help="build and validate the auditor bundle", allow_abbrev=False)
    common(bundle)
    bundle.add_argument("--questions", type=Path, required=True, help="the auditor questions written by audit-sample")
    bundle.add_argument("--private-map", type=Path, required=True,
                        help="the private map file written by audit-sample (read only; never copied into the bundle)")
    bundle.add_argument("--out-dir", type=Path, required=True, help="the bundle folder to create (must not exist)")

    record = commands.add_parser("audit-record", help="record the auditor's independent evidence", allow_abbrev=False)
    common(record)
    record.add_argument("--spec", type=Path, required=True)
    record.add_argument("--auditor", required=True)
    record.add_argument("--date", default=None)
    record.add_argument("--alias-map", type=Path, default=None,
                        help="the private map: the spec then names questions by audit id (each once) and documents by neutral id")
    return parser


def _print_issues(issues) -> None:
    for item in issues:
        print(f"{item.severity.upper():7} {item.code} at {item.location}: {item.message}")


def _read_spec(path: Path) -> list[dict]:
    try:
        return labelling.read_spec(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError):
        raise labelling.LabellingError("the spec file cannot be read as UTF-8") from None


def _show(args: argparse.Namespace) -> int:
    manifest = labelling.load_manifest(args.corpus_dir)
    texts = labelling.load_texts(args.corpus_dir, manifest)
    if args.key is None:
        for entry in manifest.documents:
            print(f"{entry.corpus_key}\t{entry.language}\t{len(texts[entry.corpus_key])} characters")
        return 0
    if args.key not in texts:
        raise labelling.LabellingError("the corpus key is not a document of the manifest")
    text = texts[args.key]
    if args.find is not None:
        offsets, position = [], text.find(args.find) if args.find else -1
        while position != -1:
            offsets.append(position)
            position = text.find(args.find, position + 1)
        print(f"{len(offsets)} occurrence(s)")
        for number, offset in enumerate(offsets, start=1):
            print(f"  occurrence {number}: start {offset}, end {offset + len(args.find)}")
        return 0
    start = max(0, args.start)
    end = min(len(text), start + 2000 if args.end is None else args.end)
    if start >= end:
        raise labelling.LabellingError("the slice is empty")
    print(f"[{args.key} offsets {start}-{end} of {len(text)}]")
    print(text[start:end])
    return 0


def _add(args: argparse.Namespace, *, answerable: bool) -> int:
    review = labelling.Review(
        evidence_reviewer=args.evidence_reviewer,
        date=args.date or _date.today().isoformat(),
        hindi_reviewer=args.hindi_reviewer,
        semantic_reviewer=None if answerable else args.semantic_reviewer,
        ai_semantic_basis=None if answerable else args.ai_semantic_basis,
        hindi_review_date=args.hindi_review_date,
    )
    result = labelling.add_questions(
        args.dataset_dir, args.corpus_dir, _read_spec(args.spec),
        answerable=answerable, review=review, threshold=args.near_duplicate_threshold,
    )
    _print_issues(result.issues)
    if not result.written:
        print("nothing was written")
        return 1
    print(f"added {len(result.added)} question(s); {len(result.issues)} warning(s)")
    return 0


def _check(args: argparse.Namespace) -> int:
    issues = labelling.check_all(args.dataset_dir, args.corpus_dir, args.near_duplicate_threshold)
    _print_issues(issues)
    errors = sum(1 for item in issues if item.severity == SEVERITY_ERROR)
    dataset = labelling.load_dataset(args.dataset_dir)
    print(f"{len(dataset.dev)} dev and {len(dataset.test)} test question(s); "
          f"{errors} error(s), {len(issues) - errors} warning(s)")
    print("structural validation only; run readiness for the state of the human reviews")
    return 1 if errors else 0


def _readiness(args: argparse.Namespace) -> int:
    dataset = labelling.load_dataset(args.dataset_dir)
    r = labelling.review_readiness(dataset, labelling.load_events(args.dataset_dir))
    print(f"{r.questions} question(s); {r.current_evidence_reviews} with a current human evidence review")
    if r.missing_evidence_review:
        print(f"MISSING human evidence review ({len(r.missing_evidence_review)}): {', '.join(r.missing_evidence_review)}")
    for qid, reason in r.stale_evidence_review:
        print(f"STALE   human evidence review for {qid}: {reason}")
    if r.hindi_review_pending:
        print(f"PENDING native-speaker Hindi review ({len(r.hindi_review_pending)} of {r.hindi_questions} Hindi): "
              f"{', '.join(r.hindi_review_pending)}")
    if r.ai_only_semantic_review:
        print(f"AI-ONLY semantic review, no human reader; informational, does not block ({len(r.ai_only_semantic_review)}): "
              f"{', '.join(r.ai_only_semantic_review)}")
    if r.semantic_review_unverified:
        print(f"NO semantic review provenance ({len(r.semantic_review_unverified)}): "
              f"{', '.join(r.semantic_review_unverified)}")
    print("ready for promotion" if r.ready else "not ready for promotion; nothing was changed")
    return 0 if r.ready else 1


def _record_ai_semantic_review(args: argparse.Namespace) -> int:
    count = labelling.record_ai_semantic_review(args.dataset_dir, args.questions, args.basis, args.date)
    print(f"recorded an AI semantic review for {count} question(s); it is not a human review")
    return 0


def _record_evidence_review(args: argparse.Namespace) -> int:
    count = labelling.record_evidence_review(
        args.dataset_dir, args.questions, args.evidence_reviewer, args.date or _date.today().isoformat())
    print(f"recorded a human evidence review for {count} question(s); earlier events are unchanged")
    return 0


def _inside(path: Path, folder: Path) -> bool:
    resolved = path.resolve()
    return resolved.parent == folder.resolve() or folder.resolve() in resolved.parents


def _audit_sample(args: argparse.Namespace) -> int:
    date = args.date or _date.today().isoformat()
    if not labelling.is_valid_iso_date(date):
        raise labelling.LabellingError("the date must be a real YYYY-MM-DD date")
    out, private = args.out, args.private_dir
    if out.exists() or private.exists():
        raise labelling.LabellingError("an output file already exists")
    if _inside(out, private):
        raise labelling.LabellingError("the private map must be in a different folder from the auditor questions")
    if _inside(private, args.dataset_dir) or _inside(private, args.corpus_dir):
        raise labelling.LabellingError("the private folder must not be inside the dataset or corpus folder")
    dataset = labelling.load_dataset(args.dataset_dir)
    if any(e["event"] == labelling.EVENT_AUDIT_SAMPLE for e in labelling.load_events(args.dataset_dir)):
        raise labelling.LabellingError("an audit sample is already recorded")
    ids = labelling.draw_audit_sample(dataset, args.seed)
    aliases = audit_bundle.make_aliases(dataset, ids, args.alias_seed)
    documents = audit_bundle.make_document_aliases(labelling.load_manifest(args.corpus_dir), args.alias_seed)
    questions = audit_bundle.auditor_questions(dataset, aliases)
    # Fails closed: the folder is only kept if its owner-only access could be applied and verified.
    audit_bundle.create_private_dir(private)
    map_path = private / audit_bundle.MAP_FILE
    try:
        audit_bundle.write_private_map(map_path, aliases, documents, date)
        audit_bundle.verify_private(map_path, directory=False)
        audit_bundle.write_questions(out, questions)
        # The sample is recorded last. If anything fails, nothing is deleted: an interrupted run leaves its files for
        # manual recovery (guidelines section 9), and a sample that was recorded needs the map that was written.
        recorded = labelling.record_audit_sample(args.dataset_dir, dataset, args.seed, date)
    except BaseException:
        print(f"audit-sample did not finish; nothing was deleted. Check {private} and {out}, and see 'Interrupted "
              "sampling' in eval/dataset/guidelines.md before running it again.", file=sys.stderr)
        raise
    if recorded != ids:
        raise labelling.LabellingError("the recorded sample differs from the drawn sample; run check")
    print(f"sampled {len(ids)} question(s) with seed {args.seed}; auditor questions and private map written")
    return 0


def _audit_bundle(args: argparse.Namespace) -> int:
    count = audit_bundle.build_bundle(args.dataset_dir, args.corpus_dir, args.questions, args.private_map, args.out_dir)
    print(f"built and validated the auditor bundle for {count} question(s)")
    return 0


def _audit_record(args: argparse.Namespace) -> int:
    aliases = documents = None
    if args.alias_map is not None:
        aliases, documents = audit_bundle.read_private_map(args.alias_map), audit_bundle.read_document_map(args.alias_map)
    results = labelling.record_audit_results(
        args.dataset_dir, args.corpus_dir, _read_spec(args.spec),
        auditor=args.auditor, date=args.date or _date.today().isoformat(), alias_map=aliases,
        document_map=documents,
    )
    disagreements = sum(1 for r in results if r["outcome"] != "agree")
    print(f"recorded {len(results)} result(s); {disagreements} disagreement(s)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "show":
            return _show(args)
        if args.command == "add-answerable":
            return _add(args, answerable=True)
        if args.command == "add-unanswerable":
            return _add(args, answerable=False)
        if args.command == "check":
            return _check(args)
        if args.command == "readiness":
            return _readiness(args)
        if args.command == "record-evidence-review":
            return _record_evidence_review(args)
        if args.command == "record-ai-semantic-review":
            return _record_ai_semantic_review(args)
        if args.command == "audit-sample":
            return _audit_sample(args)
        if args.command == "audit-bundle":
            return _audit_bundle(args)
        return _audit_record(args)
    except EvaluationDataError as error:
        _print_issues(error.issues)
        return 2
    except labelling.LabellingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
