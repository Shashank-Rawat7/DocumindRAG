"""C2.3d benchmark: real tokenizer behaviour, 512-token safety, CPU speed and memory
for intfloat/multilingual-e5-small at the pinned revision.

Measurement only. It does NOT touch PostgreSQL, does not import app.core
(settings / database), does not index anything and writes no files unless
--json-out is given (and refuses a path inside the project).

Uploaded documents are optional (--docs-dir). For them only counts are
reported: never text, token strings or file names.

Usage (PowerShell, project root):
    venv\\Scripts\\python.exe -X utf8 scripts\\benchmark_embedder.py
    venv\\Scripts\\python.exe -X utf8 scripts\\benchmark_embedder.py --docs-dir uploads --json-out $env:USERPROFILE\\DocuMindC23\\c23d.json
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

# Never touch the network: the model must already be in the local cache.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.chunking import DEFAULT_CHUNKER_CONFIG, ChunkingError, chunk_text  # noqa: E402

MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EMBEDDING_DIM = 384
MAX_MODEL_TOKENS = 512
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
BATCH_SIZES = (1, 8, 32)

# ---------- built-in samples (original text written for this benchmark) ----------

EN_PARAGRAPHS = [
    "PostgreSQL stores table rows in heap files and uses indexes to find rows quickly. "
    "A B-tree index keeps keys in sorted order, so equality and range queries can skip "
    "most of the table.",
    "FastAPI builds request validation on top of Python type hints. When a request body "
    "does not match the declared model, the framework returns a 422 response that lists "
    "every invalid field.",
    "Docker images are built in layers. Each instruction in a Dockerfile creates a new "
    "layer, and unchanged layers are reused from the build cache, which keeps rebuilds fast.",
    "Connection pooling lets an application reuse a small number of database connections "
    "instead of opening a new connection for every request. This reduces latency and "
    "protects the database from overload.",
    "Background tasks run after the response has been sent. They are useful for work such "
    "as sending email or processing an uploaded file, but long jobs should move to a "
    "dedicated worker.",
    "Vector search compares embeddings with a distance function such as cosine distance. "
    "Exact search scans every row, which is accurate and fast enough for small collections.",
]

# Hindi translations of EN_PARAGRAPHS, same order (used for cross-lingual pairs).
HI_PARAGRAPHS = [
    "PostgreSQL तालिका की पंक्तियों को हीप फ़ाइलों में संग्रहीत करता है और पंक्तियों को जल्दी "
    "खोजने के लिए इंडेक्स का उपयोग करता है। B-tree इंडेक्स कुंजियों को क्रम में रखता है, इसलिए "
    "समानता और सीमा वाली क्वेरी तालिका के अधिकांश भाग को छोड़ सकती हैं।",
    "FastAPI अनुरोध सत्यापन को Python टाइप हिंट्स पर आधारित करता है। जब अनुरोध का डेटा घोषित "
    "मॉडल से मेल नहीं खाता, तो फ़्रेमवर्क 422 प्रतिक्रिया लौटाता है जिसमें हर अमान्य फ़ील्ड की "
    "सूची होती है।",
    "Docker इमेज परतों में बनाई जाती हैं। Dockerfile का हर निर्देश एक नई परत बनाता है, और बिना "
    "बदली परतें बिल्ड कैश से दोबारा उपयोग की जाती हैं, जिससे दोबारा बिल्ड तेज़ रहता है।",
    "कनेक्शन पूलिंग से एप्लिकेशन हर अनुरोध के लिए नया कनेक्शन खोलने के बजाय कुछ डेटाबेस "
    "कनेक्शनों का दोबारा उपयोग करता है। इससे विलंब कम होता है और डेटाबेस ओवरलोड से सुरक्षित "
    "रहता है।",
    "बैकग्राउंड कार्य प्रतिक्रिया भेजे जाने के बाद चलते हैं। ये ईमेल भेजने या अपलोड की गई फ़ाइल "
    "को संसाधित करने जैसे कामों के लिए उपयोगी हैं, लेकिन लंबे कामों को अलग वर्कर में ले जाना "
    "चाहिए।",
    "वेक्टर खोज कोसाइन दूरी जैसे दूरी फ़ंक्शन से एम्बेडिंग की तुलना करती है। सटीक खोज हर "
    "पंक्ति को स्कैन करती है, जो छोटे संग्रहों के लिए सटीक और पर्याप्त तेज़ है।",
]

HI_PURE = (
    "यह अनुच्छेद केवल हिंदी में लिखा गया है ताकि टोकन की गिनती पर लैटिन शब्दों का असर न पड़े। "
    "हर वाक्य सरल है और उसमें आम शब्दों का प्रयोग किया गया है। संयुक्ताक्षर, मात्राएँ और "
    "अनुस्वार जैसे क्षत्रिय, प्रश्न, श्रद्धा और संस्कृति भी शामिल हैं।"
)

MIXED_TEXTS = [
    "Server restart करने के बाद भी API 500 error दे रहा है, इसलिए पहले logs check करो।",
    "Database migration चलाने से पहले backup लेना ज़रूरी है, वरना data loss हो सकता है।",
    "Kal meeting mein humne indexing ka plan discuss kiya aur sabne agree kiya.",
]

EMOJI_TEXTS = [
    "Deploy finished \U0001F680 and all tests passed \u2705",
    "Family \U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466 thumbs "
    "\U0001F44D\U0001F3FD flags \U0001F1EE\U0001F1F3 \U0001F1FA\U0001F1F8",
    "Keycap 1\uFE0F\u20E3 heart \u2764\uFE0F accent e\u0301 done",
    "बधाई हो \U0001F389 परीक्षा पास हो गई \U0001F44F",
]

QUERIES = {
    "en": QUERY_PREFIX + "how does an index make database lookups faster?",
    "hi": QUERY_PREFIX + "इंडेक्स डेटाबेस खोज को तेज़ कैसे बनाता है?",
}
QUERY_TARGET = 0  # both queries are about paragraph 0
UNRELATED = "The recipe needs two cups of flour, one egg and a pinch of salt."

# Controlled whitespace cases, reported one by one (section 3b), never mixed into 3a.
WHITESPACE_SAMPLES = {
    "leading_spaces": "   leading spaces",
    "trailing_spaces": "trailing spaces   ",
    "multiple_spaces": "multiple   spaces    between",
    "tabs": "tab\tseparated\t\twords",
    "newline": "line one\nline two",
    "blank_line": "paragraph one\n\nparagraph two",
    "crlf": "windows line\r\nnext line",
    "spaces_around_newline": "end of line  \n  start of next",
    "only_whitespace_between": "a \t \n b",
    "nbsp": "non\u00a0breaking\u00a0space",
    "hindi_whitespace": "  हिंदी   शब्द\tऔर\nपंक्ति  ",
}


def long_document(paragraphs: list[str], copies: int) -> str:
    """A long synthetic document: numbered copies of the paragraphs."""
    blocks = []
    for k in range(copies):
        for p in paragraphs:
            blocks.append(f"{k + 1}. {p}")
    return "\n\n".join(blocks)


# ---------- pure helpers (no torch) ----------

def summarize(values) -> dict:
    values = sorted(values)
    if not values:
        return {"n": 0}
    def pct(q):
        return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]
    return {
        "n": len(values),
        "min": values[0],
        "p50": pct(0.50),
        "p95": pct(0.95),
        "max": values[-1],
        "mean": round(statistics.fmean(values), 3),
    }


def analyze_offsets(text: str, offsets: list[tuple[int, int]]) -> dict:
    """Validity and shape of tokenizer character offsets for ``text``."""
    n = len(text)
    bad_type = out_of_range = overlaps = out_of_order = leading_ws = trailing_ws = 0
    zero_len = whitespace_only = 0
    zero_followed_same_start = zero_followed_after_gap = zero_trailing = 0
    covered = bytearray(n)
    prev_start = prev_end = 0
    # next_start[i]: start of the first non-empty token after token i (None if none)
    next_start: list[int | None] = [None] * len(offsets)
    upcoming = None
    for i in range(len(offsets) - 1, -1, -1):
        next_start[i] = upcoming
        s, e = offsets[i]
        if type(s) is int and type(e) is int and e > s:
            upcoming = s
    for i, (s, e) in enumerate(offsets):
        if type(s) is not int or type(e) is not int:
            bad_type += 1
            continue
        if not 0 <= s <= e <= n:
            out_of_range += 1
            continue
        if s < prev_start:
            out_of_order += 1
        prev_start = s  # every valid offset, zero-length included, takes part in ordering
        if s == e:
            zero_len += 1
            if next_start[i] is None:
                zero_trailing += 1
            elif next_start[i] == s:
                zero_followed_same_start += 1
            else:
                zero_followed_after_gap += 1
            continue
        if s < prev_end:
            overlaps += 1
        if text[s:e].isspace():
            whitespace_only += 1
        if text[s].isspace():
            leading_ws += 1
        if text[e - 1].isspace():
            trailing_ws += 1
        covered[s:e] = b"\x01" * (e - s)
        prev_end = max(prev_end, e)  # overlap state: non-zero-length spans only
    uncovered_content = sum(
        1 for pos, ch in enumerate(text) if not covered[pos] and not ch.isspace()
    )
    total = len(offsets)
    return {
        "tokens": total,
        "zero_length": zero_len,
        "zero_length_pct": round(100.0 * zero_len / total, 2) if total else 0.0,
        "zero_then_token_at_same_start": zero_followed_same_start,
        "zero_then_token_after_gap": zero_followed_after_gap,
        "zero_trailing": zero_trailing,
        "non_int_offsets": bad_type,
        "out_of_range": out_of_range,
        "out_of_order": out_of_order,
        "overlapping": overlaps,
        "span_starts_with_whitespace": leading_ws,
        "span_ends_with_whitespace": trailing_ws,
        "whitespace_only_spans": whitespace_only,
        "uncovered_non_whitespace_chars": uncovered_content,
    }

def spans_for_chunker(offsets: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    """Offsets -> C2.1 token spans (one span per real token), for this benchmark only.

    Returns (spans, zero_length, overlapping). Zero-length tokens are left out
    (the "attach to next real token" option under evaluation for C2.4; the
    chunker cannot count them, so the true model count is measured separately
    by re-tokenizing each chunk). Overlapping offsets cannot be represented by
    the C2.1 one-span-per-token API without changing the token count, so when
    ``overlapping > 0`` the caller must treat chunking as UNAVAILABLE.
    """
    spans: list[tuple[int, int]] = []
    zero_length = overlapping = 0
    prev_end = 0
    for s, e in offsets:
        if e <= s:
            zero_length += 1
            continue
        if s < prev_end:
            overlapping += 1
        spans.append((s, e))
        prev_end = max(prev_end, e)
    return spans, zero_length, overlapping


def pack_passages(paragraphs: list[str], count_tokens, target: int) -> list[str]:
    """Greedy paragraph packing up to ``target`` content tokens (speed inputs only;
    independent of the chunker so speed is measured even if chunking is unavailable)."""
    passages: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        candidate = "\n\n".join(current + [paragraph])
        if current and count_tokens(candidate) > target:
            passages.append("\n\n".join(current))
            current = [paragraph]
        else:
            current.append(paragraph)
    if current:
        passages.append("\n\n".join(current))
    return passages


def positive_int(value: str) -> int:
    try:
        number = int(value, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return number


# ---------- memory ----------

class PeakSampler:
    """Samples this process's RSS in a background thread while active."""

    def __init__(self, process, interval: float = 0.005):
        self.process = process
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.peak = self.process.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def mib(n: int) -> float:
    return round(n / (1024 * 1024), 1)


def memory_snapshot(process) -> dict:
    info = process.memory_info()
    snap = {"rss_mib": mib(info.rss)}
    peak = getattr(info, "peak_wset", None)  # Windows only: lifetime peak working set
    if peak is not None:
        snap["peak_working_set_mib"] = mib(peak)
    return snap


# ---------- output ----------

def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def show(label: str, value) -> None:
    print(f"  {label:42} {value}")


# ---------- uploaded documents (counts only) ----------

def load_docs(docs_dir: Path, max_docs: int) -> tuple[list[str], Counter, Counter]:
    """Texts of the .pdf/.txt/.md files in ``docs_dir``, extracted exactly as
    app.core.processing.extract_text() does (that module is not imported: it
    imports app.core.database). Returns (texts, read_by_suffix, skipped_by_reason);
    no names, no per-file rows.

    Parity with extract_text():
      .txt/.md  path.read_text(encoding="utf-8-sig", errors="replace")
      .pdf      "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
                (via app.rag.pdf_pages.read_pdf_page_texts, kept identical by the
                C2.2 drift tests)
      then      text.replace("\x00", ""); whitespace-only text counts as "no text"
      .docx     NOT measured here (skipped and counted)
    Any read failure is counted by exception type, as processing._run() turns any
    exception into a failed document; messages are never printed.
    """
    from app.rag.pdf_pages import read_pdf_page_texts

    # pypdf warnings can quote raw bytes of a malformed file; keep them out of the output.
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)

    texts: list[str] = []
    read: Counter = Counter()
    skipped: Counter = Counter()
    for path in sorted(p for p in docs_dir.iterdir() if p.is_file()):
        if len(texts) >= max_docs:
            skipped["over --max-docs"] += 1
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in (".txt", ".md"):
                text = path.read_text(encoding="utf-8-sig", errors="replace")
            elif suffix == ".pdf":
                text = "\n".join(read_pdf_page_texts(path))
            elif suffix == ".docx":
                skipped[".docx not measured"] += 1
                continue
            else:
                skipped["unsupported type"] += 1
                continue
        except Exception as exc:  # benchmark only: count, never print the message
            skipped[f"read error {type(exc).__name__}"] += 1
            continue
        text = text.replace("\x00", "")
        if not text.strip():
            skipped["no text"] += 1
            continue
        texts.append(text)
        read[suffix] += 1
    return texts, read, skipped


# ---------- model load probe (runs in a child process) ----------

def run_load_probe() -> int:
    """Import + load once in a fresh process and print one JSON line."""
    t0 = time.perf_counter()
    import psutil
    import torch  # noqa: F401
    from transformers import AutoModel, AutoTokenizer
    t1 = time.perf_counter()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    kw = dict(revision=REVISION, local_files_only=True, trust_remote_code=False)
    with PeakSampler(process) as peak:
        t2 = time.perf_counter()
        AutoTokenizer.from_pretrained(MODEL_ID, **kw)
        t3 = time.perf_counter()
        model = AutoModel.from_pretrained(MODEL_ID, use_safetensors=True, **kw)
        model.eval()
        t4 = time.perf_counter()
    print(json.dumps({
        "import_s": round(t1 - t0, 3),
        "tokenizer_load_s": round(t3 - t2, 3),
        "model_load_s": round(t4 - t3, 3),
        "rss_before_load_mib": mib(rss_before),
        "rss_after_load_mib": mib(process.memory_info().rss),
        "peak_rss_during_load_mib": mib(peak.peak),
    }))
    return 0


def measure_loads(runs: int) -> list[dict]:
    """Run the load probe ``runs`` times, sequentially, each in its own process.
    Only one model instance exists at a time and none in this process."""
    out = []
    for _ in range(runs):
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--load-probe"],
            cwd=str(PROJECT_ROOT), env=os.environ.copy(),
            capture_output=True, text=True, encoding="utf-8", errors="backslashreplace",
            timeout=900,
        )
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
            raise RuntimeError(f"load probe failed (exit {proc.returncode}):\n{tail}")
        out.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    return out


# ---------- benchmark ----------

def main(argv=None) -> int:
    started_at = time.time()  # wall clock at start, used for the uptime figure
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs-dir", type=Path, help="optional folder of .pdf/.txt/.md files")
    parser.add_argument("--max-docs", type=positive_int, default=100)
    parser.add_argument("--threads", type=positive_int, help="torch.set_num_threads (default: torch's choice)")
    parser.add_argument("--passages", type=positive_int, default=64,
                        help=f"passages per language for speed (a multiple of {max(BATCH_SIZES)})")
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--query-runs", type=positive_int, default=30)
    parser.add_argument("--load-runs", type=positive_int, default=3,
                        help="separate child processes that each load the model once")
    parser.add_argument("--json-out", type=Path, help="write results as JSON (outside the project)")
    parser.add_argument("--load-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.load_probe:
        return run_load_probe()
    if args.passages % max(BATCH_SIZES) != 0:
        parser.error(f"--passages must be a multiple of {max(BATCH_SIZES)} so every batch is full")

    if args.json_out is not None:
        out = args.json_out.resolve()
        if out == PROJECT_ROOT or PROJECT_ROOT in out.parents:
            parser.error("--json-out must be outside the project folder")
    if args.docs_dir is not None and not args.docs_dir.is_dir():
        parser.error("--docs-dir must be an existing folder")

    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except AttributeError:
        pass

    import psutil
    import regex
    import tokenizers
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModel, AutoTokenizer

    process = psutil.Process()
    results: dict = {"model": MODEL_ID, "revision": REVISION}
    checks: dict[str, bool | None] = {}  # None = UNAVAILABLE (could not be measured)

    # ---- 1. environment ----
    section("1. Environment")
    if args.threads:
        torch.set_num_threads(args.threads)
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "regex": regex.__version__,
        "psutil": psutil.__version__,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }
    for k, v in env.items():
        show(k, v)
    results["environment"] = env
    checks["cpu_only_torch_build"] = torch.version.cuda is None

    # ---- 2. load ----
    section("2. Model load (offline, pinned revision, safetensors, no remote code)")
    probes = measure_loads(args.load_runs)
    for k, probe in enumerate(probes, start=1):
        show(f"process {k}", ", ".join(f"{key}={val}" for key, val in probe.items()))
    load = {
        "uptime_at_start_min": round((started_at - psutil.boot_time()) / 60, 1),
        "first_process": probes[0],
        "later_processes_median": {
            key: round(statistics.median(p[key] for p in probes[1:]), 3) for key in probes[0]
        } if len(probes) > 1 else None,
    }
    show("system uptime at start (min)", load["uptime_at_start_min"])
    show("later processes, median", load["later_processes_median"])
    print("  Process 1 is a cold load only if the model files were not already in the OS")
    print("  file cache (e.g. first run after a reboot); processes 2+ are warm loads.")
    print("  True cold-cache state cannot be forced or verified from this script.")

    # The model used below is loaded once in this process; its memory is the baseline.
    gc.collect()
    mem_before = memory_snapshot(process)
    kw = dict(revision=REVISION, local_files_only=True, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, **kw)
    model = AutoModel.from_pretrained(MODEL_ID, use_safetensors=True, **kw)
    model.eval()
    gc.collect()
    mem_after = memory_snapshot(process)
    load.update({
        "tokenizer_class": type(tok).__name__,
        "tokenizer_is_fast": tok.is_fast,
        "tokenizer_model_max_length": tok.model_max_length,
        "model_class": type(model).__name__,
        "hidden_size": model.config.hidden_size,
        "dtype": str(next(model.parameters()).dtype),
        "parameters": sum(p.numel() for p in model.parameters()),
        "rss_after_load_mib": mem_after["rss_mib"],
        "rss_added_by_load_mib": round(mem_after["rss_mib"] - mem_before["rss_mib"], 1),
    })
    for key in ("tokenizer_class", "tokenizer_is_fast", "tokenizer_model_max_length", "model_class",
                "hidden_size", "dtype", "parameters", "rss_after_load_mib", "rss_added_by_load_mib"):
        show(key, load[key])
    results["load"] = load
    checks["fast_tokenizer"] = bool(tok.is_fast)
    checks["hidden_size_384"] = model.config.hidden_size == EMBEDDING_DIM
    special = len(tok("x")["input_ids"]) - len(tok("x", add_special_tokens=False)["input_ids"])
    show("special tokens added per input", special)
    results["special_tokens_per_input"] = special

    def content_offsets(text: str) -> list[tuple[int, int]]:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(o) for o in enc["offset_mapping"]]

    def model_input_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])

    # ---- 3. tokenizer behaviour ----
    section("3. Tokenizer offsets (content tokens, add_special_tokens=False)")
    categories = {
        "english": EN_PARAGRAPHS,
        "hindi": HI_PARAGRAPHS + [HI_PURE],
        "hindi_pure": [HI_PURE],
        "mixed": MIXED_TEXTS,
        "emoji": EMOJI_TEXTS,
    }
    tokenizer_results = {}
    zero_len_tokens: Counter = Counter()
    for name, texts in categories.items():
        agg: Counter = Counter()
        chars = words = 0
        for text in texts:
            offsets = content_offsets(text)
            agg.update(analyze_offsets(text, offsets))
            chars += len(text)
            words += len(text.split())
            ids = tok(text, add_special_tokens=False)["input_ids"]
            for (s, e), tid in zip(offsets, ids):
                if s == e:
                    zero_len_tokens[tok.convert_ids_to_tokens(tid)] += 1
        row = dict(agg)
        row.pop("zero_length_pct", None)
        row["zero_length_pct"] = round(100.0 * row["zero_length"] / row["tokens"], 2)
        row["tokens_per_char"] = round(row["tokens"] / chars, 3)
        row["tokens_per_word"] = round(row["tokens"] / words, 3)
        tokenizer_results[name] = row
        print(f"  [{name}]")
        for k, v in row.items():
            show("  " + k, v)
    show("zero-length token strings (built-in samples)",
         ascii(dict(zero_len_tokens.most_common(10))))
    results["tokenizer"] = tokenizer_results
    results["zero_length_token_strings"] = {ascii(k): v for k, v in zero_len_tokens.most_common(10)}
    checks["offsets_valid"] = all(
        r["non_int_offsets"] == 0 and r["out_of_range"] == 0 and r["out_of_order"] == 0
        for r in tokenizer_results.values()
    )

    section("3b. Whitespace boundaries (built-in, one sample at a time, not in 3a totals)")
    whitespace_results = {}
    for name, text in WHITESPACE_SAMPLES.items():
        offsets = content_offsets(text)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        stats = analyze_offsets(text, offsets)
        tokens = [
            [s, e, ascii(text[s:e]), ascii(tok.convert_ids_to_tokens(tid))]
            for (s, e), tid in zip(offsets, ids)
        ]
        whitespace_results[name] = {"text": ascii(text), "stats": stats, "tokens": tokens}
        print(f"  [{name}] text={ascii(text)}")
        show("  non-zero stats", {k: v for k, v in stats.items() if v and k != "tokens"})
        show("  tokens", stats["tokens"])
        for s, e, piece, token in tokens:
            print(f"      ({s:>2},{e:>2}) slice={piece:24} token={token}")
    results["tokenizer_whitespace"] = whitespace_results
    checks["whitespace_offsets_valid"] = all(
        r["stats"]["non_int_offsets"] == 0 and r["stats"]["out_of_range"] == 0
        and r["stats"]["out_of_order"] == 0 for r in whitespace_results.values()
    )

    # ---- 4. 512-token safety with the C2.1 chunker ----
    section(f"4. 512-token safety: '{PASSAGE_PREFIX}' + chunk, special tokens included "
            f"(chunker {DEFAULT_CHUNKER_CONFIG.target_tokens}/"
            f"{DEFAULT_CHUNKER_CONFIG.max_tokens}/{DEFAULT_CHUNKER_CONFIG.overlap_tokens})")
    corpora = {
        "english_long": long_document(EN_PARAGRAPHS, 12),
        "hindi_long": long_document(HI_PARAGRAPHS + [HI_PURE], 12),
        "mixed_long": long_document(MIXED_TEXTS + EMOJI_TEXTS + EN_PARAGRAPHS[:2] + HI_PARAGRAPHS[:2], 12),
    }
    doc_texts: list[str] = []
    if args.docs_dir is not None:
        doc_texts, doc_read, doc_skipped = load_docs(args.docs_dir, args.max_docs)
        show("uploaded files read (by type)", dict(doc_read))
        show("uploaded files skipped (by reason)", dict(doc_skipped))
        results["documents_read"] = dict(doc_read)
        results["documents_skipped"] = dict(doc_skipped)

    def safety_for(text: str, verbose: bool) -> dict:
        offsets = content_offsets(text)
        spans, zero_length, overlapping = spans_for_chunker(offsets)
        row = {
            "chars": len(text),
            "content_tokens": len(offsets),
            "zero_length_tokens_not_counted_by_chunker": zero_length,
            "overlapping_tokens": overlapping,
        }
        if overlapping:
            row["status"] = "unavailable"
            row["reason"] = ("overlapping tokenizer offsets cannot be represented by the C2.1 "
                             "one-span-per-token API without changing the token count")
            return row
        try:
            chunks = chunk_text(text, spans, DEFAULT_CHUNKER_CONFIG)
        except ChunkingError as exc:
            row["status"] = "chunking_error"
            row["error"] = (type(exc).__name__ + ": " + str(exc)[:120]) if verbose else type(exc).__name__
            return row
        model_tokens = [model_input_tokens(PASSAGE_PREFIX + c.text) for c in chunks]
        retokenized = [len(content_offsets(c.text)) for c in chunks]
        groups = [c.token_count for c in chunks]
        row.update({
            "status": "measured",
            "chunks": len(chunks),
            "chunker_tokens_per_chunk": summarize(groups),
            "retokenized_chunk_tokens": summarize(retokenized),
            "model_input_tokens": summarize(model_tokens),
            "over_512": sum(1 for t in model_tokens if t > MAX_MODEL_TOKENS),
            "min_margin_to_512": MAX_MODEL_TOKENS - max(model_tokens) if model_tokens else None,
            "max_retokenized_minus_chunker": max(
                (r - g for r, g in zip(retokenized, groups)), default=0
            ),
        })
        return row

    safety = {}
    for label, text in corpora.items():
        safety[label] = safety_for(text, verbose=True)
        print(f"  [{label}]")
        for k, v in safety[label].items():
            show("  " + k, v)

    doc_rows: list[dict] = []
    if doc_texts:
        doc_rows = [safety_for(text, verbose=False) for text in doc_texts]
        measured = [r for r in doc_rows if r["status"] == "measured"]
        doc_offsets: Counter = Counter()
        for text in doc_texts:
            doc_offsets.update(analyze_offsets(text, content_offsets(text)))
        agg = dict(doc_offsets)
        agg["zero_length_pct"] = (
            round(100.0 * agg["zero_length"] / agg["tokens"], 2) if agg.get("tokens") else 0.0
        )
        docs_summary = {
            "documents": len(doc_rows),
            "status_counts": dict(Counter(r["status"] for r in doc_rows)),
            "chunking_errors": dict(Counter(r["error"] for r in doc_rows
                                            if r["status"] == "chunking_error")),
            "measured_chunks": sum(r["chunks"] for r in measured),
            "max_model_input_tokens": max((r["model_input_tokens"]["max"] for r in measured
                                           if r["chunks"]), default=None),
            "chunks_over_512": sum(r["over_512"] for r in measured),
            "max_retokenized_minus_chunker": max((r["max_retokenized_minus_chunker"]
                                                  for r in measured), default=None),
            "offsets": agg,
        }
        print("  [uploaded files, aggregate counts only]")
        for k, v in docs_summary.items():
            show("  " + k, v)
        safety["uploaded_documents"] = docs_summary
    results["safety_512"] = safety

    all_rows = [safety[k] for k in corpora] + doc_rows
    statuses = {r["status"] for r in all_rows}
    over = sum(r.get("over_512", 0) for r in all_rows)
    checks["chunking_measured"] = (
        False if "chunking_error" in statuses else None if "unavailable" in statuses else True
    )
    checks["no_chunk_over_512"] = (
        False if over else None if statuses != {"measured"} else True
    )

    # Speed inputs: paragraph packing up to the chunk target, independent of the chunker.
    def count_content(text: str) -> int:
        return len(content_offsets(text))

    perf_passages = {
        "english": [PASSAGE_PREFIX + p for p in pack_passages(
            corpora["english_long"].split("\n\n"), count_content, DEFAULT_CHUNKER_CONFIG.target_tokens)],
        "hindi": [PASSAGE_PREFIX + p for p in pack_passages(
            corpora["hindi_long"].split("\n\n"), count_content, DEFAULT_CHUNKER_CONFIG.target_tokens)],
    }
    results["speed_passage_tokens"] = {
        lang: summarize([model_input_tokens(p) for p in ps]) for lang, ps in perf_passages.items()
    }

    # ---- embedding helper ----
    def embed(texts: list[str]):
        batch = tok(texts, padding=True, truncation=False, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"]
        pooled = hidden.masked_fill(~mask[..., None].bool(), 0.0).sum(dim=1) / mask.sum(dim=1)[..., None]
        return F.normalize(pooled, p=2, dim=1), int(mask.sum()), int(mask.numel())

    # ---- 5. speed ----
    section("5. CPU speed (torch.inference_mode, padding to the longest in each batch)")
    speed = {}
    with PeakSampler(process) as encode_peak:
        for lang in ("english", "hindi"):
            show(f"{lang} speed passage tokens", results["speed_passage_tokens"][lang])
            base = perf_passages[lang]
            passages = [base[i % len(base)] for i in range(args.passages)]
            for bs in BATCH_SIZES:
                batches = [passages[i:i + bs] for i in range(0, len(passages), bs)]
                embed(batches[0])  # warm-up
                times, real_tokens, padded_tokens = [], 0, 0
                for _ in range(args.repeats):
                    for b in batches:
                        t0 = time.perf_counter()
                        _, real, padded = embed(b)
                        times.append(time.perf_counter() - t0)
                        real_tokens += real
                        padded_tokens += padded
                total = sum(times)
                n = len(passages) * args.repeats
                row = {
                    "passages_per_s": round(n / total, 2),
                    "real_tokens_per_s": round(real_tokens / total, 1),
                    "ms_per_batch_p50": round(1000 * statistics.median(times), 1),
                    "ms_per_batch_max": round(1000 * max(times), 1),
                    "padding_overhead_pct": round(100.0 * (padded_tokens - real_tokens) / padded_tokens, 1),
                }
                speed[f"{lang}/batch_{bs}"] = row
                print(f"  [{lang}, batch {bs}]  " + ", ".join(f"{k}={v}" for k, v in row.items()))

        query_times = []
        embed([QUERIES["en"]])  # warm-up
        for i in range(args.query_runs):
            q = QUERIES["en"] if i % 2 == 0 else QUERIES["hi"]
            t0 = time.perf_counter()
            embed([q])
            query_times.append(1000 * (time.perf_counter() - t0))
    speed["single_query_ms"] = {k: (round(v, 2) if isinstance(v, float) else v)
                                for k, v in summarize(query_times).items()}
    show("single query latency (ms)", speed["single_query_ms"])
    results["speed"] = speed

    mem_end = memory_snapshot(process)
    memory = {
        "rss_after_load_mib": load["rss_after_load_mib"],
        "peak_rss_during_encoding_mib": mib(encode_peak.peak),
        "encoding_peak_above_loaded_mib": round(mib(encode_peak.peak) - load["rss_after_load_mib"], 1),
        "rss_at_end_mib": mem_end["rss_mib"],
    }
    if "peak_working_set_mib" in mem_end:
        memory["process_lifetime_peak_working_set_mib"] = mem_end["peak_working_set_mib"]
    section("6. Memory")
    for k, v in memory.items():
        show(k, v)
    results["memory"] = memory

    # ---- 7. sanity: repeatability, normalisation, batch consistency, multilingual ----
    section("7. Sanity checks (not an evaluation)")
    en = [PASSAGE_PREFIX + p for p in EN_PARAGRAPHS]
    hi = [PASSAGE_PREFIX + p for p in HI_PARAGRAPHS]
    all_passages = en + hi + [PASSAGE_PREFIX + UNRELATED]
    e1, _, _ = embed(all_passages)
    e2, _, _ = embed(all_passages)
    norms = e1.norm(dim=1)
    single = torch.cat([embed([p])[0] for p in all_passages])
    batch_diff = float((single - e1).abs().max())
    batch_cos_min = float((single * e1).sum(dim=1).min())

    sims = e1[: len(en)] @ e1[len(en): len(en) + len(hi)].T
    pairs_ok = sum(1 for i in range(len(en)) if int(sims[i].argmax()) == i)
    diag = [round(float(sims[i, i]), 4) for i in range(len(en))]
    off = [float(sims[i, j]) for i in range(len(en)) for j in range(len(hi)) if i != j]

    q_emb, _, _ = embed([QUERIES["en"], QUERIES["hi"]])
    q_scores = q_emb @ e1.T
    targets = {QUERY_TARGET, len(en) + QUERY_TARGET}
    q_top1 = {lang: int(q_scores[k].argmax()) in targets for k, lang in enumerate(("en", "hi"))}
    q_unrelated_lowest = {
        lang: int(q_scores[k].argmin()) == len(all_passages) - 1 for k, lang in enumerate(("en", "hi"))
    }

    sanity = {
        "shape": list(e1.shape),
        "norm_max_abs_error": float((norms - 1).abs().max()),
        "identical_on_repeat": bool(torch.equal(e1, e2)),
        "single_vs_batch_max_abs_diff": batch_diff,
        "single_vs_batch_min_cosine": round(batch_cos_min, 6),
        "translation_pairs_top1": f"{pairs_ok}/{len(en)}",
        "translation_pair_similarity": diag,
        "mismatched_pair_similarity_mean": round(statistics.fmean(off), 4),
        "query_top1_is_target": q_top1,
        "query_unrelated_scores_lowest": q_unrelated_lowest,
        "query_scores_en": [round(float(s), 4) for s in q_scores[0]],
        "query_scores_hi": [round(float(s), 4) for s in q_scores[1]],
    }
    for k, v in sanity.items():
        show(k, v)
    results["sanity"] = sanity
    checks["shape_n_x_384"] = list(e1.shape) == [len(all_passages), EMBEDDING_DIM]
    checks["unit_norm"] = sanity["norm_max_abs_error"] < 1e-5
    checks["repeatable"] = sanity["identical_on_repeat"]
    checks["batch_consistent"] = batch_cos_min > 0.9999
    checks["translation_pairs_all_top1"] = pairs_ok == len(en)
    checks["queries_find_target"] = all(q_top1.values())

    # ---- summary ----
    section("Checks")
    for k, v in checks.items():
        show(k, "UNAVAILABLE" if v is None else "PASS" if v else "FAIL")
    results["checks"] = checks

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"\nResults written to {args.json_out}")
    if any(v is False for v in checks.values()):
        return 1
    if any(v is None for v in checks.values()):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())