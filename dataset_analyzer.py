"""
dataset_analyzer.py

Analyzes uploaded datasets (CSV, JSON, TXT, PDF-as-text) and streams
cognitive signal frames to a WebSocket-compatible async callable.

Each frame simulates dual-brain (human + AI) cognitive activity
as it processes the dataset in real-time.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import random
import re
import string
import time
from collections import Counter
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional numpy import — used for enhanced stats if available
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Optional pandas import — used for CSV analysis if available
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASES = ["encoding", "pattern_detection", "integration", "consolidation", "insight"]

PHASE_BOUNDARIES = [0.0, 0.20, 0.45, 0.70, 0.90, 1.0]

NUM_TRANSFORMER_LAYERS = 32

FIRING_PATTERNS = ["burst", "tonic", "synchronized", "theta", "gamma"]

# Positive / negative sentiment words for TXT analysis
POSITIVE_WORDS = frozenset([
    "good", "great", "excellent", "best", "love", "wonderful", "fantastic",
    "amazing", "positive", "success", "happy", "joy", "beautiful", "perfect",
    "outstanding", "remarkable", "brilliant", "superb", "splendid", "awesome",
    "nice", "better", "improved", "gain", "benefit", "win", "boost", "strong",
])
NEGATIVE_WORDS = frozenset([
    "bad", "worst", "terrible", "hate", "awful", "negative", "failure",
    "fail", "sad", "ugly", "poor", "broken", "wrong", "error", "issue",
    "problem", "loss", "weak", "decline", "crash", "bug", "defect", "flaw",
    "worse", "difficult", "hard", "complicated", "frustrating", "annoying",
])


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _noise(amplitude: float = 0.05) -> float:
    return (random.random() * 2 - 1) * amplitude


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t)


def _phase_for_progress(progress: float) -> str:
    for i, (lo, hi) in enumerate(zip(PHASE_BOUNDARIES, PHASE_BOUNDARIES[1:])):
        if progress <= hi:
            return PHASES[i]
    return PHASES[-1]


def _phase_local_t(progress: float) -> float:
    """0..1 within the current phase."""
    for lo, hi in zip(PHASE_BOUNDARIES, PHASE_BOUNDARIES[1:]):
        if progress <= hi:
            if hi == lo:
                return 0.0
            return _clamp((progress - lo) / (hi - lo))
    return 1.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# ---------------------------------------------------------------------------
# Dataset parsing — CSV
# ---------------------------------------------------------------------------

class ColumnInfo:
    """Metadata for a single column."""
    __slots__ = [
        "name", "dtype", "n_unique", "missing_frac", "numeric_values",
        "mean", "std", "is_date", "cardinality_score",
    ]

    def __init__(self):
        self.name: str = ""
        self.dtype: str = "unknown"
        self.n_unique: int = 0
        self.missing_frac: float = 0.0
        self.numeric_values: List[float] = []
        self.mean: float = 0.0
        self.std: float = 0.0
        self.is_date: bool = False
        self.cardinality_score: float = 0.0  # 0=low, 1=high


_DATE_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    re.compile(r"\d{2}/\d{2}/\d{4}"),
    re.compile(r"\d{2}-\d{2}-\d{4}"),
    re.compile(r"\d{4}/\d{2}/\d{2}"),
]


def _looks_like_date(s: str) -> bool:
    return any(p.search(s) for p in _DATE_PATTERNS)


def _parse_csv_stdlib(content: str) -> Tuple[List[ColumnInfo], dict]:
    """Parse CSV using stdlib csv module. Returns (columns, meta)."""
    reader = csv.DictReader(io.StringIO(content))
    rows: List[dict] = []
    try:
        for row in reader:
            rows.append(row)
    except Exception:
        pass

    if not rows:
        return [], {"shape": [0, 0], "missing_frac": 0.0}

    fieldnames = list(rows[0].keys())
    n_rows = len(rows)
    n_cols = len(fieldnames)
    columns: List[ColumnInfo] = []
    total_missing = 0

    for col_name in fieldnames:
        col = ColumnInfo()
        col.name = col_name
        values = [r.get(col_name, "") for r in rows]
        missing = sum(1 for v in values if v == "" or v is None)
        col.missing_frac = missing / n_rows if n_rows else 0.0
        total_missing += missing

        non_missing = [v for v in values if v not in ("", None)]
        unique_vals = set(non_missing)
        col.n_unique = len(unique_vals)
        col.cardinality_score = _clamp(col.n_unique / max(n_rows, 1))

        # Try numeric
        numeric = []
        for v in non_missing:
            try:
                numeric.append(float(v.replace(",", "")))
            except (ValueError, AttributeError):
                pass

        if len(numeric) / max(len(non_missing), 1) > 0.8:
            col.dtype = "numeric"
            col.numeric_values = numeric
            col.mean = _mean(numeric)
            col.std = _std(numeric)
        else:
            # Check date
            sample = [v for v in non_missing[:20] if isinstance(v, str)]
            if sample and sum(1 for s in sample if _looks_like_date(s)) > len(sample) * 0.5:
                col.dtype = "date"
                col.is_date = True
            else:
                col.dtype = "categorical"

        columns.append(col)

    overall_missing = total_missing / max(n_rows * n_cols, 1)
    return columns, {"shape": [n_rows, n_cols], "missing_frac": overall_missing}


def _parse_csv_pandas(content: str) -> Tuple[List[ColumnInfo], dict]:
    """Parse CSV using pandas if available."""
    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception:
        return _parse_csv_stdlib(content)

    columns: List[ColumnInfo] = []
    n_rows, n_cols = df.shape

    for col_name in df.columns:
        col = ColumnInfo()
        col.name = str(col_name)
        series = df[col_name]
        col.missing_frac = float(series.isna().mean())
        col.n_unique = int(series.nunique())
        col.cardinality_score = _clamp(col.n_unique / max(n_rows, 1))

        dtype_str = str(series.dtype)
        if "int" in dtype_str or "float" in dtype_str:
            col.dtype = "numeric"
            nums = series.dropna().tolist()
            col.numeric_values = [float(v) for v in nums]
            col.mean = float(series.mean()) if col.numeric_values else 0.0
            col.std = float(series.std()) if len(col.numeric_values) > 1 else 0.0
        elif "datetime" in dtype_str:
            col.dtype = "date"
            col.is_date = True
        else:
            # Check if it looks like dates
            sample = series.dropna().astype(str).head(20).tolist()
            if sample and sum(1 for s in sample if _looks_like_date(s)) > len(sample) * 0.5:
                col.dtype = "date"
                col.is_date = True
            else:
                col.dtype = "categorical"

        columns.append(col)

    overall_missing = float(df.isna().mean().mean())
    return columns, {"shape": [n_rows, n_cols], "missing_frac": overall_missing}


def parse_csv(content_str: str) -> Tuple[List[ColumnInfo], dict]:
    if _HAS_PANDAS:
        return _parse_csv_pandas(content_str)
    return _parse_csv_stdlib(content_str)


# ---------------------------------------------------------------------------
# Dataset parsing — JSON
# ---------------------------------------------------------------------------

class JSONDatasetInfo:
    __slots__ = [
        "is_tabular", "n_records", "keys", "value_types",
        "max_nesting_depth", "array_lengths", "mixed_types",
        "complexity_score",
    ]

    def __init__(self):
        self.is_tabular: bool = False
        self.n_records: int = 0
        self.keys: List[str] = []
        self.value_types: Dict[str, str] = {}
        self.max_nesting_depth: int = 0
        self.array_lengths: List[int] = []
        self.mixed_types: bool = False
        self.complexity_score: float = 0.0


def _measure_depth(obj: Any, depth: int = 0) -> int:
    if isinstance(obj, dict):
        if not obj:
            return depth
        return max(_measure_depth(v, depth + 1) for v in obj.values())
    if isinstance(obj, list):
        if not obj:
            return depth
        return max(_measure_depth(item, depth + 1) for item in obj[:10])
    return depth


def _collect_array_lengths(obj: Any, acc: List[int]) -> None:
    if isinstance(obj, list):
        acc.append(len(obj))
        for item in obj[:5]:
            _collect_array_lengths(item, acc)
    elif isinstance(obj, dict):
        for v in list(obj.values())[:10]:
            _collect_array_lengths(v, acc)


def _infer_type(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    if v is None:
        return "null"
    return "unknown"


def parse_json(content_str: str) -> JSONDatasetInfo:
    info = JSONDatasetInfo()
    try:
        data = json.loads(content_str)
    except Exception:
        return info

    info.max_nesting_depth = _measure_depth(data)
    arr_lengths: List[int] = []
    _collect_array_lengths(data, arr_lengths)
    info.array_lengths = arr_lengths

    if isinstance(data, list) and data:
        # Check if list-of-dicts (tabular)
        sample = data[:5]
        if all(isinstance(r, dict) for r in sample):
            info.is_tabular = True
            info.n_records = len(data)
            # Gather keys from first record
            keys = list(sample[0].keys())
            info.keys = keys
            for k in keys:
                types = set(_infer_type(r.get(k)) for r in sample)
                info.value_types[k] = list(types)[0] if len(types) == 1 else "mixed"
            info.mixed_types = any(v == "mixed" for v in info.value_types.values())
        else:
            info.n_records = len(data)
    elif isinstance(data, dict):
        info.keys = list(data.keys())[:50]
        for k in info.keys[:10]:
            info.value_types[k] = _infer_type(data[k])

    # Complexity: depth + mixed types + array diversity
    depth_score = _clamp(info.max_nesting_depth / 8.0)
    mixed_score = 0.3 if info.mixed_types else 0.0
    array_score = _clamp(_mean(arr_lengths) / 1000.0) if arr_lengths else 0.0
    info.complexity_score = _clamp(depth_score * 0.5 + mixed_score + array_score * 0.2)

    return info


# ---------------------------------------------------------------------------
# Dataset parsing — TXT
# ---------------------------------------------------------------------------

class TextDatasetInfo:
    __slots__ = [
        "word_count", "sentence_count", "vocab_size", "avg_sentence_length",
        "top_words", "lexical_diversity", "has_code", "has_numbers",
        "positive_ratio", "negative_ratio", "emotional_valence",
        "complexity_score", "has_named_entities",
    ]

    def __init__(self):
        self.word_count: int = 0
        self.sentence_count: int = 0
        self.vocab_size: int = 0
        self.avg_sentence_length: float = 0.0
        self.top_words: List[Tuple[str, int]] = []
        self.lexical_diversity: float = 0.0
        self.has_code: bool = False
        self.has_numbers: bool = False
        self.positive_ratio: float = 0.0
        self.negative_ratio: float = 0.0
        self.emotional_valence: float = 0.5
        self.complexity_score: float = 0.0
        self.has_named_entities: bool = False


_CODE_PATTERNS = [
    re.compile(r"def\s+\w+\s*\("),
    re.compile(r"function\s+\w+\s*\("),
    re.compile(r"import\s+\w+"),
    re.compile(r"class\s+\w+[\s(:]"),
    re.compile(r"if\s+.+:\s*$", re.MULTILINE),
    re.compile(r"for\s+\w+\s+in\s+"),
    re.compile(r"=>\s*\{"),
    re.compile(r"console\.log"),
    re.compile(r"#include\s*<"),
    re.compile(r"SELECT\s+.+FROM", re.IGNORECASE),
]

_NAMED_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_SENTENCE_END = re.compile(r"[.!?]+\s+")


def parse_text(content_str: str) -> TextDatasetInfo:
    info = TextDatasetInfo()

    # Sentences
    sentences = _SENTENCE_END.split(content_str.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    info.sentence_count = max(len(sentences), 1)

    # Words (lowercase, strip punctuation)
    translator = str.maketrans("", "", string.punctuation)
    words = content_str.lower().translate(translator).split()
    info.word_count = len(words)
    info.avg_sentence_length = info.word_count / info.sentence_count

    # Vocabulary
    word_counts = Counter(words)
    info.vocab_size = len(word_counts)
    info.lexical_diversity = info.vocab_size / max(info.word_count, 1)

    # Top 20 words (excluding very short stop words)
    stop = {"the", "a", "an", "is", "in", "of", "and", "to", "it",
            "this", "that", "was", "for", "on", "are", "with", "as",
            "at", "be", "by", "or", "from", "not", "but", "its"}
    filtered = [(w, c) for w, c in word_counts.most_common(100) if w not in stop and len(w) > 2]
    info.top_words = filtered[:20]

    # Sentiment
    pos = sum(word_counts.get(w, 0) for w in POSITIVE_WORDS)
    neg = sum(word_counts.get(w, 0) for w in NEGATIVE_WORDS)
    total_sentiment = pos + neg
    if total_sentiment > 0:
        info.positive_ratio = pos / total_sentiment
        info.negative_ratio = neg / total_sentiment
        # 0=negative, 0.5=neutral, 1=positive
        info.emotional_valence = _clamp(0.5 + 0.5 * (pos - neg) / (total_sentiment + 1))
    else:
        info.emotional_valence = 0.5

    # Code detection
    info.has_code = any(p.search(content_str) for p in _CODE_PATTERNS)

    # Numbers
    info.has_numbers = bool(re.search(r"\d{3,}", content_str))

    # Named entities (simple heuristic: Title Case multi-word sequences)
    entities = _NAMED_ENTITY_PATTERN.findall(content_str)
    info.has_named_entities = len(entities) > 3

    # Complexity
    diversity_score = info.lexical_diversity
    length_score = _clamp(info.avg_sentence_length / 30.0)
    vocab_score = _clamp(info.vocab_size / 5000.0)
    info.complexity_score = _clamp(diversity_score * 0.4 + length_score * 0.3 + vocab_score * 0.3)

    return info


# ---------------------------------------------------------------------------
# Unified parsed dataset container
# ---------------------------------------------------------------------------

class ParsedDataset:
    """Holds all parsed information about the uploaded file."""

    def __init__(self):
        self.file_type: str = "unknown"   # csv | json | txt
        self.raw_size: int = 0

        # CSV fields
        self.csv_columns: List[ColumnInfo] = []
        self.csv_meta: dict = {}

        # JSON fields
        self.json_info: Optional[JSONDatasetInfo] = None

        # TXT fields
        self.txt_info: Optional[TextDatasetInfo] = None

        # Derived
        self.complexity_score: float = 0.5
        self.n_logical_units: int = 1   # columns, keys, or sentences
        self.key_findings: List[str] = []
        self.shape: Optional[List[int]] = None


def detect_file_type(filename: str, content_bytes: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith((".txt", ".md", ".rst", ".pdf")):
        return "txt"

    # Sniff content
    snippet = content_bytes[:512]
    try:
        text = snippet.decode("utf-8", errors="replace").strip()
    except Exception:
        return "txt"

    if text.startswith("{") or text.startswith("["):
        return "json"
    if "," in text and "\n" in text:
        return "csv"
    return "txt"


def parse_dataset(content_bytes: bytes, filename: str) -> ParsedDataset:
    ds = ParsedDataset()
    ds.raw_size = len(content_bytes)
    ds.file_type = detect_file_type(filename, content_bytes)

    try:
        content_str = content_bytes.decode("utf-8", errors="replace")
    except Exception:
        content_str = ""

    if ds.file_type == "csv":
        ds.csv_columns, ds.csv_meta = parse_csv(content_str)
        ds.n_logical_units = max(len(ds.csv_columns), 1)
        ds.shape = ds.csv_meta.get("shape", [0, 0])
        ds.complexity_score = _compute_csv_complexity(ds)
        ds.key_findings = _build_csv_findings(ds)

    elif ds.file_type == "json":
        ds.json_info = parse_json(content_str)
        ds.n_logical_units = max(len(ds.json_info.keys), 1)
        ds.complexity_score = ds.json_info.complexity_score
        ds.key_findings = _build_json_findings(ds.json_info)

    else:  # txt / unknown
        ds.file_type = "txt"
        ds.txt_info = parse_text(content_str)
        # Logical units = sentences (capped at 200 for frame planning)
        ds.n_logical_units = min(ds.txt_info.sentence_count, 200)
        ds.complexity_score = ds.txt_info.complexity_score
        ds.key_findings = _build_txt_findings(ds.txt_info)

    return ds


def _compute_csv_complexity(ds: ParsedDataset) -> float:
    cols = ds.csv_columns
    if not cols:
        return 0.3
    missing_score = ds.csv_meta.get("missing_frac", 0.0)
    n_numeric = sum(1 for c in cols if c.dtype == "numeric")
    n_cat = sum(1 for c in cols if c.dtype == "categorical")
    high_cardinality = sum(1 for c in cols if c.cardinality_score > 0.5)
    high_variance = sum(1 for c in cols if c.std > c.mean * 0.5 and c.mean != 0)

    col_count_score = _clamp(len(cols) / 50.0)
    row_count_score = _clamp(ds.shape[0] / 100000.0) if ds.shape else 0.0
    cardinality_score = _clamp(high_cardinality / max(len(cols), 1))
    variance_score = _clamp(high_variance / max(n_numeric, 1)) if n_numeric else 0.0

    return _clamp(
        col_count_score * 0.2
        + row_count_score * 0.2
        + cardinality_score * 0.25
        + variance_score * 0.15
        + missing_score * 0.1
        + (n_cat / max(len(cols), 1)) * 0.1
    )


def _build_csv_findings(ds: ParsedDataset) -> List[str]:
    findings = []
    cols = ds.csv_columns
    shape = ds.shape or [0, 0]
    findings.append(f"{shape[0]} rows × {shape[1]} columns detected")

    n_numeric = sum(1 for c in cols if c.dtype == "numeric")
    n_cat = sum(1 for c in cols if c.dtype == "categorical")
    n_date = sum(1 for c in cols if c.dtype == "date")
    if n_numeric:
        findings.append(f"{n_numeric} numeric column{'s' if n_numeric > 1 else ''}")
    if n_cat:
        findings.append(f"{n_cat} categorical column{'s' if n_cat > 1 else ''}")
    if n_date:
        findings.append(f"{n_date} date/time column{'s' if n_date > 1 else ''}")

    missing = ds.csv_meta.get("missing_frac", 0.0)
    if missing > 0.01:
        findings.append(f"{missing*100:.1f}% missing data overall")

    high_var = [c for c in cols if c.dtype == "numeric" and c.mean and c.std > c.mean * 0.5]
    if high_var:
        names = ", ".join(c.name for c in high_var[:3])
        findings.append(f"High variance in: {names}")

    high_card = [c for c in cols if c.cardinality_score > 0.5]
    if high_card:
        findings.append(f"High cardinality in {len(high_card)} column(s)")

    return findings


def _build_json_findings(info: JSONDatasetInfo) -> List[str]:
    findings = []
    if info.is_tabular:
        findings.append(f"Tabular JSON: {info.n_records} records, {len(info.keys)} fields")
    else:
        findings.append(f"Nested JSON object with {len(info.keys)} top-level keys")
    findings.append(f"Maximum nesting depth: {info.max_nesting_depth}")
    if info.mixed_types:
        findings.append("Mixed value types detected across fields")
    if info.array_lengths:
        avg_len = _mean(info.array_lengths)
        findings.append(f"Arrays present, avg length: {avg_len:.0f}")
    return findings


def _build_txt_findings(info: TextDatasetInfo) -> List[str]:
    findings = []
    findings.append(f"{info.word_count} words, {info.sentence_count} sentences")
    findings.append(f"Vocabulary size: {info.vocab_size} unique words")
    findings.append(f"Lexical diversity: {info.lexical_diversity:.2%}")
    if info.has_code:
        findings.append("Code-like patterns detected")
    if info.has_named_entities:
        findings.append("Named entities / proper nouns detected")
    top = [w for w, _ in info.top_words[:5]]
    if top:
        findings.append(f"Top terms: {', '.join(top)}")
    valence_label = "positive" if info.emotional_valence > 0.6 else ("negative" if info.emotional_valence < 0.4 else "neutral")
    findings.append(f"Emotional tone: {valence_label}")
    return findings


# ---------------------------------------------------------------------------
# Cognitive model: per-phase lobe activations
# ---------------------------------------------------------------------------

class LobeActivations:
    __slots__ = ["frontal", "parietal", "temporal", "occipital",
                 "limbic", "cerebellum", "brainstem"]

    def __init__(self, **kwargs):
        self.frontal = kwargs.get("frontal", 0.3)
        self.parietal = kwargs.get("parietal", 0.3)
        self.temporal = kwargs.get("temporal", 0.3)
        self.occipital = kwargs.get("occipital", 0.3)
        self.limbic = kwargs.get("limbic", 0.3)
        self.cerebellum = kwargs.get("cerebellum", 0.3)
        self.brainstem = kwargs.get("brainstem", 0.2)

    def as_dict(self) -> dict:
        return {
            "frontal": round(_clamp(self.frontal), 4),
            "parietal": round(_clamp(self.parietal), 4),
            "temporal": round(_clamp(self.temporal), 4),
            "occipital": round(_clamp(self.occipital), 4),
            "limbic": round(_clamp(self.limbic), 4),
            "cerebellum": round(_clamp(self.cerebellum), 4),
            "brainstem": round(_clamp(self.brainstem), 4),
        }

    def mean(self) -> float:
        return _mean([self.frontal, self.parietal, self.temporal,
                      self.occipital, self.limbic, self.cerebellum, self.brainstem])

    def copy(self) -> "LobeActivations":
        return LobeActivations(**self.as_dict())


def _phase_base_lobes(phase: str, t: float, ds: ParsedDataset) -> LobeActivations:
    """Return ideal lobe activations for a given phase and intra-phase time t."""

    la = LobeActivations()

    if phase == "encoding":
        # Sensory intake: high occipital + temporal; frontal ramps up
        la.occipital = _lerp(0.60, 0.85, t)
        la.temporal = _lerp(0.55, 0.80, t)
        la.frontal = _lerp(0.30, 0.65, t)
        la.parietal = _lerp(0.25, 0.45, t)
        la.limbic = _lerp(0.20, 0.40, t)
        la.cerebellum = _lerp(0.30, 0.45, t)
        la.brainstem = _lerp(0.25, 0.35, t)

    elif phase == "pattern_detection":
        # Peak parietal + frontal; gamma when patterns found
        la.parietal = _lerp(0.65, 0.95, t)
        la.frontal = _lerp(0.65, 0.90, t)
        la.temporal = _lerp(0.55, 0.75, t)
        la.occipital = _lerp(0.40, 0.55, t)
        la.limbic = _lerp(0.30, 0.50, t)
        la.cerebellum = _lerp(0.40, 0.55, t)
        la.brainstem = _lerp(0.25, 0.35, t)

    elif phase == "integration":
        # Temporal peaks; hippocampal/limbic; plasticity peaks
        la.temporal = _lerp(0.70, 0.95, t)
        la.limbic = _lerp(0.55, 0.85, t)
        la.frontal = _lerp(0.70, 0.80, t)
        la.parietal = _lerp(0.60, 0.75, t)
        la.occipital = _lerp(0.30, 0.40, t)
        la.cerebellum = _lerp(0.45, 0.55, t)
        la.brainstem = _lerp(0.25, 0.30, t)

    elif phase == "consolidation":
        # Global decreases; limbic stays active
        la.limbic = _lerp(0.65, 0.55, t)
        la.frontal = _lerp(0.55, 0.35, t)
        la.parietal = _lerp(0.50, 0.30, t)
        la.temporal = _lerp(0.55, 0.40, t)
        la.occipital = _lerp(0.25, 0.15, t)
        la.cerebellum = _lerp(0.40, 0.30, t)
        la.brainstem = _lerp(0.25, 0.20, t)

    elif phase == "insight":
        # Frontal + parietal flash; brief gamma; dopamine spike
        flash = math.sin(t * math.pi)  # peaks at midpoint
        la.frontal = _lerp(0.60, 0.95, flash)
        la.parietal = _lerp(0.55, 0.90, flash)
        la.temporal = _lerp(0.50, 0.75, flash)
        la.occipital = _lerp(0.20, 0.55, flash)
        la.limbic = _lerp(0.45, 0.70, flash)
        la.cerebellum = _lerp(0.35, 0.60, flash)
        la.brainstem = _lerp(0.20, 0.35, flash)

    return la


def _firing_pattern_for_phase(phase: str, t: float, dopamine: float) -> str:
    if phase == "encoding":
        return "theta"
    if phase == "pattern_detection":
        return "gamma" if dopamine > 0.65 else "burst"
    if phase == "integration":
        return "synchronized"
    if phase == "consolidation":
        return "tonic"
    if phase == "insight":
        return "gamma"
    return "tonic"


# ---------------------------------------------------------------------------
# Per-column / per-unit spike logic
# ---------------------------------------------------------------------------

class SpikeEvent:
    """A transient spike affecting certain lobes for a brief window."""

    def __init__(self, lobe: str, amplitude: float, duration_frames: int, frame_start: int):
        self.lobe = lobe
        self.amplitude = amplitude
        self.duration = duration_frames
        self.frame_start = frame_start

    def value_at(self, frame: int) -> float:
        rel = frame - self.frame_start
        if rel < 0 or rel >= self.duration:
            return 0.0
        # Smooth bell: sin(π * rel / duration)
        return self.amplitude * math.sin(math.pi * rel / self.duration)


def _plan_column_spikes(ds: ParsedDataset, total_frames: int) -> List[SpikeEvent]:
    """Distribute column-processing spikes over the encoding+pattern_detection phases."""
    spikes: List[SpikeEvent] = []
    n = ds.n_logical_units

    # Spikes happen during encoding (0..20%) and pattern_detection (20..45%)
    spike_start_frac = 0.05
    spike_end_frac = 0.45

    for i in range(n):
        t = i / max(n - 1, 1)  # 0..1 within the logical units
        frame_frac = spike_start_frac + t * (spike_end_frac - spike_start_frac)
        frame_idx = int(frame_frac * total_frames)

        if ds.file_type == "csv" and i < len(ds.csv_columns):
            col = ds.csv_columns[i]
            # Temporal spike for every column (reading)
            spikes.append(SpikeEvent("temporal", 0.20, 4, frame_idx))

            if col.is_date:
                spikes.append(SpikeEvent("temporal", 0.25, 5, frame_idx + 1))
                spikes.append(SpikeEvent("limbic", 0.15, 4, frame_idx + 1))

            if col.dtype == "numeric" and col.std > 0:
                # High variance → parietal spike
                var_factor = _clamp(col.std / max(abs(col.mean), 1e-9), 0.0, 2.0) / 2.0
                spikes.append(SpikeEvent("parietal", 0.15 + var_factor * 0.20, 5, frame_idx + 2))

            if col.dtype == "categorical" and col.cardinality_score > 0.3:
                spikes.append(SpikeEvent("temporal", 0.15, 4, frame_idx + 2))

            if col.missing_frac > 0.05:
                # Limbic stress
                spikes.append(SpikeEvent("limbic", 0.12, 3, frame_idx + 1))
                spikes.append(SpikeEvent("frontal", 0.10, 3, frame_idx + 2))

        elif ds.file_type == "json":
            spikes.append(SpikeEvent("temporal", 0.15, 3, frame_idx))
            if ds.json_info and ds.json_info.max_nesting_depth > 3:
                spikes.append(SpikeEvent("frontal", 0.20, 4, frame_idx + 1))
            if ds.json_info and ds.json_info.mixed_types:
                spikes.append(SpikeEvent("frontal", 0.12, 3, frame_idx))

        else:  # txt
            # Each logical unit = a sentence segment
            spikes.append(SpikeEvent("temporal", 0.10, 3, frame_idx))
            if ds.txt_info and ds.txt_info.has_code:
                spikes.append(SpikeEvent("frontal", 0.18, 4, frame_idx + 1))
            if ds.txt_info and ds.txt_info.has_named_entities:
                spikes.append(SpikeEvent("temporal", 0.15, 3, frame_idx + 2))

    return spikes


def _sum_spikes(spikes: List[SpikeEvent], lobe: str, frame: int) -> float:
    return sum(s.value_at(frame) for s in spikes if s.lobe == lobe)


# ---------------------------------------------------------------------------
# Thought labels per phase
# ---------------------------------------------------------------------------

_HUMAN_THOUGHTS: Dict[str, List[str]] = {
    "encoding": [
        "Reading file structure...",
        "Parsing initial tokens...",
        "Perceiving data layout...",
        "Recognizing column headers...",
        "Loading data into working memory...",
        "Scanning field types...",
        "Orienting to schema...",
        "Building mental model of structure...",
    ],
    "pattern_detection": [
        "Scanning for numeric patterns...",
        "Detecting distributions...",
        "Searching for correlations...",
        "Identifying outliers...",
        "Mapping value ranges...",
        "Recognizing categorical groupings...",
        "Computing statistical signatures...",
        "Flagging anomalies...",
        "Finding temporal trends...",
    ],
    "integration": [
        "Linking patterns to prior knowledge...",
        "Consolidating schema understanding...",
        "Binding column relationships...",
        "Associating semantics to values...",
        "Constructing holistic dataset model...",
        "Connecting disparate features...",
        "Semantic encoding in long-term memory...",
    ],
    "consolidation": [
        "Compressing key findings...",
        "Tagging emotionally significant patterns...",
        "Transferring to long-term memory...",
        "Pruning redundant representations...",
        "Stabilizing memory trace...",
        "Rehearsing important statistics...",
    ],
    "insight": [
        "Synthesizing all observations...",
        "Eureka: connecting hidden structures!",
        "Emerging high-level understanding...",
        "Formulating final conclusions...",
        "Crystallizing insight...",
    ],
}

_AI_THOUGHTS: Dict[str, List[str]] = {
    "encoding": [
        "Tokenizing input bytes...",
        "Embedding feature vectors...",
        "Attention: attending to header tokens...",
        "Building token index...",
        "Activating embedding layers...",
        "Encoding positional features...",
    ],
    "pattern_detection": [
        "Running convolution over numeric sequences...",
        "Attention: cross-field pattern heads active...",
        "Clustering feature distributions...",
        "Gradient-boosting anomaly signals...",
        "Computing pairwise correlations...",
        "Activating pattern-detection layers 8-19...",
        "Statistical model inference running...",
    ],
    "integration": [
        "Cross-attention: merging semantic layers...",
        "Late layers: integrating context...",
        "Attention entropy: broad focus...",
        "Constructing unified representation...",
        "Deep residual features propagating...",
        "Global context pooling active...",
    ],
    "consolidation": [
        "Compressing latent representation...",
        "Key-value cache stabilizing...",
        "Reducing gradient noise...",
        "Output logits converging...",
        "Final layer normalization...",
    ],
    "insight": [
        "Max confidence achieved on pattern...",
        "High-confidence output generation...",
        "Finalizing prediction vector...",
        "Synthesizing multi-head output...",
        "Generating summary representation...",
    ],
}


def _pick_thought(phase: str, thoughts_map: Dict[str, List[str]], frame: int) -> str:
    pool = thoughts_map.get(phase, ["Processing..."])
    return pool[frame % len(pool)]


# ---------------------------------------------------------------------------
# AI layer activations derived from human lobes
# ---------------------------------------------------------------------------

def _compute_layer_activations(la: LobeActivations, imitation_quality: float) -> List[float]:
    """32 transformer layer activations derived from human lobe activations."""
    layers = []
    for i in range(NUM_TRANSFORMER_LAYERS):
        t = i / (NUM_TRANSFORMER_LAYERS - 1)  # 0..1

        if i < 8:
            # Early layers: occipital + temporal
            base = _lerp(la.occipital, la.temporal, t / (8 / NUM_TRANSFORMER_LAYERS))
        elif i < 20:
            # Mid layers: parietal + frontal
            mid_t = (i - 8) / 11.0
            base = _lerp(la.parietal, la.frontal, mid_t)
        else:
            # Late layers: frontal + limbic
            late_t = (i - 20) / 11.0
            base = _lerp(la.frontal, la.limbic, late_t)

        noise = _noise(0.05 * (1 - imitation_quality))
        val = _clamp(base * imitation_quality + (0.5 + noise) * (1 - imitation_quality) + noise)
        layers.append(round(val, 4))

    return layers


# ---------------------------------------------------------------------------
# Dataset-specific per-frame insight text
# ---------------------------------------------------------------------------

def _current_finding_csv(ds: ParsedDataset, columns_processed: int) -> str:
    if not ds.csv_columns:
        return "Analyzing CSV structure..."
    if columns_processed < len(ds.csv_columns):
        col = ds.csv_columns[columns_processed]
        if col.dtype == "numeric":
            return f"Numeric column '{col.name}': mean={col.mean:.2f}, std={col.std:.2f}"
        if col.dtype == "categorical":
            return f"Categorical column '{col.name}': {col.n_unique} unique values"
        if col.is_date:
            return f"Date column '{col.name}' detected"
        return f"Analyzing column '{col.name}'..."
    return f"All {len(ds.csv_columns)} columns processed"


def _current_finding_json(ds: ParsedDataset, units_done: int) -> str:
    info = ds.json_info
    if not info:
        return "Parsing JSON..."
    if info.is_tabular:
        return f"Tabular JSON: processing record group {units_done}/{max(ds.n_logical_units, 1)}"
    if units_done < len(info.keys):
        k = info.keys[units_done]
        vtype = info.value_types.get(k, "?")
        return f"Key '{k}' ({vtype}) at depth {info.max_nesting_depth}"
    return f"JSON fully parsed: depth={info.max_nesting_depth}"


def _current_finding_txt(ds: ParsedDataset, units_done: int) -> str:
    info = ds.txt_info
    if not info:
        return "Reading text..."
    pct = units_done / max(ds.n_logical_units, 1)
    if pct < 0.25:
        return f"Scanning vocabulary... {info.vocab_size} unique words so far"
    if pct < 0.5:
        if info.has_code:
            return "Code-like patterns detected in text"
        return f"Average sentence length: {info.avg_sentence_length:.1f} words"
    if pct < 0.75:
        top = [w for w, _ in info.top_words[:3]]
        return f"Top terms: {', '.join(top)}"
    return f"Lexical diversity: {info.lexical_diversity:.2%}"


# ---------------------------------------------------------------------------
# Main FrameGenerator
# ---------------------------------------------------------------------------

class FrameGenerator:
    """Generates the sequence of cognitive signal frames for a parsed dataset."""

    def __init__(self, ds: ParsedDataset, total_frames: int = 300):
        self.ds = ds
        self.total_frames = total_frames
        self.spikes = _plan_column_spikes(ds, total_frames)
        self._patterns_found: List[str] = []
        self._insight_count: int = 0
        self._sync_moments: int = 0

        # Imitation quality: AI matches human at ~71% by default, varies with complexity
        self._base_imitation = _clamp(0.85 - ds.complexity_score * 0.30)

        # Pre-plan pattern discoveries
        self._patterns_found_set: List[Tuple[int, str]] = self._plan_patterns()

    def _plan_patterns(self) -> List[Tuple[int, str]]:
        """Pre-plan when patterns are 'discovered' and what label they get."""
        events: List[Tuple[int, str]] = []
        ds = self.ds

        if ds.file_type == "csv":
            # One event per high-variance or high-cardinality column
            for i, col in enumerate(ds.csv_columns):
                t = 0.20 + i / max(len(ds.csv_columns), 1) * 0.25
                frame = int(t * self.total_frames)
                if col.dtype == "numeric" and col.std > col.mean * 0.3:
                    events.append((frame, f"high variance in '{col.name}'"))
                if col.cardinality_score > 0.5:
                    events.append((frame, f"high cardinality in '{col.name}'"))
                if col.missing_frac > 0.1:
                    events.append((frame, f"missing data in '{col.name}' ({col.missing_frac:.0%})"))
                if col.is_date:
                    events.append((frame, f"temporal trend in '{col.name}'"))

        elif ds.file_type == "json":
            info = ds.json_info
            if info:
                if info.max_nesting_depth > 3:
                    events.append((int(0.25 * self.total_frames), f"deep nesting: {info.max_nesting_depth} levels"))
                if info.mixed_types:
                    events.append((int(0.30 * self.total_frames), "mixed value types across fields"))
                if info.array_lengths:
                    events.append((int(0.40 * self.total_frames), f"variable-length arrays (avg {_mean(info.array_lengths):.0f})"))

        else:  # txt
            info = ds.txt_info
            if info:
                events.append((int(0.22 * self.total_frames), f"vocabulary size: {info.vocab_size} unique words"))
                if info.has_code:
                    events.append((int(0.30 * self.total_frames), "code-like syntax patterns present"))
                if info.has_named_entities:
                    events.append((int(0.35 * self.total_frames), "named entities detected"))
                top = [w for w, _ in info.top_words[:3]]
                if top:
                    events.append((int(0.40 * self.total_frames), f"dominant terms: {', '.join(top)}"))
                events.append((int(0.45 * self.total_frames), f"lexical diversity: {info.lexical_diversity:.2%}"))

        return sorted(events, key=lambda x: x[0])

    def _get_patterns_up_to_frame(self, frame: int) -> List[str]:
        return [label for (f, label) in self._patterns_found_set if f <= frame]

    def _dopamine_at(self, frame: int, phase: str, t: float, lobes: LobeActivations) -> float:
        """Compute dopamine signal."""
        # Base: high at novelty start, pulses with pattern discoveries
        base = 0.40

        if phase == "encoding" and t < 0.3:
            base = _lerp(0.80, 0.50, t / 0.3)  # novelty spike at start

        # Pulse on pattern discovery events
        for (event_frame, _) in self._patterns_found_set:
            dist = abs(frame - event_frame)
            if dist <= 5:
                pulse = 0.30 * math.exp(-0.5 * (dist ** 2))
                base = _clamp(base + pulse)

        if phase == "insight":
            base = _clamp(base + 0.30 * math.sin(t * math.pi))

        return _clamp(base + _noise(0.05))

    def generate_frame(self, frame: int) -> dict:
        progress = frame / max(self.total_frames - 1, 1)
        phase = _phase_for_progress(progress)
        phase_t = _phase_local_t(progress)

        # --- Fatigue ---
        fatigue = _clamp(progress * 0.25)

        # --- Base lobe activations ---
        la = _phase_base_lobes(phase, phase_t, self.ds)

        # --- Apply spike overlays ---
        la.temporal = _clamp(la.temporal + _sum_spikes(self.spikes, "temporal", frame))
        la.frontal = _clamp(la.frontal + _sum_spikes(self.spikes, "frontal", frame))
        la.parietal = _clamp(la.parietal + _sum_spikes(self.spikes, "parietal", frame))
        la.limbic = _clamp(la.limbic + _sum_spikes(self.spikes, "limbic", frame))
        la.occipital = _clamp(la.occipital + _sum_spikes(self.spikes, "occipital", frame))

        # --- Organic noise ---
        la.frontal = _clamp(la.frontal + _noise(0.05))
        la.parietal = _clamp(la.parietal + _noise(0.05))
        la.temporal = _clamp(la.temporal + _noise(0.05))
        la.occipital = _clamp(la.occipital + _noise(0.05))
        la.limbic = _clamp(la.limbic + _noise(0.05))
        la.cerebellum = _clamp(la.cerebellum + _noise(0.04))
        la.brainstem = _clamp(la.brainstem + _noise(0.03))

        # --- Dataset-specific modifiers ---
        la = self._apply_dataset_modifiers(la, phase, progress)

        global_activity = la.mean()
        dopamine = self._dopamine_at(frame, phase, phase_t, la)

        # Attention, memory, plasticity, cognitive load
        attention_focus = _clamp(la.frontal * 0.6 + la.parietal * 0.4 + _noise(0.04))
        memory_load = _clamp(
            la.frontal * 0.45 + la.temporal * 0.45 + la.limbic * 0.10 + _noise(0.04)
        )
        plasticity = _clamp(
            (0.3 if phase not in ("integration",) else 0.75)
            + (0.20 if phase == "pattern_detection" else 0.0)
            + _noise(0.05)
        )
        cognitive_load = _clamp(global_activity * 0.7 + (1 - fatigue) * 0.3 + _noise(0.04))
        novelty = _clamp(dopamine * 0.7 + _noise(0.08))
        emotional_valence = self._emotional_valence(phase, phase_t, dopamine)
        arousal = _clamp(global_activity * 0.8 + dopamine * 0.2 + _noise(0.04))
        pulse = _clamp(0.35 + arousal * 0.40 + math.sin(frame * 0.4) * 0.08 + _noise(0.03))
        firing_pattern = _firing_pattern_for_phase(phase, phase_t, dopamine)
        thought_label = _pick_thought(phase, _HUMAN_THOUGHTS, frame)

        human_block = {
            "lobe_activations": la.as_dict(),
            "global_activity": round(global_activity, 4),
            "pulse": round(pulse, 4),
            "attention_focus": round(attention_focus, 4),
            "plasticity": round(plasticity, 4),
            "dopamine": round(dopamine, 4),
            "memory_load": round(memory_load, 4),
            "cognitive_load": round(cognitive_load, 4),
            "novelty": round(novelty, 4),
            "fatigue": round(fatigue, 4),
            "emotional_valence": round(emotional_valence, 4),
            "arousal": round(arousal, 4),
            "thought_label": thought_label,
            "firing_pattern": firing_pattern,
        }

        # --- AI Brain ---
        imitation_quality = self._imitation_quality(phase, phase_t)
        ai_la = self._compute_ai_lobes(la, phase, imitation_quality)
        layer_activations = _compute_layer_activations(ai_la, imitation_quality)
        attention_entropy = _clamp(
            0.4 + (0.4 if phase == "encoding" else 0.0)
            + (0.35 if phase in ("integration", "consolidation") else 0.0)
            + _noise(0.06)
        )
        ai_confidence = _clamp(1.0 - novelty * 0.5 + _noise(0.05))
        loss_estimate = _clamp(1.0 - ai_confidence + _noise(0.04))
        gradient_magnitude = _clamp(
            (0.75 if phase == "pattern_detection" else 0.40)
            + _noise(0.06)
        )
        ai_processing_speed = self._ai_processing_speed(phase, phase_t)
        ai_thought_label = _pick_thought(phase, _AI_THOUGHTS, frame)
        ai_firing = "tonic" if phase in ("consolidation",) else ("gamma" if phase == "pattern_detection" else "burst")

        prediction_error = _clamp(1.0 - imitation_quality + _noise(0.06))
        working_memory_ai = _clamp(memory_load * 0.85 + _noise(0.05))
        pattern_match = _clamp(imitation_quality * 0.8 + novelty * 0.1 + _noise(0.05))
        creativity_index = _clamp(
            (0.2 if phase not in ("insight",) else 0.65)
            + ds_complexity_bonus(self.ds) * 0.2
            + _noise(0.06)
        )

        ai_block = {
            "layer_activations": layer_activations,
            "attention_entropy": round(attention_entropy, 4),
            "confidence": round(ai_confidence, 4),
            "loss_estimate": round(loss_estimate, 4),
            "gradient_magnitude": round(gradient_magnitude, 4),
            "processing_speed": round(ai_processing_speed, 4),
            "imitation_score": round(imitation_quality, 4),
            "prediction_error": round(prediction_error, 4),
            "working_memory": round(working_memory_ai, 4),
            "pattern_match": round(pattern_match, 4),
            "creativity_index": round(creativity_index, 4),
            "thought_label": ai_thought_label,
            "firing_pattern": ai_firing,
        }

        # --- Comparison ---
        sync_score = _clamp(imitation_quality * 0.7 + (1 - abs(global_activity - _mean(layer_activations))) * 0.3 + _noise(0.04))
        ai_global = _mean(layer_activations)
        human_ahead = global_activity >= ai_global
        race_delta = _clamp(abs(global_activity - ai_global))

        # Divergence/convergence regions based on phase
        divergence_regions, convergence_regions = self._region_divergence(phase)

        insight_moment = (phase == "insight" and phase_t > 0.3 and phase_t < 0.7
                          and global_activity > 0.75 and ai_global > 0.65)
        if insight_moment:
            self._insight_count += 1
        if sync_score > 0.80:
            self._sync_moments += 1

        comparison_block = {
            "sync_score": round(sync_score, 4),
            "human_ahead": human_ahead,
            "divergence_regions": divergence_regions,
            "convergence_regions": convergence_regions,
            "race_delta": round(race_delta, 4),
            "insight_moment": insight_moment,
            "imitation_quality": round(imitation_quality, 4),
        }

        # --- Dataset insight ---
        ds = self.ds
        units_done = int(progress * ds.n_logical_units)
        current_finding = self._current_finding(units_done, frame)
        patterns_so_far = self._get_patterns_up_to_frame(frame)

        n_cols_total = len(ds.csv_columns) if ds.file_type == "csv" else ds.n_logical_units
        n_cols_processed = min(units_done, n_cols_total)

        dataset_insight_block = {
            "current_finding": current_finding,
            "columns_processed": n_cols_processed,
            "total_columns": n_cols_total,
            "patterns_found": patterns_so_far[-6:],  # last 6 patterns
            "complexity_score": round(ds.complexity_score, 4),
        }

        return {
            "type": "cognitive_frame",
            "frame": frame,
            "total_frames": self.total_frames,
            "progress": round(progress, 6),
            "phase": phase,
            "human": human_block,
            "ai": ai_block,
            "comparison": comparison_block,
            "dataset_insight": dataset_insight_block,
        }

    def _apply_dataset_modifiers(self, la: LobeActivations, phase: str, progress: float) -> LobeActivations:
        ds = self.ds
        if ds.file_type == "csv":
            missing = ds.csv_meta.get("missing_frac", 0.0)
            if missing > 0.1:
                la.limbic = _clamp(la.limbic + missing * 0.3)
                la.frontal = _clamp(la.frontal + missing * 0.15)

        elif ds.file_type == "json" and ds.json_info:
            if ds.json_info.max_nesting_depth > 4:
                la.frontal = _clamp(la.frontal + 0.15)
            if ds.json_info.mixed_types:
                la.frontal = _clamp(la.frontal + 0.10)
                la.temporal = _clamp(la.temporal + 0.08)

        elif ds.file_type == "txt" and ds.txt_info:
            info = ds.txt_info
            if info.has_code:
                la.frontal = _clamp(la.frontal + 0.15)
                la.parietal = _clamp(la.parietal + 0.10)
            # Emotional valence → limbic
            limbic_boost = abs(info.emotional_valence - 0.5) * 0.40
            la.limbic = _clamp(la.limbic + limbic_boost)
            # High diversity → temporal
            la.temporal = _clamp(la.temporal + info.lexical_diversity * 0.25)

        return la

    def _emotional_valence(self, phase: str, t: float, dopamine: float) -> float:
        ds = self.ds
        base = 0.5
        if ds.file_type == "txt" and ds.txt_info:
            base = ds.txt_info.emotional_valence
        # Dopamine boosts positive valence
        return _clamp(base + (dopamine - 0.5) * 0.20 + _noise(0.04))

    def _imitation_quality(self, phase: str, t: float) -> float:
        """AI imitation quality varies by phase."""
        base = self._base_imitation
        if phase == "encoding":
            # AI is good at encoding
            return _clamp(base + 0.08 + _noise(0.03))
        if phase == "pattern_detection":
            return _clamp(base + 0.05 + _noise(0.04))
        if phase == "integration":
            # AI lags on semantic integration
            return _clamp(base - 0.10 + _noise(0.04))
        if phase == "consolidation":
            return _clamp(base + 0.05 + _noise(0.03))
        if phase == "insight":
            # AI struggles with true insight
            return _clamp(base - 0.15 + _noise(0.05))
        return _clamp(base + _noise(0.04))

    def _compute_ai_lobes(self, human_la: LobeActivations, phase: str, iq: float) -> LobeActivations:
        """AI lobe activations derived from human + imitation noise."""
        ai = LobeActivations()
        # AI diverges on emotional/creative aspects
        ai.frontal = _clamp(human_la.frontal * iq + (1 - iq) * random.random() * 0.6 + _noise(0.04))
        ai.parietal = _clamp(human_la.parietal * iq + (1 - iq) * random.random() * 0.6 + _noise(0.04))
        ai.temporal = _clamp(human_la.temporal * (iq * 0.95) + (1 - iq) * random.random() * 0.5 + _noise(0.04))
        ai.occipital = _clamp(human_la.occipital * iq + (1 - iq) * random.random() * 0.5 + _noise(0.03))
        # AI substantially weaker on limbic (emotion)
        ai.limbic = _clamp(human_la.limbic * (iq * 0.60) + (1 - iq) * random.random() * 0.3 + _noise(0.04))
        ai.cerebellum = _clamp(human_la.cerebellum * iq + (1 - iq) * random.random() * 0.5 + _noise(0.03))
        ai.brainstem = _clamp(human_la.brainstem * iq + _noise(0.02))
        return ai

    def _ai_processing_speed(self, phase: str, t: float) -> float:
        """AI processing speed relative to human baseline."""
        if phase == "encoding":
            return _clamp(1.0 + 0.30 + _noise(0.04))  # 30% faster
        if phase == "integration":
            return _clamp(1.0 - 0.20 + _noise(0.04))  # 20% slower
        if phase == "consolidation":
            return _clamp(1.0 + 0.40 + _noise(0.04))  # 40% faster
        return _clamp(1.0 + _noise(0.05))

    def _region_divergence(self, phase: str) -> Tuple[List[str], List[str]]:
        ds_type = self.ds.file_type
        if phase == "encoding":
            return ["limbic", "frontal"], ["occipital", "temporal"]
        if phase == "pattern_detection":
            return ["temporal", "limbic"], ["parietal", "frontal"]
        if phase == "integration":
            return ["limbic", "temporal"], ["parietal"]
        if phase == "consolidation":
            return ["limbic", "occipital"], ["frontal", "temporal"]
        if phase == "insight":
            return ["limbic", "temporal"], ["frontal", "parietal"]
        return ["limbic"], ["parietal"]

    def _current_finding(self, units_done: int, frame: int) -> str:
        ds = self.ds
        if ds.file_type == "csv":
            return _current_finding_csv(ds, units_done)
        if ds.file_type == "json":
            return _current_finding_json(ds, units_done)
        return _current_finding_txt(ds, units_done)

    def generate_summary_frame(self) -> dict:
        ds = self.ds
        total_iq = self._base_imitation
        insight_c = max(self._insight_count, 1 if (self.total_frames > 100) else 0)

        # Peak lobe
        lobe_names = ["frontal", "parietal", "temporal", "occipital", "limbic", "cerebellum", "brainstem"]
        # Check which phase had highest activation — integration peaks temporal, pattern_detection peaks parietal
        if ds.complexity_score > 0.6:
            peak_lobe = "parietal"
        else:
            peak_lobe = "temporal"

        # Peak AI layer: mid layers for numerical data, late for text
        if ds.file_type == "csv":
            peak_layer = 14
        elif ds.file_type == "json":
            peak_layer = 11
        else:
            peak_layer = 24

        dominant_phase = "pattern_detection" if ds.complexity_score > 0.5 else "encoding"
        cognitive_cost = _clamp(ds.complexity_score * 0.7 + total_iq * 0.15 + 0.10)

        recommendation = (
            f"Dataset is {'highly' if ds.complexity_score > 0.7 else 'moderately'} complex. "
            f"AI imitation achieved {total_iq*100:.0f}% fidelity."
        )

        summary = {
            "dataset_type": ds.file_type,
            "shape": ds.shape or [0, 0],
            "key_findings": ds.key_findings[:6],
            "human_peak_lobe": peak_lobe,
            "ai_peak_layer": peak_layer,
            "total_imitation_score": round(total_iq, 4),
            "sync_moments": self._sync_moments,
            "insight_moments": insight_c,
            "complexity_score": round(ds.complexity_score, 4),
            "cognitive_cost": round(cognitive_cost, 4),
            "dominant_phase": dominant_phase,
            "recommendation": recommendation,
        }

        return {"type": "analysis_complete", "summary": summary}


def ds_complexity_bonus(ds: ParsedDataset) -> float:
    return ds.complexity_score * 0.5


# ---------------------------------------------------------------------------
# Frame timing: variable interval for dramatic pacing
# ---------------------------------------------------------------------------

def _frame_delay(frame: int, total_frames: int, base_delay: float = 0.05) -> float:
    """
    Vary the inter-frame delay to create pacing:
    - Encoding (0-20%): slightly slower (0.06s)
    - Pattern detection (20-45%): fast (0.04s)
    - Integration (45-70%): medium (0.05s)
    - Consolidation (70-90%): slower (0.07s)
    - Insight (90-100%): fast then slow (0.04..0.08s)
    """
    progress = frame / max(total_frames - 1, 1)
    phase = _phase_for_progress(progress)

    if phase == "encoding":
        return 0.060
    if phase == "pattern_detection":
        return 0.042
    if phase == "integration":
        return 0.052
    if phase == "consolidation":
        return 0.068
    if phase == "insight":
        t = _phase_local_t(progress)
        return 0.040 + t * 0.040
    return base_delay


# ---------------------------------------------------------------------------
# Frame count calculation
# ---------------------------------------------------------------------------

def _compute_frame_count(ds: ParsedDataset) -> int:
    """
    Aim for 200-600 frames. Scale with dataset complexity and size.
    """
    base = 300
    complexity_bonus = int(ds.complexity_score * 200)
    size_bonus = int(_clamp(ds.raw_size / 1_000_000) * 100)  # up to 100 extra for 1MB+
    total = base + complexity_bonus + size_bonus
    return max(200, min(600, total))


# ---------------------------------------------------------------------------
# Public DatasetAnalyzer class
# ---------------------------------------------------------------------------

class DatasetAnalyzer:
    """
    Analyzes uploaded datasets and streams cognitive signal frames.

    Usage::

        analyzer = DatasetAnalyzer()
        await analyzer.analyze(content_bytes, "data.csv", ws.send)
        summary = analyzer.get_summary()
    """

    def __init__(self):
        self._summary: Optional[dict] = None
        self._parsed: Optional[ParsedDataset] = None

    async def analyze(
        self,
        content: bytes,
        filename: str,
        ws_send: Callable[[str], Coroutine],
    ) -> None:
        """
        Parse the file, then stream cognitive signal frames to ws_send.

        Parameters
        ----------
        content : bytes
            Raw file content.
        filename : str
            Original filename (used to detect type).
        ws_send : async callable
            An async function that accepts a JSON string; each frame is sent as:
            ``await ws_send(json.dumps(frame))``
        """
        # 1. Parse dataset
        ds = parse_dataset(content, filename)
        self._parsed = ds

        # 2. Determine frame count
        total_frames = _compute_frame_count(ds)

        # 3. Create frame generator
        gen = FrameGenerator(ds, total_frames)

        # 4. Stream frames
        for frame_idx in range(total_frames):
            frame_data = gen.generate_frame(frame_idx)
            await ws_send(json.dumps(frame_data))
            delay = _frame_delay(frame_idx, total_frames)
            await asyncio.sleep(delay)

        # 5. Send final summary frame
        summary_frame = gen.generate_summary_frame()
        await ws_send(json.dumps(summary_frame))
        self._summary = summary_frame

    def get_summary(self) -> dict:
        """Return the completed analysis summary (after analyze() finishes)."""
        if self._summary is None:
            return {"error": "analyze() has not been called yet"}
        return self._summary


# ---------------------------------------------------------------------------
# Standalone test / demo
# ---------------------------------------------------------------------------

async def _demo():
    """Run a quick demo to verify the module works."""
    import sys

    # Build a small CSV sample
    csv_sample = (
        "id,name,age,salary,department,hire_date\n"
        + "\n".join(
            f"{i},Employee_{i},{20+i%40},{30000+i*500},{'Eng' if i%3==0 else 'Sales'},2020-{(i%12)+1:02d}-01"
            for i in range(50)
        )
    )
    content = csv_sample.encode()
    filename = "employees.csv"

    frames_received = []
    final_summary = {}

    async def fake_ws_send(msg: str):
        data = json.loads(msg)
        if data["type"] == "cognitive_frame":
            frames_received.append(data["frame"])
            if data["frame"] % 50 == 0:
                pct = data["progress"] * 100
                phase = data["phase"]
                ga = data["human"]["global_activity"]
                tl = data["human"]["thought_label"]
                print(f"  Frame {data['frame']:3d}/{data['total_frames']} ({pct:5.1f}%) [{phase:20s}] "
                      f"global={ga:.2f} | {tl}")
        else:
            print(f"\nFinal summary: {json.dumps(data['summary'], indent=2)}")
            final_summary.update(data)

    print("=" * 70)
    print("DatasetAnalyzer demo — CSV file")
    print("=" * 70)
    t0 = time.time()
    analyzer = DatasetAnalyzer()
    await analyzer.analyze(content, filename, fake_ws_send)
    elapsed = time.time() - t0
    print(f"\nTotal frames: {len(frames_received)}, elapsed: {elapsed:.1f}s")
    summary = analyzer.get_summary()
    print(f"Summary type: {summary['summary']['dataset_type']}")
    print(f"Complexity:   {summary['summary']['complexity_score']}")
    print(f"Imitation:    {summary['summary']['total_imitation_score']}")

    # Test TXT
    print("\n" + "=" * 70)
    print("DatasetAnalyzer demo — TXT file")
    print("=" * 70)
    txt = (
        "The quick brown fox jumps over the lazy dog. "
        "Natural language processing is a fascinating field of artificial intelligence. "
        "Dr. Alan Turing proposed the famous imitation game in 1950. "
        "Researchers have made tremendous progress in deep learning, particularly with transformer architectures. "
        "The BERT model introduced bidirectional attention, while GPT pioneered causal language modeling. "
        "import numpy as np\nfor i in range(100):\n    print(i)\n"
        "Positive outcomes are expected from the successful implementation of these techniques."
    ) * 5

    txt_frames = []

    async def txt_ws(msg: str):
        data = json.loads(msg)
        if data["type"] == "cognitive_frame":
            txt_frames.append(data["frame"])

    await DatasetAnalyzer().analyze(txt.encode(), "notes.txt", txt_ws)
    print(f"TXT frames streamed: {len(txt_frames)}")


if __name__ == "__main__":
    asyncio.run(_demo())
