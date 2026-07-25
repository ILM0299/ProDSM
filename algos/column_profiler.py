"""
column_profiler.py
==================
Pre-compute per-column statistical profiles for source CSV files and serialize
the results to disk as a cache.

Design Motivation
-----------------
The source database consists of multiple CSV tables, each potentially containing
millions of rows. Scanning raw CSV files directly during every schema matching
pass would result in a computational complexity of
O(N * target_cols * source_cols) across four similarity signals
(value domain, type, distribution, semantics), which is entirely unacceptable.

Core idea: **one O(N) scan, permanent O(1) queries**.
Each table is scanned once in a streaming fashion, compressing all statistics
required by the signals into a lightweight dictionary (Profile), which is then
serialized to disk. Subsequent matching stages read only the Profiles and never
touch the raw CSV files again.

Per-Column Profile Fields
-------------------------
dtype          : Semantic data type, 'int' | 'float' | 'str' | 'bool' | 'mixed'
n_rows         : Total number of rows scanned for this column (including nulls)
n_unique       : Number of unique values (exact count or HLL estimate)
null_rate      : Null rate in [0, 1]
num_min        : Minimum value for numeric columns (None for non-numeric)
num_max        : Maximum value for numeric columns
num_p25        : 25th percentile
num_p50        : Median
num_p75        : 75th percentile
str_avg_len    : Average string length for string columns (None for non-string)
str_char_class : Character set type, 'alpha' | 'digit' | 'alnum' | 'mixed' | None
unique_vals    : List of unique values (complete set when cardinality <= UNIQUE_VALS_CAP,
                 otherwise a random sample)
exact_unique   : True if unique_vals is the complete exact set; False if approximate sample
hll_cardinality: HyperLogLog estimate when cardinality exceeds the threshold, otherwise None
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────

# Upper bound for storing exact unique values.
# Low-cardinality columns (enums, status codes, gender, etc.) typically have
# far fewer unique values than this and can be stored in full.
# High-cardinality columns (IDs, timestamps, etc.) switch to HLL + random
# sampling mode once this limit is exceeded.
UNIQUE_VALS_CAP  = 500

# Number of random samples to retain after exceeding UNIQUE_VALS_CAP.
# Used for edit-distance computation in semantic similarity (Signal 4);
# 200 samples are sufficient to cover the major value domain patterns.
SAMPLE_VALS_SIZE = 200

# Regular expression for identifying numeric strings.
# Supports integers, decimals, and scientific notation, with an optional
# leading minus sign.
NUM_PATTERN = re.compile(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?$')

# Regular expression to distinguish pure integers from floats (no decimal
# point or scientific notation).
INT_PATTERN = re.compile(r'^-?\d+$')


# ─────────────────────────────────────────────────────────────
# Lightweight HyperLogLog cardinality estimation (no external dependencies)
# ─────────────────────────────────────────────────────────────

class _HLL:
    """
    HyperLogLog cardinality estimator with approximately +/-2% accuracy.

    Algorithm Overview
    ------------------
    Hash each input value into the [0, 2^64) space. Use the lowest b bits as
    the bucket index (M = 2^b buckets in total), and estimate the maximum
    "luck" observed per bucket using the number of leading zeros (rho) among
    the remaining bits. The final cardinality is estimated via the harmonic
    mean formula.

    Parameter Choice
    ----------------
    b = 12, yielding M = 4096 buckets.
    Memory footprint: 4096 * 1 byte = 4 KB per column (numpy uint8 array).
    Standard error is approximately 1.04 / sqrt(M) ~ 1.6%.

    When to Use
    -----------
    Activated only when a column's unique value count exceeds UNIQUE_VALS_CAP
    (default 500). Low-cardinality columns use the exact set directly and do
    not require HLL.
    """

    _B = 12          # Number of bits for bucket index
    _M = 1 << _B    # Number of buckets = 4096

    def __init__(self):
        # Each bucket stores the maximum rho (number of leading zeros + 1) observed
        self._regs = np.zeros(self._M, dtype=np.uint8)

    def add(self, val: str):
        """Add a string value to the HLL sketch."""
        # Use MD5 to obtain a 128-bit hash, then convert to integer
        h   = int(hashlib.md5(val.encode(), usedforsecurity=False).hexdigest(), 16)
        # Use the lowest _B bits as the bucket index
        idx = h & (self._M - 1)
        # Right-shift by _B bits, then compute "leading zeros + 1" (i.e., rho) of the remainder
        w   = h >> self._B
        rho = min(64, (w == 0) and 64 or (65 - w.bit_length()))
        # Update only if rho is larger (core HLL update rule)
        if rho > self._regs[idx]:
            self._regs[idx] = rho

    def count(self) -> int:
        """
        Estimate the number of distinct values seen so far.

        Uses the HyperLogLog++ corrected formula:
          raw = alpha * M^2 * Z
        where Z = (sum of 2^(-reg[i]))^(-1) is the inverse of the harmonic mean.

        Applies LinearCounting correction for small cardinalities to improve
        accuracy at the low end.
        """
        # alpha_M is the bias correction coefficient for M=4096
        alpha = 0.7213 / (1 + 1.079 / self._M)
        z     = 1.0 / np.sum(2.0 ** (-self._regs.astype(float)))
        raw   = alpha * self._M * self._M * z

        # LinearCounting small-range correction: more accurate when estimate <= 2.5M and empty buckets exist
        if raw <= 2.5 * self._M:
            zeros = int(np.sum(self._regs == 0))
            if zeros:
                raw = self._M * math.log(self._M / zeros)

        return max(1, int(raw))


# ─────────────────────────────────────────────────────────────
# Type inference utility functions
# ─────────────────────────────────────────────────────────────

def _detect_dtype(sample_vals: list[str]) -> str:
    """
    Infer the semantic data type from a sample of raw CSV strings.

    Why This Function Is Needed
    ---------------------------
    All values in a CSV file are stored as strings, and Python's csv module
    does not perform type conversion. While pandas offers dtype inference,
    it is unavailable in a streaming, row-by-row reading mode. This function
    performs batch inference on the first 1000 non-null samples collected,
    avoiding the overhead of per-row inference.

    Inference Rules (by priority)
    ------------------------------
    1. All values match the pure integer pattern (-?\\d+)            -> 'int'
    2. All values match the numeric pattern (incl. decimals/sci notation) -> 'float'
    3. All values belong to the boolean vocabulary (true/false/yes/no, etc.) -> 'bool'
    4. Otherwise                                                    -> 'str'
    5. If multiple types are present (e.g., some numeric, some string), returns 'mixed'

    Parameters
    ----------
    sample_vals : List of non-null raw strings (up to 1000 entries)

    Returns
    -------
    'int' | 'float' | 'bool' | 'str' | 'mixed'
    """
    if not sample_vals:
        return "str"

    all_int = True     # Whether all values are pure integer strings
    all_num = True     # Whether all values can be parsed as numeric
    all_bool = True    # Whether all values belong to the boolean vocabulary
    has_non_num = False# Whether any non-numeric, non-boolean string has appeared

    # Accepted boolean value representations, case-insensitive
    bool_vals = {"true", "false", "yes", "no", "0", "1", "t", "f", "y", "n"}

    for v in sample_vals:
        vl = v.strip()
        vl_lower = vl.lower()

        # Check if it is a valid pure integer string
        if not INT_PATTERN.match(vl):
            all_int = False
        # Check if it is a valid numeric string
        if not NUM_PATTERN.match(vl):
            all_num = False
        # Check if it is a boolean token
        if vl_lower not in bool_vals:
            all_bool = False

        # Flag whether a value that is neither numeric nor boolean (plain string) has appeared
        is_num = NUM_PATTERN.match(vl) is not None
        is_bool = vl_lower in bool_vals
        if not is_num and not is_bool:
            has_non_num = True

    # ===================== Rule 5: Mixed type detection =====================
    # If both numeric/boolean and plain string values are present -> mixed type
    has_mixed = (all_num or all_bool) and has_non_num

    # Apply rules by priority
    if has_mixed:
        return "mixed"
    if all_int:
        return "int"
    if all_num:
        return "float"
    if all_bool:
        return "bool"
    return "str"


def _char_class(vals: list[str]) -> str:
    """
    Determine the character set type of a string column, used in Signal 3
    (distribution similarity) for the string branch.

    Classification Logic
    --------------------
    'alpha' : All values contain only letters (and spaces/hyphens), e.g., names, cities
    'digit' : All values contain only digits, e.g., postal codes, numeric IDs
    'alnum' : All values contain only letters and digits, e.g., product codes like "ABC123"
    'mixed' : Contains punctuation, special characters, etc., e.g., phone numbers, addresses

    Parameters
    ----------
    vals : List of string samples (typically the first 500 non-null values)

    Returns
    -------
    'alpha' | 'digit' | 'alnum' | 'mixed'
    """
    if not vals:
        return "mixed"
    # After ignoring spaces and hyphens, check if all values are purely alphabetic
    # (relaxed alpha definition to accommodate "New York", "Sao Paulo")
    all_alpha = all(v.replace(" ", "").replace("-", "").isalpha() for v in vals if v)
    all_digit = all(v.isdigit() for v in vals if v)
    all_alnum = all(v.replace(" ", "").isalnum() for v in vals if v)
    if all_alpha:
        return "alpha"
    if all_digit:
        return "digit"
    if all_alnum:
        return "alnum"
    return "mixed"


# ─────────────────────────────────────────────────────────────
# Per-column incremental aggregator
# ─────────────────────────────────────────────────────────────

class _ColAgg:
    """
    Streaming single-column statistical aggregator.

    Design Goal
    -----------
    Perform a single linear scan over the CSV file, collecting all Profile
    statistics for a column with O(1) extra memory (not growing with the
    number of rows).

    Internal State Overview
    -----------------------
    Unique value tracking uses a "two-phase" strategy:
      - Phase 1 (exact mode): All unique values are stored in a set until
        UNIQUE_VALS_CAP is exceeded.
      - Phase 2 (approximate mode): After exceeding the limit, switches to
        an HLL sketch + Reservoir random sampling.
        At the transition point, the existing exact set is migrated to HLL.

    Numeric quantiles are likewise estimated via Reservoir Sampling
    (capacity 10,000), yielding quantile errors typically < 1% for
    million-row files.

    Parameters
    ----------
    col_name : Column name, used as the identifier in the final Profile
    """

    def __init__(self, col_name: str):
        self.col_name    = col_name
        self.n_rows      = 0       # Total rows processed (including nulls)
        self.n_null      = 0       # Number of null rows

        # ── Unique value tracking ──
        self._exact      : set[str] = set()   # Exact unique value set (Phase 1)
        self._exact_full = False              # True once switched to approximate mode
        self._hll        : _HLL | None = None # HLL sketch (initialized in Phase 2 only)
        self._sample     : list[str] = []     # Reservoir random sample (Phase 2)

        # ── Numeric statistics ──
        # Reservoir sampling with capacity 10,000 for estimating min/max/p25/p50/p75
        self._num_reservoir : list[float] = []
        self._reservoir_n   = 0             # Total numeric rows seen by Reservoir

        # ── String statistics ──
        self._str_len_sum = 0               # Cumulative string length (for computing mean)
        self._str_len_cnt = 0               # Count of non-null strings
        self._str_samples : list[str] = []  # First 500 strings (for char_class classification)

        # ── Type inference samples ──
        self._type_samples : list[str] = [] # First 1000 non-null values (passed to _detect_dtype)

    def feed(self, raw: str):
        """
        Process the raw string value of this column from one row.

        Null recognition: empty strings, null, none, na, nan, n/a
        (case-insensitive) are all treated as null values.
        Non-null values sequentially update the type samples, unique value set,
        numeric Reservoir, and string statistics.

        Parameters
        ----------
        raw : Raw string read from CSV (no preprocessing applied)
        """
        self.n_rows += 1
        v = raw.strip()

        # Null detection
        if v == "" or v.lower() in ("null", "none", "na", "nan", "n/a"):
            self.n_null += 1
            return

        # Type inference samples (only the first 1000, sufficient to cover common type distributions)
        if len(self._type_samples) < 1000:
            self._type_samples.append(v)

        # ── Unique value / HLL update ──
        if not self._exact_full:
            # Phase 1: exact set
            self._exact.add(v)
            if len(self._exact) > UNIQUE_VALS_CAP:
                # Trigger switch: migrate exact set to HLL, clear set to free memory
                self._exact_full = True
                self._hll = _HLL()
                for ev in self._exact:
                    self._hll.add(ev)
                self._exact.clear()
                # Initialize random sample with the first SAMPLE_VALS_SIZE entries
                # from the exact set (at this point _exact has been cleared, so the
                # sample starts empty and will be filled by subsequent Reservoir steps)
                self._sample = []
        else:
            # Phase 2: update HLL + Reservoir sampling
            self._hll.add(v)
            self._reservoir_n += 1
            if len(self._sample) < SAMPLE_VALS_SIZE:
                # Sample not yet full, simply append
                self._sample.append(v)
            else:
                # Reservoir Sampling: replace existing sample with probability
                # SAMPLE_VALS_SIZE / n, ensuring each element is selected with
                # equal probability
                j = np.random.randint(0, self._reservoir_n)
                if j < SAMPLE_VALS_SIZE:
                    self._sample[j] = v

        # ── Numeric Reservoir (for quantile estimation) ──
        if NUM_PATTERN.match(v):
            fv = float(v)
            self._reservoir_n += 1
            if len(self._num_reservoir) < 10_000:
                self._num_reservoir.append(fv)
            else:
                # Likewise use Reservoir Sampling to maintain uniformity
                j = np.random.randint(0, self._reservoir_n)
                if j < 10_000:
                    self._num_reservoir[j] = fv

        # ── String length statistics ──
        self._str_len_sum += len(v)
        self._str_len_cnt += 1
        if len(self._str_samples) < 500:
            self._str_samples.append(v)

    def build(self) -> dict:
        """
        Called after scanning is complete to aggregate all intermediate state
        into the final Profile dictionary.

        Returns
        -------
        A dict containing all statistical fields for this column, structured
        as described in the module docstring.
        """
        # Infer semantic type
        dtype = _detect_dtype(self._type_samples)

        # ── Numeric quantiles ──
        num_stats: dict[str, float | None] = dict.fromkeys(
            ["num_min", "num_max", "num_p25", "num_p50", "num_p75"], None
        )
        if dtype in ("int", "float") and self._num_reservoir:
            arr = np.array(self._num_reservoir)
            # numpy's percentile computes approximate quantiles from Reservoir samples
            num_stats["num_min"] = float(np.min(arr))
            num_stats["num_max"] = float(np.max(arr))
            num_stats["num_p25"] = float(np.percentile(arr, 25))
            num_stats["num_p50"] = float(np.percentile(arr, 50))
            num_stats["num_p75"] = float(np.percentile(arr, 75))

        # ── String statistics ──
        str_stats: dict[str, Any] = {"str_avg_len": None, "str_char_class": None}
        if self._str_len_cnt > 0:
            str_stats["str_avg_len"]    = self._str_len_sum / self._str_len_cnt
            str_stats["str_char_class"] = _char_class(self._str_samples)

        # ── Unique value list and cardinality ──
        if not self._exact_full:
            # Exact mode: return the complete sorted set
            unique_vals     = sorted(str(v) for v in self._exact)
            hll_cardinality = None
        else:
            # Approximate mode: return deduplicated Reservoir samples, along with HLL cardinality estimate
            unique_vals     = list(dict.fromkeys(self._sample))[:SAMPLE_VALS_SIZE]
            hll_cardinality = self._hll.count()

        # Unified cardinality field (exact count or HLL estimate)
        n_unique = hll_cardinality if hll_cardinality is not None else len(unique_vals)

        return {
            "col":             self.col_name,
            "dtype":           dtype,
            "n_rows":          self.n_rows,
            "n_unique":        n_unique,
            "null_rate":       self.n_null / max(self.n_rows, 1),
            **num_stats,
            **str_stats,
            "unique_vals":     unique_vals,
            "exact_unique":    not self._exact_full,  # Used by Signal 1 to determine if exact subset check is feasible
            "hll_cardinality": hll_cardinality,
        }


# ─────────────────────────────────────────────────────────────
# Single-table Profile computation entry point
# ─────────────────────────────────────────────────────────────

def profile_csv(
    csv_path: str | Path,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> dict[str, dict]:
    """
    Compute full column Profiles for a single CSV table.

    Implementation
    --------------
    Uses Python's standard csv.DictReader for streaming, row-by-row reading.
    Each column is assigned a _ColAgg instance; after all rows are processed,
    build() is called to generate the Profile.

    Memory Characteristics
    ----------------------
    Memory usage ~ O(num_cols * UNIQUE_VALS_CAP) + O(num_cols * 10000 * 8 bytes),
    independent of the number of rows. Peak memory for million-row tables is
    typically < 100 MB.

    Parameters
    ----------
    csv_path  : Path to the CSV file
    encoding  : File encoding, defaults to 'utf-8'
    delimiter : Field delimiter, defaults to comma

    Returns
    -------
    A dict of {col_name: profile_dict}, where col_name corresponds to the
    column names from the CSV header
    """
    csv_path = Path(csv_path)
    aggs: dict[str, _ColAgg] = {}
    t0 = time.time()

    with open(csv_path, encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return {}

        # Create an aggregator for each column name
        for col in reader.fieldnames:
            aggs[col] = _ColAgg(col)

        # Stream-process row by row
        for row in reader:
            for col, agg in aggs.items():
                agg.feed(row.get(col, ""))

    elapsed = time.time() - t0
    profiles = {col: agg.build() for col, agg in aggs.items()}
    n_rows   = next(iter(profiles.values()), {}).get("n_rows", 0)
    print(f"  [profiler] {csv_path.name}: {n_rows:,} rows x {len(profiles)} cols  ({elapsed:.1f}s)")
    return profiles


# ─────────────────────────────────────────────────────────────
# Multi-table Profile batch management (with disk caching)
# ─────────────────────────────────────────────────────────────

def _csv_fingerprint(path: Path) -> str:
    """
    Compute a lightweight "fingerprint" string for a CSV file to determine
    whether the file has been modified.

    Constructed by combining the file size (in bytes) and last modification
    time (Unix timestamp, second precision). This is an O(1) file change
    detection method that avoids the O(N) overhead of full-file MD5 hashing.

    Limitation: modifications within the same second that coincidentally
    produce the same file size cannot be detected. For offline batch
    processing scenarios, this probability is negligible and acceptable.

    Parameters
    ----------
    path : Path to the CSV file

    Returns
    -------
    A string in the format "{file_size}_{mtime}"
    """
    st = path.stat()
    return f"{st.st_size}_{st.st_mtime:.0f}"


def load_or_build_profiles(
    data_dir: str | Path,
    cache_dir: str | Path | None = None,
    encoding: str = "utf-8",
    delimiter: str = ",",
    force_rebuild: bool = False,
) -> dict[str, dict[str, dict]]:
    """
    Batch-load column Profiles for all CSV tables in a directory, using disk
    cache when available.

    Caching Strategy
    ----------------
    - Each table's Profile is individually serialized as a JSON file under cache_dir.
    - The JSON file includes the CSV file's fingerprint (the _fingerprint field).
    - On load, the fingerprint is compared first: if it matches, the cached data
      is deserialized directly (millisecond-level); if not, the Profile is
      recomputed and the cache is updated.
    - force_rebuild=True forces all caches to be ignored, useful for debugging
      or manually refreshing after data changes.

    Cache Directory
    ---------------
    Defaults to data_dir/.profile_cache/, customizable via the cache_dir parameter.
    Created automatically if it does not exist.

    Parameters
    ----------
    data_dir      : Path to the directory containing data files
    cache_dir     : Directory for Profile JSON cache; None uses data_dir/.profile_cache/
    encoding      : CSV file encoding
    delimiter     : CSV field delimiter
    force_rebuild : If True, ignore cache and re-scan all CSV files

    Returns
    -------
    {table_name: {col_name: profile_dict}}
    table_name is the CSV filename without the .csv extension
    """
    data_dir   = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir else data_dir / ".profile_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_profiles: dict[str, dict[str, dict]] = {}

    for csv_path in sorted(data_dir.glob("*.csv")):
        table_name = csv_path.stem
        fp         = _csv_fingerprint(csv_path)
        cache_path = cache_dir / f"{table_name}.json"

        # Attempt to load from cache
        if not force_rebuild and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("_fingerprint") == fp:
                # Fingerprint matches: use cache directly
                all_profiles[table_name] = cached["profiles"]
                print(f"  [cache hit] {table_name}  ({len(cached['profiles'])} cols)")
                continue
            # Fingerprint mismatch (file has changed): fall through to recompute

        # Recompute and write to cache
        profiles = profile_csv(csv_path, encoding=encoding, delimiter=delimiter)
        payload  = {"_fingerprint": fp, "profiles": profiles}
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        all_profiles[table_name] = profiles

    return all_profiles
