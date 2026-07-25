"""
dependency_validator.py
========================
Validates three classes of data dependencies over a relational instance:

  1. Functional Dependencies   (FDs)   — X → Y
  2. Conditional FDs           (CFDs)  — (X → Y, tp) with a pattern tuple
  3. Inclusion Dependencies    (INDs)  — R[X] ⊆ S[Y]

────────────────────────────────────────────────────────────────────────────
DATA MODEL
────────────────────────────────────────────────────────────────────────────
A *relation* is represented as a list of dicts, e.g.:
    [{"name": "Alice", "age": 30, "city": "Sydney"},
     {"name": "Bob",   "age": 25, "city": "Melbourne"}]

Dependency constructors:

  FD(lhs, rhs)
      lhs / rhs : list[str]  — attribute names
      Semantics : ∀ t1,t2 ∈ R,  t1[lhs] = t2[lhs]  ⟹  t1[rhs] = t2[rhs]

  CFD(lhs, rhs, pattern)
      pattern   : dict mapping some lhs/rhs attrs to *constant* values
                  (use '_' or omit a key to mean "any value")
      Semantics : only tuples that match `pattern` on lhs attributes
                  are checked; among those, lhs → rhs must hold.

  IND(lhs_attrs, rhs_attrs, lhs_relation=None, rhs_relation=None)
      lhs_attrs / rhs_attrs : list[str]
      Semantics : the projection of lhs_relation on lhs_attrs must be
                  a subset of the projection of rhs_relation on rhs_attrs
                  (positional column alignment).

ValidationResult bundles:
  • satisfied   : bool
  • violations  : list of human-readable strings
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

Relation = List[Dict[str, Any]]   # a table as a list of row-dicts
Tuple_   = Dict[str, Any]         # a single row


@dataclass
class FD:
    """Functional Dependency  X → Y."""
    lhs: List[str]
    rhs: List[str]

    def __str__(self) -> str:
        return f"{', '.join(self.lhs)} → {', '.join(self.rhs)}"


@dataclass
class CFD:
    """Conditional Functional Dependency  (X → Y, tp).

    pattern maps attribute names to either a constant value or '_'
    (wildcard).  Attributes not listed in the pattern are treated as
    wildcards automatically.
    """
    lhs: List[str]
    rhs: List[str]
    pattern: Dict[str, Any]   # attr → constant | '_'

    def __str__(self) -> str:
        pat_str = ", ".join(
            f"{k}={v}" for k, v in self.pattern.items() if v != "_"
        )
        return (f"({', '.join(self.lhs)} → {', '.join(self.rhs)}"
                f"  |  pattern: {{{pat_str}}})")


@dataclass
class IND:
    """Inclusion Dependency  R[X] ⊆ S[Y].

    lhs_relation / rhs_relation are supplied at validation time if not
    embedded here.
    """
    lhs_attrs: List[str]
    rhs_attrs: List[str]
    lhs_relation: Optional[Relation] = field(default=None, repr=False)
    rhs_relation: Optional[Relation] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if len(self.lhs_attrs) != len(self.rhs_attrs):
            raise ValueError(
                "IND: lhs_attrs and rhs_attrs must have the same arity."
            )

    def __str__(self) -> str:
        return (f"R[{', '.join(self.lhs_attrs)}]"
                f" ⊆ S[{', '.join(self.rhs_attrs)}]")


@dataclass
class ValidationResult:
    dependency: Any
    satisfied: bool
    violations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "✔  SATISFIED" if self.satisfied else "✘  VIOLATED"
        lines  = [f"{status}  —  {self.dependency}"]
        for v in self.violations:
            lines.append(f"   ↳ {v}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _project(t: Tuple_, attrs: List[str]) -> tuple:
    """Return the values of *attrs* from tuple *t* as a hashable tuple."""
    return tuple(t[a] for a in attrs)


def _matches_pattern(t: Tuple_, pattern: Dict[str, Any]) -> bool:
    """Return True iff tuple *t* agrees with every constant in *pattern*."""
    for attr, val in pattern.items():
        if val == "_":
            continue
        if attr not in t or t.get(attr) != val:
            return False
    return True


def _fmt_tuple(t: Tuple_, attrs: List[str]) -> str:
    parts = ", ".join(f"{a}={t[a]!r}" for a in attrs)
    return f"({parts})"


# ──────────────────────────────────────────────────────────────────────────────
# Core validators
# ──────────────────────────────────────────────────────────────────────────────

def validate_fd(relation: Relation, fd: FD) -> ValidationResult:
    """
    Check that  lhs → rhs  holds in *relation*.

    Algorithm:
      Group rows by lhs-projection value.  Within each group, the
      rhs-projection must be the same for every row.
    """
    violations: List[str] = []
    groups: Dict[tuple, List[int]] = {}

    for idx, row in enumerate(relation):
        key = _project(row, fd.lhs)
        groups.setdefault(key, []).append(idx)

    for key, indices in groups.items():
        rhs_values = {_project(relation[i], fd.rhs) for i in indices}
        if len(rhs_values) > 1:
            # Collect the first two violating pairs for readability
            pairs = list(itertools.combinations(indices, 2))
            for i, j in pairs[:3]:
                t1, t2 = relation[i], relation[j]
                if _project(t1, fd.rhs) != _project(t2, fd.rhs):
                    violations.append(
                        f"Rows {i} and {j} share lhs "
                        f"{_fmt_tuple(t1, fd.lhs)} but differ on rhs: "
                        f"{_fmt_tuple(t1, fd.rhs)} vs {_fmt_tuple(t2, fd.rhs)}"
                    )

    return ValidationResult(fd, satisfied=not violations, violations=violations)


def validate_cfd(relation: Relation, cfd: CFD) -> ValidationResult:
    """
    Check that  (lhs → rhs, pattern)  holds in *relation*.

    Only tuples that match the pattern on the lhs attributes (and any
    rhs constant in the pattern) are considered.  Among those, lhs → rhs
    must hold as an ordinary FD.
    """
    violations: List[str] = []

    # Filter rows that satisfy the pattern
    relevant: List[Tuple[int, Tuple_]] = [
        (idx, row)
        for idx, row in enumerate(relation)
        if _matches_pattern(row, cfd.pattern)
    ]

    if not relevant:
        # No tuples match the pattern — vacuously satisfied
        return ValidationResult(
            cfd, satisfied=True,
            violations=["(no tuples match the pattern — vacuously satisfied)"]
        )

    # Within relevant tuples check lhs → rhs
    groups: Dict[tuple, List[int]] = {}
    for idx, row in relevant:
        key = _project(row, cfd.lhs)
        groups.setdefault(key, []).append(idx)

    for key, indices in groups.items():
        rhs_values = {_project(relation[i], cfd.rhs) for i in indices}
        if len(rhs_values) > 1:
            for i, j in list(itertools.combinations(indices, 2))[:3]:
                t1, t2 = relation[i], relation[j]
                if _project(t1, cfd.rhs) != _project(t2, cfd.rhs):
                    violations.append(
                        f"Rows {i} and {j} both match pattern, share lhs "
                        f"{_fmt_tuple(t1, cfd.lhs)} but differ on rhs: "
                        f"{_fmt_tuple(t1, cfd.rhs)} vs {_fmt_tuple(t2, cfd.rhs)}"
                    )

    return ValidationResult(cfd, satisfied=not violations, violations=violations)


def validate_ind(
    ind: IND,
    lhs_relation: Optional[Relation] = None,
    rhs_relation: Optional[Relation] = None,
) -> ValidationResult:
    """
    Check that  R[lhs_attrs] ⊆ S[rhs_attrs].

    Relations may be supplied either in the IND object or as arguments
    (arguments take precedence).
    """
    R = lhs_relation if lhs_relation is not None else ind.lhs_relation
    S = rhs_relation if rhs_relation is not None else ind.rhs_relation

    if R is None or S is None:
        raise ValueError(
            "validate_ind: both lhs_relation and rhs_relation must be provided."
        )

    violations: List[str] = []

    # Build the set of rhs projections
    rhs_set = {_project(row, ind.rhs_attrs) for row in S}

    for idx, row in enumerate(R):
        lhs_val = _project(row, ind.lhs_attrs)
        if lhs_val not in rhs_set:
            violations.append(
                f"Row {idx} of R has "
                f"{_fmt_tuple(row, ind.lhs_attrs)} "
                f"which has no match in S[{', '.join(ind.rhs_attrs)}]"
            )

    return ValidationResult(ind, satisfied=not violations, violations=violations)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: validate all dependencies at once
# ──────────────────────────────────────────────────────────────────────────────

def validate_all(
    relation: Relation,
    fds:  List[FD]  = (),
    cfds: List[CFD] = (),
    inds: List[IND] = (),
    *,
    lhs_relation: Optional[Relation] = None,
    rhs_relation: Optional[Relation] = None,
) -> List[ValidationResult]:
    """
    Validate all dependency types and return a list of ValidationResult objects.

    For INDs, pass *lhs_relation* / *rhs_relation* (or embed them in each IND).
    If only one relation is being tested, pass the same relation for both.
    """
    results = []
    for fd   in fds:   results.append(validate_fd(relation, fd))
    for cfd  in cfds:  results.append(validate_cfd(relation, cfd))
    for ind  in inds:
        lr = lhs_relation if lhs_relation is not None else relation
        rr = rhs_relation if rhs_relation is not None else relation
        results.append(validate_ind(ind, lr, rr))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Attribute-closure utility (bonus: used for FD implication / key detection)
# ──────────────────────────────────────────────────────────────────────────────

def attribute_closure(attrs: List[str], fds: List[FD]) -> set:
    """
    Compute the closure of *attrs* under *fds* using the standard
    iterative algorithm.  Useful for checking FD implication and
    identifying keys / super-keys.
    """
    closure = set(attrs)
    changed = True
    while changed:
        changed = False
        for fd in fds:
            if set(fd.lhs).issubset(closure) and not set(fd.rhs).issubset(closure):
                closure.update(fd.rhs)
                changed = True
    return closure


def is_superkey(attrs: List[str], all_attrs: List[str], fds: List[FD]) -> bool:
    """Return True if *attrs* is a super-key of the relation schema."""
    return attribute_closure(attrs, fds) >= set(all_attrs)


# ──────────────────────────────────────────────────────────────────────────────
# Demo / self-test
# ──────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    print("=" * 68)
    print("  DEPENDENCY VALIDATOR — DEMONSTRATION")
    print("=" * 68)

    # ── Relation R ────────────────────────────────────────────────────────────
    R: Relation = [
        {"zip": "2000", "city": "Sydney",    "country": "AU", "status": "active"},
        {"zip": "2000", "city": "Sydney",    "country": "AU", "status": "active"},
        {"zip": "3000", "city": "Melbourne", "country": "AU", "status": "active"},
        {"zip": "4000", "city": "Brisbane",  "country": "AU", "status": "inactive"},
        {"zip": "5000", "city": "Adelaide",  "country": "AU", "status": "active"},
        # Intentional FD violation: same zip, different city
        {"zip": "2000", "city": "Parramatta","country": "AU", "status": "active"},
    ]

    # ── Relation S (for IND tests) ─────────────────────────────────────────────
    S: Relation = [
        {"zip": "2000", "region": "NSW"},
        {"zip": "3000", "region": "VIC"},
        {"zip": "4000", "region": "QLD"},
        {"zip": "5000", "region": "SA"},
        # Note: 9999 is not in R — R[zip] ⊆ S[zip] will still hold (direction matters)
        {"zip": "9999", "region": "WA"},
    ]

    # ── Define dependencies ───────────────────────────────────────────────────
    fds = [
        FD(["zip"],           ["city"]),        # zip → city  (VIOLATED by row 5)
        FD(["zip"],           ["country"]),     # zip → country  (satisfied)
        FD(["zip", "status"], ["city"]),        # {zip,status} → city  (still violated)
    ]

    cfds = [
        # Among active rows: zip → city  (also violated)
        CFD(["zip"], ["city"],
            pattern={"status": "active"}),

        # Among inactive rows: zip → city  (satisfied — only row 3 is inactive)
        CFD(["zip"], ["city"],
            pattern={"status": "inactive"}),

        # Pattern with wildcard: country=AU, status=_ → zip → city
        CFD(["zip"], ["city"],
            pattern={"country": "AU", "status": "_"}),
    ]

    inds = [
        # R[zip] ⊆ S[zip]  — satisfied (every R-zip exists in S)
        IND(["zip"], ["zip"]),

        # S[zip] ⊆ R[zip]  — violated (9999 not in R)
        IND(["zip"], ["zip"],
            lhs_relation=S, rhs_relation=R),
    ]

    # ── Run validation ────────────────────────────────────────────────────────
    print("\n── Relation R ──")
    for i, row in enumerate(R):
        print(f"  [{i}] {row}")

    print("\n── Relation S ──")
    for i, row in enumerate(S):
        print(f"  [{i}] {row}")

    print("\n── Functional Dependencies ──────────────────────────────────")
    fd_results = [validate_fd(R, fd) for fd in fds]
    for r in fd_results:
        print(r)

    print("\n── Conditional Functional Dependencies ──────────────────────")
    cfd_results = [validate_cfd(R, cfd) for cfd in cfds]
    for r in cfd_results:
        print(r)

    print("\n── Inclusion Dependencies ───────────────────────────────────")
    ind_results = [
        validate_ind(inds[0], lhs_relation=R, rhs_relation=S),
        validate_ind(inds[1]),          # relations embedded in IND object
    ]
    for r in ind_results:
        print(r)

    # ── Attribute closure demo ────────────────────────────────────────────────
    print("\n── Attribute Closure & Key Detection ────────────────────────")
    schema_attrs = ["zip", "city", "country", "status"]
    clean_fds = [FD(["zip"], ["city"]), FD(["zip"], ["country"])]
    closure = attribute_closure(["zip"], clean_fds)
    print(f"  Closure of {{zip}} under FDs: {sorted(closure)}")
    print(f"  Is {{zip}} a super-key?  "
          f"{is_superkey(['zip'], schema_attrs, clean_fds)}")
    print(f"  Is {{zip, status}} a super-key?  "
          f"{is_superkey(['zip', 'status'], schema_attrs, clean_fds)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_results = fd_results + cfd_results + ind_results
    n_sat = sum(1 for r in all_results if r.satisfied)
    print(f"\n── Summary: {n_sat}/{len(all_results)} dependencies satisfied ──")
    print("=" * 68)


if __name__ == "__main__":
    _demo()
