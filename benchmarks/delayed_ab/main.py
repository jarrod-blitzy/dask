"""A/B performance runner for ``dask.delayed`` graph construction.

The runner measures two implementations of ``delayed`` against each other inside a
single interpreter, proves that they behave identically *before* it records a
single timing, prints a pass/fail checklist and writes the suite's two committed
artefacts. It is equivalence evidence as much as performance evidence: a speedup
reported for an implementation that does something different is worthless, so the
equivalence assertions run first and no artefact is written when one of them
fails.

Arms:
    Arm A -- the frozen baseline: ``benchmarks.delayed_ab.baseline_delayed``, a
        verbatim capture of ``dask/delayed.py`` taken before the first edit of the
        refactor. It is never edited, and neither is any module it imports.
    Arm B -- the live candidate: the module object returned by
        ``importlib.import_module("dask.delayed")``. The attribute form
        ``dask.delayed`` is deliberately not used: ``dask/__init__.py`` executes
        ``from dask.delayed import delayed``, which rebinds the attribute
        ``delayed`` on the ``dask`` package, so ``import dask.delayed as m`` hands
        back the ``delayed`` curry rather than the module.

Arm activation:
    Two frozen modules recognise ``Delayed`` by identity through a late-bound
    import, so the arm under test has to be installed as ``dask.delayed`` while it
    runs. ``dask/base.py`` lines 451-453 execute ``from dask.delayed import
    Delayed`` inside the per-collection loop of ``collections_to_expr`` and route
    anything that is not an instance of *that* class down the
    ``getattr(coll, "expr", None)`` / ``__dask_exprs__`` branch;
    ``dask/_expr.py`` lines 1336-1345 (``HLGFinalizeCompute._simplify_down``)
    compares ``self.dsk.postcompute`` with ``Delayed.__dask_postcompute__(...)``,
    which is the identity of the live module's ``single_key``. Without activation
    the baseline arm's objects are foreign collections, the finalize skip does not
    fire, and its graphs acquire extra ``finalize-hlgfinalizecompute-*`` layers.
    ``activate`` therefore swaps ``sys.modules["dask.delayed"]`` for the duration
    of every arm operation -- construction, equivalence extraction and every
    compute -- and it is used for both arms so that the two run under identical
    conditions. It exists precisely so that no frozen ``dask`` module has to be
    edited.

Paired protocol:
    A round is the block A, B, B, A: four timed regions, two per arm, so that
    drift inside the round cancels. Consecutive rounds alternate which arm starts,
    so odd-numbered rounds run B, A, A, B. The per-round paired ratio is
    ``(A1 + A2) / (B1 + B2)`` -- baseline over candidate, so a ratio above 1 means
    the candidate is faster. Warmup rounds are discarded. ``time.perf_counter_ns``
    brackets the case's ``build`` call and nothing else; the garbage collector is
    collected and disabled around every region and collected again between rounds;
    every computation runs on the main thread under the synchronous scheduler and
    outside every timed region.

Allocation figures:
    ``sys.getallocatedblocks`` deltas are recorded per timed region.
    ``tracemalloc`` peak bytes -- the gate-bearing allocation figure -- come from
    separate untimed rounds taken after the timing rounds, because tracing slows
    execution two to three times. Two block figures accompany them, each under its
    own definition and neither of them a peak: ``live_blocks_end`` and
    ``max_observed_blocks``. The standard library exposes no peak block count at
    all; that requirement conflict and its resolution are recorded in the
    artefacts under ``environment.peak_block_count_conflict``.

Artefacts:
    ``<output>/baseline_vs_candidate.json`` holds every raw measurement, the
    statistics, the gate checklist and the environment block.
    ``<output>/report.md`` holds the environment summary, one table for the gated
    cases, one for the informational sub-series and a single closing
    ``OVERALL:`` line. Both contain the real measured numbers of the run that
    wrote them and are never hand-edited. The default output directory lives
    inside the repository, and writing there is refused while the working tree is
    dirty, so a committed artefact always describes an identifiable commit. The
    live SHA it records is the commit whose ``dask/delayed.py`` was measured, not
    the commit that adds the artefacts -- the sources are committed first, the
    runner is executed from that clean commit, and its two result files are
    committed afterwards.

Invocation, from the repository root:
    ``python -m benchmarks.delayed_ab``

    ``DASK_DELAYED_AB=1 pytest dask/tests/test_delayed_ab_gate.py``

Exit codes -- the run's own outcomes. A command line that cannot be parsed is
rejected by ``argparse`` before the run starts, with its conventional status 2:

    0: every gate item held.
    1: the gate failed, or the run could not be configured (an unreadable
        ``--calibration`` file, for instance). The checklist names the item.
    2: an equivalence mismatch. No artefact is written, and no performance
        verdict is produced.
    3: artefacts were requested inside a dirty repository working tree. The
        offending paths are printed.

Halt and report -- these are reported, never worked around:
    * The two arms cannot be imported side by side.
    * An A/A calibration run on unmodified code fails the equivalence assertions
      with activation in place.
    * A measurement appears to require a change outside ``benchmarks/delayed_ab``.
    * A canonical graph carries a value that cannot be reproduced across
      processes, such as an ``id()``-derived layer name from
      ``dask/highlevelgraph.py`` lines 1002-1010.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import pathlib
import platform
import random
import re
import statistics
import subprocess
import sys
import sysconfig
import time
import timeit
import tracemalloc

# Typing-support standard library modules. They carry no runtime behaviour and
# exist only so that every definition below can be annotated.
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from dask.base import is_dask_collection
from dask.hashing import hashers

# Relative imports: ``benchmarks/`` is a PEP 420 namespace package (no
# ``__init__.py``), so an absolute ``benchmarks.delayed_ab.*`` import would make
# mypy see each sibling module under two names when the whole repository is
# checked. The suite is always started as ``python -m benchmarks.delayed_ab``.
from .canon import canonical_graph, canonical_result, normalize_key
from .cases import CASES, RATIO_CASES, SUBSERIES, Case

# ---------------------------------------------------------------------------
# Every magic number of the protocol, auditable in one place.
# ---------------------------------------------------------------------------

#: Measured rounds of a full run, and the floor the gate test's reduced run uses.
_DEFAULT_ROUNDS = 15
_MIN_ROUNDS = 7

#: Discarded warmup rounds, and their floor.
_DEFAULT_WARMUP = 3
_MIN_WARMUP = 2

#: Percentile bootstrap on the median of the per-round ratios.
_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_SEED = 0
_CI_LOWER_PERCENTILE = 0.025
_CI_UPPER_PERCENTILE = 0.975

#: Gate thresholds.
_RATIO_THRESHOLD = 1.25
_CI_LOWER_FLOOR = 1.0
_REGRESSION_CI_UPPER = 0.98
_MIN_IMPROVED_CASES = 4
_PEAK_ALLOC_TOLERANCE = 1.05

#: Provenance of arm A, cross-checked against the capture's own header.
_BASELINE_SHA = "c9d1df34ccba182ddf43c2dbe4315c4d9c8c44e1"

#: Artefact location, relative to the repository root, and the JSON schema
#: version that ``dask/tests/test_delayed_ab_gate.py`` parses.
_DEFAULT_OUTPUT = "benchmarks/delayed_ab/results"
_SCHEMA_VERSION = 1

#: Exit codes. ``__main__.py`` raises ``SystemExit(main())``.
_EXIT_PASS = 0
_EXIT_GATE_FAIL = 1
_EXIT_EQUIVALENCE = 2
_EXIT_DIRTY = 3

#: Profile events between two ``sys.getallocatedblocks()`` samples in the
#: ``max_observed_blocks`` run. The call itself walks the allocator's pools, so
#: sampling it on every event costs roughly forty times the region it observes
#: (measured: 0.21 s of construction becomes 16.8 s, while an empty hook costs
#: 0.48 s). At this stride the same run costs 0.9 s and reports a figure within
#: 0.5% of the every-event value, which is why the figure is documented as a
#: sampled lower bound and never as a peak.
_BLOCK_SAMPLE_STRIDE = 64

#: Iterations of the ``is_dask_collection`` micro-benchmark that records the cost
#: of the probe ordering the refactor deliberately did not change.
_MICROBENCH_ITERATIONS = 1_000_000

#: Constructions performed by the ``flat_loop`` case. Its size is frozen in
#: ``cases.py`` as a private constant, so it is restated -- not imported -- here
#: to turn that case's median into a per-construction cost.
_FLAT_LOOP_CONSTRUCTIONS = 10_000

#: Cases whose keys are inherently random for part of their workload, so that the
#: ``pure=True`` variant legitimately still needs key placeholders. Every other
#: case must produce none.
_IMPURE_BY_DEFINITION = frozenset({"pure_vs_impure", "pure_false"})

#: How far an A/A calibration ratio median may sit from 1.0 before the report
#: flags that case as arm or order bias for the reviewer to weigh.
_CALIBRATION_DEVIATION = 0.05

#: Distributions surfaced at the top of the environment block.
_KEY_PACKAGES = ("dask", "toolz", "cloudpickle", "numpy", "pandas")

#: The arm labels used by every payload, checklist line and report row.
_BASELINE = "baseline"
_CANDIDATE = "candidate"

#: Longest repr the mismatch report prints for one side of a difference.
_MISMATCH_REPR_LIMIT = 400

#: The standing statement of the peak-block-count requirement conflict. It is
#: recorded in the JSON and printed into the report rather than being resolved
#: silently.
_PEAK_BLOCK_COUNT_CONFLICT = (
    "The suite was asked to record a peak block count alongside peak bytes. The "
    "standard library exposes no such API: tracemalloc tracks peak bytes only "
    "(get_traced_memory()[1]) and a snapshot exposes the live block count, while "
    "the exact alternative would be a third-party allocation tracer, which the "
    "no-new-dependency directive forbids. The harder directive wins. The gate is "
    "evaluated on tracemalloc peak bytes, which the standard library supplies "
    "exactly, and two block figures are recorded as supporting evidence, each "
    "under its own definition and neither of them a peak: 'live_blocks_end' is "
    "the number of traced blocks still alive at the end of the region (end "
    "snapshot trace count minus the baseline count), and 'max_observed_blocks' is "
    "a sampled lower bound -- the maximum of sys.getallocatedblocks() read every "
    f"{_BLOCK_SAMPLE_STRIDE}th sys.setprofile event of one further untimed run, "
    "floored by the count taken immediately after the region while the "
    "constructed objects are still alive, minus the count taken before it."
)


# ---------------------------------------------------------------------------
# Records. Every measurement travels through one of these, so the payload
# writers never have to guess what a bare tuple meant.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """One timed region: a single ``build`` call of one case under one arm.

    Attributes:
        timing_ns: Wall time of the ``build`` call in nanoseconds, from
            ``time.perf_counter_ns`` brackets that contain nothing else.
        blocks_delta: ``sys.getallocatedblocks()`` after the region minus before
            it, sampled while the constructed objects are still alive.
    """

    timing_ns: int
    blocks_delta: int


@dataclass(frozen=True)
class Round:
    """One A, B, B, A block -- or B, A, A, B when the candidate starts.

    Attributes:
        first_arm: ``"baseline"`` or ``"candidate"``, whichever ran first.
        baseline: The arm-A regions, in the order they were measured.
        candidate: The arm-B regions, in the order they were measured.
    """

    first_arm: str
    baseline: tuple[Region, Region]
    candidate: tuple[Region, Region]

    @property
    def ratio(self) -> float:
        """Paired ratio ``(A1 + A2) / (B1 + B2)``: above 1 means B is faster."""
        baseline_total = sum(region.timing_ns for region in self.baseline)
        candidate_total = sum(region.timing_ns for region in self.candidate)
        if candidate_total <= 0:
            # Unreachable for the millisecond-scale regions of this corpus; the
            # guard keeps a zero denominator from producing a division error or
            # an ``inf`` that JSON cannot express.
            return float(baseline_total)
        return baseline_total / candidate_total


@dataclass(frozen=True)
class ArmAllocation:
    """The three allocation figures of one arm on one case.

    Attributes:
        tracemalloc_peak_bytes: ``tracemalloc.get_traced_memory()[1]`` after the
            region, following ``reset_peak()`` before it. This is the only
            gate-bearing allocation figure.
        live_blocks_end: Traced blocks still alive at the end of the region: the
            end snapshot's trace count minus the baseline snapshot's. Not a peak.
        max_observed_blocks: Sampled lower bound on the peak block delta of the
            region, as defined in ``_PEAK_BLOCK_COUNT_CONFLICT``. Not a peak.
    """

    tracemalloc_peak_bytes: int
    live_blocks_end: int
    max_observed_blocks: int

    def payload(self) -> dict[str, int]:
        """Return the JSON shape of these figures."""
        return {
            "tracemalloc_peak_bytes": self.tracemalloc_peak_bytes,
            "live_blocks_end": self.live_blocks_end,
            "max_observed_blocks": self.max_observed_blocks,
        }


@dataclass(frozen=True)
class Extraction:
    """The comparable surface of one case built once under one arm.

    Attributes:
        count: Number of constructed objects.
        keys: ``o.key`` per object, mapped through ``normalize_key`` with one
            table per arm and threaded in list order, so that a placeholder's
            number follows first appearance.
        graphs: ``canonical_graph(o)`` per object, in the same order.
        results: ``canonical_result(objs)`` -- the objects computed under the
            synchronous scheduler, outside every timed region.
        placeholders: Size of the normalisation table, i.e. how many distinct
            keys carried an inherently random token.
    """

    count: int
    keys: tuple[object, ...]
    graphs: tuple[dict[str, Any], ...]
    results: tuple[Any, ...]
    placeholders: int


@dataclass(frozen=True)
class Equivalence:
    """The equivalence verdict for one case, over both ``pure`` variants.

    Attributes:
        case: The case name.
        native: Whether the native ``pure=None`` variant matched.
        pure_true: Whether the ``pure=True`` variant matched.
        placeholders_pure_true: Key placeholders needed by the ``pure=True``
            variant. Zero for every case except those in
            ``_IMPURE_BY_DEFINITION``, and asserted as such.
        mismatch: ``None`` when the case is equivalent, otherwise the formatted
            report naming the variant, the first differing object and the field.
    """

    case: str
    native: bool
    pure_true: bool
    placeholders_pure_true: int
    mismatch: str | None

    @property
    def ok(self) -> bool:
        """Whether both variants matched."""
        return self.mismatch is None

    def payload(self) -> dict[str, Any]:
        """Return the JSON shape of this verdict."""
        return {
            "native": self.native,
            "pure_true": self.pure_true,
            "placeholders_pure_true": self.placeholders_pure_true,
        }


@dataclass(frozen=True)
class CaseResult:
    """Everything measured for one case, plus the statistics derived from it.

    Attributes:
        name: The case name, and the key it is recorded under in both artefacts.
        pure: The ``pure`` variant that was timed -- ``None``, each case's native
            keying. The ``pure=True`` variant is built for the exact-key
            equivalence check and never timed.
        gated: Whether the case takes part in the gate. ``False`` for the
            informational sub-series.
        rounds: The measured rounds, warmup already discarded.
        baseline_allocation: Arm A's allocation figures.
        candidate_allocation: Arm B's allocation figures.
        equivalence: The case's equivalence verdict.
        ratio_median: Median of the per-round paired ratios.
        ci_low: Lower bound of the percentile-bootstrap 95% interval on that
            median.
        ci_high: Upper bound of the same interval.
    """

    name: str
    pure: bool | None
    gated: bool
    rounds: tuple[Round, ...]
    baseline_allocation: ArmAllocation
    candidate_allocation: ArmAllocation
    equivalence: Equivalence
    ratio_median: float
    ci_low: float
    ci_high: float

    @property
    def ratios(self) -> tuple[float, ...]:
        """The per-round paired ratios, in round order."""
        return tuple(round_.ratio for round_ in self.rounds)

    def timings(self, arm: str) -> tuple[int, ...]:
        """Return one arm's region timings in measurement order."""
        return tuple(
            region.timing_ns
            for round_ in self.rounds
            for region in self._regions(round_, arm)
        )

    def blocks(self, arm: str) -> tuple[int, ...]:
        """Return one arm's per-region allocated-block deltas, same order."""
        return tuple(
            region.blocks_delta
            for round_ in self.rounds
            for region in self._regions(round_, arm)
        )

    def allocation(self, arm: str) -> ArmAllocation:
        """Return one arm's allocation figures."""
        return (
            self.baseline_allocation if arm == _BASELINE else self.candidate_allocation
        )

    @property
    def peak_bytes_ratio(self) -> float:
        """Candidate peak bytes over baseline peak bytes: 1.05 is the tolerance."""
        baseline_peak = self.baseline_allocation.tracemalloc_peak_bytes
        if baseline_peak <= 0:
            # Only reachable if tracing recorded nothing at all, which cannot
            # happen for a region that constructs a graph; treat it as neutral
            # rather than inventing an unbounded ratio.
            return 1.0
        return self.candidate_allocation.tracemalloc_peak_bytes / baseline_peak

    @staticmethod
    def _regions(round_: Round, arm: str) -> tuple[Region, Region]:
        """Return the two regions one arm contributed to a round."""
        return round_.baseline if arm == _BASELINE else round_.candidate


@dataclass(frozen=True)
class GateCheck:
    """One item of the gate checklist.

    Attributes:
        name: The item's stable name, which is part of the JSON parsing contract.
        case: The case the item is about, or ``None`` for a cross-case item.
        measured: The value measured for the item.
        threshold: The value it was required to reach.
        passed: Whether it held.
    """

    name: str
    case: str | None
    measured: float
    threshold: float
    passed: bool

    def payload(self) -> dict[str, Any]:
        """Return the JSON shape of this item."""
        return {
            "name": self.name,
            "case": self.case,
            "measured": self.measured,
            "threshold": self.threshold,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class RepositoryState:
    """Provenance of the candidate arm, read once before anything is written.

    Attributes:
        root: The repository root, derived from this file's location rather than
            from the working directory.
        git_head: ``git rev-parse HEAD``, or ``None`` when git is unavailable.
        dirty: Whether ``git status --porcelain --untracked-files=all`` reported
            anything. Read at start-up, so artefacts the runner itself writes are
            never counted.
        dirty_paths: The porcelain lines behind ``dirty``, for the exit-3 message.
        delayed_py_sha256: Hex digest of ``dask/delayed.py`` -- the file the
            refactor changes -- or ``None`` if it could not be read.
        notes: Anything that degraded, such as a missing git executable or a
            capture header that disagrees with ``_BASELINE_SHA``.
    """

    root: pathlib.Path
    git_head: str | None
    dirty: bool
    dirty_paths: tuple[str, ...]
    delayed_py_sha256: str | None
    notes: tuple[str, ...]


# ---------------------------------------------------------------------------
# Arms and arm activation
# ---------------------------------------------------------------------------


class ArmLoadError(RuntimeError):
    """The two arms could not be loaded side by side in one interpreter.

    This is a halt-and-report condition. Neither the frozen capture nor any
    ``dask`` module may be edited to make the import work; the failure is printed
    and the run ends non-zero.
    """


def load_arms() -> tuple[ModuleType, ModuleType]:
    """Import both arms and return them as ``(baseline, candidate)``.

    Returns:
        The frozen baseline module ``benchmarks.delayed_ab.baseline_delayed`` and
        the live ``dask.delayed`` module, as two distinct module objects.

    Raises:
        ArmLoadError: If either import fails, or if the two turn out to be the
            same object -- meaning they cannot coexist in one process. Both are
            halt-and-report conditions.
    """
    try:
        baseline = importlib.import_module("benchmarks.delayed_ab.baseline_delayed")
    except Exception as exc:
        # Reported, never worked around: the capture and every frozen dask module
        # stay untouched.
        raise ArmLoadError(
            "arm A (benchmarks.delayed_ab.baseline_delayed) could not be imported: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        # ``importlib.import_module`` is mandatory here. ``import dask.delayed as
        # m`` would bind the *attribute* ``dask.delayed``, and ``dask/__init__.py``
        # executes ``from dask.delayed import delayed``, which rebinds that
        # attribute on the package to the ``delayed`` curry -- so the attribute
        # form yields the function, not the module.
        candidate = importlib.import_module("dask.delayed")
    except Exception as exc:
        raise ArmLoadError(
            f"arm B (dask.delayed) could not be imported: {type(exc).__name__}: {exc}"
        ) from exc
    if baseline is candidate:
        raise ArmLoadError(
            "arm A and arm B resolved to the same module object, so the two "
            "implementations cannot be measured side by side"
        )
    return baseline, candidate


@contextlib.contextmanager
def activate(mod: ModuleType) -> Iterator[ModuleType]:
    """Install one arm as ``dask.delayed`` for the duration of the block.

    Frozen engine code recognises ``Delayed`` by identity through a late-bound
    ``from dask.delayed import Delayed`` (``dask/base.py`` lines 451-453) and
    compares ``single_key`` identity (``dask/_expr.py`` lines 1336-1345), so an
    arm only behaves like *the* ``delayed`` implementation while it occupies that
    module slot. Every arm operation -- construction, equivalence extraction and
    every compute -- runs inside this context, for both arms, so that the two are
    measured under identical conditions.

    The previous entry is always restored, including on the error paths that end
    the run, and the key is deleted again if there was none. Nesting is therefore
    safe, and no patched ``sys.modules`` is ever left behind.

    Args:
        mod: The arm to activate.

    Yields:
        The activated module, so that ``with activate(arm) as mod`` reads
        naturally at the call site.
    """
    had_previous = "dask.delayed" in sys.modules
    # Typed ``Any`` so that restoring whatever occupied the slot needs no cast and
    # no suppression comment.
    previous: Any = sys.modules.get("dask.delayed")
    sys.modules["dask.delayed"] = mod
    try:
        yield mod
    finally:
        if had_previous:
            sys.modules["dask.delayed"] = previous
        else:
            del sys.modules["dask.delayed"]


# ---------------------------------------------------------------------------
# Equivalence, asserted before a single timing is recorded
# ---------------------------------------------------------------------------


def _variant_label(pure: bool | None) -> str:
    """Name a ``pure`` variant the way the mismatch report and checklist do."""
    return f"pure={pure}"


def _truncate(value: object) -> str:
    """Return a bounded ``repr`` so a mismatch report stays readable."""
    text = repr(value)
    if len(text) <= _MISMATCH_REPR_LIMIT:
        return text
    return f"{text[:_MISMATCH_REPR_LIMIT]}... ({len(text)} chars)"


def _extract(case: Case, mod: ModuleType, *, pure: bool | None) -> Extraction:
    """Build one case under one arm and extract its comparable surface.

    Construction, key extraction, canonicalisation and the computation all run
    under that arm's activation, so every late-bound ``from dask.delayed import
    Delayed`` inside the engine sees the arm being measured. Nothing here is
    timed: ``canonical_result`` computes the whole batch under the synchronous
    scheduler, which must never happen inside a timed region.

    Args:
        case: The case to build.
        mod: The arm to build it with.
        pure: The ``pure`` value threaded into the case's ``delayed`` calls.

    Returns:
        The case's comparable surface: object count, normalised keys, canonical
        graphs, computed results and the number of key placeholders needed.
    """
    with activate(mod):
        state = case.setup(mod, pure=pure)
        objs = case.build(mod, state, pure=pure)
        # One table per arm, threaded in list order, so that a placeholder's
        # number follows the first appearance of its key.
        table: dict[str, str] = {}
        keys = tuple(normalize_key(obj.key, table) for obj in objs)
        graphs = tuple(canonical_graph(obj) for obj in objs)
        results = canonical_result(objs)
    return Extraction(
        count=len(objs),
        keys=keys,
        graphs=graphs,
        results=results,
        placeholders=len(table),
    )


def _first_graph_difference(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    """Name the first canonical-graph field that differs, with both values.

    The field order of ``canonical_graph`` is fixed, so the field reported is
    stable for a given pair of graphs. ``layer_order`` and ``dependency_order``
    are the order-sensitive fields; a difference in one of them means the two
    arms built the same layers in a different insertion order, which is a
    behaviour change because ``tokenize`` hashes pickled slot state.
    """
    for field, baseline_value in baseline.items():
        candidate_value = candidate.get(field)
        if baseline_value != candidate_value:
            return (
                f"field {field!r}: baseline={_truncate(baseline_value)} "
                f"candidate={_truncate(candidate_value)}"
            )
    extra = sorted(set(candidate) - set(baseline))
    if extra:
        return f"candidate carries fields the baseline does not: {extra}"
    # Reached only if the two dicts compare unequal while every field compares
    # equal, which no canonical form of this corpus produces.
    return (
        f"the two canonical graphs compare unequal with no differing field: "
        f"baseline={_truncate(baseline)} candidate={_truncate(candidate)}"
    )


def _compare(
    case_name: str,
    baseline: Extraction,
    candidate: Extraction,
    *,
    pure: bool | None,
) -> str | None:
    """Compare two extractions of one case and return the first mismatch.

    The four steps run in the order the contract fixes: counts, keys, canonical
    graphs, results. The ``pure=True`` variant additionally has to be fully
    deterministic -- every key compared as an exact string, no placeholder at all
    -- for every case except those whose workload is impure by definition.

    Args:
        case_name: The case being compared, named in the report.
        baseline: Arm A's extraction.
        candidate: Arm B's extraction.
        pure: The variant the two extractions were built with.

    Returns:
        ``None`` when the two are equivalent, otherwise a formatted report naming
        the case, the variant, the index of the first differing object and the
        differing field.
    """
    variant = _variant_label(pure)
    prefix = f"{case_name} [{variant}]"

    # (0) Counts, before anything else is compared.
    if baseline.count != candidate.count:
        return (
            f"{prefix}: object count differs -- baseline built {baseline.count}, "
            f"candidate built {candidate.count}"
        )

    # (1) Keys. Deterministic keys are compared as exact strings because
    # ``normalize_key`` returns them verbatim; only inherently random tokens
    # become placeholders.
    if baseline.keys != candidate.keys:
        index = next(
            (
                position
                for position, (left, right) in enumerate(
                    zip(baseline.keys, candidate.keys)
                )
                if left != right
            ),
            min(len(baseline.keys), len(candidate.keys)),
        )
        return (
            f"{prefix}: key differs at object {index} -- "
            f"baseline={_truncate(baseline.keys[index])} "
            f"candidate={_truncate(candidate.keys[index])}"
        )
    if pure is True and case_name not in _IMPURE_BY_DEFINITION:
        placeholders = max(baseline.placeholders, candidate.placeholders)
        if placeholders:
            return (
                f"{prefix}: {placeholders} key(s) still carried a random token "
                f"under pure=True, so the exact-key comparison did not happen "
                f"(baseline={baseline.placeholders}, "
                f"candidate={candidate.placeholders})"
            )

    # (2) Canonical graphs, pairwise and in order.
    for index, (left, right) in enumerate(zip(baseline.graphs, candidate.graphs)):
        if left != right:
            return (
                f"{prefix}: canonical graph differs at object {index} "
                f"(key {_truncate(baseline.keys[index])}) -- "
                f"{_first_graph_difference(left, right)}"
            )

    # (3) Results, computed under the synchronous scheduler outside every timed
    # region.
    if baseline.results != candidate.results:
        index = next(
            (
                position
                for position, (left_result, right_result) in enumerate(
                    zip(baseline.results, candidate.results)
                )
                if left_result != right_result
            ),
            min(len(baseline.results), len(candidate.results)),
        )
        return (
            f"{prefix}: computed result differs at object {index} -- "
            f"baseline={_truncate(baseline.results[index])} "
            f"candidate={_truncate(candidate.results[index])}"
        )
    return None


def assert_equivalent(
    case: Case, baseline: ModuleType, live: ModuleType
) -> Equivalence:
    """Prove one case behaves identically under both arms, for both variants.

    Both ``pure`` variants are built and compared: the native ``pure=None``
    keying and ``pure=True``, which is what makes the key comparison an exact
    string comparison. The check is never skipped, sampled or made conditional on
    a flag -- it is the suite's own guard against reporting a speedup for an
    implementation that does something different.

    It reports rather than raises: a mismatch has to end the run with exit status
    2 and a legible message naming the case, the variant, the object and the
    field, not with a traceback. An exception escaping a build or a computation is
    reported the same way, because it too means the two arms could not be shown
    equivalent.

    Args:
        case: The case to check.
        baseline: Arm A.
        live: Arm B.

    Returns:
        The case's :class:`Equivalence` verdict, whose ``mismatch`` is ``None``
        exactly when both variants matched.
    """
    placeholders_pure_true = 0
    matched: dict[bool | None, bool] = {None: False, True: False}
    for pure in (None, True):
        try:
            extracted_baseline = _extract(case, baseline, pure=pure)
            extracted_candidate = _extract(case, live, pure=pure)
            if pure is True:
                placeholders_pure_true = extracted_candidate.placeholders
            mismatch = _compare(
                case.name, extracted_baseline, extracted_candidate, pure=pure
            )
        except Exception as exc:
            return Equivalence(
                case=case.name,
                native=matched[None],
                pure_true=matched[True],
                placeholders_pure_true=placeholders_pure_true,
                mismatch=(
                    f"{case.name} [{_variant_label(pure)}]: equivalence extraction "
                    f"raised {type(exc).__name__}: {exc}"
                ),
            )
        if mismatch is not None:
            return Equivalence(
                case=case.name,
                native=matched[None],
                pure_true=matched[True],
                placeholders_pure_true=placeholders_pure_true,
                mismatch=mismatch,
            )
        matched[pure] = True
    return Equivalence(
        case=case.name,
        native=matched[None],
        pure_true=matched[True],
        placeholders_pure_true=placeholders_pure_true,
        mismatch=None,
    )


# ---------------------------------------------------------------------------
# The paired measurement protocol
# ---------------------------------------------------------------------------


def _ensure_fixed_hash_seed() -> None:
    """Re-execute the runner with ``PYTHONHASHSEED=0`` unless it is already set.

    Hash randomisation changes dict and set iteration order, which changes how
    much work the construction path does over containers, so both arms must run
    under a fixed seed. The seed can only be fixed before the interpreter starts,
    hence the re-exec. It happens before any measurement and at most once: after
    the exec the variable is ``"0"``, so the replacement process returns
    immediately from here.
    """
    if os.environ.get("PYTHONHASHSEED") == "0":
        return
    # The command is rebuilt as ``-m benchmarks.delayed_ab`` rather than from
    # ``sys.argv[0]``, which is the path of ``__main__.py``. Like the original
    # invocation, this requires the repository root as the working directory.
    os.execve(
        sys.executable,
        [sys.executable, "-m", "benchmarks.delayed_ab", *sys.argv[1:]],
        {**os.environ, "PYTHONHASHSEED": "0"},
    )


def _bounded_round_count(value: str, *, minimum: int, flag: str) -> int:
    """Parse a round count and reject anything below the protocol's floor."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{flag} expects an integer number of rounds, got {value!r}"
        )
    if parsed < minimum:
        raise argparse.ArgumentTypeError(
            f"{flag} must be at least {minimum} for the paired protocol to be "
            f"meaningful, got {parsed}"
        )
    return parsed


def _measured_rounds_argument(value: str) -> int:
    """Argparse type for ``--rounds``: at least ``_MIN_ROUNDS``."""
    return _bounded_round_count(value, minimum=_MIN_ROUNDS, flag="--rounds")


def _warmup_rounds_argument(value: str) -> int:
    """Argparse type for ``--warmup``: at least ``_MIN_WARMUP``."""
    return _bounded_round_count(value, minimum=_MIN_WARMUP, flag="--warmup")


def _time_region(case: Case, mod: ModuleType, *, pure: bool | None) -> Region:
    """Measure one ``build`` call of one case under one arm.

    The region is exactly what the protocol prescribes and nothing more. Setup
    runs untimed, the collector is collected and disabled around the region, the
    allocated-block count is read on both sides of it -- the closing read while
    the constructed objects are still alive, which is what makes the delta mean
    anything -- and the ``perf_counter_ns`` brackets contain the ``build`` call
    alone: no setup, no logging, no allocation sampling, not even a ``len()``.

    Args:
        case: The case to build.
        mod: The arm to build it with.
        pure: The ``pure`` value threaded into the case's ``delayed`` calls.

    Returns:
        The region's timing and allocated-block delta.
    """
    with activate(mod):
        state = case.setup(mod, pure=pure)
        gc.collect()
        gc.disable()
        try:
            blocks_before = sys.getallocatedblocks()
            start = time.perf_counter_ns()
            objs = case.build(mod, state, pure=pure)
            end = time.perf_counter_ns()
            blocks_after = sys.getallocatedblocks()
        finally:
            gc.enable()
        # Drop the constructed graph before the next region starts, so no region
        # begins with the previous one's objects still resident.
        del objs, state
    return Region(timing_ns=end - start, blocks_delta=blocks_after - blocks_before)


def time_case(
    case: Case,
    baseline: ModuleType,
    live: ModuleType,
    *,
    warmup: int,
    rounds: int,
    pure: bool | None = None,
) -> tuple[Round, ...]:
    """Run the warmup and measured rounds of one case and return the measured ones.

    A round is the block A, B, B, A -- four regions, two per arm -- so that drift
    inside the round cancels, and consecutive rounds alternate which arm starts,
    so odd-numbered rounds run B, A, A, B. The alternation counts warmup rounds
    too, so it carries on unbroken into the measured ones. The collector is
    collected again between rounds.

    Args:
        case: The case to measure.
        baseline: Arm A.
        live: Arm B.
        warmup: Rounds to run and discard.
        rounds: Rounds to run and keep.
        pure: The variant to time. The suite times each case's native keying.

    Returns:
        The measured rounds, in round order.
    """
    measured: list[Round] = []
    for index in range(warmup + rounds):
        if index % 2 == 0:
            first = _time_region(case, baseline, pure=pure)
            second = _time_region(case, live, pure=pure)
            third = _time_region(case, live, pure=pure)
            fourth = _time_region(case, baseline, pure=pure)
            completed = Round(
                first_arm=_BASELINE,
                baseline=(first, fourth),
                candidate=(second, third),
            )
        else:
            first = _time_region(case, live, pure=pure)
            second = _time_region(case, baseline, pure=pure)
            third = _time_region(case, baseline, pure=pure)
            fourth = _time_region(case, live, pure=pure)
            completed = Round(
                first_arm=_CANDIDATE,
                baseline=(second, third),
                candidate=(first, fourth),
            )
        gc.collect()
        if index >= warmup:
            measured.append(completed)
    return tuple(measured)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def arm_stats(values: Sequence[float]) -> dict[str, float]:
    """Summarise one arm's timings over all its measured regions.

    Args:
        values: The arm's region timings, two per measured round.

    Returns:
        ``median``, ``min``, ``p95`` and ``iqr``. ``p95`` is
        ``statistics.quantiles(values, n=20)[18]`` and ``iqr`` is the third minus
        the first quartile of ``statistics.quantiles(values, n=4)``, both of which
        need at least two data points -- the enforced minimum of seven rounds
        supplies fourteen, and the degenerate shapes are handled here rather than
        left to raise.
    """
    if not values:
        # Unreachable while ``--rounds`` is floored at seven; kept so that the
        # function is total and never puts a NaN into the artefact.
        return {"median": 0.0, "min": 0.0, "p95": 0.0, "iqr": 0.0}
    if len(values) == 1:
        only = float(values[0])
        return {"median": only, "min": only, "p95": only, "iqr": 0.0}
    quartiles = statistics.quantiles(values, n=4)
    return {
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "p95": float(statistics.quantiles(values, n=20)[18]),
        "iqr": float(quartiles[2] - quartiles[0]),
    }


def bootstrap_ci(
    ratios: Sequence[float],
    *,
    resamples: int = _BOOTSTRAP_RESAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile-bootstrap 95% confidence interval on the median paired ratio.

    ``resamples`` resamples of the same size as the input are drawn with
    replacement from the per-round ratios, the median of each is taken, and the
    interval endpoints are read off the sorted resample medians with a fixed index
    convention: ``low = ordered[int(0.025 * resamples)]`` and
    ``high = ordered[int(0.975 * resamples) - 1]``. That convention is part of the
    committed evidence and does not change.

    With seven measured rounds the endpoints are order statistics of those seven
    ratios -- coarse, but well defined; the fifteen-round default tightens the
    interval. A dedicated ``random.Random`` seeded per call keeps the interval
    reproducible and never touches the global random state.

    Args:
        ratios: The per-round paired ratios.
        resamples: Bootstrap resamples to draw.
        seed: Seed of the local generator.

    Returns:
        ``(low, high)``. Degenerate inputs collapse to the only value available,
        so the artefact never carries a NaN.
    """
    if not ratios:
        return 0.0, 0.0
    if len(ratios) == 1 or resamples < 1:
        only = float(ratios[0])
        return only, only
    rng = random.Random(seed)
    population = list(ratios)
    size = len(population)
    medians = sorted(
        statistics.median(rng.choices(population, k=size)) for _ in range(resamples)
    )
    low = medians[int(_CI_LOWER_PERCENTILE * resamples)]
    high = medians[int(_CI_UPPER_PERCENTILE * resamples) - 1]
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Allocation measurement. Every figure here comes from an untimed round: tracing
# and profiling slow execution several times over, so none of this may ever run
# inside a timed region.
# ---------------------------------------------------------------------------


def _tracemalloc_round(
    case: Case, mod: ModuleType, *, pure: bool | None
) -> tuple[int, int]:
    """Measure peak traced bytes and surviving traced blocks of one build.

    The peak is read from ``tracemalloc.get_traced_memory()[1]`` after
    ``reset_peak()`` was called immediately before the build, so it is the peak of
    the build alone and not of the process. It is the only gate-bearing
    allocation figure. The surviving-block figure is the end snapshot's trace
    count minus the baseline snapshot's, taken while the constructed objects are
    still alive -- it counts what the build left behind, and it is not a peak.

    Args:
        case: The case to build.
        mod: The arm to build it with.
        pure: The ``pure`` value threaded into the case's ``delayed`` calls.

    Returns:
        ``(tracemalloc_peak_bytes, live_blocks_end)``.
    """
    with activate(mod):
        state = case.setup(mod, pure=pure)
        gc.collect()
        tracemalloc.start()
        try:
            baseline_snapshot = tracemalloc.take_snapshot()
            baseline_traces = len(baseline_snapshot.traces)
            tracemalloc.reset_peak()
            objs = case.build(mod, state, pure=pure)
            peak_bytes = tracemalloc.get_traced_memory()[1]
            end_snapshot = tracemalloc.take_snapshot()
            live_blocks_end = len(end_snapshot.traces) - baseline_traces
        finally:
            tracemalloc.stop()
        del objs, state, baseline_snapshot, end_snapshot
    return peak_bytes, live_blocks_end


def _sampled_max_blocks(case: Case, mod: ModuleType, *, pure: bool | None) -> int:
    """Sample ``sys.getallocatedblocks()`` through a profile hook during one build.

    The result is a *sampled lower bound* on the region's peak block delta and is
    never labelled a peak. ``sys.getallocatedblocks()`` walks the allocator's
    pools, so reading it on every profile event costs roughly forty times the
    region it observes; it is therefore read every ``_BLOCK_SAMPLE_STRIDE``th
    event, which measured within 0.5% of the every-event figure at a fortieth of
    the cost. The maximum is additionally floored by the count taken immediately
    after the region, while the constructed objects are still alive, so the bound
    can never come out below what the build demonstrably left allocated.

    Both arms are sampled identically, which is what keeps the two figures
    comparable.

    Args:
        case: The case to build.
        mod: The arm to build it with.
        pure: The ``pure`` value threaded into the case's ``delayed`` calls.

    Returns:
        The sampled maximum block count minus the count taken before the region.
    """
    with activate(mod):
        state = case.setup(mod, pure=pure)
        get_blocks = sys.getallocatedblocks
        sampled = 0
        events = 0

        def hook(frame: Any, event: str, arg: Any) -> None:
            """Update the sampled maximum; never raise, whatever happens."""
            nonlocal sampled, events
            try:
                events += 1
                if events % _BLOCK_SAMPLE_STRIDE == 0:
                    current = get_blocks()
                    if current > sampled:
                        sampled = current
            except Exception:
                # An exception inside a profile hook silently disables profiling,
                # which would turn a supporting figure into a silent hole. There
                # is nothing here that can raise, and this makes sure of it.
                pass

        gc.collect()
        gc.disable()
        try:
            before = get_blocks()
            sys.setprofile(hook)
            try:
                objs = case.build(mod, state, pure=pure)
            finally:
                sys.setprofile(None)
            after = get_blocks()
        finally:
            gc.enable()
        del objs, state
    return max(sampled, after) - before


def measure_allocations(
    case: Case,
    baseline: ModuleType,
    live: ModuleType,
    *,
    pure: bool | None = None,
) -> dict[str, ArmAllocation]:
    """Collect both arms' allocation figures for one case, in untimed rounds.

    Args:
        case: The case to measure.
        baseline: Arm A.
        live: Arm B.
        pure: The variant to measure -- the same one the timing rounds used.

    Returns:
        The arm-keyed allocation figures.
    """
    figures: dict[str, ArmAllocation] = {}
    for arm, mod in ((_BASELINE, baseline), (_CANDIDATE, live)):
        peak_bytes, live_blocks_end = _tracemalloc_round(case, mod, pure=pure)
        figures[arm] = ArmAllocation(
            tracemalloc_peak_bytes=peak_bytes,
            live_blocks_end=live_blocks_end,
            max_observed_blocks=_sampled_max_blocks(case, mod, pure=pure),
        )
    return figures


# ---------------------------------------------------------------------------
# Environment and provenance
# ---------------------------------------------------------------------------


def _probe_target(x: int) -> int:
    """A plain function: the subject of the ``is_dask_collection`` probe.

    It is a module-level function rather than a lambda or a local so that the
    micro-benchmark measures the probe against exactly the kind of object
    ``delayed(f)`` wraps in the ``flat_loop`` case.
    """
    return x


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _repository_root() -> pathlib.Path:
    """Return the repository root, derived from this file rather than the cwd.

    ``benchmarks/delayed_ab/main.py`` sits two directories below the root, so the
    root is reachable without trusting the working directory -- which matters for
    the provenance commands and for the inside-the-worktree test that guards the
    artefacts.
    """
    return pathlib.Path(__file__).resolve().parents[2]


def _git(root: pathlib.Path, *args: str) -> tuple[str | None, str | None]:
    """Run one git command and return ``(stdout, note)``.

    Provenance must never crash the run, so every failure -- a missing git
    executable, a non-zero exit, a directory that is not a repository -- comes
    back as ``(None, note)`` and the note is recorded in the environment block.

    Args:
        root: Repository to run the command in.
        *args: The git arguments, without the executable.

    Returns:
        The command's stdout and ``None``, or ``None`` and an explanatory note.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return None, f"git could not be executed ({type(exc).__name__}: {exc})"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        return None, f"`git {' '.join(args)}` failed: {detail}"
    return completed.stdout, None


def repository_state() -> RepositoryState:
    """Read the candidate arm's provenance, before any output file is written.

    The dirty flag is deliberately evaluated at start-up: the two artefacts the
    runner writes live inside the repository by default, so inspecting the tree
    afterwards would count its own output. When the tree cannot be inspected at
    all, the run is treated as clean and a note says so -- refusing to measure
    over a missing git executable would be a worse failure than recording that
    the flag is unknown.

    Returns:
        The repository root, ``git rev-parse HEAD``, the dirty flag with the
        porcelain lines behind it, the sha256 of ``dask/delayed.py`` and any note
        about something that degraded.
    """
    root = _repository_root()
    notes: list[str] = []

    head, note = _git(root, "rev-parse", "HEAD")
    if note is not None:
        notes.append(note)

    status, note = _git(root, "status", "--porcelain", "--untracked-files=all")
    if note is not None:
        notes.append(note)
    if status is None:
        dirty = False
        dirty_paths: tuple[str, ...] = ()
        notes.append(
            "the working tree could not be inspected, so the dirty-tree refusal "
            "did not run and the artefacts may not describe a clean commit"
        )
    else:
        dirty_paths = tuple(line for line in status.splitlines() if line.strip())
        dirty = bool(dirty_paths)

    delayed_py = root / "dask" / "delayed.py"
    try:
        digest: str | None = hashlib.sha256(delayed_py.read_bytes()).hexdigest()
    except OSError as exc:
        digest = None
        notes.append(f"sha256({delayed_py}) unavailable: {type(exc).__name__}: {exc}")

    return RepositoryState(
        root=root,
        git_head=head.strip() if head is not None else None,
        dirty=dirty,
        dirty_paths=dirty_paths,
        delayed_py_sha256=digest,
        notes=tuple(notes),
    )


def _cpu_model() -> str:
    """Return the CPU model: ``/proc/cpuinfo`` on Linux, ``platform`` elsewhere.

    The file is read without a platform test and its absence is simply a miss, so
    this never fails on a system that does not provide it.
    """
    try:
        cpuinfo = pathlib.Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        if line.lower().startswith("model name"):
            _, _, value = line.partition(":")
            if value.strip():
                return value.strip()
    return platform.processor() or "unknown"


def _distributions() -> dict[str, str]:
    """Return the complete resolved distribution name to version map.

    A distribution whose metadata cannot be read is skipped rather than allowed
    to abort the run: the map is provenance, not a measurement.
    """
    resolved: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        try:
            name = dist.metadata["Name"]
            version = dist.version
        except Exception:
            continue
        if not name:
            continue
        resolved.setdefault(str(name), str(version))
    return dict(sorted(resolved.items(), key=lambda item: item[0].lower()))


def _baseline_arm() -> dict[str, Any]:
    """Describe arm A, reading its commit SHA out of the capture's own header.

    The SHA in the header is the authority -- it says which commit the capture was
    taken from -- and it is cross-checked against ``_BASELINE_SHA``. A
    disagreement is recorded rather than raised, because it is a provenance
    finding for the reader, not a reason to abandon a valid measurement.
    """
    capture = pathlib.Path(__file__).with_name("baseline_delayed.py")
    sha: str | None = None
    note: str | None = None
    try:
        header = capture.read_text(encoding="utf-8")
    except OSError as exc:
        note = f"the capture could not be read: {type(exc).__name__}: {exc}"
    else:
        match = re.search(
            r"^#\s*Source commit:\s*([0-9a-f]{40})\s*$", header, re.MULTILINE
        )
        if match is None:
            note = "the capture's header carries no `# Source commit: <sha>` line"
        else:
            sha = match.group(1)
            if sha != _BASELINE_SHA:
                note = (
                    f"the capture's header records {sha}, which is not the "
                    f"expected {_BASELINE_SHA}"
                )
    return {
        "module": "benchmarks.delayed_ab.baseline_delayed",
        "sha": sha,
        "expected_sha": _BASELINE_SHA,
        "sha_matches_expected": sha == _BASELINE_SHA,
        "note": note,
    }


def _rejected_optimizations(flat_loop_median_ns: float | None) -> dict[str, Any]:
    """Re-measure the cost of the one probe the refactor deliberately kept.

    ``delayed()`` evaluates ``is_dask_collection(obj) or traverse`` in that order.
    Reordering it would skip the probe whenever ``traverse`` is true, but for a
    user object that exposes the collection protocol the first probe is
    observable -- it reads ``x.expr`` / calls ``__dask_graph__()`` before
    ``unpack_collections`` does -- so reordering could change warning counts,
    mutations or exceptions in user wrappers. The optimization is therefore not
    taken, and this is where the measured cost of not taking it enters the
    artefacts instead of living only in the plan.

    Args:
        flat_loop_median_ns: The candidate's median ``flat_loop`` region, used to
            express the probe as a share of one construction. ``None`` leaves the
            derived fields null.

    Returns:
        The record for ``environment.rejected_optimizations``.
    """
    elapsed = timeit.timeit(
        "is_dask_collection(target)",
        number=_MICROBENCH_ITERATIONS,
        globals={"is_dask_collection": is_dask_collection, "target": _probe_target},
    )
    ns_per_call = elapsed * 1e9 / _MICROBENCH_ITERATIONS
    per_construction_ns: float | None = None
    share: float | None = None
    if flat_loop_median_ns:
        per_construction_ns = flat_loop_median_ns / _FLAT_LOOP_CONSTRUCTIONS
        if per_construction_ns > 0:
            share = ns_per_call / per_construction_ns
    return {
        "traverse_probe_order": {
            "decision": "not reordered",
            "site": "dask/delayed.py -- `if is_dask_collection(obj) or traverse:`",
            "reason": (
                "for a non-Delayed object exposing the collection protocol the "
                "first probe calls x.expr / __dask_graph__() once before "
                "unpack_collections calls it again, so dropping it could change "
                "warning counts, mutation or exceptions in user wrappers"
            ),
            "ns_per_call": ns_per_call,
            "flat_loop_ns_per_construction": per_construction_ns,
            "share_of_flat_loop_construction": share,
            "method": (
                "timeit.timeit('is_dask_collection(target)', "
                f"number={_MICROBENCH_ITERATIONS}) against a plain module-level "
                "function, in this process and this environment"
            ),
        }
    }


def environment(
    *,
    repository: RepositoryState | None = None,
    pythonhashseed: str | None = None,
    flat_loop_median_ns: float | None = None,
) -> dict[str, Any]:
    """Assemble the environment block that both artefacts carry.

    Args:
        repository: Provenance read at start-up. When omitted it is read now,
            which is only correct before any output file exists.
        pythonhashseed: The effective seed. When omitted the live value of
            ``PYTHONHASHSEED`` is recorded.
        flat_loop_median_ns: The candidate's median ``flat_loop`` region, for the
            rejected-optimization cost record.

    Returns:
        A JSON-serialisable description of the interpreter, the machine, the
        resolved dependency set, both arms' provenance, the cost record of the
        rejected probe reordering and the standing peak-block-count conflict.
    """
    state = repository if repository is not None else repository_state()
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    distributions = _distributions()
    lowered = {name.lower(): version for name, version in distributions.items()}
    # ``hashers`` is a list of plain functions; ``getattr`` keeps the lookup
    # tolerant of any callable that carries no ``__name__``.
    active_hasher = (
        getattr(hashers[0], "__name__", repr(hashers[0])) if hashers else None
    )
    return {
        "python_version": platform.python_version(),
        "sys_version": sys.version,
        "implementation": {
            "name": sys.implementation.name,
            "version": ".".join(str(part) for part in sys.implementation.version[:3]),
            "cache_tag": sys.implementation.cache_tag,
        },
        "free_threading": {
            "py_gil_disabled": sysconfig.get_config_var("Py_GIL_DISABLED"),
            "gil_enabled": gil_probe() if gil_probe is not None else None,
        },
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "pythonhashseed": (
            pythonhashseed
            if pythonhashseed is not None
            else os.environ.get("PYTHONHASHSEED")
        ),
        "hasher": active_hasher,
        "distributions": distributions,
        "packages": {name: lowered.get(name) for name in _KEY_PACKAGES},
        "repository_root": str(state.root),
        "arms": {
            _BASELINE: _baseline_arm(),
            _CANDIDATE: {
                "module": "dask.delayed",
                "git_head": state.git_head,
                "dirty": state.dirty,
                "delayed_py_sha256": state.delayed_py_sha256,
            },
        },
        "generated_at": _utc_now_iso(),
        "rejected_optimizations": _rejected_optimizations(flat_loop_median_ns),
        "peak_block_count_conflict": _PEAK_BLOCK_COUNT_CONFLICT,
        "notes": list(state.notes),
    }


# ---------------------------------------------------------------------------
# Artefacts: the JSON payload, its report, and the calibration block
# ---------------------------------------------------------------------------


class CalibrationError(RuntimeError):
    """``--calibration`` named a file that could not be used.

    Raised with a message for the operator instead of letting a traceback out.
    """


def _load_calibration(path: pathlib.Path) -> dict[str, Any]:
    """Load an earlier A/A run's per-case ratio medians and intervals.

    Only the figures the report and the ``calibration`` block need are kept, so a
    calibration file from a different schema version is still usable as long as it
    carries per-case ratios.

    Args:
        path: The A/A JSON written by an earlier run, normally outside the
            checkout so that run left the working tree clean.

    Returns:
        The calibration block: the source path, its timestamp and the per-case
        ratio median with its interval, for cases and sub-series.

    Raises:
        CalibrationError: If the file cannot be read, is not JSON, or carries no
            per-case ratio figures.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalibrationError(
            f"--calibration {path}: cannot be read ({type(exc).__name__}: {exc})"
        )
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"--calibration {path}: is not valid JSON ({exc})")
    if not isinstance(raw, dict):
        raise CalibrationError(
            f"--calibration {path}: expected a JSON object at the top level, got "
            f"{type(raw).__name__}"
        )

    def figures(section: object) -> dict[str, Any]:
        collected: dict[str, Any] = {}
        if isinstance(section, dict):
            for name, entry in section.items():
                if isinstance(entry, dict) and "ratio_median" in entry:
                    collected[str(name)] = {
                        "ratio_median": entry.get("ratio_median"),
                        "ci_low": entry.get("ci_low"),
                        "ci_high": entry.get("ci_high"),
                    }
        return collected

    cases = figures(raw.get("cases"))
    if not cases:
        raise CalibrationError(
            f"--calibration {path}: carries no per-case ratio figures under "
            f"'cases', so it is not an A/B suite result file"
        )
    return {
        "path": str(path),
        "generated_at": raw.get("generated_at"),
        "cases": cases,
        "subseries": figures(raw.get("subseries")),
    }


def _case_payload(result: CaseResult, failed: Sequence[str]) -> dict[str, Any]:
    """Serialise one case exactly as the JSON schema fixes it.

    The key names here are the parsing contract of
    ``dask/tests/test_delayed_ab_gate.py`` and of any later reader of the
    committed artefact, so they do not change.

    Args:
        result: The case's measurements and statistics.
        failed: The gate thresholds this case missed, already formatted. Empty
            for a passing case and for every informational sub-series.

    Returns:
        The case's JSON object.
    """
    if not result.gated:
        verdict = "INFORMATIONAL"
    else:
        verdict = "FAIL" if failed else "PASS"
    return {
        "name": result.name,
        "pure": result.pure,
        "timings_ns": {
            _BASELINE: list(result.timings(_BASELINE)),
            _CANDIDATE: list(result.timings(_CANDIDATE)),
        },
        "allocated_blocks_delta": {
            _BASELINE: list(result.blocks(_BASELINE)),
            _CANDIDATE: list(result.blocks(_CANDIDATE)),
        },
        "round_timings_ns": [
            {
                "a1": round_.baseline[0].timing_ns,
                "b1": round_.candidate[0].timing_ns,
                "b2": round_.candidate[1].timing_ns,
                "a2": round_.baseline[1].timing_ns,
                "first_arm": round_.first_arm,
                "ratio": round_.ratio,
            }
            for round_ in result.rounds
        ],
        "ratios": list(result.ratios),
        "ratio_median": result.ratio_median,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "stats": {
            _BASELINE: arm_stats(result.timings(_BASELINE)),
            _CANDIDATE: arm_stats(result.timings(_CANDIDATE)),
        },
        "allocation": {
            _BASELINE: result.baseline_allocation.payload(),
            _CANDIDATE: result.candidate_allocation.payload(),
            "peak_bytes_ratio": result.peak_bytes_ratio,
        },
        "equivalence": result.equivalence.payload(),
        "verdict": verdict,
        "failed_thresholds": list(failed),
    }


def build_payload(
    results: Sequence[CaseResult],
    *,
    warmup: int,
    rounds: int,
    gate_passed: bool,
    checks: Sequence[GateCheck],
    environment_block: dict[str, Any],
    calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the complete JSON payload of one run.

    Args:
        results: Every measured case, gated and informational alike.
        warmup: Discarded rounds per case.
        rounds: Measured rounds per case.
        gate_passed: Whether every gate item held.
        checks: The gate checklist, in checklist order.
        environment_block: The block from :func:`environment`.
        calibration: The loaded A/A block, or ``None`` when ``--calibration`` was
            not given.

    Returns:
        The payload that ``write_json`` serialises and ``write_report`` renders.
    """
    cases: dict[str, Any] = {}
    subseries: dict[str, Any] = {}
    for result in results:
        failed = _failed_thresholds(result) if result.gated else ()
        target = cases if result.gated else subseries
        target[result.name] = _case_payload(result, failed)
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "rounds": {"warmup": warmup, "measured": rounds},
        "verdict": "PASS" if gate_passed else "FAIL",
        "gate": {
            "passed": gate_passed,
            "checks": [check.payload() for check in checks],
        },
        "environment": environment_block,
        "cases": cases,
        "subseries": subseries,
        "calibration": calibration,
    }


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Write the payload as sorted, indented JSON with a trailing newline.

    The trailing newline matters: the repository's ``end-of-file-fixer``
    pre-commit hook covers the committed artefact too.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _format_ms(nanoseconds: float) -> str:
    """Render a nanosecond figure as milliseconds."""
    return f"{nanoseconds / 1e6:.3f}"


def _format_ci(low: float, high: float) -> str:
    """Render a confidence interval."""
    return f"[{low:.3f}, {high:.3f}]"


def _format_peak_delta(ratio: float) -> str:
    """Render a peak-allocation ratio as a signed percentage change."""
    return f"{(ratio - 1.0) * 100:+.1f}"


def _case_table(payload: dict[str, Any], section: str, *, verdicts: bool) -> list[str]:
    """Render one markdown table of case rows.

    Args:
        payload: The run payload.
        section: ``"cases"`` or ``"subseries"``.
        verdicts: Whether to carry the verdict column. The informational
            sub-series table has none, because they take no part in the gate.

    Returns:
        The table's lines, header included.
    """
    header = [
        "case",
        "baseline median (ms)",
        "candidate median (ms)",
        "paired ratio median",
        "95% CI",
        "peak-allocation delta (%)",
    ]
    if verdicts:
        header.append("verdict")
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for name, case in payload[section].items():
        row = [
            name,
            _format_ms(case["stats"][_BASELINE]["median"]),
            _format_ms(case["stats"][_CANDIDATE]["median"]),
            f"{case['ratio_median']:.3f}",
            _format_ci(case["ci_low"], case["ci_high"]),
            _format_peak_delta(case["allocation"]["peak_bytes_ratio"]),
        ]
        if verdicts:
            failed = case["failed_thresholds"]
            row.append(
                f"{case['verdict']} ({'; '.join(failed)})"
                if failed
                else case["verdict"]
            )
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_report(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Render the human-readable report of one run.

    The report carries the environment summary, the peak-block-count conflict
    note, one "A/A calibration" line per case when a calibration file was given,
    the gate checklist, a table of the gated cases whose verdict cell names the
    threshold a failing case missed and by how much, a second table of the
    informational sub-series without a verdict column, and a single closing
    ``OVERALL:`` line. Every number in it is measured by the run that writes it.
    """
    env = payload["environment"]
    rounds = payload["rounds"]
    baseline_arm = env["arms"][_BASELINE]
    candidate_arm = env["arms"][_CANDIDATE]
    probe = env["rejected_optimizations"]["traverse_probe_order"]
    packages = env["packages"]
    lines: list[str] = [
        "# dask.delayed A/B performance report",
        "",
        "Arm A is the frozen pre-refactor capture "
        "`benchmarks/delayed_ab/baseline_delayed.py`; arm B is the live "
        "`dask.delayed`. Both were measured in one interpreter, each under "
        'activation as `sys.modules["dask.delayed"]`, and every case was proven '
        "equivalent -- keys, canonical graphs and computed results, under both "
        "`pure=None` and `pure=True` -- before a single timing was recorded.",
        "",
        "## Environment",
        "",
        f"- Generated (UTC): {payload['generated_at']}",
        f"- Schema version: {payload['schema_version']}",
        f"- Rounds: {rounds['warmup']} warmup + {rounds['measured']} measured; "
        "each round is A,B,B,A with the starting arm alternating, and the paired "
        "ratio is (A1+A2)/(B1+B2), baseline over candidate",
        f"- Interpreter: {env['implementation']['name']} {env['python_version']} "
        f"({env['sys_version'].splitlines()[0].strip()})",
        f"- Free-threaded build: Py_GIL_DISABLED="
        f"{env['free_threading']['py_gil_disabled']}, "
        f"sys._is_gil_enabled()={env['free_threading']['gil_enabled']}",
        f"- Platform: {env['platform']}",
        f"- CPU: {env['cpu_model']}",
        f"- PYTHONHASHSEED: {env['pythonhashseed']}",
        f"- Active tokenize hasher: {env['hasher']} (it changes every "
        "pickle-sensitive token, so it is part of the result)",
        "- Key packages: "
        + ", ".join(f"{name} {packages.get(name)}" for name in _KEY_PACKAGES),
        f"- Arm A provenance: commit {baseline_arm['sha']} per the capture's own "
        f"header (expected {baseline_arm['expected_sha']}, matches="
        f"{baseline_arm['sha_matches_expected']})",
        f"- Arm B provenance: git HEAD {candidate_arm['git_head']}, "
        f"dirty={candidate_arm['dirty']}, "
        f"sha256(dask/delayed.py)={candidate_arm['delayed_py_sha256']}",
        "- The arm B SHA identifies the commit whose `dask/delayed.py` was "
        "measured, not the commit that adds these artefacts: the runner refuses "
        "to write inside a dirty working tree, so the sources are committed "
        "first, the runner is executed from that clean commit, and these two "
        "files are committed afterwards.",
        f"- Rejected optimization on record: the `is_dask_collection(obj) or "
        f"traverse` probe order in `delayed()` was {probe['decision']} -- "
        f"{probe['reason']}. Measured cost of keeping it: "
        f"{probe['ns_per_call']:.1f} ns per probe against "
        f"{_format_ratio_or_none(probe['flat_loop_ns_per_construction'])} ns per "
        f"`flat_loop` construction "
        f"({_format_share(probe['share_of_flat_loop_construction'])}), by "
        f"{probe['method']}.",
        f"- Peak block count: {env['peak_block_count_conflict']}",
    ]
    for note in env["notes"]:
        lines.append(f"- Note: {note}")

    calibration = payload["calibration"]
    if calibration is not None:
        lines.append(
            f"- A/A calibration source: {calibration['path']} "
            f"(generated {calibration['generated_at']}). The A/A ratios are the "
            "noise floor of this machine; they are diagnostic, not a gate."
        )
        for section in ("cases", "subseries"):
            for name, figures in calibration[section].items():
                lines.append(
                    f"- A/A calibration ({name}): ratio median "
                    f"{_format_ratio_or_none(figures['ratio_median'])}, 95% CI "
                    f"[{_format_ratio_or_none(figures['ci_low'])}, "
                    f"{_format_ratio_or_none(figures['ci_high'])}]"
                    f"{_calibration_bias_note(figures)}"
                )

    lines.extend(["", "## Gate checklist", ""])
    for check in payload["gate"]["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        target = f" [{check['case']}]" if check["case"] else ""
        lines.append(
            f"- {status} {check['name']}{target}: measured "
            f"{_format_measurement(check['measured'])}, required "
            f"{_check_relation(check['name'])} "
            f"{_format_measurement(check['threshold'])}"
        )

    lines.extend(["", "## Gated cases", ""])
    lines.extend(_case_table(payload, "cases", verdicts=True))
    lines.extend(["", "## Informational sub-series (not gated)", ""])
    lines.extend(_case_table(payload, "subseries", verdicts=False))
    lines.extend(["", f"OVERALL: {payload['verdict']}", ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_ratio_or_none(value: object) -> str:
    """Render a float figure, or ``n/a`` when it could not be derived."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.3f}"
    return "n/a"


def _numeric(value: object) -> float | None:
    """Return a JSON number as a ``float``, or ``None`` for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _calibration_bias_note(figures: dict[str, Any]) -> str:
    """Flag an A/A case whose paired ratio is not centred on 1.0.

    An A/A run compares the baseline against itself, so its ratios are the noise
    floor and should straddle 1.0. A case whose median is more than
    ``_CALIBRATION_DEVIATION`` away from 1.0, or whose interval excludes 1.0
    altogether, is measuring something other than the implementations -- arm or
    order bias on that machine -- and says so in the report's environment section
    for the reviewer to weigh. It is diagnostic: no gate item depends on it.

    Args:
        figures: One case's ``ratio_median``, ``ci_low`` and ``ci_high`` from the
            calibration file.

    Returns:
        The note to append to that case's calibration line, empty when the case
        is centred on 1.0.
    """
    median = _numeric(figures.get("ratio_median"))
    low = _numeric(figures.get("ci_low"))
    high = _numeric(figures.get("ci_high"))
    reasons: list[str] = []
    if median is not None and abs(median - 1.0) > _CALIBRATION_DEVIATION:
        reasons.append(
            f"the median deviates from 1.0 by more than "
            f"{_CALIBRATION_DEVIATION * 100:.0f}%"
        )
    if low is not None and high is not None and (low > 1.0 or high < 1.0):
        reasons.append("the interval excludes 1.0")
    if not reasons:
        return ""
    return f" -- arm/order bias to weigh: {'; '.join(reasons)}"


def _format_share(value: object) -> str:
    """Render a share of one construction as a percentage, or ``n/a``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value) * 100:.2f}% of one construction"
    return "share n/a"


def _resolve_output(value: str | None) -> pathlib.Path:
    """Resolve the output directory to an absolute path.

    The default lives inside the repository and is resolved against the
    repository root, not the working directory, so it lands in the same place
    however the runner was started. An explicit ``--output`` is resolved against
    the working directory, which is what makes ``--output /tmp/...`` -- the A/A
    calibration idiom that keeps the tree clean -- behave as written.
    """
    if value is None:
        return _repository_root() / _DEFAULT_OUTPUT
    return pathlib.Path(value).expanduser().resolve()


def _dirty_tree_refusal(
    output: pathlib.Path, state: RepositoryState
) -> tuple[str, ...] | None:
    """Return the offending paths when artefacts must not be written, else ``None``.

    Writing inside the repository is refused while the working tree is dirty,
    because a committed artefact has to describe an identifiable commit: the
    ``git HEAD`` it records would otherwise not be the code that was measured.
    Writing into a directory outside the checkout is always allowed -- that is how
    the A/A calibration run happens before the first commit without dirtying
    anything.

    Args:
        output: The resolved output directory.
        state: Provenance read at start-up, before any file was written.

    Returns:
        The porcelain lines behind the refusal, or ``None`` when writing is fine.
    """
    if not state.dirty:
        return None
    try:
        inside = output.resolve().is_relative_to(state.root)
    except OSError:
        # An unresolvable path is not inside the tree in any useful sense.
        inside = False
    if not inside:
        return None
    return state.dirty_paths


# ---------------------------------------------------------------------------
# The gate, its checklist and the command line
# ---------------------------------------------------------------------------

#: Artefact file names. They are part of the parsing contract of
#: ``dask/tests/test_delayed_ab_gate.py``.
_JSON_NAME = "baseline_vs_candidate.json"
_REPORT_NAME = "report.md"

#: The case whose median expresses the cost of one construction. It belongs with
#: ``_FLAT_LOOP_CONSTRUCTIONS``: the divisor is that case's frozen size.
_FLAT_LOOP_CASE = "flat_loop"

#: The relation each checklist item asserts between its measured value and its
#: threshold, for legible output. It is presentation only; the verdict comes from
#: the check itself.
_CHECK_RELATIONS = {
    "ratio": ">=",
    "ci_lower": ">",
    "improved_case_count": ">=",
    "no_regression": ">=",
    "peak_allocation": "<=",
    "equivalence": "==",
}


def _format_measurement(value: float) -> str:
    """Render a measured value: whole numbers stay whole, ratios get decimals."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}"


def _check_relation(name: str) -> str:
    """Return the relation a checklist item asserts, by item name."""
    for prefix, relation in _CHECK_RELATIONS.items():
        if name == prefix or name.startswith(f"{prefix}_"):
            return relation
    return "=="


def _failed_thresholds(result: CaseResult) -> tuple[str, ...]:
    """Name every gate threshold one case missed, and by how much.

    A failing case is reported with its cause named and stays in the corpus; it is
    never dropped, resized or re-parameterised. The paired-ratio and
    confidence-interval floors apply only to the cases in ``RATIO_CASES``; the
    regression floor and the allocation tolerance apply to every gated case.

    Args:
        result: The gated case to judge.

    Returns:
        The formatted failures, empty when the case passed everything that
        applies to it.
    """
    failures: list[str] = []
    if result.name in RATIO_CASES:
        if result.ratio_median < _RATIO_THRESHOLD:
            failures.append(
                f"ratio_median {result.ratio_median:.3f} < {_RATIO_THRESHOLD:.2f} "
                f"(short by {_RATIO_THRESHOLD - result.ratio_median:.3f})"
            )
        if result.ci_low <= _CI_LOWER_FLOOR:
            failures.append(
                f"ci_low {result.ci_low:.3f} <= {_CI_LOWER_FLOOR:.2f} "
                f"(short by {_CI_LOWER_FLOOR - result.ci_low:.3f})"
            )
    if result.ci_high < _REGRESSION_CI_UPPER:
        failures.append(
            f"ci_high {result.ci_high:.3f} < {_REGRESSION_CI_UPPER:.2f}, a measured "
            f"regression (by {_REGRESSION_CI_UPPER - result.ci_high:.3f})"
        )
    if result.peak_bytes_ratio > _PEAK_ALLOC_TOLERANCE:
        failures.append(
            f"peak_bytes_ratio {result.peak_bytes_ratio:.3f} > "
            f"{_PEAK_ALLOC_TOLERANCE:.2f} "
            f"(over by {result.peak_bytes_ratio - _PEAK_ALLOC_TOLERANCE:.3f})"
        )
    if not result.equivalence.ok:
        # Unreachable in a produced artefact: an equivalence mismatch ends the run
        # with status 2 before any gate verdict exists. Kept so that the per-case
        # record can never claim a pass it did not earn.
        failures.append("equivalence mismatch")
    return tuple(failures)


def evaluate_gate(
    results: Sequence[CaseResult],
) -> tuple[bool, tuple[GateCheck, ...]]:
    """Evaluate the gate over the measured cases.

    The eight items, in checklist order: the paired-ratio floor and the interval
    floor for each of the two cases named by ``RATIO_CASES`` (taken from
    ``cases.py``, never from a list repeated here), the count of gated cases whose
    interval clears 1.0, the absence of a measured regression, the peak-allocation
    tolerance, and the equivalence of every case. The informational sub-series
    take part in none of them.

    Args:
        results: Every measured case, gated and informational.

    Returns:
        Whether every item held, and the checklist in order.
    """
    gated = [result for result in results if result.gated]
    by_name = {result.name: result for result in gated}
    checks: list[GateCheck] = []

    for name in RATIO_CASES:
        result = by_name.get(name)
        if result is None:
            # A gated case named by the gate but absent from the corpus is a
            # failure of the run, not something to pass over silently.
            checks.append(
                GateCheck(
                    name=f"ratio_{name}",
                    case=name,
                    measured=0.0,
                    threshold=_RATIO_THRESHOLD,
                    passed=False,
                )
            )
            checks.append(
                GateCheck(
                    name=f"ci_lower_{name}",
                    case=name,
                    measured=0.0,
                    threshold=_CI_LOWER_FLOOR,
                    passed=False,
                )
            )
            continue
        checks.append(
            GateCheck(
                name=f"ratio_{name}",
                case=name,
                measured=result.ratio_median,
                threshold=_RATIO_THRESHOLD,
                passed=result.ratio_median >= _RATIO_THRESHOLD,
            )
        )
        checks.append(
            GateCheck(
                name=f"ci_lower_{name}",
                case=name,
                measured=result.ci_low,
                threshold=_CI_LOWER_FLOOR,
                passed=result.ci_low > _CI_LOWER_FLOOR,
            )
        )

    improved = sum(1 for result in gated if result.ci_low > _CI_LOWER_FLOOR)
    checks.append(
        GateCheck(
            name="improved_case_count",
            case=None,
            measured=float(improved),
            threshold=float(_MIN_IMPROVED_CASES),
            passed=improved >= _MIN_IMPROVED_CASES,
        )
    )

    if gated:
        weakest = min(gated, key=lambda result: result.ci_high)
        checks.append(
            GateCheck(
                name="no_regression",
                case=weakest.name,
                measured=weakest.ci_high,
                threshold=_REGRESSION_CI_UPPER,
                passed=weakest.ci_high >= _REGRESSION_CI_UPPER,
            )
        )
        heaviest = max(gated, key=lambda result: result.peak_bytes_ratio)
        checks.append(
            GateCheck(
                name="peak_allocation",
                case=heaviest.name,
                measured=heaviest.peak_bytes_ratio,
                threshold=_PEAK_ALLOC_TOLERANCE,
                passed=heaviest.peak_bytes_ratio <= _PEAK_ALLOC_TOLERANCE,
            )
        )
    else:
        checks.append(
            GateCheck(
                name="no_regression",
                case=None,
                measured=0.0,
                threshold=_REGRESSION_CI_UPPER,
                passed=False,
            )
        )
        checks.append(
            GateCheck(
                name="peak_allocation",
                case=None,
                measured=0.0,
                threshold=_PEAK_ALLOC_TOLERANCE,
                passed=False,
            )
        )

    equivalent = sum(1 for result in results if result.equivalence.ok)
    checks.append(
        GateCheck(
            name="equivalence",
            case=None,
            measured=float(equivalent),
            threshold=float(len(results)),
            passed=bool(results) and equivalent == len(results),
        )
    )
    return all(check.passed for check in checks), tuple(checks)


def print_checklist(payload: dict[str, Any]) -> None:
    """Print the gate checklist, every case's verdict and the overall line.

    Each item carries its measured value next to the threshold it had to reach,
    and a failing case names the thresholds it missed and by how much.
    """
    print()
    print("Gate checklist:")
    for check in payload["gate"]["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        target = f" [{check['case']}]" if check["case"] else ""
        print(
            f"  [{status}] {check['name']}{target}: measured "
            f"{_format_measurement(check['measured'])}, required "
            f"{_check_relation(check['name'])} "
            f"{_format_measurement(check['threshold'])}"
        )

    print()
    print("Gated cases:")
    for name, case in payload["cases"].items():
        failed = case["failed_thresholds"]
        cause = f" -- {'; '.join(failed)}" if failed else ""
        print(
            f"  [{case['verdict']}] {name}: ratio median "
            f"{case['ratio_median']:.3f}, 95% CI "
            f"{_format_ci(case['ci_low'], case['ci_high'])}, peak allocation "
            f"{_format_peak_delta(case['allocation']['peak_bytes_ratio'])}%"
            f"{cause}"
        )

    if payload["subseries"]:
        print()
        print("Informational sub-series (not gated):")
        for name, case in payload["subseries"].items():
            print(
                f"  {name}: ratio median {case['ratio_median']:.3f}, 95% CI "
                f"{_format_ci(case['ci_low'], case['ci_high'])}, peak allocation "
                f"{_format_peak_delta(case['allocation']['peak_bytes_ratio'])}%"
            )

    print()
    print(f"OVERALL: {payload['verdict']}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the runner's five options.

    Args:
        argv: The argument list, or ``None`` to read ``sys.argv``.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.delayed_ab",
        description=(
            "Measure the frozen pre-refactor delayed (arm A) against the live "
            "dask.delayed (arm B) in one interpreter, prove the two behave "
            "identically before timing anything, print a pass/fail checklist and "
            "write the suite's two artefacts."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=_measured_rounds_argument,
        default=_DEFAULT_ROUNDS,
        metavar="N",
        help=(
            f"measured rounds per case, at least {_MIN_ROUNDS} "
            f"(default: {_DEFAULT_ROUNDS})"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=_warmup_rounds_argument,
        default=_DEFAULT_WARMUP,
        metavar="N",
        help=(
            f"discarded warmup rounds per case, at least {_MIN_WARMUP} "
            f"(default: {_DEFAULT_WARMUP})"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help=(
            "directory for the two artefacts (default: "
            f"{_DEFAULT_OUTPUT} inside the repository). Writing there is refused "
            "while the working tree is dirty; a directory outside the checkout is "
            "always allowed, which is how the A/A calibration run keeps the tree "
            "clean"
        ),
    )
    parser.add_argument(
        "--no-artefacts",
        action="store_true",
        help="print the checklist and skip both writers, keeping the exit code",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        metavar="PATH",
        help=(
            "an earlier A/A result JSON whose per-case ratio medians and "
            "intervals are copied into this run's calibration block and report"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the suite and return the process exit status.

    The order is fixed and each step is a precondition for the next: fix the hash
    seed, parse the options, load the calibration file, load both arms, read the
    repository provenance, refuse a dirty tree before spending a minute on
    measurement, prove every case equivalent, time the cases, measure their
    allocations, evaluate the gate, print the checklist and only then write the
    artefacts.

    Args:
        argv: The argument list, or ``None`` to read ``sys.argv``.

    Returns:
        0 when every gate item held, 1 when the gate failed or the run could not
        be configured, 2 on an equivalence mismatch, 3 on a refusal to write into
        a dirty working tree.
    """
    _ensure_fixed_hash_seed()
    args = parse_args(argv)
    output = _resolve_output(args.output)
    write_artefacts = not args.no_artefacts

    calibration: dict[str, Any] | None = None
    if args.calibration is not None:
        try:
            calibration = _load_calibration(pathlib.Path(args.calibration).expanduser())
        except CalibrationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return _EXIT_GATE_FAIL

    try:
        baseline, live = load_arms()
    except ArmLoadError as exc:
        # Halt and report: neither the capture nor any frozen dask module is
        # edited to make this work.
        print(f"halt: {exc}", file=sys.stderr)
        return _EXIT_GATE_FAIL

    state = repository_state()
    if write_artefacts:
        offending = _dirty_tree_refusal(output, state)
        if offending is not None:
            print(
                "refusing to write artefacts inside the repository while the "
                "working tree is dirty, because the recorded git HEAD would not "
                "identify the code that was measured. Offending paths:",
                file=sys.stderr,
            )
            for line in offending:
                print(f"  {line}", file=sys.stderr)
            print(
                "commit the sources first and rerun, or pass --output with a "
                "directory outside the checkout, or pass --no-artefacts.",
                file=sys.stderr,
            )
            return _EXIT_DIRTY

    corpus: tuple[tuple[Case, bool], ...] = tuple(
        [(case, True) for case in CASES] + [(case, False) for case in SUBSERIES]
    )
    print(
        f"arms: A={baseline.__name__} B={live.__name__}; "
        f"{args.warmup} warmup + {args.rounds} measured rounds per case; "
        f"artefacts={'on' if write_artefacts else 'off'} ({output})"
    )

    print()
    print("Equivalence (asserted before any timing is recorded):")
    equivalences: dict[str, Equivalence] = {}
    mismatched: list[Equivalence] = []
    for case, _gated in corpus:
        verdict = assert_equivalent(case, baseline, live)
        equivalences[case.name] = verdict
        if verdict.ok:
            print(
                f"  [ok] {case.name}: keys, canonical graphs and results match "
                f"under pure=None and pure=True "
                f"({verdict.placeholders_pure_true} key placeholder(s) needed "
                f"under pure=True)",
                flush=True,
            )
        else:
            print(f"  [MISMATCH] {case.name}", flush=True)
            mismatched.append(verdict)
    if mismatched:
        print(
            "equivalence mismatch: no artefact is written and no performance "
            "verdict is produced.",
            file=sys.stderr,
        )
        for verdict in mismatched:
            print(f"  {verdict.mismatch}", file=sys.stderr)
        return _EXIT_EQUIVALENCE

    print()
    print("Timing and allocation:")
    results: list[CaseResult] = []
    for case, gated in corpus:
        rounds = time_case(case, baseline, live, warmup=args.warmup, rounds=args.rounds)
        allocation = measure_allocations(case, baseline, live)
        ratios = [round_.ratio for round_ in rounds]
        ci_low, ci_high = bootstrap_ci(ratios)
        result = CaseResult(
            name=case.name,
            pure=None,
            gated=gated,
            rounds=rounds,
            baseline_allocation=allocation[_BASELINE],
            candidate_allocation=allocation[_CANDIDATE],
            equivalence=equivalences[case.name],
            ratio_median=float(statistics.median(ratios)) if ratios else 0.0,
            ci_low=ci_low,
            ci_high=ci_high,
        )
        results.append(result)
        print(
            f"  {case.name}: baseline median "
            f"{_format_ms(arm_stats(result.timings(_BASELINE))['median'])} ms, "
            f"candidate median "
            f"{_format_ms(arm_stats(result.timings(_CANDIDATE))['median'])} ms, "
            f"ratio median {result.ratio_median:.3f}, 95% CI "
            f"{_format_ci(result.ci_low, result.ci_high)}",
            flush=True,
        )

    gate_passed, checks = evaluate_gate(results)
    flat_loop_median = next(
        (
            arm_stats(result.timings(_CANDIDATE))["median"]
            for result in results
            if result.name == _FLAT_LOOP_CASE
        ),
        None,
    )
    payload = build_payload(
        results,
        warmup=args.warmup,
        rounds=args.rounds,
        gate_passed=gate_passed,
        checks=checks,
        environment_block=environment(
            repository=state,
            pythonhashseed=os.environ.get("PYTHONHASHSEED"),
            flat_loop_median_ns=flat_loop_median,
        ),
        calibration=calibration,
    )
    print_checklist(payload)

    if write_artefacts:
        json_path = output / _JSON_NAME
        report_path = output / _REPORT_NAME
        write_json(json_path, payload)
        write_report(report_path, payload)
        print()
        print(f"wrote {json_path}")
        print(f"wrote {report_path}")

    return _EXIT_PASS if gate_passed else _EXIT_GATE_FAIL
