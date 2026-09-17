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
    1: the gate failed, or the run could not be configured or trusted -- an
        unreadable or non-A/A ``--calibration`` file, a capture that is not the
        frozen arm A, a measurement that cannot be interpreted, or a failure
        while writing the artefact pair. The message names the item.
    2: an equivalence mismatch. No artefact is written, and no performance
        verdict is produced.
    3: artefacts were requested inside a repository working tree that is dirty,
        whose state could not be established, or whose provenance changed while
        the run was in progress. The offending paths are printed.

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

# Intra-package imports, in the relative form. ``from .canon import ...`` inside
# ``benchmarks.delayed_ab.main`` binds the single ``sys.modules`` entry
# ``benchmarks.delayed_ab.canon`` -- the same module object the characterisation
# test binds -- so the harness and the test share one canonicaliser definition and
# the two bodies of evidence cannot drift apart.
#
# The absolute spelling ``from benchmarks.delayed_ab.canon import ...`` is not used
# here for a type-checker reason rather than a preference. ``benchmarks/`` is a PEP
# 420 namespace package (no ``__init__.py``), so mypy maps this file from its path
# to ``delayed_ab.main`` while the absolute name resolves the same file a second
# time as ``benchmarks.delayed_ab.main``, and it halts with ``Source file found
# twice under different module names`` followed by ``errors prevented further
# checking`` -- which silences the type check of the entire repository, not just
# this tree. That error is raised while the module graph is assembled and carries
# no error code, so no inline suppression reaches it, and both remedies mypy names
# are out of scope here: ``benchmarks/__init__.py`` falls outside the paths the
# run's structural criterion permits (and would make ``[tool.setuptools.packages]
# find = {namespaces = false}`` ship this evidence tree inside the wheel), while
# ``explicit_package_bases`` would edit the frozen ``pyproject.toml``. Applying
# either one makes the absolute spelling work unchanged.
#
# The suite is always started as ``python -m benchmarks.delayed_ab``.
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

#: sha256 of ``dask/delayed.py`` at ``_BASELINE_SHA``, which is byte-for-byte what
#: the capture's body -- everything below its header comment block -- has to be.
#: The capture's own header records this digest too, and it is re-derived from git
#: history (``git show <sha>:dask/delayed.py``) whenever git can answer, so the
#: constant is auditable rather than merely asserted. Arm A is the measurement's
#: denominator: if it is not this source, every ratio in the run compares the
#: candidate against something unknown.
_BASELINE_BODY_SHA256 = (
    "4c0000e204ea5b701cbef0879f6af74e3b547249edede4a1003ef4d78493d6f1"
)

#: The frozen source arm A must reproduce, as git addresses it.
_BASELINE_SOURCE_PATH = "dask/delayed.py"

#: Artefact location, relative to the repository root, and the JSON schema
#: version that ``dask/tests/test_delayed_ab_gate.py`` parses.
_DEFAULT_OUTPUT = "benchmarks/delayed_ab/results"
_SCHEMA_VERSION = 1

#: Suffix of the staging files the artefact pair is rendered into before either
#: destination is replaced, so a failure between the two writes cannot leave a
#: JSON from this run beside a report from the previous one.
_STAGING_SUFFIX = ".staging"

#: Suffix of the copy each destination is set aside under while the pair is
#: published. Publication is two renames and the second one can fail, so the
#: first destination has to be restorable from its previous contents -- without
#: it, a failed publication leaves this run's JSON beside the previous report.
_BACKUP_SUFFIX = ".previous"

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
# Failure modes. Each is raised by the code that detects the problem and caught
# once, at the orchestration boundary in ``main``, which turns it into a legible
# message and an exit code. Nothing here is recovered from: a measurement that
# cannot be interpreted, a capture that is not the frozen arm A and a half-written
# artefact pair are all conditions under which this suite has no verdict to give,
# and saying so is the whole point of raising rather than substituting a value.
# ---------------------------------------------------------------------------


class _MeasurementError(RuntimeError):
    """A measurement came back in a shape that cannot be interpreted.

    A region that took no measurable time, an arm whose traced peak is zero, a
    sampling hook that failed: each would have to be turned into an invented
    number before the gate could judge it, and an invented number is exactly
    what a performance gate must never see. The run ends with status 1 and the
    message names the case, the arm and the figure.
    """


class _ProvenanceError(RuntimeError):
    """Arm A is not the frozen capture the suite claims to measure.

    The capture's header records the commit it was taken from and its body must
    be byte-identical to that commit's ``dask/delayed.py``. If either is not
    true, every ratio in the run compares the candidate against something
    unknown, so the run ends with status 1 before a single timing is taken.
    """


class _ArtefactError(RuntimeError):
    """The artefact pair could not be written as one consistent pair.

    The two files are one piece of evidence. Rather than leave a JSON from this
    run beside a report from the previous one, the writer stages both, validates
    both and only then replaces the committed pair; any failure in that sequence
    ends the run with status 1 and leaves the previous pair untouched.
    """


# ---------------------------------------------------------------------------
# Records. Every measurement travels through one of these, so the payload
# writers never have to guess what a bare tuple meant.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Region:
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
class _Round:
    """One A, B, B, A block -- or B, A, A, B when the candidate starts.

    Attributes:
        first_arm: ``"baseline"`` or ``"candidate"``, whichever ran first.
        baseline: The arm-A regions, in the order they were measured.
        candidate: The arm-B regions, in the order they were measured.
    """

    first_arm: str
    baseline: tuple[_Region, _Region]
    candidate: tuple[_Region, _Region]

    @property
    def ratio(self) -> float:
        """Paired ratio ``(A1 + A2) / (B1 + B2)``: above 1 means B is faster.

        Returns:
            The round's paired ratio, baseline total over candidate total.

        Raises:
            _MeasurementError: If either arm's total is not positive. The
                protocol's ratio is only defined for two positive totals: a
                zero candidate total would have to be replaced by some other
                number to yield a ratio at all -- and any such substitution
                manufactures a speedup out of a failed measurement -- while a
                zero baseline total manufactures a regression the same way.
                Neither is reported as a ratio; the round is rejected instead.
        """
        baseline_total = sum(region.timing_ns for region in self.baseline)
        candidate_total = sum(region.timing_ns for region in self.candidate)
        if baseline_total <= 0 or candidate_total <= 0:
            raise _MeasurementError(
                "invalid paired measurement: a round needs a positive timing "
                "total for both arms, but this round "
                f"(first arm {self.first_arm}) measured baseline "
                f"{baseline_total} ns and candidate {candidate_total} ns. No "
                "ratio is defined for it, so none is reported"
            )
        return baseline_total / candidate_total


@dataclass(frozen=True)
class _ArmAllocation:
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
class _Extraction:
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
class _Equivalence:
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
class _CaseResult:
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
    rounds: tuple[_Round, ...]
    baseline_allocation: _ArmAllocation
    candidate_allocation: _ArmAllocation
    equivalence: _Equivalence
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

    def allocation(self, arm: str) -> _ArmAllocation:
        """Return one arm's allocation figures."""
        return (
            self.baseline_allocation if arm == _BASELINE else self.candidate_allocation
        )

    @property
    def peak_bytes_ratio(self) -> float:
        """Candidate peak bytes over baseline peak bytes: 1.05 is the tolerance.

        Returns:
            The gate-bearing allocation ratio of this case.

        Raises:
            _MeasurementError: If either arm's traced peak is not positive. A
                region that constructs a graph always allocates, so a
                non-positive peak means tracing recorded nothing -- and the
                allocation tolerance cannot be judged against a figure that was
                never measured. Returning a neutral 1.0 would pass the
                allocation item on the strength of a failed measurement, so the
                case is rejected instead.
        """
        baseline_peak = self.baseline_allocation.tracemalloc_peak_bytes
        candidate_peak = self.candidate_allocation.tracemalloc_peak_bytes
        if baseline_peak <= 0 or candidate_peak <= 0:
            raise _MeasurementError(
                f"invalid allocation measurement for case {self.name!r}: the "
                "allocation gate compares two traced peaks and both must be "
                f"positive, but tracemalloc recorded {baseline_peak} bytes for "
                f"the baseline arm and {candidate_peak} bytes for the candidate "
                "arm. The allocation tolerance cannot be evaluated"
            )
        return candidate_peak / baseline_peak

    @staticmethod
    def _regions(round_: _Round, arm: str) -> tuple[_Region, _Region]:
        """Return the two regions one arm contributed to a round."""
        return round_.baseline if arm == _BASELINE else round_.candidate


@dataclass(frozen=True)
class _GateCheck:
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
class _RepositoryState:
    """Provenance of the candidate arm, read once before anything is written.

    Attributes:
        root: The repository root, derived from this file's location rather than
            from the working directory.
        git_head: ``git rev-parse HEAD``, or ``None`` when git is unavailable.
        dirty: Whether ``git status --porcelain --untracked-files=all`` reported
            anything. Read at start-up, so artefacts the runner itself writes are
            never counted. It only means "clean" when ``provenance_gaps`` is
            empty: a tree whose state could not be read is not a clean one.
        dirty_paths: The porcelain lines behind ``dirty``, for the exit-3 message.
        delayed_py_sha256: Hex digest of ``dask/delayed.py`` -- the file the
            refactor changes -- or ``None`` if it could not be read.
        provenance_gaps: Every provenance figure that could not be established --
            an invalid repository root, an unavailable ``git rev-parse HEAD``, an
            uninspectable working tree, an unreadable ``dask/delayed.py``. Empty
            means the whole record was read successfully, and nothing but an
            empty tuple permits a write inside the repository.
        notes: Anything that degraded, such as a missing git executable.
    """

    root: pathlib.Path
    git_head: str | None
    dirty: bool
    dirty_paths: tuple[str, ...]
    delayed_py_sha256: str | None
    provenance_gaps: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def trustworthy(self) -> bool:
        """Whether every provenance figure of this record was established."""
        return not self.provenance_gaps


# ---------------------------------------------------------------------------
# Arms and arm activation
# ---------------------------------------------------------------------------


class _ArmLoadError(RuntimeError):
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
        _ArmLoadError: If either import fails, or if the two turn out to be the
            same object -- meaning they cannot coexist in one process. Both are
            halt-and-report conditions.
    """
    try:
        baseline = importlib.import_module("benchmarks.delayed_ab.baseline_delayed")
    except Exception as exc:
        # Reported, never worked around: the capture and every frozen dask module
        # stay untouched.
        raise _ArmLoadError(
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
        raise _ArmLoadError(
            f"arm B (dask.delayed) could not be imported: {type(exc).__name__}: {exc}"
        ) from exc
    if baseline is candidate:
        raise _ArmLoadError(
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


def _extract(case: Case, mod: ModuleType, *, pure: bool | None) -> _Extraction:
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
    return _Extraction(
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
    baseline: _Extraction,
    candidate: _Extraction,
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
) -> _Equivalence:
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
        The case's :class:`_Equivalence` verdict, whose ``mismatch`` is ``None``
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
            return _Equivalence(
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
            return _Equivalence(
                case=case.name,
                native=matched[None],
                pure_true=matched[True],
                placeholders_pure_true=placeholders_pure_true,
                mismatch=mismatch,
            )
        matched[pure] = True
    return _Equivalence(
        case=case.name,
        native=matched[None],
        pure_true=matched[True],
        placeholders_pure_true=placeholders_pure_true,
        mismatch=None,
    )


# ---------------------------------------------------------------------------
# The paired measurement protocol
# ---------------------------------------------------------------------------


def _ensure_fixed_hash_seed(argv: Sequence[str]) -> None:
    """Re-execute the runner with ``PYTHONHASHSEED=0`` unless it is already set.

    Hash randomisation changes dict and set iteration order, which changes how
    much work the construction path does over containers, so both arms must run
    under a fixed seed. The seed can only be fixed before the interpreter starts,
    hence the re-exec. It happens before any measurement and at most once: after
    the exec the variable is ``"0"``, so the replacement process returns
    immediately from here.

    Args:
        argv: The effective argument list of this run -- what the caller passed
            to :func:`main`, or ``sys.argv[1:]`` when it passed nothing. It is
            what the replacement process receives, so a programmatic
            ``main([...])`` runs the options it was given rather than whatever
            happens to be on the real command line.
    """
    if os.environ.get("PYTHONHASHSEED") == "0":
        return
    # The command is rebuilt as ``-m benchmarks.delayed_ab`` rather than from
    # ``sys.argv[0]``, which is the path of ``__main__.py``. Like the original
    # invocation, this requires the repository root as the working directory.
    os.execve(
        sys.executable,
        [sys.executable, "-m", "benchmarks.delayed_ab", *argv],
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


def _time_region(case: Case, mod: ModuleType, *, pure: bool | None) -> _Region:
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
    return _Region(timing_ns=end - start, blocks_delta=blocks_after - blocks_before)


def time_case(
    case: Case,
    baseline: ModuleType,
    live: ModuleType,
    *,
    warmup: int,
    rounds: int,
    pure: bool | None = None,
) -> tuple[_Round, ...]:
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
    measured: list[_Round] = []
    for index in range(warmup + rounds):
        if index % 2 == 0:
            first = _time_region(case, baseline, pure=pure)
            second = _time_region(case, live, pure=pure)
            third = _time_region(case, live, pure=pure)
            fourth = _time_region(case, baseline, pure=pure)
            completed = _Round(
                first_arm=_BASELINE,
                baseline=(first, fourth),
                candidate=(second, third),
            )
        else:
            first = _time_region(case, live, pure=pure)
            second = _time_region(case, baseline, pure=pure)
            third = _time_region(case, baseline, pure=pure)
            fourth = _time_region(case, live, pure=pure)
            completed = _Round(
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


def _arm_stats(values: Sequence[float]) -> dict[str, float]:
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


def _bootstrap_ci(
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

    Raises:
        _MeasurementError: If the sampling hook failed. The hook cannot raise --
            an exception inside a profile hook silently disables profiling for
            the rest of the region -- so it records the failure in a closure cell
            instead, and the failure is reported here, once profiling is off. A
            figure whose sampling broke halfway through is stale, and a stale
            figure in a committed artefact is worse than a run that stops.
    """
    with activate(mod):
        state = case.setup(mod, pure=pure)
        get_blocks = sys.getallocatedblocks
        sampled = 0
        events = 0
        sampling_failure: BaseException | None = None

        def hook(frame: Any, event: str, arg: Any) -> None:
            """Update the sampled maximum, recording any failure for the caller.

            Args:
                frame: The frame the profile event fired in. Unused: the figure
                    is process-wide, so no frame data enters it.
                event: The profile event name. Unused, for the same reason --
                    every call and return event is one tick of the stride.
                arg: The event's argument. Unused.
            """
            nonlocal sampled, events, sampling_failure
            try:
                events += 1
                if events % _BLOCK_SAMPLE_STRIDE == 0:
                    current = get_blocks()
                    if current > sampled:
                        sampled = current
            except Exception as exc:
                # Recorded, never raised: raising here would disable profiling
                # and leave the figure silently stale. The first failure is kept
                # and reported by the caller once the hook is uninstalled.
                if sampling_failure is None:
                    sampling_failure = exc

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
    if sampling_failure is not None:
        raise _MeasurementError(
            f"block sampling failed for case {case.name!r} under arm "
            f"{mod.__name__}: the profile hook raised "
            f"{type(sampling_failure).__name__}: {sampling_failure}. The "
            "'max_observed_blocks' figure would be stale, so it is not reported"
        )
    return max(sampled, after) - before


def measure_allocations(
    case: Case,
    baseline: ModuleType,
    live: ModuleType,
    *,
    pure: bool | None = None,
) -> dict[str, _ArmAllocation]:
    """Collect both arms' allocation figures for one case, in untimed rounds.

    Args:
        case: The case to measure.
        baseline: Arm A.
        live: Arm B.
        pure: The variant to measure -- the same one the timing rounds used.

    Returns:
        The arm-keyed allocation figures.
    """
    figures: dict[str, _ArmAllocation] = {}
    for arm, mod in ((_BASELINE, baseline), (_CANDIDATE, live)):
        peak_bytes, live_blocks_end = _tracemalloc_round(case, mod, pure=pure)
        figures[arm] = _ArmAllocation(
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
    ``delayed(f)`` wraps in the ``flat_loop`` case. The body is the identity so
    that nothing but the probe itself is ever measured; the micro-benchmark
    passes the function object to ``is_dask_collection`` and never calls it.

    Args:
        x: The value to return. Unused by the micro-benchmark, which measures
            the probe against this function *object*.

    Returns:
        ``x`` unchanged.
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


def _invalid_root_reason(root: pathlib.Path) -> str | None:
    """Return why ``root`` is not this repository's root, or ``None`` if it is.

    The root is derived from ``__file__``, so it is wrong only when the suite has
    been copied out of the checkout -- and in that case every provenance figure
    taken from it describes some other tree. The two markers checked are the ones
    the run actually depends on: a ``.git`` entry, so the provenance commands mean
    something, and ``dask/delayed.py``, the file whose digest the artefacts carry.

    Args:
        root: The candidate repository root.

    Returns:
        A reason naming the missing marker, or ``None`` when both are present.
    """
    if not root.is_dir():
        return f"the derived repository root {root} is not a directory"
    if not (root / ".git").exists():
        return f"the derived repository root {root} contains no .git entry"
    if not (root / "dask" / "delayed.py").is_file():
        return f"the derived repository root {root} contains no dask/delayed.py"
    return None


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


def _repository_state() -> _RepositoryState:
    """Read the candidate arm's provenance, before any output file is written.

    The dirty flag is deliberately evaluated at start-up: the two artefacts the
    runner writes live inside the repository by default, so inspecting the tree
    afterwards would count its own output.

    Every figure that cannot be established is recorded as a provenance gap
    rather than replaced by an optimistic default. That distinction is the whole
    control: a missing git executable, a directory that is not this repository or
    a ``git status`` that failed all leave cleanliness *unknown*, and an unknown
    tree is not a clean one. Reading it as clean would let the runner write a
    committed artefact whose recorded ``git HEAD`` and ``dirty=false`` describe
    nothing that was ever checked. Provenance still never crashes the run -- a
    run that writes outside the checkout, which is how the A/A calibration works,
    proceeds with the gaps recorded in the environment block.

    Returns:
        The repository root, ``git rev-parse HEAD``, the dirty flag with the
        porcelain lines behind it, the sha256 of ``dask/delayed.py``, every
        provenance gap, and any note about something that degraded.
    """
    root = _repository_root()
    notes: list[str] = []
    gaps: list[str] = []

    root_problem = _invalid_root_reason(root)
    if root_problem is not None:
        gaps.append(root_problem)

    head, note = _git(root, "rev-parse", "HEAD")
    if note is not None:
        notes.append(note)
    if head is None or not head.strip():
        gaps.append("`git rev-parse HEAD` did not report a commit")

    status, note = _git(root, "status", "--porcelain", "--untracked-files=all")
    if note is not None:
        notes.append(note)
    if status is None:
        dirty = False
        dirty_paths: tuple[str, ...] = ()
        gaps.append(
            "the working tree could not be inspected, so whether it is clean is "
            "unknown"
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
        gaps.append(f"sha256({delayed_py}) could not be computed")

    return _RepositoryState(
        root=root,
        git_head=head.strip() if head is not None else None,
        dirty=dirty,
        dirty_paths=dirty_paths,
        delayed_py_sha256=digest,
        provenance_gaps=tuple(gaps),
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


def _sanitise_path(text: str, root: pathlib.Path) -> str:
    """Replace the checkout's own location in ``text`` with ``<repository>``.

    Both artefacts are committed, so nothing that describes the machine that
    produced them belongs in either: a checkout path names an ephemeral workspace
    layout rather than durable provenance, and the commit, the ``dask/delayed.py``
    digest and the capture's SHA already identify the measured code exactly. Only
    the root prefix is rewritten, so the repository-relative remainder of a
    message -- which is the part that says what happened -- survives intact.

    Args:
        text: A message destined for an artefact, such as a provenance note.
        root: The repository root to redact.

    Returns:
        The message with every occurrence of the root path replaced by the
        literal ``<repository>``.
    """
    return text.replace(str(root), "<repository>")


def _git_show_bytes(root: pathlib.Path, spec: str) -> tuple[bytes | None, str | None]:
    """Read one blob out of git history verbatim, as bytes.

    ``_git`` decodes to text with universal newlines, which would silently
    rewrite line endings; a digest has to be taken over the bytes git stored.

    Args:
        root: Repository to read from.
        spec: A ``<commit>:<path>`` revision specification.

    Returns:
        The blob's bytes and ``None``, or ``None`` and an explanatory note.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "show", spec],
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return None, f"git could not be executed ({type(exc).__name__}: {exc})"
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip() or (
            f"exit status {completed.returncode}"
        )
        return None, f"`git show {spec}` failed: {detail}"
    return completed.stdout, None


def _capture_body(text: str) -> str:
    """Return the capture's body: everything below its leading comment header.

    The capture is ``dask/delayed.py`` with a comment block prepended -- a block
    rather than a docstring, deliberately, because a docstring would displace the
    module's first statement. The body therefore starts at the first line that is
    neither blank nor a comment, which is the module's own
    ``from __future__ import annotations``.

    Args:
        text: The capture file's full text.

    Returns:
        The body text, empty when the file holds nothing but comments.
    """
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return "".join(lines[index:])
    return ""


def _baseline_arm() -> dict[str, Any]:
    """Describe arm A, validating that it is the frozen source it claims to be.

    Two things are checked, and both are conditions of the run rather than
    observations about it: the capture's header must record ``_BASELINE_SHA`` in a
    ``# Source commit:`` line, and its body -- everything below that comment
    header -- must hash to ``_BASELINE_BODY_SHA256``, the digest of
    ``dask/delayed.py`` at that commit. When git can read the commit, the
    expected digest is re-derived from history as well, so a tampered constant is
    caught alongside a tampered capture.

    Neither condition is a note for the reader to weigh. Arm A is the denominator
    of every ratio the suite reports: a capture that is not the frozen source
    turns the whole run into a comparison against something unknown, which is why
    this is called before any timing is taken and raises rather than degrades.

    Returns:
        The record for ``environment.arms.baseline``: the module path, the
        header's SHA and the expected one, both body digests, whether each
        matched, and a note when git could not corroborate the digest.

    Raises:
        _ProvenanceError: If the capture cannot be read, carries no
            ``# Source commit:`` line, records a different commit, or has a body
            that does not hash to the frozen source's digest.
    """
    capture = pathlib.Path(__file__).with_name("baseline_delayed.py")
    try:
        text = capture.read_text(encoding="utf-8")
    except OSError as exc:
        raise _ProvenanceError(
            f"arm A ({capture}) could not be read: {type(exc).__name__}: {exc}"
        ) from exc

    match = re.search(r"^#\s*Source commit:\s*([0-9a-f]{40})\s*$", text, re.MULTILINE)
    if match is None:
        raise _ProvenanceError(
            f"arm A ({capture}) carries no `# Source commit: <sha>` header line, "
            "so the commit it was captured from cannot be established"
        )
    sha = match.group(1)
    if sha != _BASELINE_SHA:
        raise _ProvenanceError(
            f"arm A ({capture}) records source commit {sha}, but this suite "
            f"measures the capture of {_BASELINE_SHA}. The frozen arm was "
            "replaced, and no ratio taken against it would mean what it says"
        )

    body_sha256 = hashlib.sha256(_capture_body(text).encode("utf-8")).hexdigest()
    if body_sha256 != _BASELINE_BODY_SHA256:
        raise _ProvenanceError(
            f"arm A ({capture}) has a body that hashes to {body_sha256}, not to "
            f"{_BASELINE_BODY_SHA256} -- the digest of {_BASELINE_SOURCE_PATH} at "
            f"{_BASELINE_SHA}. The capture is not a verbatim copy of the frozen "
            "source, so it is not the baseline this suite reports against"
        )

    note: str | None = None
    frozen_source, git_note = _git_show_bytes(
        _repository_root(), f"{_BASELINE_SHA}:{_BASELINE_SOURCE_PATH}"
    )
    if frozen_source is None:
        note = (
            f"the frozen source could not be re-read from git history "
            f"({git_note}), so the body digest was checked against the recorded "
            f"constant {_BASELINE_BODY_SHA256} only"
        )
    else:
        frozen_sha256 = hashlib.sha256(frozen_source).hexdigest()
        if frozen_sha256 != _BASELINE_BODY_SHA256:
            raise _ProvenanceError(
                f"{_BASELINE_SOURCE_PATH} at {_BASELINE_SHA} hashes to "
                f"{frozen_sha256} in git history, but this suite expects "
                f"{_BASELINE_BODY_SHA256}. The expected digest and the "
                "repository's own history disagree, so arm A's provenance "
                "cannot be established from either"
            )

    return {
        "module": "benchmarks.delayed_ab.baseline_delayed",
        "sha": sha,
        "expected_sha": _BASELINE_SHA,
        "sha_matches_expected": sha == _BASELINE_SHA,
        "body_sha256": body_sha256,
        "expected_body_sha256": _BASELINE_BODY_SHA256,
        "body_matches_expected": body_sha256 == _BASELINE_BODY_SHA256,
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
    repository: _RepositoryState | None = None,
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
        Every recorded fact is reachable under its own direct key -- the
        free-threading flags, the five key package versions and both arms'
        provenance included -- so a reader never has to know the grouping first;
        the grouped objects (``implementation``, ``distributions``, ``packages``,
        ``free_threading``, ``arms``) are kept beside them because they carry
        detail the direct keys do not, such as the capture's expected SHA. No
        filesystem path of the machine that ran the suite appears anywhere in the
        block: a workspace layout is not durable provenance, and the commit,
        ``dask/delayed.py`` digest and capture SHA identify the measured code
        exactly.
    """
    state = repository if repository is not None else _repository_state()
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    py_gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
    gil_enabled = gil_probe() if gil_probe is not None else None
    distributions = _distributions()
    lowered = {name.lower(): version for name, version in distributions.items()}
    packages = {name: lowered.get(name) for name in _KEY_PACKAGES}
    # ``hashers`` is a list of plain functions; ``getattr`` keeps the lookup
    # tolerant of any callable that carries no ``__name__``.
    active_hasher = (
        getattr(hashers[0], "__name__", repr(hashers[0])) if hashers else None
    )
    baseline_arm = _baseline_arm()
    # A capture that could not be read reports the operating system's message,
    # which names the file: redact the checkout's location out of it here, at the
    # boundary where artefact content is decided.
    if isinstance(baseline_arm["note"], str):
        baseline_arm["note"] = _sanitise_path(baseline_arm["note"], state.root)
    return {
        "python_version": platform.python_version(),
        "sys_version": sys.version,
        "implementation": {
            "name": sys.implementation.name,
            "version": ".".join(str(part) for part in sys.implementation.version[:3]),
            "cache_tag": sys.implementation.cache_tag,
        },
        # The free-threading facts, each under its own name: the build flag, the
        # runtime state of the GIL on 3.13+ (``None`` on an interpreter that has
        # no probe) and the plain question a reader asks, derived from the flag
        # because a free-threaded build can still be started with the GIL on.
        "py_gil_disabled": py_gil_disabled,
        "gil_enabled": gil_enabled,
        "free_threaded": bool(py_gil_disabled),
        "free_threading": {
            "py_gil_disabled": py_gil_disabled,
            "gil_enabled": gil_enabled,
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
        "packages": packages,
        # The five key versions, surfaced next to the complete map so that a
        # reader after one of them does not have to index into ``distributions``.
        # ``None`` means the distribution is not installed in this environment.
        "dask": packages["dask"],
        "toolz": packages["toolz"],
        "cloudpickle": packages["cloudpickle"],
        "numpy": packages["numpy"],
        "pandas": packages["pandas"],
        # Provenance of both arms, direct: the SHA the capture's own header
        # records for arm A, and for arm B the commit, the repository-wide dirty
        # flag read before anything was written, and the digest of the one
        # production file the refactor changes.
        "baseline_sha": baseline_arm["sha"],
        "git_head": state.git_head,
        "dirty": state.dirty,
        "delayed_py_sha256": state.delayed_py_sha256,
        "arms": {
            _BASELINE: baseline_arm,
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
        "notes": [_sanitise_path(note, state.root) for note in state.notes],
    }


# ---------------------------------------------------------------------------
# Artefacts: the JSON payload, its report, and the calibration block
# ---------------------------------------------------------------------------


class _CalibrationError(RuntimeError):
    """``--calibration`` named a file that could not be used.

    Raised with a message for the operator instead of letting a traceback out.
    """


def _calibration_number(
    entry: dict[str, Any], key: str, *, where: str, path: pathlib.Path
) -> float:
    """Read one required numeric figure out of a calibration entry.

    Args:
        entry: The case or sub-series object the figure belongs to.
        key: The figure to read: ``ratio_median``, ``ci_low`` or ``ci_high``.
        where: How to name the entry in a failure message.
        path: The calibration file, for the same message.

    Returns:
        The figure as a ``float``.

    Raises:
        _CalibrationError: If the key is absent, does not hold a number, or holds
            one that cannot be a paired ratio. ``bool`` is rejected because it is
            an ``int`` subclass and a boolean in a numeric slot means the file is
            not an A/B result; non-finite and non-positive values are rejected
            because every figure here is a quotient of two positive durations,
            and a NaN would silently disable the report's bias comparisons.
    """
    if key not in entry:
        raise _CalibrationError(
            f"--calibration {path}: {where} carries no {key!r}, so it is not an "
            "A/B suite result file"
        )
    value = entry[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _CalibrationError(
            f"--calibration {path}: {where} has {key}={value!r} "
            f"({type(value).__name__}), which is not a number"
        )
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        # JSON admits integers of unbounded size, and ``float`` raises on one too
        # large to represent. That is malformed input, not a runner fault, so it
        # is reported like every other malformed field instead of escaping.
        raise _CalibrationError(
            f"--calibration {path}: {where} has a {key} that is not a usable "
            f"number ({type(exc).__name__}: {exc})"
        ) from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise _CalibrationError(
            f"--calibration {path}: {where} has a non-finite {key} ({value!r})"
        )
    if number <= 0.0:
        raise _CalibrationError(
            f"--calibration {path}: {where} has {key}={number!r}, but a paired "
            "ratio and its interval bounds are quotients of two positive "
            "durations and cannot be zero or negative"
        )
    return number


def _require_aa_provenance(raw: dict[str, Any], path: pathlib.Path) -> None:
    """Reject a calibration file that is not an A/A run of this suite.

    ``--calibration`` exists to carry the machine's *noise floor* into the
    committed artefacts, and only an A/A run measures that: arm A against a live
    ``dask/delayed.py`` that is still byte-identical to the frozen capture, whose
    ratios therefore straddle 1.0. An A/B run's ratios are the very figures under
    test, so copying them in as the noise floor would compare this run's speedup
    against a previous run's speedup while labelling it "A/A calibration". The
    identity is exact and needs no heuristic: the recorded candidate digest has to
    be the frozen source's, the recorded baseline commit has to be
    ``_BASELINE_SHA``, every arm-A identity field the file carries has to agree
    with the frozen source rather than contradict it, and the commit the run
    measured has to have been established at the time.

    Args:
        raw: The parsed calibration JSON.
        path: The calibration file, for the failure messages.

    Raises:
        _CalibrationError: If the provenance block is missing or malformed, if it
            shows the candidate arm was not the unmodified frozen source, if it
            contradicts arm A's identity, or if it records no commit for the arm
            it measured.
    """
    environment_block = raw.get("environment")
    if not isinstance(environment_block, dict):
        raise _CalibrationError(
            f"--calibration {path}: carries no 'environment' object, so its "
            "provenance cannot be checked"
        )
    arms = environment_block.get("arms")
    if not isinstance(arms, dict):
        raise _CalibrationError(
            f"--calibration {path}: carries no 'environment.arms' object, so "
            "neither arm's provenance can be checked"
        )
    baseline = arms.get(_BASELINE)
    candidate = arms.get(_CANDIDATE)
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise _CalibrationError(
            f"--calibration {path}: 'environment.arms' does not describe both "
            f"the {_BASELINE!r} and the {_CANDIDATE!r} arm"
        )
    if baseline.get("sha") != _BASELINE_SHA:
        raise _CalibrationError(
            f"--calibration {path}: was measured against baseline capture "
            f"{baseline.get('sha')!r}, not against {_BASELINE_SHA}, so its "
            "ratios are not this suite's noise floor"
        )
    # Every identity field the recording run wrote about arm A has to agree with
    # the frozen source. They are checked when present rather than required,
    # because an A/A run can only be taken before the production edit -- the live
    # module has to still equal the capture -- so a legitimate noise floor may
    # predate the fields a later revision of this runner records. What is never
    # accepted is a file that carries them and contradicts them.
    for field, expected in (
        ("expected_sha", _BASELINE_SHA),
        ("body_sha256", _BASELINE_BODY_SHA256),
        ("expected_body_sha256", _BASELINE_BODY_SHA256),
    ):
        recorded = baseline.get(field)
        if recorded is not None and recorded != expected:
            raise _CalibrationError(
                f"--calibration {path}: its baseline arm records "
                f"{field}={recorded!r} instead of {expected!r}, so arm A of that "
                "run was not the frozen source and its ratios are not this "
                "suite's noise floor"
            )
    for flag in ("sha_matches_expected", "body_matches_expected"):
        declared = baseline.get(flag)
        if declared is not None and declared is not True:
            raise _CalibrationError(
                f"--calibration {path}: its baseline arm records "
                f"{flag}={declared!r}, i.e. that run reported arm A was not the "
                "frozen source, so its ratios are not this suite's noise floor"
            )
    measured = candidate.get("delayed_py_sha256")
    if measured != _BASELINE_BODY_SHA256:
        raise _CalibrationError(
            f"--calibration {path}: is not an A/A run. Its candidate arm "
            f"measured {_BASELINE_SOURCE_PATH} with digest {measured!r}, not the "
            f"frozen source's {_BASELINE_BODY_SHA256}, so its ratios are a "
            "speedup measurement and not the machine's noise floor"
        )
    head = candidate.get("git_head")
    if not isinstance(head, str) or re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise _CalibrationError(
            f"--calibration {path}: its candidate arm records git_head={head!r}, "
            "so the commit it measured was never established and its provenance "
            "cannot be cited in this run's artefacts"
        )


def _load_calibration(path: pathlib.Path) -> dict[str, Any]:
    """Load and validate an earlier A/A run's ratio medians and intervals.

    The file is an external input to a run whose output is committed evidence, so
    it is validated in full rather than probed: the schema version this reader was
    written against, the timestamp, every case and sub-series this corpus defines
    with no name missing and none unknown, three finite positive numbers per
    entry with a coherent interval, and the A/A provenance that makes those
    numbers a noise floor at all. Anything else is rejected with a message naming
    what was wrong, because a calibration block silently built from one arbitrary
    case -- or from a previous A/B run -- would be published in the report as this
    machine's noise floor.

    Args:
        path: The A/A JSON written by an earlier run, normally outside the
            checkout so that run left the working tree clean.

    Returns:
        The calibration block: the logical label of the source, the sha256 digest
        of its bytes, the timestamp it recorded for itself, and the per-case ratio
        median with its interval, for cases and sub-series. The file's path is
        deliberately not carried into the block: that run's output lives in an
        ephemeral scratch directory outside the checkout, so the path describes a
        workspace layout rather than the calibration, while the digest identifies
        the exact bytes these figures were read from.

    Raises:
        _CalibrationError: If the file cannot be read, is not JSON, is not this
            suite's schema, is missing or malformed in any validated field, or
            was not produced by an A/A run.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise _CalibrationError(
            f"--calibration {path}: cannot be read ({type(exc).__name__}: {exc})"
        )
    except json.JSONDecodeError as exc:
        raise _CalibrationError(f"--calibration {path}: is not valid JSON ({exc})")
    if not isinstance(raw, dict):
        raise _CalibrationError(
            f"--calibration {path}: expected a JSON object at the top level, got "
            f"{type(raw).__name__}"
        )

    schema_version = raw.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise _CalibrationError(
            f"--calibration {path}: carries no integer 'schema_version', so it "
            "is not an A/B suite result file"
        )
    if schema_version != _SCHEMA_VERSION:
        raise _CalibrationError(
            f"--calibration {path}: is schema version {schema_version}, and this "
            f"runner reads version {_SCHEMA_VERSION}. Rerun the A/A calibration "
            "with this revision of the suite rather than copying figures across "
            "schemas"
        )

    generated_at = raw.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.strip():
        raise _CalibrationError(
            f"--calibration {path}: carries no 'generated_at' timestamp, so the "
            "report could not say when the noise floor was measured"
        )

    def figures(
        section: object, *, expected: tuple[str, ...], label: str
    ) -> dict[str, Any]:
        """Extract one section's validated ratio figures, keyed by entry name.

        Args:
            section: The JSON value found under ``label`` in the calibration
                file -- expected to be an object keyed by case name.
            expected: Every name this corpus defines for the section, in
                registry order. All of them must be present and nothing else
                may be, so that a calibration file from a different corpus is
                rejected instead of silently covering part of this one.
            label: The section's key (``"cases"`` or ``"subseries"``), used in
                every failure message.

        Returns:
            One entry per expected name, in registry order, each holding that
            entry's ``ratio_median``, ``ci_low`` and ``ci_high`` as floats.

        Raises:
            _CalibrationError: If the section is not an object, a name is
                missing or unknown, an entry is not an object, or any of its
                three figures is absent, non-numeric, non-finite, non-positive
                or describes an interval whose bounds are inverted.
        """
        if not isinstance(section, dict):
            raise _CalibrationError(
                f"--calibration {path}: carries no {label!r} object, so it is "
                "not an A/B suite result file"
            )
        missing = [name for name in expected if name not in section]
        if missing:
            raise _CalibrationError(
                f"--calibration {path}: {label} is missing "
                f"{', '.join(missing)} -- this corpus defines "
                f"{len(expected)} {label} and a calibration file has to cover "
                "every one of them"
            )
        unknown = sorted(set(map(str, section)) - set(expected))
        if unknown:
            raise _CalibrationError(
                f"--calibration {path}: {label} carries unknown "
                f"{', '.join(unknown)}, so it was measured on a different "
                "corpus than this run"
            )
        collected: dict[str, Any] = {}
        for name in expected:
            entry = section[name]
            where = f"{label}[{name!r}]"
            if not isinstance(entry, dict):
                raise _CalibrationError(
                    f"--calibration {path}: {where} holds "
                    f"{type(entry).__name__}, not an object"
                )
            ratio_median = _calibration_number(
                entry, "ratio_median", where=where, path=path
            )
            ci_low = _calibration_number(entry, "ci_low", where=where, path=path)
            ci_high = _calibration_number(entry, "ci_high", where=where, path=path)
            if ci_low > ci_high:
                raise _CalibrationError(
                    f"--calibration {path}: {where} has ci_low {ci_low} above "
                    f"ci_high {ci_high}, which is not an interval"
                )
            collected[name] = {
                "ratio_median": ratio_median,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        return collected

    # Identity before figures: passing this run's own A/B result is the likeliest
    # mistake, and its corpus matches, so the identity check is what names it.
    _require_aa_provenance(raw, path)
    cases = figures(
        raw.get("cases"),
        expected=tuple(case.name for case in CASES),
        label="cases",
    )
    subseries = figures(
        raw.get("subseries"),
        expected=tuple(case.name for case in SUBSERIES),
        label="subseries",
    )
    try:
        digest: str | None = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        # The file parsed a moment ago, so a read that fails now leaves the
        # figures perfectly usable and only the digest unknown.
        digest = None
    return {
        "label": "A/A calibration run",
        "source_sha256": digest,
        "generated_at": generated_at,
        "cases": cases,
        "subseries": subseries,
    }


def _allocation_payload(result: _CaseResult) -> dict[str, Any]:
    """Serialise one case's allocation figures under the schema's direct keys.

    Each of the three measured figures is a direct key of the object whose value
    is the per-arm map, so a reader takes
    ``allocation["tracemalloc_peak_bytes"]["candidate"]`` without having to know
    the arm layout first, and ``peak_bytes_ratio`` -- the only gate-bearing
    allocation figure, candidate peak bytes over baseline peak bytes -- sits
    beside them. The per-arm objects are kept as well, so
    ``allocation["candidate"]`` still yields that one arm's three figures
    together; the two forms carry the same numbers.

    Args:
        result: The case's measurements, holding both arms' allocation figures.

    Returns:
        The case's ``allocation`` object.
    """
    baseline = result.baseline_allocation
    candidate = result.candidate_allocation
    return {
        _BASELINE: baseline.payload(),
        _CANDIDATE: candidate.payload(),
        "tracemalloc_peak_bytes": {
            _BASELINE: baseline.tracemalloc_peak_bytes,
            _CANDIDATE: candidate.tracemalloc_peak_bytes,
        },
        "live_blocks_end": {
            _BASELINE: baseline.live_blocks_end,
            _CANDIDATE: candidate.live_blocks_end,
        },
        "max_observed_blocks": {
            _BASELINE: baseline.max_observed_blocks,
            _CANDIDATE: candidate.max_observed_blocks,
        },
        "peak_bytes_ratio": result.peak_bytes_ratio,
    }


def _case_payload(result: _CaseResult, failed: Sequence[str]) -> dict[str, Any]:
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
            _BASELINE: _arm_stats(result.timings(_BASELINE)),
            _CANDIDATE: _arm_stats(result.timings(_CANDIDATE)),
        },
        "allocation": _allocation_payload(result),
        "equivalence": result.equivalence.payload(),
        "verdict": verdict,
        "failed_thresholds": list(failed),
    }


def _build_payload(
    results: Sequence[_CaseResult],
    *,
    warmup: int,
    rounds: int,
    gate_passed: bool,
    checks: Sequence[_GateCheck],
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

    The report's surface is fixed and this is all of it: the environment summary,
    the peak-block-count conflict note, one "A/A calibration" line per gated case
    when a calibration file was given, a table of the gated cases whose verdict
    cell names the threshold a failing case missed and by how much, a second table
    of the informational sub-series without a verdict column, and a single closing
    ``OVERALL:`` line. Every number in it is measured by the run that writes it.

    The gate checklist is deliberately not rendered here: the runner prints it to
    stdout through :func:`_print_checklist` and the JSON carries it item by item
    under ``gate.checks``, so repeating it in the report would add a section the
    report contract does not have.
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
            f"- A/A calibration source: {calibration['label']}, sha256 "
            f"{calibration['source_sha256']} (generated "
            f"{calibration['generated_at']}). Its path is not recorded: that run "
            "writes outside the checkout so the tree stays clean, which makes the "
            "location ephemeral and the digest the durable identifier. The A/A "
            "ratios are the noise floor of this machine; they are diagnostic, not "
            "a gate."
        )
        # One line per gated case, which is what the calibration section is for.
        # The informational sub-series keep their A/A figures in the JSON's
        # ``calibration.subseries`` block and take no part in the gate, so they do
        # not appear here.
        for name, figures in calibration["cases"].items():
            lines.append(
                f"- A/A calibration ({name}): ratio median "
                f"{_format_ratio_or_none(figures['ratio_median'])}, 95% CI "
                f"[{_format_ratio_or_none(figures['ci_low'])}, "
                f"{_format_ratio_or_none(figures['ci_high'])}]"
                f"{_calibration_bias_note(figures)}"
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


def _first_payload_difference(parsed: dict[str, Any], expected: dict[str, Any]) -> str:
    """Name the first top-level payload field whose serialisation differs.

    Args:
        parsed: The payload read back out of the staged artefact.
        expected: The payload the run produced, round-tripped through JSON.

    Returns:
        A description of the first differing or missing field, or of the extra
        fields when every shared field matches.
    """
    for field, value in expected.items():
        if field not in parsed:
            return f"field {field!r} is missing from the staged file"
        if parsed[field] != value:
            return f"field {field!r} differs from the payload the run produced"
    extra = sorted(set(parsed) - set(expected))
    if extra:
        return f"the staged file carries fields the run did not produce: {extra}"
    return "the two payloads compare unequal with no differing field"


def _validate_staged_json(staged: pathlib.Path, payload: dict[str, Any]) -> None:
    """Prove a staged JSON artefact reads back as the payload that was written.

    The whole payload is compared, not a summary of it: the run's own numbers are
    what the artefact exists to publish, so the check is that reparsing the file
    yields exactly the payload the run produced, round-tripped through JSON so
    that both sides are plain JSON types.

    Args:
        staged: The temporary file ``write_json`` just wrote.
        payload: The payload it was given.

    Raises:
        _ArtefactError: If the file cannot be read back, has no trailing
            newline, is not valid JSON, is not an object, or does not reparse to
            this run's payload -- which would mean the writer and the payload
            disagree, and that is not something to discover from a committed
            artefact. A payload that cannot be serialised at all is reported the
            same way rather than as a bare ``TypeError``.
    """
    try:
        text = staged.read_text(encoding="utf-8")
    except OSError as exc:
        raise _ArtefactError(
            f"the staged JSON {staged} could not be read back: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not text.endswith("\n"):
        raise _ArtefactError(f"the staged JSON {staged} has no trailing newline")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _ArtefactError(
            f"the staged JSON {staged} is not valid JSON ({exc})"
        ) from exc
    if not isinstance(parsed, dict):
        raise _ArtefactError(
            f"the staged JSON {staged} is a {type(parsed).__name__}, not an object"
        )
    try:
        expected = json.loads(json.dumps(payload, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise _ArtefactError(
            f"the run payload cannot be represented as JSON, so the staged "
            f"artefact {staged} cannot be verified: {type(exc).__name__}: {exc}"
        ) from exc
    if parsed != expected:
        raise _ArtefactError(
            f"the staged JSON {staged} does not describe this run: "
            f"{_first_payload_difference(parsed, expected)}"
        )


def _validate_staged_report(staged: pathlib.Path, payload: dict[str, Any]) -> None:
    """Prove a staged report artefact carries this run's closing verdict.

    Args:
        staged: The temporary file ``write_report`` just wrote.
        payload: The payload it was given.

    Raises:
        _ArtefactError: If the file cannot be read back, has no trailing
            newline, does not end with the single ``OVERALL:`` line of this
            run's verdict, or was rendered from a payload carrying no verdict at
            all.
    """
    try:
        text = staged.read_text(encoding="utf-8")
    except OSError as exc:
        raise _ArtefactError(
            f"the staged report {staged} could not be read back: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    verdict = payload.get("verdict")
    if not isinstance(verdict, str) or not verdict:
        raise _ArtefactError(
            f"the run payload carries no verdict, so the staged report {staged} "
            f"cannot be verified (found {verdict!r})"
        )
    if not text.endswith("\n"):
        raise _ArtefactError(f"the staged report {staged} has no trailing newline")
    expected = f"OVERALL: {verdict}"
    overall = [line for line in text.splitlines() if line.startswith("OVERALL:")]
    if overall != [expected]:
        raise _ArtefactError(
            f"the staged report {staged} does not close with this run's verdict: "
            f"expected exactly one {expected!r} line, found {overall}"
        )


def _publish_artefact(
    staged: pathlib.Path, destination: pathlib.Path
) -> pathlib.Path | None:
    """Move one staged artefact onto its destination, keeping what it replaced.

    The previous contents are renamed aside first, so the move can be undone if a
    later one in the same pair fails. If the move itself fails, this function
    puts the previous contents straight back before propagating, leaving the
    destination as it found it.

    Args:
        staged: The validated staging file.
        destination: The published path it replaces.

    Returns:
        The path the previous contents were set aside under, or ``None`` when the
        destination did not exist.

    Raises:
        OSError: If the rename could not be performed. The destination is
            unchanged when this happens.
    """
    backup: pathlib.Path | None = None
    if destination.exists():
        backup = destination.with_name(destination.name + _BACKUP_SUFFIX)
        os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except OSError:
        if backup is not None:
            with contextlib.suppress(OSError):
                os.replace(backup, destination)
        raise
    return backup


def _unpublish_artefact(destination: pathlib.Path, backup: pathlib.Path | None) -> None:
    """Undo one published artefact, restoring what was there before it.

    Args:
        destination: The path that was published.
        backup: The set-aside previous contents, or ``None`` when the
            destination did not exist before publication -- in which case
            undoing means removing it again.
    """
    with contextlib.suppress(OSError):
        if backup is None:
            destination.unlink(missing_ok=True)
        else:
            os.replace(backup, destination)


def _write_artefact_pair(
    json_path: pathlib.Path, report_path: pathlib.Path, payload: dict[str, Any]
) -> None:
    """Write the JSON and the report as one pair, or leave both as they were.

    The two files are a single piece of evidence: a JSON from this run beside a
    report from the previous one is worse than no artefact at all, because nothing
    in either file says they disagree. Two things are needed to rule that out, and
    the sequence here does both.

    First, both artefacts are rendered into staging files beside their
    destinations and read back -- the JSON reparsed and compared against the
    whole payload, the report checked for this run's closing verdict -- so a
    rendering failure never reaches a destination at all.

    Second, publication itself is undone on failure. Moving two files is two
    renames, and the second one can fail after the first has succeeded; that is
    exactly how a mixed pair would appear. So each destination's previous
    contents are set aside before it is replaced, and if a later move in the pair
    fails, the earlier ones are put back. Whatever happens, the caller is left
    with both artefacts from this run or both from before it, and no staging or
    set-aside file behind.

    Args:
        json_path: Destination of the JSON artefact.
        report_path: Destination of the report.
        payload: The run payload both artefacts describe.

    Raises:
        _ArtefactError: If either artefact could not be written, could not be
            read back, does not describe this run, or could not be published.
            Both destinations hold their previous contents in every one of those
            cases.
    """
    staged_json = json_path.with_name(json_path.name + _STAGING_SUFFIX)
    staged_report = report_path.with_name(report_path.name + _STAGING_SUFFIX)
    published: list[tuple[pathlib.Path, pathlib.Path | None]] = []
    try:
        try:
            write_json(staged_json, payload)
            write_report(staged_report, payload)
        except Exception as exc:
            # Deliberately broad: the writers render whatever the run produced,
            # and every way that can fail -- a full disk, a read-only directory,
            # a payload field the report expects and cannot find -- has to leave
            # the committed pair untouched and say why, rather than surface as a
            # traceback from between the two writes.
            raise _ArtefactError(
                f"the artefact pair could not be staged in {json_path.parent}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        _validate_staged_json(staged_json, payload)
        _validate_staged_report(staged_report, payload)
        try:
            for staged, destination in (
                (staged_json, json_path),
                (staged_report, report_path),
            ):
                published.append((destination, _publish_artefact(staged, destination)))
        except OSError as exc:
            # Roll the pair back in reverse order, so neither destination is
            # left holding this run's output next to the previous run's.
            for destination, backup in reversed(published):
                _unpublish_artefact(destination, backup)
            raise _ArtefactError(
                f"the staged artefacts could not replace the committed pair in "
                f"{json_path.parent}, which was left as it was: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    finally:
        # Neither a staging file nor a set-aside copy that cannot be removed may
        # mask the failure being raised; both are visible to the next run's
        # dirty check anyway.
        for leftover in (
            staged_json,
            staged_report,
            json_path.with_name(json_path.name + _BACKUP_SUFFIX),
            report_path.with_name(report_path.name + _BACKUP_SUFFIX),
        ):
            with contextlib.suppress(OSError):
                leftover.unlink(missing_ok=True)


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


def _writes_inside_repository(output: pathlib.Path, state: _RepositoryState) -> bool:
    """Whether the resolved output directory lies inside the repository tree.

    Args:
        output: The resolved output directory.
        state: Provenance carrying the repository root.

    Returns:
        ``True`` when the artefacts would land inside the checkout, so that the
        provenance rules apply to them.
    """
    try:
        return output.resolve().is_relative_to(state.root)
    except OSError:
        # An unresolvable path is not inside the tree in any useful sense.
        return False


def _dirty_tree_refusal(
    output: pathlib.Path, state: _RepositoryState
) -> tuple[str, ...] | None:
    """Return the reasons artefacts must not be written, or ``None`` if they may.

    Writing inside the repository is refused unless the tree is provably clean,
    because a committed artefact has to describe an identifiable commit: the
    ``git HEAD`` it records would otherwise not be the code that was measured.
    Two conditions refuse, not one. A dirty tree is the obvious one. The other is
    a tree whose state could not be established at all -- no git executable, a
    directory that is not this repository, a failed ``git status`` -- because
    "unknown" is not "clean", and treating it as clean is precisely how an
    artefact acquires a ``dirty=false`` flag that nothing ever checked.

    Writing into a directory outside the checkout is always allowed -- that is how
    the A/A calibration run happens before the first commit without dirtying
    anything, and such output is not committed evidence.

    Args:
        output: The resolved output directory.
        state: Provenance read at start-up, before any file was written.

    Returns:
        The provenance gaps and porcelain lines behind the refusal, or ``None``
        when writing is fine.
    """
    if not _writes_inside_repository(output, state):
        return None
    reasons = (*state.provenance_gaps, *state.dirty_paths)
    return reasons or None


def _provenance_drift(
    before: _RepositoryState, after: _RepositoryState
) -> tuple[str, ...]:
    """Name every provenance figure that moved between two readings.

    The dirty flag, the commit and the source digest are read before the
    measurement and used again afterwards, in the artefacts. A tree that changed
    in between -- a commit, a checkout, an edit to ``dask/delayed.py`` -- would be
    described by an artefact recording the state that no longer holds, and the
    digest it publishes would not be the code that was measured. The second
    reading is taken while no runner output exists yet, so nothing this run wrote
    can appear as drift.

    Args:
        before: The reading taken at start-up.
        after: The reading taken immediately before the first write.

    Returns:
        One line per figure that differs, empty when the two readings agree.
    """
    drift: list[str] = []
    if after.provenance_gaps:
        drift.extend(
            f"provenance can no longer be established: {gap}"
            for gap in after.provenance_gaps
        )
    if before.git_head != after.git_head:
        drift.append(
            f"git HEAD moved from {before.git_head} to {after.git_head} during "
            "the run"
        )
    if before.delayed_py_sha256 != after.delayed_py_sha256:
        drift.append(
            f"sha256({_BASELINE_SOURCE_PATH}) changed from "
            f"{before.delayed_py_sha256} to {after.delayed_py_sha256} during the "
            "run, so the measured code is not the code on disk"
        )
    if after.dirty:
        drift.extend(
            f"the working tree became dirty during the run: {line}"
            for line in after.dirty_paths
        )
    return tuple(drift)


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


def _failed_thresholds(result: _CaseResult) -> tuple[str, ...]:
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


def _ratio_case_checks(
    name: str, result: _CaseResult | None
) -> tuple[_GateCheck, _GateCheck]:
    """Build the paired-ratio floor and interval floor items for one case.

    Args:
        name: The case named by ``RATIO_CASES``.
        result: Its measurements, or ``None`` when the corpus did not produce
            it. An absent case is a failure of the run, not something to pass
            over silently, so both items are emitted as failed with a measured
            value of zero.

    Returns:
        The ``ratio_<case>`` and ``ci_lower_<case>`` items, in checklist order.
    """
    ratio_measured = 0.0 if result is None else result.ratio_median
    ci_measured = 0.0 if result is None else result.ci_low
    return (
        _GateCheck(
            name=f"ratio_{name}",
            case=name,
            measured=ratio_measured,
            threshold=_RATIO_THRESHOLD,
            passed=result is not None and ratio_measured >= _RATIO_THRESHOLD,
        ),
        _GateCheck(
            name=f"ci_lower_{name}",
            case=name,
            measured=ci_measured,
            threshold=_CI_LOWER_FLOOR,
            passed=result is not None and ci_measured > _CI_LOWER_FLOOR,
        ),
    )


def _improvement_check(gated: Sequence[_CaseResult]) -> _GateCheck:
    """Build the item counting gated cases whose interval clears 1.0.

    Args:
        gated: The gated cases. The informational sub-series are excluded by the
            caller and take no part in the count.

    Returns:
        The ``improved_case_count`` item.
    """
    improved = sum(1 for result in gated if result.ci_low > _CI_LOWER_FLOOR)
    return _GateCheck(
        name="improved_case_count",
        case=None,
        measured=float(improved),
        threshold=float(_MIN_IMPROVED_CASES),
        passed=improved >= _MIN_IMPROVED_CASES,
    )


def _regression_check(gated: Sequence[_CaseResult]) -> _GateCheck:
    """Build the item asserting no gated case is a measured regression.

    Args:
        gated: The gated cases. The weakest interval upper bound carries the
            item, because the gate fails on *any* case below the floor.

    Returns:
        The ``no_regression`` item, failed with no case attributed when the
        corpus produced no gated case at all.
    """
    if not gated:
        return _GateCheck(
            name="no_regression",
            case=None,
            measured=0.0,
            threshold=_REGRESSION_CI_UPPER,
            passed=False,
        )
    weakest = min(gated, key=lambda result: result.ci_high)
    return _GateCheck(
        name="no_regression",
        case=weakest.name,
        measured=weakest.ci_high,
        threshold=_REGRESSION_CI_UPPER,
        passed=weakest.ci_high >= _REGRESSION_CI_UPPER,
    )


def _allocation_check(gated: Sequence[_CaseResult]) -> _GateCheck:
    """Build the peak-allocation tolerance item.

    Args:
        gated: The gated cases. The heaviest peak ratio carries the item, since
            the tolerance applies to every case.

    Returns:
        The ``peak_allocation`` item, failed with no case attributed when the
        corpus produced no gated case at all.

    Raises:
        _MeasurementError: If a case's traced peaks cannot be compared, via
            :attr:`_CaseResult.peak_bytes_ratio`. An unevaluable allocation
            figure is a failed measurement, never a passing item.
    """
    if not gated:
        return _GateCheck(
            name="peak_allocation",
            case=None,
            measured=0.0,
            threshold=_PEAK_ALLOC_TOLERANCE,
            passed=False,
        )
    heaviest = max(gated, key=lambda result: result.peak_bytes_ratio)
    return _GateCheck(
        name="peak_allocation",
        case=heaviest.name,
        measured=heaviest.peak_bytes_ratio,
        threshold=_PEAK_ALLOC_TOLERANCE,
        passed=heaviest.peak_bytes_ratio <= _PEAK_ALLOC_TOLERANCE,
    )


def _equivalence_check(results: Sequence[_CaseResult]) -> _GateCheck:
    """Build the item asserting every case was proven equivalent.

    Args:
        results: Every measured case, gated and informational -- equivalence
            applies to all of them.

    Returns:
        The ``equivalence`` item, which cannot pass for an empty corpus.
    """
    equivalent = sum(1 for result in results if result.equivalence.ok)
    return _GateCheck(
        name="equivalence",
        case=None,
        measured=float(equivalent),
        threshold=float(len(results)),
        passed=bool(results) and equivalent == len(results),
    )


def evaluate_gate(
    results: Sequence[_CaseResult],
) -> tuple[bool, tuple[_GateCheck, ...]]:
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

    Raises:
        _MeasurementError: If a case's allocation figures cannot be compared.
            The gate has no verdict to give over a failed measurement.
    """
    gated = [result for result in results if result.gated]
    by_name = {result.name: result for result in gated}
    checks: list[_GateCheck] = []
    for name in RATIO_CASES:
        checks.extend(_ratio_case_checks(name, by_name.get(name)))
    checks.append(_improvement_check(gated))
    checks.append(_regression_check(gated))
    checks.append(_allocation_check(gated))
    checks.append(_equivalence_check(results))
    return all(check.passed for check in checks), tuple(checks)


def _print_checklist(payload: dict[str, Any]) -> None:
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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


def _corpus() -> tuple[tuple[Case, bool], ...]:
    """Return every case to run, paired with whether it takes part in the gate.

    Returns:
        The six gated cases in gate order followed by the three informational
        sub-series, each with its ``gated`` flag.
    """
    return tuple(
        [(case, True) for case in CASES] + [(case, False) for case in SUBSERIES]
    )


def _report_dirty_tree_refusal(reasons: Sequence[str]) -> None:
    """Explain a refusal to write artefacts inside the repository.

    Args:
        reasons: The provenance gaps and porcelain lines behind the refusal.
    """
    print(
        "refusing to write artefacts inside the repository: a committed "
        "artefact has to describe an identifiable commit, and the recorded git "
        "HEAD would not identify the code that was measured. Offending paths "
        "and unestablished provenance:",
        file=sys.stderr,
    )
    for line in reasons:
        print(f"  {line}", file=sys.stderr)
    print(
        "commit the sources first and rerun, or pass --output with a directory "
        "outside the checkout, or pass --no-artefacts.",
        file=sys.stderr,
    )


@dataclass(frozen=True)
class _Startup:
    """What a run needs once it is known to be configured and trustworthy.

    Attributes:
        baseline: Arm A, the frozen capture, already verified against the source
            it claims to be.
        live: Arm B, the live ``dask.delayed``.
        calibration: The validated A/A calibration block, or ``None`` when
            ``--calibration`` was not given.
        state: Provenance read before any output file exists.
    """

    baseline: ModuleType
    live: ModuleType
    calibration: dict[str, Any] | None
    state: _RepositoryState


def _prepare_run(
    args: argparse.Namespace, output: pathlib.Path, *, write_artefacts: bool
) -> _Startup | int:
    """Configure and vet the run before any measurement is taken.

    Everything that can invalidate a whole run is settled here, in the order
    that costs least: the calibration file, then the two arms, then arm A's
    provenance, then the working tree. Each is a precondition of the next, and
    all of them precede the minutes of measurement -- a run that cannot produce
    trustworthy evidence should not spend that time first.

    Args:
        args: The parsed options.
        output: The resolved output directory.
        write_artefacts: Whether the run intends to write its artefacts, which
            is what makes the working tree's state matter.

    Returns:
        The prepared run, or the exit status the run should end with.
    """
    try:
        calibration = (
            None
            if args.calibration is None
            else _load_calibration(pathlib.Path(args.calibration).expanduser())
        )
    except _CalibrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_GATE_FAIL

    try:
        baseline, live = load_arms()
        # Arm A's provenance is a precondition of every ratio this run reports,
        # so it is verified here: before equivalence, before any timing.
        _baseline_arm()
    except (_ArmLoadError, _ProvenanceError) as exc:
        # Halt and report: neither the capture nor any frozen dask module is
        # edited to make this work.
        print(f"halt: {exc}", file=sys.stderr)
        return _EXIT_GATE_FAIL

    state = _repository_state()
    if write_artefacts:
        refusal = _dirty_tree_refusal(output, state)
        if refusal is not None:
            _report_dirty_tree_refusal(refusal)
            return _EXIT_DIRTY

    print(
        f"arms: A={baseline.__name__} B={live.__name__}; "
        f"{args.warmup} warmup + {args.rounds} measured rounds per case; "
        f"artefacts={'on' if write_artefacts else 'off'} ({output})"
    )
    return _Startup(baseline=baseline, live=live, calibration=calibration, state=state)


def _report_equivalence_mismatch(mismatched: Sequence[_Equivalence]) -> None:
    """Print every equivalence mismatch that ended the run.

    Args:
        mismatched: The verdicts whose ``mismatch`` names the case, the variant,
            the first differing object and the differing field.
    """
    print(
        "equivalence mismatch: no artefact is written and no performance "
        "verdict is produced.",
        file=sys.stderr,
    )
    for verdict in mismatched:
        print(f"  {verdict.mismatch}", file=sys.stderr)


def _check_corpus_equivalence(
    corpus: Sequence[tuple[Case, bool]], baseline: ModuleType, live: ModuleType
) -> tuple[dict[str, _Equivalence], tuple[_Equivalence, ...]]:
    """Prove every case equivalent under both arms, printing each verdict.

    Args:
        corpus: The cases to check, with their gate flags.
        baseline: Arm A.
        live: Arm B.

    Returns:
        The verdict per case name, and the mismatched verdicts. A non-empty
        second element means no timing may be recorded and no artefact written.
    """
    print()
    print("Equivalence (asserted before any timing is recorded):")
    verdicts: dict[str, _Equivalence] = {}
    mismatched: list[_Equivalence] = []
    for case, _gated in corpus:
        verdict = assert_equivalent(case, baseline, live)
        verdicts[case.name] = verdict
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
    return verdicts, tuple(mismatched)


def _validate_rounds(case_name: str, rounds: Sequence[_Round]) -> None:
    """Reject a case whose timing rounds cannot yield a paired ratio.

    Args:
        case_name: The case the rounds belong to.
        rounds: Its measured rounds.

    Raises:
        _MeasurementError: If no round was measured, or if any region recorded a
            non-positive duration. ``perf_counter_ns`` has nanosecond resolution
            and every region of this corpus takes microseconds at least, so a
            zero is a broken measurement rather than a fast one -- and the
            per-round ratio it would feed is not defined. The failure names the
            case, the round, the arm and the region so it can be reproduced.
    """
    if not rounds:
        raise _MeasurementError(
            f"case {case_name!r} produced no measured round, so it has no paired "
            "ratio"
        )
    for index, round_ in enumerate(rounds):
        arms = ((_BASELINE, round_.baseline), (_CANDIDATE, round_.candidate))
        for arm, regions in arms:
            for position, region in enumerate(regions):
                if region.timing_ns <= 0:
                    raise _MeasurementError(
                        f"case {case_name!r}, measured round {index}, arm {arm}, "
                        f"region {position}: recorded {region.timing_ns} ns. A "
                        "region that constructs a graph cannot take no time, so "
                        "this is a failed measurement and not a result"
                    )


def _validate_allocation(case_name: str, allocation: dict[str, _ArmAllocation]) -> None:
    """Reject a case whose traced peaks cannot be compared.

    Args:
        case_name: The case the figures belong to.
        allocation: The arm-keyed allocation figures.

    Raises:
        _MeasurementError: If an arm is missing, or if either arm's traced peak
            is not positive. The allocation item of the gate is evaluated on
            those two numbers, and it must never pass on a figure that was never
            measured.
    """
    for arm in (_BASELINE, _CANDIDATE):
        figures = allocation.get(arm)
        if figures is None:
            raise _MeasurementError(
                f"case {case_name!r}: no allocation figures were measured for "
                f"arm {arm}"
            )
        if figures.tracemalloc_peak_bytes <= 0:
            raise _MeasurementError(
                f"case {case_name!r}, arm {arm}: tracemalloc recorded a peak of "
                f"{figures.tracemalloc_peak_bytes} bytes for a region that "
                "constructs a graph, so the allocation tolerance cannot be "
                "evaluated"
            )


def _measure_corpus(
    corpus: Sequence[tuple[Case, bool]],
    baseline: ModuleType,
    live: ModuleType,
    equivalences: dict[str, _Equivalence],
    *,
    warmup: int,
    rounds: int,
) -> list[_CaseResult]:
    """Time and measure every case, validating each result as it is produced.

    Args:
        corpus: The cases to measure, with their gate flags.
        baseline: Arm A.
        live: Arm B.
        equivalences: The verdicts proven before any timing was recorded.
        warmup: Rounds to discard per case.
        rounds: Rounds to keep per case.

    Returns:
        One result per case, in corpus order.

    Raises:
        _MeasurementError: If a case's timings or allocation figures cannot be
            interpreted. Validation happens per case, as it is measured, so the
            failure names the case rather than surfacing later as an arithmetic
            oddity in the payload.
    """
    print()
    print("Timing and allocation:")
    results: list[_CaseResult] = []
    for case, gated in corpus:
        measured = time_case(case, baseline, live, warmup=warmup, rounds=rounds)
        _validate_rounds(case.name, measured)
        allocation = measure_allocations(case, baseline, live)
        _validate_allocation(case.name, allocation)
        ratios = [round_.ratio for round_ in measured]
        ci_low, ci_high = _bootstrap_ci(ratios)
        result = _CaseResult(
            name=case.name,
            pure=None,
            gated=gated,
            rounds=measured,
            baseline_allocation=allocation[_BASELINE],
            candidate_allocation=allocation[_CANDIDATE],
            equivalence=equivalences[case.name],
            ratio_median=float(statistics.median(ratios)),
            ci_low=ci_low,
            ci_high=ci_high,
        )
        results.append(result)
        print(
            f"  {case.name}: baseline median "
            f"{_format_ms(_arm_stats(result.timings(_BASELINE))['median'])} ms, "
            f"candidate median "
            f"{_format_ms(_arm_stats(result.timings(_CANDIDATE))['median'])} ms, "
            f"ratio median {result.ratio_median:.3f}, 95% CI "
            f"{_format_ci(result.ci_low, result.ci_high)}",
            flush=True,
        )
    return results


def _run_payload(
    results: Sequence[_CaseResult],
    state: _RepositoryState,
    calibration: dict[str, Any] | None,
    *,
    warmup: int,
    rounds: int,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate the gate and assemble the payload both artefacts describe.

    Args:
        results: Every measured case.
        state: Provenance read at start-up.
        calibration: The validated A/A block, or ``None``.
        warmup: Discarded rounds per case.
        rounds: Measured rounds per case.

    Returns:
        Whether the gate passed, and the payload.

    Raises:
        _MeasurementError: If a case's allocation figures cannot be compared.
        _ProvenanceError: If arm A stopped being the frozen capture while the
            run was in progress -- the environment block re-reads it, and a
            capture edited mid-run invalidates every ratio just as one edited
            beforehand would.
    """
    gate_passed, checks = evaluate_gate(results)
    flat_loop_median = next(
        (
            _arm_stats(result.timings(_CANDIDATE))["median"]
            for result in results
            if result.name == _FLAT_LOOP_CASE
        ),
        None,
    )
    payload = _build_payload(
        results,
        warmup=warmup,
        rounds=rounds,
        gate_passed=gate_passed,
        checks=checks,
        environment_block=environment(
            repository=state,
            pythonhashseed=os.environ.get("PYTHONHASHSEED"),
            flat_loop_median_ns=flat_loop_median,
        ),
        calibration=calibration,
    )
    return gate_passed, payload


def _emit_artefacts(
    output: pathlib.Path, payload: dict[str, Any], state: _RepositoryState
) -> int | None:
    """Re-check provenance, then write the artefact pair transactionally.

    The provenance in the payload was read before a measurement run that takes
    minutes. This is the last moment at which it can be confirmed, and the only
    moment at which it can be confirmed *cleanly*: no runner output exists yet,
    so the second reading sees only what the tree did on its own. A commit, a
    checkout or an edit to ``dask/delayed.py`` during the run would otherwise be
    published as this run's ``dirty=false``, ``git_head`` and source digest --
    provenance describing a state that no longer holds.

    Drift refuses only for an in-repository write, where the artefacts are
    committed evidence. Output outside the checkout -- the A/A calibration idiom,
    which legitimately runs before the first commit -- is written with the drift
    reported on stderr, because those files are diagnostics, not evidence.

    Args:
        output: The resolved output directory.
        payload: The run payload.
        state: Provenance read at start-up.

    Returns:
        ``_EXIT_DIRTY`` when an in-repository write was refused because
        provenance drifted, otherwise ``None``.

    Raises:
        _ArtefactError: If the pair could not be written as one consistent pair.
    """
    inside = _writes_inside_repository(output, state)
    drift = _provenance_drift(state, _repository_state())
    if drift:
        if inside:
            print(
                "refusing to write artefacts: the repository's provenance "
                "changed while the run was in progress, so the recorded HEAD, "
                "dirty flag and source digest no longer describe the measured "
                "code. Drift:",
                file=sys.stderr,
            )
            for line in drift:
                print(f"  {line}", file=sys.stderr)
            print(
                "rerun from a clean commit so the artefacts describe the code "
                "they measured.",
                file=sys.stderr,
            )
            return _EXIT_DIRTY
        print(
            "warning: the repository's provenance changed during the run, and "
            "the recorded values are the ones read at start-up:",
            file=sys.stderr,
        )
        for line in drift:
            print(f"  {line}", file=sys.stderr)

    json_path = output / _JSON_NAME
    report_path = output / _REPORT_NAME
    _write_artefact_pair(json_path, report_path, payload)
    print()
    print(f"wrote {json_path}")
    print(f"wrote {report_path}")
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the suite and return the process exit status.

    The order is fixed and each step is a precondition for the next: fix the hash
    seed, parse the options, load and validate the calibration file, load both
    arms, verify that arm A is the frozen capture, read the repository
    provenance, refuse an untrustworthy tree before spending minutes on
    measurement, prove every case equivalent, time the cases, measure their
    allocations, evaluate the gate, print the checklist, re-check provenance and
    only then write the artefacts.

    Args:
        argv: The argument list, or ``None`` to read ``sys.argv``. An explicit
            list is threaded through the hash-seed re-exec, so a programmatic
            call runs the options it was given rather than whatever happens to
            be on the real command line.

    Returns:
        0 when every gate item held, 1 when the gate failed or the run could not
        be configured or trusted, 2 on an equivalence mismatch, 3 on a refusal to
        write into a working tree that is dirty, unverifiable, or changed while
        the run was in progress.
    """
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    _ensure_fixed_hash_seed(effective_argv)
    args = _parse_args(effective_argv)
    output = _resolve_output(args.output)
    write_artefacts = not args.no_artefacts

    prepared = _prepare_run(args, output, write_artefacts=write_artefacts)
    if isinstance(prepared, int):
        return prepared

    corpus = _corpus()
    equivalences, mismatched = _check_corpus_equivalence(
        corpus, prepared.baseline, prepared.live
    )
    if mismatched:
        _report_equivalence_mismatch(mismatched)
        return _EXIT_EQUIVALENCE

    try:
        results = _measure_corpus(
            corpus,
            prepared.baseline,
            prepared.live,
            equivalences,
            warmup=args.warmup,
            rounds=args.rounds,
        )
        gate_passed, payload = _run_payload(
            results,
            prepared.state,
            prepared.calibration,
            warmup=args.warmup,
            rounds=args.rounds,
        )
    except (_MeasurementError, _ProvenanceError) as exc:
        # A measurement that cannot be interpreted, or an arm A that stopped
        # being the frozen capture while the run was in progress, has no
        # verdict: no artefact is written, and the gate is reported as neither
        # passed nor failed.
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_GATE_FAIL

    _print_checklist(payload)

    if write_artefacts:
        try:
            refused = _emit_artefacts(output, payload, prepared.state)
        except _ArtefactError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return _EXIT_GATE_FAIL
        if refused is not None:
            return refused

    return _EXIT_PASS if gate_passed else _EXIT_GATE_FAIL
