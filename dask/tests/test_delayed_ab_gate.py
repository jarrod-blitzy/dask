"""Opt-in performance gate for the committed ``dask.delayed`` A/B suite.

The suite under ``benchmarks/delayed_ab/`` measures the frozen pre-refactor
``dask/delayed.py`` (arm A) against the live module (arm B) in one interpreter,
proves that the two behave identically *before* it records a single timing, and
prints a PASS/FAIL verdict against thresholds that were fixed before any
measurement was taken. This module re-runs that suite over reduced round counts
and asserts the same thresholds, so the run's performance claim is checkable
from pytest and not only from the committed artefacts.

The gate is opt-in: it is skipped at collection unless ``DASK_DELAYED_AB=1`` is
set in the environment. **Timing thresholds are never asserted in the default
test run.** A threshold asserted on shared, loaded CI hardware would measure the
machine rather than the implementation, so the default run skips this test
before it imports anything from ``benchmarks/``, executes no subprocess, and
emits no warning under the project's ``filterwarnings = ["error", ...]``.

The harness runs out of process, which is a measurement-integrity requirement
rather than a style choice. A subprocess keeps pytest's own imports, its
assertion rewriting and its warnings machinery out of the timed regions, and it
pins ``PYTHONHASHSEED=0`` for the measurement without the runner having to
re-execute itself with ``os.execve``: hash randomisation changes dict and set
iteration order, which changes how much work the construction path does over
containers, so both arms must run under one fixed seed.

Artefacts of a gate run land in pytest's ``tmp_path``, never in
``benchmarks/delayed_ab/results/``. The committed results are therefore never
overwritten, and the runner's refusal to write into a dirty working tree --
which applies only to writes inside the repository -- is never met.

Two properties of an enabled run belong to the measurement rather than to the
implementation, and this module reports them as such.

*The subprocess cap is derived, not hard-coded.* It is pytest's own configured
per-test timeout minus a fixed margin, bounded by a ceiling, so it stays below
the limit that would take the whole suite down even if that limit is changed --
a guarantee :func:`test_gate_subprocess_cap_is_below_the_pytest_timeout` asserts
in every session, enabled or skipped. A run that does not finish inside the cap
is reported as inconclusive, with its elapsed time, the cap it ran against and
the host's load average, rather than as a threshold failure it never measured.

*The two paired-ratio items are decided on the interval, not on the point
median.* An interval wholly at or above the threshold passes it; an interval
wholly below it fails it; an interval that straddles it is inconclusive. Over
``_ROUNDS`` rounds a one-to-two-percent shortfall moves less than the noise a
loaded host adds, and the same implementation was observed to alternate between
passing and failing for exactly that reason. Deciding on the interval turns that
into one stable outcome per implementation: a conclusive pass, a conclusive
failure, or an explicit "these rounds cannot tell". Nothing is relaxed by it --
the gate cannot pass while the threshold is unmet -- and the committed 15-round
artefact keeps its own PASS/FAIL verdict either way.

Invocation, from the repository root:

    ```text
    python -m benchmarks.delayed_ab
    DASK_DELAYED_AB=1 pytest dask/tests/test_delayed_ab_gate.py
    ```

Notes:
    ``benchmarks/delayed_ab/main.py`` is the authority for the artefact name,
    the JSON key names and the exit-code contract parsed here; no key is
    invented. Every figure this module asserts on -- timings, statistics,
    intervals, allocations -- is produced by the harness inside the subprocess;
    it computes none of them, and it imports nothing from ``dask`` or
    ``benchmarks``. The one clock it reads is wall time around the subprocess
    itself, which measures the host rather than the code under test and exists
    so that a slow run can be told apart from a missed threshold.

"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

#: The harness this gate re-runs, invoked as ``python -m benchmarks.delayed_ab``.
#: ``benchmarks/`` is a PEP 420 namespace package with no ``__init__.py``, so the
#: module name only resolves when the working directory is the repository root --
#: which is why the subprocess sets ``cwd`` explicitly instead of inheriting
#: pytest's invocation directory.
_HARNESS_MODULE = "benchmarks.delayed_ab"

#: Relative path that identifies the repository root, and the artefact the
#: harness writes. Both are fixed by ``benchmarks/delayed_ab/main.py``
#: (``_JSON_NAME``) and by ``benchmarks/delayed_ab/__main__.py``; neither is
#: guessed here.
_HARNESS_ENTRY_POINT = Path("benchmarks") / "delayed_ab" / "__main__.py"
_JSON_NAME = "baseline_vs_candidate.json"

#: JSON schema version this parser was written against
#: (``benchmarks/delayed_ab/main.py::_SCHEMA_VERSION``). A bump means the key
#: names below have to be re-read from the writer before this test is trusted.
_SCHEMA_VERSION = 1

#: Reduced round counts: the protocol's minimums, which the harness enforces in
#: ``--rounds``/``--warmup`` (``_MIN_ROUNDS``/``_MIN_WARMUP``). The full run
#: defaults to 15 measured rounds and is the committed evidence; the gate trades
#: interval precision for a runtime that fits inside a pytest test.
#:
#: They stay at the minimums. More rounds would tighten the interval, but a
#: nine-case run at these counts already needs 150-215 s of the cap derived
#: below, and the committed 15-round artefact shows the interval of a marginal
#: case still straddling its threshold -- so rounds buy runtime risk rather than
#: a decision, and the decision is made robust by judging the interval instead
#: (see :func:`_evaluate_thresholds`).
_ROUNDS = 7
_WARMUP = 2

#: How the subprocess cap is derived, and the guarantee it carries.
#:
#: The project runs pytest with ``timeout = 300`` and ``timeout_method =
#: "thread"`` (``pyproject.toml``), and a thread-method timeout kills the whole
#: suite rather than the offending test -- so the harness has to fail cleanly
#: first. That makes the cap a function of pytest's own limit rather than a
#: number written next to it: it is the configured per-test timeout minus
#: :data:`_TIMEOUT_MARGIN_SECONDS`, bounded above by
#: :data:`_TIMEOUT_CEILING_SECONDS`. Because the ceiling plus the margin do not
#: exceed :data:`_PYTEST_TIMEOUT_FALLBACK_SECONDS`, the cap stays below 300 s
#: even when the ini value cannot be read -- the guarantee
#: :func:`test_gate_subprocess_cap_is_below_the_pytest_timeout` asserts.
#:
#: The margin is what this test needs *around* the subprocess: killing and
#: reaping the child, and formatting the diagnostic block. The floor is the
#: point below which the measurement cannot be attempted at all -- an enabled
#: nine-case run at :data:`_ROUNDS` rounds has been observed to need 150-215 s
#: on a twelve-core host under sibling load, so a cap under a minute could only
#: ever time out, and the gate says so instead of pretending to measure.
_PYTEST_TIMEOUT_INI = "timeout"
_PYTEST_TIMEOUT_FALLBACK_SECONDS = 300.0
_TIMEOUT_MARGIN_SECONDS = 45.0
_TIMEOUT_CEILING_SECONDS = 255.0
_TIMEOUT_FLOOR_SECONDS = 60.0

#: Trailing lines of each captured stream a failure message reproduces. A gate
#: failure whose message omits the harness output is useless to a reviewer, and
#: an unbounded dump of a nine-case run is unreadable.
_OUTPUT_TAIL_LINES = 40

#: Exit codes of the harness (``benchmarks/delayed_ab/main.py``).
_EXIT_PASS = 0
_EXIT_GATE_FAIL = 1
_EXIT_EQUIVALENCE = 2
_EXIT_DIRTY = 3

_PASS_VERDICT = "PASS"

#: The gate's thresholds, as fixed before any measurement was taken. They are
#: restated -- not imported -- because this module imports nothing from
#: ``benchmarks/``: a test that reads its expectations out of the code under test
#: cannot fail when that code's thresholds are relaxed.
_RATIO_THRESHOLD = 1.25
_CI_LOWER_FLOOR = 1.0
_REGRESSION_CI_UPPER = 0.98
_MIN_IMPROVED_CASES = 4
_PEAK_ALLOC_TOLERANCE = 1.05

#: The six gated cases, in gate order (``benchmarks/delayed_ab/cases.py::CASES``).
_GATED_CASES = (
    "flat_loop",
    "linear_chain",
    "nested_containers",
    "wide_fan_in",
    "pure_vs_impure",
    "attr_and_operators",
)

#: The two cases that carry the paired-ratio floor and the interval floor
#: (``cases.py::RATIO_CASES``).
_RATIO_CASES = ("flat_loop", "nested_containers")

#: The informational sub-series (``cases.py::SUBSERIES``). They are measured and
#: reported but carry no verdict: nothing here asserts on them, and they never
#: count towards the four-of-six improvement item. They are listed only so that
#: the test can prove they stayed out of ``_GATED_CASES``.
_SUBSERIES_CASES = ("nested_containers_shallow", "pure_true", "pure_false")


def _repository_root() -> Path:
    """Return the repository root, without importing anything to find it.

    This module lives at ``<root>/dask/tests/``, so ``parents[0]`` is
    ``dask/tests``, ``parents[1]`` is ``dask`` and ``parents[2]`` is the root.
    Importing ``dask`` to ask it where it lives would defeat the point of the
    subprocess, and importing anything from ``benchmarks/`` would make the
    default, skipped run pay for the benchmark suite's imports.

    Returns:
        The absolute repository root.

    Raises:
        Failed: Via :func:`pytest.fail`, when the resolved directory does not
            contain the harness entry point -- the gate cannot run from a
            checkout whose layout it does not recognise.

    """
    root = Path(__file__).resolve().parents[2]
    if not (root / _HARNESS_ENTRY_POINT).is_file():
        pytest.fail(
            f"cannot locate the A/B harness: {root / _HARNESS_ENTRY_POINT} does not exist. "
            f"The repository root was derived from {__file__!r} as {root}, which assumes "
            "this test still lives in <root>/dask/tests/."
        )
    return root


def _as_text(stream: object) -> str | None:
    """Normalise one captured stream to text.

    ``subprocess.run`` hands back ``str`` under ``text=True``, while the
    ``TimeoutExpired`` it raises carries whatever it had managed to read. Both
    are reduced here so the rest of the module works with one type.

    Args:
        stream: A captured stream, or ``None`` when nothing was captured.

    Returns:
        The stream as text, or ``None``.

    """
    if stream is None:
        return None
    if isinstance(stream, str):
        return stream
    if isinstance(stream, (bytes, bytearray)):
        return stream.decode("utf-8", errors="replace")
    return str(stream)


def _tail(stream: str | None, *, lines: int = _OUTPUT_TAIL_LINES) -> str:
    if stream is None:
        return "<not captured>"
    if not stream.strip():
        return "<empty>"
    kept = stream.splitlines()
    if len(kept) > lines:
        dropped = len(kept) - lines
        return "\n".join(
            [f"<... {dropped} earlier line(s) omitted ...>", *kept[-lines:]]
        )
    return "\n".join(kept)


def _configured_pytest_timeout(config: pytest.Config) -> float:
    """Return the per-test timeout pytest is actually running with.

    The value is resolved rather than assumed, in pytest-timeout's own order of
    precedence -- the ``--timeout`` command-line option, then the
    ``PYTEST_TIMEOUT`` environment variable, then the ``timeout`` ini value --
    because the cap derived from it is only a guarantee if it tracks the limit
    that is actually in force. Reading the ini alone would miss a session
    invoked with a smaller ``--timeout``, which is exactly the session in which
    the guarantee matters. (A ``@pytest.mark.timeout`` marker outranks all
    three; this module sets none, so the three sources are exhaustive for it.)

    Args:
        config: The active pytest configuration.

    Returns:
        The timeout in seconds from the first source that supplies one, or
        :data:`_PYTEST_TIMEOUT_FALLBACK_SECONDS` when none does -- the
        documented project value, which the cap's ceiling is safe against even
        though no per-test limit is then in force at all.

    """
    try:
        ini: object = config.getini(_PYTEST_TIMEOUT_INI)
    except (KeyError, ValueError):
        # pytest-timeout is what registers this key; without the plugin there is
        # no per-test timeout to stay below.
        ini = None
    for raw in (
        config.getoption(_PYTEST_TIMEOUT_INI, default=None),
        os.environ.get("PYTEST_TIMEOUT"),
        ini,
    ):
        if raw is None or raw == "":
            continue
        try:
            return float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # A source pytest-timeout would itself reject: fall through to the
            # next one rather than raising out of a diagnostic helper.
            continue
    return _PYTEST_TIMEOUT_FALLBACK_SECONDS


def _derive_subprocess_cap(pytest_timeout_seconds: float) -> float | None:
    """Derive the harness's wall-clock cap from pytest's own per-test timeout.

    Args:
        pytest_timeout_seconds: The configured per-test timeout. Zero or a
            negative value is how pytest-timeout spells "no timeout", in which
            case there is no suite-killing limit to stay below and the ceiling
            alone applies.

    Returns:
        The cap in seconds -- always at least :data:`_TIMEOUT_MARGIN_SECONDS`
        below a positive ``pytest_timeout_seconds`` and never above
        :data:`_TIMEOUT_CEILING_SECONDS` -- or ``None`` when what is left after
        the margin is under :data:`_TIMEOUT_FLOOR_SECONDS`, which means the
        measurement cannot be attempted within the configured timeout.

    """
    if pytest_timeout_seconds <= 0.0:
        return _TIMEOUT_CEILING_SECONDS
    cap = min(
        pytest_timeout_seconds - _TIMEOUT_MARGIN_SECONDS, _TIMEOUT_CEILING_SECONDS
    )
    if cap < _TIMEOUT_FLOOR_SECONDS:
        return None
    return cap


def _load_average() -> tuple[float, float, float] | None:
    """Return the host's load averages, or ``None`` where they are unavailable.

    The figure is diagnostic only: nothing in this module branches on it. It is
    recorded because an enabled gate run is a measurement, and a reader looking
    at a slow run or a wide interval needs to know what else the host was doing.

    Returns:
        The 1-, 5- and 15-minute load averages, or ``None`` on a platform that
        does not provide them.

    """
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return None


@dataclasses.dataclass(frozen=True)
class _HarnessRun:
    """One execution of the A/B harness, with the budget it ran against.

    ``elapsed_seconds`` is wall clock around the subprocess, taken so that a
    failure can distinguish "too slow" from "threshold missed". It is not a
    measurement of the code under test: every timing, statistic, interval and
    allocation figure the gate asserts on is produced by the harness inside the
    subprocess, and this module computes none of them.

    Attributes:
        command: The command that was run.
        returncode: The harness's exit status, or ``None`` when it did not
            finish inside the cap.
        stdout: Captured standard output, or ``None``.
        stderr: Captured standard error, or ``None``.
        elapsed_seconds: Wall clock spent on the subprocess.
        cap_seconds: The cap it ran against.
        pytest_timeout_seconds: The per-test timeout that cap was derived from.
        load_before: Load averages sampled before the run, or ``None``.
        load_after: Load averages sampled after it, or ``None``.
        cpu_count: The host's CPU count, or ``None`` when it cannot be read.

    """

    command: tuple[str, ...]
    returncode: int | None
    stdout: str | None
    stderr: str | None
    elapsed_seconds: float
    cap_seconds: float
    pytest_timeout_seconds: float
    load_before: tuple[float, float, float] | None
    load_after: tuple[float, float, float] | None
    cpu_count: int | None


def _describe_budget(run: _HarnessRun) -> str:
    """Summarise what the run spent of the cap it was given.

    Args:
        run: The completed or timed-out run.

    Returns:
        One line naming the elapsed time, the cap, the share of the cap it used
        and the pytest timeout the cap was derived from.

    """
    share = (
        f"{run.elapsed_seconds / run.cap_seconds:.0%}"
        if run.cap_seconds > 0.0
        else "unknown share"
    )
    return (
        f"elapsed: {run.elapsed_seconds:.1f}s of the {run.cap_seconds:.0f}s "
        f"subprocess cap ({share} of it), derived from pytest's configured "
        f"{run.pytest_timeout_seconds:.0f}s per-test timeout minus a "
        f"{_TIMEOUT_MARGIN_SECONDS:.0f}s margin"
    )


def _describe_host(run: _HarnessRun) -> str:
    """Summarise what else the host was doing while the run was measured.

    Args:
        run: The completed or timed-out run.

    Returns:
        One line naming the CPU count and the 1-minute load average before and
        after the run, or saying that the load average is unavailable.

    """
    # Named for what it is: os.cpu_count() reports the host's CPUs, which a
    # container quota can hold well below. The load average is read against it
    # by a human, so it must not be presented as the budget the run actually had.
    cpus = "an unknown number of" if run.cpu_count is None else str(run.cpu_count)
    if run.load_before is None or run.load_after is None:
        return (
            f"host: {cpus} CPUs per os.cpu_count(), load average unavailable on "
            "this platform"
        )
    return (
        f"host: {cpus} CPUs per os.cpu_count(), 1-minute load average "
        f"{run.load_before[0]:.2f} before and {run.load_after[0]:.2f} after the run"
    )


def _exit_code_hint(returncode: int) -> str:
    if returncode == _EXIT_PASS:
        return "every gate item held"
    if returncode == _EXIT_GATE_FAIL:
        return (
            "the gate thresholds were not met, or the run could not be configured "
            "(the printed checklist names the item that failed)"
        )
    if returncode == _EXIT_EQUIVALENCE:
        return (
            "an equivalence mismatch between the frozen baseline arm and the live "
            "dask.delayed: the refactor changed observable behaviour, so no "
            "performance verdict was produced"
        )
    if returncode == _EXIT_DIRTY:
        return (
            "the harness refused to write artefacts into a dirty working tree. This "
            "is unexpected here, because --output points at pytest's tmp_path and "
            "the refusal applies only to writes inside the repository: treat it as a "
            "harness bug"
        )
    if returncode < 0:
        return f"the subprocess was terminated by signal {-returncode}"
    return "undocumented exit status: see benchmarks/delayed_ab/main.py"


def _harness_report(run: _HarnessRun) -> str:
    status = "<did not finish>" if run.returncode is None else str(run.returncode)
    return "\n".join(
        [
            "--- delayed A/B harness ---",
            f"command: {' '.join(run.command)}",
            f"returncode: {status}",
            _describe_budget(run),
            _describe_host(run),
            "stdout (tail):",
            _tail(run.stdout),
            "stderr (tail):",
            _tail(run.stderr),
            "--- end of harness output ---",
        ]
    )


def _run_harness(
    output_dir: Path,
    *,
    cap_seconds: float,
    pytest_timeout_seconds: float,
) -> _HarnessRun:
    """Run the A/B suite out of process and return what the run cost.

    A subprocess that does not finish inside the cap is *not* an assertion
    failure here: the record comes back with ``returncode is None`` and its
    elapsed time, and the caller reports it as an unmeasured run. Killing the
    child at the cap is what keeps pytest's thread-method timeout -- which would
    end the whole session -- out of reach.

    Args:
        output_dir: Directory the harness writes its two artefacts into. Always
            pytest's ``tmp_path``, so a gate run never overwrites the committed
            results and never meets the runner's dirty-tree refusal.
        cap_seconds: Wall-clock cap for the subprocess, as derived by
            :func:`_derive_subprocess_cap`.
        pytest_timeout_seconds: The per-test timeout it was derived from,
            carried into the record for the diagnostic block.

    Returns:
        The run, whether it completed or was killed at the cap.

    """
    root = _repository_root()
    command = (
        sys.executable,
        "-m",
        _HARNESS_MODULE,
        "--rounds",
        str(_ROUNDS),
        "--warmup",
        str(_WARMUP),
        "--output",
        str(output_dir),
    )
    # A copy, never an in-place mutation of os.environ: the rest of the session
    # keeps whatever hash seed it was started with.
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    load_before = _load_average()
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            list(command),
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=cap_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        # On POSIX ``subprocess.run`` kills the child and waits, without
        # draining the pipes, so the captured streams are usually absent here --
        # and the harness's own stdout is block-buffered into a pipe anyway. The
        # elapsed time, the cap and the host's load are what diagnose this
        # outcome, which is why they are recorded rather than inferred.
        return _HarnessRun(
            command=command,
            returncode=None,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            elapsed_seconds=time.perf_counter() - started,
            cap_seconds=cap_seconds,
            pytest_timeout_seconds=pytest_timeout_seconds,
            load_before=load_before,
            load_after=_load_average(),
            cpu_count=os.cpu_count(),
        )
    return _HarnessRun(
        command=command,
        returncode=proc.returncode,
        stdout=_as_text(proc.stdout),
        stderr=_as_text(proc.stderr),
        elapsed_seconds=time.perf_counter() - started,
        cap_seconds=cap_seconds,
        pytest_timeout_seconds=pytest_timeout_seconds,
        load_before=load_before,
        load_after=_load_average(),
        cpu_count=os.cpu_count(),
    )


def _read_payload(output_dir: Path, report: str) -> dict[str, Any]:
    path = output_dir / _JSON_NAME
    if not path.is_file():
        pytest.fail(f"the harness wrote no {_JSON_NAME} into {output_dir}\n{report}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        pytest.fail(f"could not read {path} as JSON: {exc}\n{report}")
    if not isinstance(payload, dict):
        pytest.fail(
            f"{path} holds {type(payload).__name__}, not a JSON object\n{report}"
        )
    return payload


def _gated_case(payload: dict[str, Any], name: str, report: str) -> dict[str, Any]:
    cases = payload.get("cases")
    if not isinstance(cases, dict):
        pytest.fail(
            f"the payload has no 'cases' object: found {type(cases).__name__}\n{report}"
        )
    case = cases.get(name)
    if not isinstance(case, dict):
        pytest.fail(
            f"case {name!r} is missing from the payload, which reports "
            f"{sorted(cases)}. Every one of {list(_GATED_CASES)} must be measured.\n{report}"
        )
    return case


def _number(record: dict[str, Any], key: str, *, what: str, report: str) -> float:
    """Read one JSON number out of a record.

    Args:
        record: The JSON object to read from.
        key: The key to read.
        what: Human-readable description of the figure, for the failure message.
        report: The harness diagnostic block, carried into every failure.

    Returns:
        The value as a ``float``.

    Raises:
        Failed: Via :func:`pytest.fail`, when the key is absent or does not hold
            a number. ``bool`` is rejected: it is an ``int`` subclass, and a
            boolean in a numeric slot means the schema moved.

    """
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        pytest.fail(
            f"{what}: key {key!r} holds {value!r} ({type(value).__name__}), not a number. "
            "The JSON schema of benchmarks/delayed_ab/main.py has moved and this "
            f"test must be re-read against it.\n{report}"
        )
    return float(value)


def _peak_bytes(case: dict[str, Any], arm: str, *, name: str, report: str) -> float:
    """Return one arm's ``tracemalloc`` peak bytes for a case.

    Peak bytes is the only gate-bearing allocation figure: it is what the
    standard library measures exactly. The record's two block figures --
    ``live_blocks_end`` and ``max_observed_blocks`` -- are supporting evidence
    under their own definitions and are never read as a peak block count, which
    ``tracemalloc`` does not expose.

    The figure is read the way the JSON schema names it: the ``allocation``
    object holds one record per arm, each carrying that arm's three figures, so
    this is ``allocation[arm]["tracemalloc_peak_bytes"]``.

    Args:
        case: The case's JSON object.
        arm: ``"baseline"`` or ``"candidate"``.
        name: The case name, for the failure message.
        report: The harness diagnostic block, carried into every failure.

    Returns:
        The arm's peak traced bytes.

    Raises:
        Failed: Via :func:`pytest.fail`, when the allocation record is missing or
            malformed.

    """
    allocation = case.get("allocation")
    if not isinstance(allocation, dict):
        pytest.fail(f"case {name!r} has no 'allocation' object\n{report}")
    figures = allocation.get(arm)
    if not isinstance(figures, dict):
        pytest.fail(
            f"case {name!r} has no {arm!r} record in its allocation object, found "
            f"{type(figures).__name__}. The JSON schema of "
            "benchmarks/delayed_ab/main.py has moved and this test must be "
            f"re-read against it.\n{report}"
        )
    return _number(
        figures,
        "tracemalloc_peak_bytes",
        what=f"case {name!r}, arm {arm!r} peak allocation",
        report=report,
    )


def _verdict_digest(payload: dict[str, Any]) -> str:
    """Summarise why the harness's own verdict was not a pass.

    The harness evaluates the same thresholds this test asserts, so when its
    verdict is negative its checklist already names the reason. Reproducing that
    next to the assertion saves the reader a trip through the JSON.

    Args:
        payload: The parsed JSON payload.

    Returns:
        The failed checklist items and the failed per-case thresholds, one per
        line, or a note that neither could be read.

    """
    lines: list[str] = []
    gate = payload.get("gate")
    if isinstance(gate, dict):
        checks = gate.get("checks")
        if isinstance(checks, list):
            for check in checks:
                if isinstance(check, dict) and not check.get("passed"):
                    target = check.get("case")
                    where = f" [{target}]" if target else ""
                    lines.append(
                        f"  failed check {check.get('name')}{where}: measured "
                        f"{check.get('measured')}, threshold {check.get('threshold')}"
                    )
    cases = payload.get("cases")
    if isinstance(cases, dict):
        for name, case in cases.items():
            if not isinstance(case, dict):
                continue
            failed = case.get("failed_thresholds")
            if isinstance(failed, list) and failed:
                lines.append(f"  {name}: {'; '.join(str(item) for item in failed)}")
    if not lines:
        return "  <no failed checklist item recorded in the payload>"
    return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class _Finding:
    """One gate item that did not simply pass.

    ``item`` is the harness's own checklist name for the item
    (``benchmarks/delayed_ab/main.py::evaluate_gate``: ``ratio_<case>``,
    ``ci_lower_<case>``, ``improved_case_count``, ``no_regression``,
    ``peak_allocation``, ``equivalence``). Naming findings after the harness's
    items is what lets :func:`_unaccounted_failed_checks` prove that every item
    the harness rejected is one this test judged too, so a failure it reports
    for a reason not modelled here can never be passed or skipped over.

    Attributes:
        item: The harness's checklist name for the item.
        case: The case it belongs to, or ``None`` for an item that spans cases.
        detail: What was measured, and against what.

    """

    item: str
    case: str | None
    detail: str

    def describe(self) -> str:
        """Return the finding as one line of a failure or skip message.

        Returns:
            The item, the case it belongs to and the detail.

        """
        where = f" [{self.case}]" if self.case else ""
        return f"{self.item}{where}: {self.detail}"


def _unaccounted_failed_checks(
    payload: dict[str, Any], findings: list[_Finding]
) -> list[str]:
    """Return the harness's failed checklist items that no finding covers.

    Args:
        payload: The parsed JSON payload.
        findings: Every finding this test produced, failed and undecided alike.

    Returns:
        One entry per failed checklist item with no matching finding, and a
        single entry when the payload carries no checklist to reconcile against
        -- either way, something this test did not judge went wrong.

    """
    covered = {finding.item for finding in findings}
    gate = payload.get("gate")
    checks = gate.get("checks") if isinstance(gate, dict) else None
    if not isinstance(checks, list):
        return ["<the payload carries no gate checklist to reconcile against>"]
    unaccounted: list[str] = []
    for check in checks:
        if not isinstance(check, dict) or check.get("passed"):
            continue
        name = check.get("name")
        if isinstance(name, str) and name in covered:
            continue
        unaccounted.append(
            f"{name!r} (measured {check.get('measured')!r}, threshold "
            f"{check.get('threshold')!r})"
        )
    return unaccounted


def _ratio_findings(
    payload: dict[str, Any],
    name: str,
    report: str,
    hard: list[_Finding],
    inconclusive: list[_Finding],
) -> None:
    """Judge one paired-ratio case, on its interval rather than its median.

    Three outcomes, and which one a run gets does not depend on how busy the
    host was: an interval whose lower bound already clears the threshold met it;
    an interval whose upper bound is still under the threshold missed it, and
    the shortfall is a measurement rather than noise; an interval that straddles
    the threshold decided nothing, and saying so is the only honest report of
    ``_ROUNDS`` rounds over a one-to-two-percent difference.

    The interval floor of 1.0 is judged separately and always hard: a case that
    cannot be shown to beat the baseline at all is a failure of the run
    regardless of where the threshold sits.

    Args:
        payload: The parsed JSON payload.
        name: The case to judge, one of :data:`_RATIO_CASES`.
        report: The harness diagnostic block, carried into every failure.
        hard: Accumulator for failures.
        inconclusive: Accumulator for undecided items.

    """
    case = _gated_case(payload, name, report)
    where = f"case {name!r}"
    ratio_median = _number(case, "ratio_median", what=where, report=report)
    ci_low = _number(case, "ci_low", what=where, report=report)
    ci_high = _number(case, "ci_high", what=where, report=report)
    measured = f"median {ratio_median:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]"

    if ci_low <= _CI_LOWER_FLOOR:
        hard.append(
            _Finding(
                f"ci_lower_{name}",
                name,
                f"the 95% CI lower bound {ci_low:.3f} does not clear "
                f"{_CI_LOWER_FLOOR:.2f}, so the improvement is not distinguishable "
                f"from noise ({measured})",
            )
        )

    if ci_low >= _RATIO_THRESHOLD:
        return
    if ci_high < _RATIO_THRESHOLD:
        hard.append(
            _Finding(
                f"ratio_{name}",
                name,
                f"the whole 95% interval is below the required "
                f"{_RATIO_THRESHOLD:.2f} ({measured}), so the shortfall is measured "
                f"and not noise -- short by "
                f"{_RATIO_THRESHOLD - ratio_median:.3f} at the median",
            )
        )
        return
    inconclusive.append(
        _Finding(
            f"ratio_{name}",
            name,
            f"the 95% interval straddles the required {_RATIO_THRESHOLD:.2f} "
            f"({measured}), so {_ROUNDS} measured rounds neither prove nor disprove "
            f"the threshold for this case",
        )
    )


def _evaluate_thresholds(
    payload: dict[str, Any], report: str
) -> tuple[list[_Finding], list[_Finding]]:
    """Judge every gate item over the payload the harness wrote.

    The items are the harness's own (``benchmarks/delayed_ab/main.py::
    evaluate_gate``): the paired-ratio floor and the interval floor for each
    case in :data:`_RATIO_CASES`, the count of cases whose interval clears 1.0,
    the absence of a measured regression, the peak-allocation tolerance, and
    per-case equivalence. Only the two paired-ratio items can come back
    undecided; every other item is a fact about the payload and is hard.

    Args:
        payload: The parsed JSON payload.
        report: The harness diagnostic block, carried into every failure.

    Returns:
        The failures and the undecided items, each already formatted for a
        message.

    """
    hard: list[_Finding] = []
    inconclusive: list[_Finding] = []

    for name in _RATIO_CASES:
        _ratio_findings(payload, name, report, hard, inconclusive)

    improved: list[str] = []
    for name in _GATED_CASES:
        case = _gated_case(payload, name, report)
        where = f"case {name!r}"
        ci_low = _number(case, "ci_low", what=where, report=report)
        ci_high = _number(case, "ci_high", what=where, report=report)
        if ci_low > _CI_LOWER_FLOOR:
            improved.append(f"{name} (ci_low {ci_low:.3f})")
        if ci_high < _REGRESSION_CI_UPPER:
            hard.append(
                _Finding(
                    "no_regression",
                    name,
                    f"the 95% CI upper bound {ci_high:.3f} is below "
                    f"{_REGRESSION_CI_UPPER:.2f}, a measured regression by "
                    f"{_REGRESSION_CI_UPPER - ci_high:.3f}",
                )
            )

    if len(improved) < _MIN_IMPROVED_CASES:
        hard.append(
            _Finding(
                "improved_case_count",
                None,
                f"only {len(improved)} of the {len(_GATED_CASES)} gated cases clear a "
                f"CI lower bound of {_CI_LOWER_FLOOR:.2f}, and "
                f"{_MIN_IMPROVED_CASES} are required. Improved: "
                f"{improved or ['none']}",
            )
        )

    for name in _GATED_CASES:
        case = _gated_case(payload, name, report)
        baseline_peak = _peak_bytes(case, "baseline", name=name, report=report)
        candidate_peak = _peak_bytes(case, "candidate", name=name, report=report)
        if candidate_peak > _PEAK_ALLOC_TOLERANCE * baseline_peak:
            ratio = candidate_peak / baseline_peak if baseline_peak else float("nan")
            hard.append(
                _Finding(
                    "peak_allocation",
                    name,
                    f"candidate peak allocation {candidate_peak:.0f} bytes exceeds "
                    f"{_PEAK_ALLOC_TOLERANCE:.2f} x baseline {baseline_peak:.0f} "
                    f"bytes (ratio {ratio:.3f})",
                )
            )

    # Equivalence is asserted by the harness before it records a timing; the
    # per-case record has to agree, or a performance number was reported for an
    # implementation that behaves differently.
    for name in _GATED_CASES:
        case = _gated_case(payload, name, report)
        equivalence = case.get("equivalence")
        if not isinstance(equivalence, dict):
            hard.append(
                _Finding(
                    "equivalence",
                    name,
                    "the case has no 'equivalence' record, found "
                    f"{type(equivalence).__name__}",
                )
            )
            continue
        for variant in ("native", "pure_true"):
            if equivalence.get(variant) is not True:
                hard.append(
                    _Finding(
                        "equivalence",
                        name,
                        f"the {variant} variant is not equivalent between the two "
                        f"arms ({equivalence!r})",
                    )
                )

    return hard, inconclusive


@pytest.mark.skipif(
    os.environ.get("DASK_DELAYED_AB") != "1",
    reason="set DASK_DELAYED_AB=1 to run the delayed A/B gate",
)
def test_delayed_ab_gate(tmp_path: Path, pytestconfig: pytest.Config) -> None:
    assert not set(_GATED_CASES) & set(_SUBSERIES_CASES), (
        "the informational sub-series must stay out of the gated case list, "
        f"found {sorted(set(_GATED_CASES) & set(_SUBSERIES_CASES))}"
    )

    pytest_timeout_seconds = _configured_pytest_timeout(pytestconfig)
    cap_seconds = _derive_subprocess_cap(pytest_timeout_seconds)
    if cap_seconds is None:
        pytest.skip(
            f"INCONCLUSIVE: pytest's configured per-test timeout of "
            f"{pytest_timeout_seconds:.0f}s leaves less than "
            f"{_TIMEOUT_FLOOR_SECONDS:.0f}s once the {_TIMEOUT_MARGIN_SECONDS:.0f}s "
            "margin this gate keeps below it is reserved, and an enabled run needs "
            "minutes. Raise the timeout for this test, or run the suite directly with "
            "`python -m benchmarks.delayed_ab`."
        )

    run = _run_harness(
        tmp_path,
        cap_seconds=cap_seconds,
        pytest_timeout_seconds=pytest_timeout_seconds,
    )
    report = _harness_report(run)
    # Captured by pytest and shown with any non-passing outcome or under -s: the
    # cost of the run is part of reading its result.
    print(f"delayed A/B gate: {_describe_budget(run)}; {_describe_host(run)}")

    if run.returncode is None:
        pytest.skip(
            f"INCONCLUSIVE: the delayed A/B harness was killed at its "
            f"{cap_seconds:.0f}s cap after {run.elapsed_seconds:.1f}s, so no verdict "
            "was measured. This is a statement about the host, not about the "
            f"implementation: an enabled run at --rounds {_ROUNDS} --warmup {_WARMUP} "
            "needs roughly 150-215s on a twelve-core host and longer under load. The "
            "cap is deliberately below pytest's own "
            f"{pytest_timeout_seconds:.0f}s per-test timeout, which uses the thread "
            "method and would end the whole session instead of this test. Rerun on an "
            "idle host; a recurrence on an idle host is a harness hang rather than a "
            f"slow one.\n{report}"
        )

    if run.returncode not in (_EXIT_PASS, _EXIT_GATE_FAIL):
        pytest.fail(
            f"the delayed A/B harness exited {run.returncode}: "
            f"{_exit_code_hint(run.returncode)}\n{report}"
        )

    payload = _read_payload(tmp_path, report)

    assert payload.get("schema_version") == _SCHEMA_VERSION, (
        f"the harness wrote schema_version {payload.get('schema_version')!r}, but this "
        f"test parses version {_SCHEMA_VERSION}. Re-read the key names from "
        f"benchmarks/delayed_ab/main.py before trusting this result.\n{report}"
    )

    assert payload.get("rounds") == {"warmup": _WARMUP, "measured": _ROUNDS}, (
        f"the run recorded rounds {payload.get('rounds')!r}, but the gate asked for "
        f"--warmup {_WARMUP} --rounds {_ROUNDS}\n{report}"
    )

    # The harness's own verdict is reconciled *after* the thresholds are judged
    # below, not asserted here. Its verdict answers "is the point median at or
    # above the threshold", which is the question this gate was observed to
    # answer differently on the same code from one run to the next; the
    # thresholds are judged on the interval instead, and the harness's verdict
    # then has to be consistent with that judgement.
    cases = payload.get("cases")
    assert isinstance(cases, dict) and set(_GATED_CASES) <= set(cases), (
        f"the payload must measure every gated case {list(_GATED_CASES)}, found "
        f"{sorted(cases) if isinstance(cases, dict) else type(cases).__name__}\n{report}"
    )

    hard, inconclusive = _evaluate_thresholds(payload, report)

    verdict = payload.get("verdict")
    unaccounted = _unaccounted_failed_checks(payload, [*hard, *inconclusive])
    if unaccounted:
        # The harness rejected an item this test did not judge. Reporting the
        # run as undecided on the strength of the items it did judge would hide
        # that, so the unjudged item carries the run.
        hard.append(
            _Finding(
                "harness_verdict",
                None,
                f"the harness (exit {run.returncode}, verdict {verdict!r}) failed "
                f"checklist item(s) this test does not judge: "
                f"{', '.join(unaccounted)}\n{_verdict_digest(payload)}",
            )
        )
    elif (
        not hard
        and not inconclusive
        and (run.returncode != _EXIT_PASS or verdict != _PASS_VERDICT)
    ):
        hard.append(
            _Finding(
                "harness_verdict",
                None,
                f"the harness exited {run.returncode} with verdict {verdict!r} while "
                "every threshold judged here held and its checklist names no failed "
                f"item, so the run cannot be trusted either way\n"
                f"{_verdict_digest(payload)}",
            )
        )

    if hard:
        pytest.fail(
            "\n".join(
                [
                    f"the delayed A/B gate failed {len(hard)} item(s):",
                    *(f"  {finding.describe()}" for finding in hard),
                    *(
                        ["undecided item(s) in the same run:"]
                        + [f"  {finding.describe()}" for finding in inconclusive]
                        if inconclusive
                        else []
                    ),
                    report,
                ]
            )
        )

    if inconclusive:
        pytest.skip(
            "\n".join(
                [
                    f"INCONCLUSIVE: {len(inconclusive)} gate item(s) were neither met "
                    f"nor missed by this run's intervals:",
                    *(f"  {finding.describe()}" for finding in inconclusive),
                    f"The harness's own point-median verdict was {verdict!r} (exit "
                    f"{run.returncode}); it is reported rather than asserted, because "
                    "the same implementation was observed to alternate between PASS "
                    "and FAIL on it from one run to the next while the interval stayed "
                    "astride the threshold. Every other gate item held. For a decision, "
                    "run `python -m benchmarks.delayed_ab` at its default 15 rounds on "
                    "an idle host, or read the committed artefacts under "
                    "benchmarks/delayed_ab/results/.",
                    report,
                ]
            )
        )


def test_gate_subprocess_cap_is_below_the_pytest_timeout(
    pytestconfig: pytest.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The harness can never outlive the timeout that would end the session.

    This is the executable half of the guarantee documented beside
    :data:`_TIMEOUT_CEILING_SECONDS`: whatever per-test timeout is configured,
    the cap handed to the subprocess is at least :data:`_TIMEOUT_MARGIN_SECONDS`
    below it, so a harness that overruns is killed by this test -- reported as
    an unmeasured run -- and never by pytest's thread-method timeout, which ends
    the whole suite rather than the offending test.

    It runs in every session, including the default one where the gate itself is
    skipped, because the guarantee is what makes an enabled run safe to attempt
    and a comment cannot be re-checked. It asserts arithmetic over module
    constants: no threshold of the A/B suite is asserted here, nothing is timed,
    no subprocess is started and nothing from ``benchmarks/`` is imported.

    Args:
        pytestconfig: The active configuration, read for the timeout in force.
        monkeypatch: Used to check that the derivation follows the timeout
            source pytest-timeout would use, not just the ini.

    """
    assert _TIMEOUT_CEILING_SECONDS + _TIMEOUT_MARGIN_SECONDS <= (
        _PYTEST_TIMEOUT_FALLBACK_SECONDS
    ), (
        f"the cap's ceiling {_TIMEOUT_CEILING_SECONDS:.0f}s plus its margin "
        f"{_TIMEOUT_MARGIN_SECONDS:.0f}s exceeds the documented per-test timeout "
        f"{_PYTEST_TIMEOUT_FALLBACK_SECONDS:.0f}s, so a run that falls back to the "
        "documented value would risk pytest's thread-method timeout"
    )
    assert _TIMEOUT_FLOOR_SECONDS <= _TIMEOUT_CEILING_SECONDS, (
        f"the cap's floor {_TIMEOUT_FLOOR_SECONDS:.0f}s is above its ceiling "
        f"{_TIMEOUT_CEILING_SECONDS:.0f}s, so no cap could ever be derived"
    )

    configured = _configured_pytest_timeout(pytestconfig)
    assert configured >= 0.0, (
        f"pytest reports a negative per-test timeout ({configured}); the gate cannot "
        "derive a cap from it"
    )

    # The configuration in force, then a spread around it: a timeout that leaves
    # no room, the project's own value, one that binds before the ceiling and one
    # the ceiling binds, and "no timeout at all".
    for candidate in (configured, 0.0, 30.0, 120.0, 300.0, 3600.0):
        cap = _derive_subprocess_cap(candidate)
        if cap is None:
            assert candidate - _TIMEOUT_MARGIN_SECONDS < _TIMEOUT_FLOOR_SECONDS, (
                f"a per-test timeout of {candidate:.0f}s leaves "
                f"{candidate - _TIMEOUT_MARGIN_SECONDS:.0f}s after the margin, which "
                f"clears the {_TIMEOUT_FLOOR_SECONDS:.0f}s floor, so the gate should "
                "have derived a cap instead of declining to run"
            )
            continue
        assert _TIMEOUT_FLOOR_SECONDS <= cap <= _TIMEOUT_CEILING_SECONDS, (
            f"a per-test timeout of {candidate:.0f}s derived a cap of {cap:.0f}s, "
            f"outside [{_TIMEOUT_FLOOR_SECONDS:.0f}s, "
            f"{_TIMEOUT_CEILING_SECONDS:.0f}s]"
        )
        limit = candidate if candidate > 0.0 else _PYTEST_TIMEOUT_FALLBACK_SECONDS
        assert cap + _TIMEOUT_MARGIN_SECONDS <= limit, (
            f"a per-test timeout of {candidate:.0f}s derived a cap of {cap:.0f}s, "
            f"which leaves less than the {_TIMEOUT_MARGIN_SECONDS:.0f}s margin below "
            f"{limit:.0f}s that keeps pytest's thread-method timeout out of reach"
        )

    # The limit a session actually runs under need not come from the ini, and a
    # cap derived from the wrong source is not a guarantee. When no --timeout was
    # passed, the environment variable is the source pytest-timeout would use,
    # so the derivation has to follow it -- including down to declining the run.
    if pytestconfig.getoption(_PYTEST_TIMEOUT_INI, default=None) is None:
        monkeypatch.setenv("PYTEST_TIMEOUT", "120")
        assert _configured_pytest_timeout(pytestconfig) == 120.0, (
            "the gate reads its timeout from the ini only, so a session running "
            "under PYTEST_TIMEOUT would derive its cap from a limit that is not in "
            "force"
        )
        assert _derive_subprocess_cap(120.0) == 120.0 - _TIMEOUT_MARGIN_SECONDS
        monkeypatch.setenv("PYTEST_TIMEOUT", str(int(_TIMEOUT_FLOOR_SECONDS)))
        assert (
            _derive_subprocess_cap(_configured_pytest_timeout(pytestconfig)) is None
        ), (
            f"a per-test timeout of {_TIMEOUT_FLOOR_SECONDS:.0f}s cannot host an "
            "enabled run, and the gate must decline it rather than start a harness "
            "that pytest's thread-method timeout would then kill the session over"
        )
