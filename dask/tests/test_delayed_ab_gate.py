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

Invocation, from the repository root:

    ```text
    python -m benchmarks.delayed_ab
    DASK_DELAYED_AB=1 pytest dask/tests/test_delayed_ab_gate.py
    ```

Notes:
    ``benchmarks/delayed_ab/main.py`` is the authority for the artefact name,
    the JSON key names and the exit-code contract parsed here; no key is
    invented. This module performs no timing, statistics, bootstrap or
    allocation measurement of its own -- it launches the harness and reads what
    the harness wrote -- and it imports nothing from ``dask`` or ``benchmarks``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
_ROUNDS = 7
_WARMUP = 2

#: Subprocess timeout in seconds. The project runs pytest with ``timeout = 300``
#: and ``timeout_method = "thread"``, and a thread-method timeout kills the whole
#: suite rather than the offending test -- so the subprocess has to fail cleanly
#: first, and this value is deliberately well below 300. Measured reference: the
#: reduced run of ``--rounds 7 --warmup 2`` over the nine cases and sub-series
#: takes a little over three minutes on a twelve-core container, so a run that
#: reaches this limit is a stuck or a badly contended one, and it is reported as
#: a failed test rather than as a hung suite.
_TIMEOUT_SECONDS = 240

#: Trailing lines of each captured stream a failure message reproduces. A gate
#: failure whose message omits the harness output is useless to a reviewer, and
#: an unbounded dump of a nine-case run is unreadable.
_OUTPUT_TAIL_LINES = 40

#: Exit codes of the harness (``benchmarks/delayed_ab/main.py``).
_EXIT_PASS = 0
_EXIT_GATE_FAIL = 1
_EXIT_EQUIVALENCE = 2
_EXIT_DIRTY = 3

#: The verdict string ``build_payload`` writes when every gate item held.
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


def _tail(stream: object, *, lines: int = _OUTPUT_TAIL_LINES) -> str:
    """Render the last ``lines`` lines of a captured stream, bounded and labelled.

    Args:
        stream: The captured text. ``None`` and non-string values -- which is
            what a timed-out or crashed subprocess can leave behind -- are
            rendered as a placeholder rather than raising.
        lines: How many trailing lines to keep.

    Returns:
        The trailing lines, prefixed with an elision marker when earlier output
        was dropped, or a placeholder when nothing was captured.
    """
    if stream is None:
        return "<not captured>"
    text = stream if isinstance(stream, str) else str(stream)
    if not text.strip():
        return "<empty>"
    kept = text.splitlines()
    if len(kept) > lines:
        dropped = len(kept) - lines
        return "\n".join(
            [f"<... {dropped} earlier line(s) omitted ...>", *kept[-lines:]]
        )
    return "\n".join(kept)


def _exit_code_hint(returncode: int) -> str:
    """Explain one of the harness's documented exit codes.

    Args:
        returncode: The status the subprocess exited with.

    Returns:
        A one-line diagnosis, so a failure is attributable without re-running
        anything.
    """
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


def _harness_report(
    command: list[str],
    returncode: int | None,
    stdout: object,
    stderr: object,
) -> str:
    """Build the diagnostic block every failure message in this module carries.

    Args:
        command: The argument list the subprocess was started with.
        returncode: Its exit status, or ``None`` when it never finished.
        stdout: Captured standard output, if any.
        stderr: Captured standard error, if any.

    Returns:
        A readable, bounded block naming the command, the return code and the
        tail of both streams.
    """
    status = "<timed out>" if returncode is None else str(returncode)
    return "\n".join(
        [
            "--- delayed A/B harness ---",
            f"command: {' '.join(command)}",
            f"returncode: {status}",
            "stdout (tail):",
            _tail(stdout),
            "stderr (tail):",
            _tail(stderr),
            "--- end of harness output ---",
        ]
    )


def _run_harness(
    output_dir: Path,
) -> tuple[list[str], subprocess.CompletedProcess[str]]:
    """Run the A/B suite out of process and return the command and its result.

    The suite runs in a subprocess so that pytest's imports, its assertion
    rewriting and its warnings filters stay out of the timed regions, and so
    that ``PYTHONHASHSEED`` can be pinned for the measurement without the
    runner having to re-execute itself: hash randomisation changes dict and set
    iteration order, which changes how much work the construction path does
    over containers, so both arms must run under one fixed seed.

    Args:
        output_dir: Directory the harness writes its two artefacts into. Always
            pytest's ``tmp_path``, so a gate run never overwrites the committed
            results and never meets the runner's dirty-tree refusal.

    Returns:
        The command that was run, and the completed process.

    Raises:
        Failed: Via :func:`pytest.fail`, when the subprocess does not finish
            within :data:`_TIMEOUT_SECONDS`.
    """
    root = _repository_root()
    command = [
        sys.executable,
        "-m",
        _HARNESS_MODULE,
        "--rounds",
        str(_ROUNDS),
        "--warmup",
        str(_WARMUP),
        "--output",
        str(output_dir),
    ]
    # A copy, never an in-place mutation of os.environ: the rest of the session
    # keeps whatever hash seed it was started with.
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    try:
        return command, subprocess.run(
            command,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"the delayed A/B harness did not finish within {_TIMEOUT_SECONDS}s. The "
            "timeout is deliberately below pytest's own 300s limit, which uses the "
            "thread method and would take the whole suite down with it.\n"
            f"{_harness_report(command, None, exc.stdout, exc.stderr)}"
        )


def _read_payload(output_dir: Path, report: str) -> dict[str, Any]:
    """Read and validate the run's JSON artefact.

    Args:
        output_dir: The directory passed to ``--output``.
        report: The harness diagnostic block, carried into every failure.

    Returns:
        The parsed payload.

    Raises:
        Failed: Via :func:`pytest.fail`, when the artefact is missing, is not
            readable, is not JSON, or is not a JSON object.
    """
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
    """Return one gated case's record from the payload.

    Args:
        payload: The parsed JSON payload.
        name: The gated case's name.
        report: The harness diagnostic block, carried into every failure.

    Returns:
        The case's JSON object.

    Raises:
        Failed: Via :func:`pytest.fail`, when ``cases`` or the case itself is
            missing or has the wrong shape.
    """
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
            f"case {name!r} has no allocation figures for arm {arm!r}\n{report}"
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


@pytest.mark.skipif(
    os.environ.get("DASK_DELAYED_AB") != "1",
    reason="set DASK_DELAYED_AB=1 to run the delayed A/B gate",
)
def test_delayed_ab_gate(tmp_path: Path) -> None:
    """Re-run the A/B suite over reduced rounds and assert the gate's thresholds.

    One test function runs the harness exactly once and asserts every item of
    the gate against the JSON it wrote: the overall verdict, the paired-ratio
    and interval floors on ``flat_loop`` and ``nested_containers``, the
    four-of-six improvement count, the absence of a measured regression, the
    peak-allocation tolerance, and the equivalence of every case. The
    informational sub-series are excluded from all of it.

    Args:
        tmp_path: pytest's per-test directory, used for ``--output`` so that the
            committed artefacts under ``benchmarks/delayed_ab/results/`` are
            never touched.
    """
    # The sub-series must never be folded into the gated corpus: they carry no
    # verdict, and counting one towards the improvement item would let an
    # informational shape stand in for a gated one.
    assert not set(_GATED_CASES) & set(_SUBSERIES_CASES), (
        "the informational sub-series must stay out of the gated case list, "
        f"found {sorted(set(_GATED_CASES) & set(_SUBSERIES_CASES))}"
    )

    command, proc = _run_harness(tmp_path)
    report = _harness_report(command, proc.returncode, proc.stdout, proc.stderr)

    assert proc.returncode == _EXIT_PASS, (
        f"the delayed A/B harness exited {proc.returncode}: "
        f"{_exit_code_hint(proc.returncode)}\n{report}"
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

    assert payload.get("verdict") == _PASS_VERDICT, (
        f"the harness verdict is {payload.get('verdict')!r}, not {_PASS_VERDICT!r}:\n"
        f"{_verdict_digest(payload)}\n{report}"
    )

    cases = payload.get("cases")
    assert isinstance(cases, dict) and set(_GATED_CASES) <= set(cases), (
        f"the payload must measure every gated case {list(_GATED_CASES)}, found "
        f"{sorted(cases) if isinstance(cases, dict) else type(cases).__name__}\n{report}"
    )

    # The paired-ratio floor and the interval floor, on the two cases that carry
    # them. Both are read per case so the message names the measured value.
    for name in _RATIO_CASES:
        case = _gated_case(payload, name, report)
        ratio_median = _number(
            case, "ratio_median", what=f"case {name!r}", report=report
        )
        ci_low = _number(case, "ci_low", what=f"case {name!r}", report=report)
        assert ratio_median >= _RATIO_THRESHOLD, (
            f"case {name!r}: paired ratio median {ratio_median:.3f} is below the "
            f"required {_RATIO_THRESHOLD:.2f} (short by "
            f"{_RATIO_THRESHOLD - ratio_median:.3f})\n{report}"
        )
        assert ci_low > _CI_LOWER_FLOOR, (
            f"case {name!r}: 95% CI lower bound {ci_low:.3f} does not clear "
            f"{_CI_LOWER_FLOOR:.2f}, so the improvement is not distinguishable from "
            f"noise\n{report}"
        )

    # At least four of the six gated cases must show a real improvement, and no
    # gated case may show a real regression.
    improved: list[str] = []
    for name in _GATED_CASES:
        case = _gated_case(payload, name, report)
        ci_low = _number(case, "ci_low", what=f"case {name!r}", report=report)
        ci_high = _number(case, "ci_high", what=f"case {name!r}", report=report)
        if ci_low > _CI_LOWER_FLOOR:
            improved.append(f"{name} (ci_low {ci_low:.3f})")
        assert ci_high >= _REGRESSION_CI_UPPER, (
            f"case {name!r}: 95% CI upper bound {ci_high:.3f} is below "
            f"{_REGRESSION_CI_UPPER:.2f}, a measured regression (by "
            f"{_REGRESSION_CI_UPPER - ci_high:.3f})\n{report}"
        )

    assert len(improved) >= _MIN_IMPROVED_CASES, (
        f"only {len(improved)} of the {len(_GATED_CASES)} gated cases clear a CI lower "
        f"bound of {_CI_LOWER_FLOOR:.2f}, and {_MIN_IMPROVED_CASES} are required. "
        f"Improved: {improved or ['none']}\n{report}"
    )

    # Peak allocation: the candidate may not exceed the baseline's tracemalloc
    # peak bytes by more than the tolerance on any gated case.
    for name in _GATED_CASES:
        case = _gated_case(payload, name, report)
        baseline_peak = _peak_bytes(case, "baseline", name=name, report=report)
        candidate_peak = _peak_bytes(case, "candidate", name=name, report=report)
        assert candidate_peak <= _PEAK_ALLOC_TOLERANCE * baseline_peak, (
            f"case {name!r}: candidate peak allocation {candidate_peak:.0f} bytes "
            f"exceeds {_PEAK_ALLOC_TOLERANCE:.2f} x baseline "
            f"{baseline_peak:.0f} bytes (ratio "
            f"{candidate_peak / baseline_peak if baseline_peak else float('nan'):.3f})"
            f"\n{report}"
        )

    # Equivalence is asserted by the harness before it records a timing; the
    # per-case record has to agree, or a performance number was reported for an
    # implementation that behaves differently.
    for name in _GATED_CASES:
        case = _gated_case(payload, name, report)
        equivalence = case.get("equivalence")
        assert isinstance(equivalence, dict), (
            f"case {name!r} has no 'equivalence' record: found "
            f"{type(equivalence).__name__}\n{report}"
        )
        assert equivalence.get("native") is True, (
            f"case {name!r}: the native pure=None variant is not equivalent between "
            f"the two arms ({equivalence!r})\n{report}"
        )
        assert equivalence.get("pure_true") is True, (
            f"case {name!r}: the pure=True variant is not equivalent between the two "
            f"arms ({equivalence!r})\n{report}"
        )
