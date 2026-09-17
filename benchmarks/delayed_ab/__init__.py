"""Committed A/B performance suite for ``dask.delayed`` graph construction.

The suite measures the pre-refactor and the post-refactor ``dask/delayed.py``
side by side inside a single interpreter, proves that the two behave
identically *before* it records a single timing, and prints a PASS/FAIL
verdict against thresholds that were fixed before any measurement was taken.
It is equivalence evidence as much as performance evidence: a speedup reported
for an implementation that does something different is worthless, so the
equivalence assertions -- the generated keys, the canonical graph of every
constructed object and the computed results, per case -- run first, and no
artefact is written and no verdict is produced when one of them fails. The
suite is re-runnable rather than a one-off measurement: its sources and both
of its result files are committed, so the claim can be reproduced instead of
being taken on trust.

Arms:
    Arm A -- the frozen baseline: ``benchmarks/delayed_ab/baseline_delayed.py``,
        a verbatim copy of ``dask/delayed.py`` captured at commit
        ``c9d1df34ccba182ddf43c2dbe4315c4d9c8c44e1``, before the first edit of
        the refactor. It is never edited, and neither is any module it imports;
        that is what keeps it importable and behaviourally fixed for as long as
        the comparison is meant to mean anything.
    Arm B -- the live candidate: the module object returned by
        ``importlib.import_module("dask.delayed")``. The attribute form
        ``import dask.delayed as m`` is deliberately not used, because
        ``dask/__init__.py`` rebinds the attribute ``delayed`` on the ``dask``
        package to the ``delayed`` curry, so that form hands back the function
        rather than the module.

Arm activation:
    Both arms are driven under *arm activation*: the runner swaps
    ``sys.modules["dask.delayed"]`` for the arm under test for the duration of
    every arm operation -- construction, equivalence extraction and every
    compute -- and restores the previous entry afterwards. Frozen engine code
    recognises ``Delayed`` by identity through a late-bound ``from dask.delayed
    import Delayed``, so without activation the baseline arm's objects are
    treated as foreign collections and its graphs acquire extra
    ``finalize-hlgfinalizecompute-*`` layers. Activation is applied to both
    arms so that the two run under identical conditions, and it exists
    precisely so that no frozen ``dask`` module has to be edited to measure
    them.

Invocation, from the repository root:
    The full suite, which writes both artefacts:

        python -m benchmarks.delayed_ab

    The same thresholds over reduced rounds, through pytest, skipped at
    collection unless the environment variable is set:

        DASK_DELAYED_AB=1 pytest dask/tests/test_delayed_ab_gate.py

    Options of the runner: ``--rounds N`` and ``--warmup N`` (measured and
    discarded rounds per case), ``--output DIR`` (default
    ``benchmarks/delayed_ab/results``), ``--no-artefacts`` (print the checklist
    and skip both writers) and ``--calibration PATH`` (an earlier A/A result
    JSON whose per-case ratios are carried into this run's report).

Artefacts:
    ``benchmarks/delayed_ab/results/baseline_vs_candidate.json``
        The raw per-round timings of both arms, the allocation figures, the
        per-arm and paired statistics with their bootstrap intervals, the gate
        checklist and the environment block that identifies the interpreter,
        the resolved dependency set and the provenance of each arm.
    ``benchmarks/delayed_ab/results/report.md``
        The environment summary, one table for the six gated cases, a second
        for the informational sub-series, the note recording that the standard
        library exposes no peak block count, and a single closing ``OVERALL:``
        line.
    Both files hold the real measured numbers of the final run that wrote them
    and are never hand-edited. They are regenerated only by re-running the
    suite from a clean working tree, which is what makes the commit they
    describe identifiable.

Exit codes of the runner -- the outcomes of a run that started. A command line
that cannot be parsed is rejected by ``argparse`` before the run starts, with
its conventional status 2:
    0
        Every gate item held.
    1
        The gate failed, or the run could not be configured; the printed
        checklist names the item that failed.
    2
        An equivalence mismatch. No artefact is written and no performance
        verdict is produced. ``argparse`` uses the same status for a command
        line it rejects before the run starts: that one prints a ``usage:``
        message, while a mismatch prints ``equivalence mismatch`` and names
        the differing case.
    3
        Artefacts were requested inside a dirty repository working tree. The
        runner refuses to write into ``results/`` and prints the offending
        paths.

Halt and report:
    The following conditions are reported rather than worked around: no frozen
    file is edited and no expectation is adjusted to accommodate them.

    * The two arms cannot be imported side by side in one interpreter.
    * An A/A calibration run on unmodified code fails the equivalence
      assertions with arm activation in place.
    * A measurement appears to require a change outside this tree.
    * A canonical graph carries a value that is not reproducible across
      processes, such as an ``id()``-derived layer name.
"""

from __future__ import annotations
