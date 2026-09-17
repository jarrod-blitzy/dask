"""Committed A/B performance suite for ``dask.delayed`` graph construction.

A re-runnable suite that measures the pre-refactor and the post-refactor
``dask/delayed.py`` side by side in one interpreter. Per case it asserts that the
two arms are behaviourally identical -- the generated keys, the canonical graph of
every constructed object and the computed results -- before it records a single
timing, then prints a PASS/FAIL verdict against thresholds fixed before any
measurement was taken. When an equivalence assertion fails, no artefact is written
and no verdict is produced.

Arms:
    Arm A -- the frozen baseline ``benchmarks/delayed_ab/baseline_delayed.py``, a
        verbatim copy of ``dask/delayed.py`` captured at commit
        ``c9d1df34ccba182ddf43c2dbe4315c4d9c8c44e1``, never edited.
    Arm B -- the live candidate, the module returned by
        ``importlib.import_module("dask.delayed")``.

    Frozen engine code recognises ``Delayed`` by identity, so the runner swaps
    ``sys.modules["dask.delayed"]`` for the arm under test for the duration of
    every arm operation and restores the previous entry afterwards. That
    activation is applied to both arms, so the two run under identical conditions.

Invocation, from the repository root:
    The full suite, which writes both artefacts:

        python -m benchmarks.delayed_ab

    The same thresholds over reduced rounds, skipped at collection unless the
    environment variable is set:

        DASK_DELAYED_AB=1 pytest dask/tests/test_delayed_ab_gate.py

Artefacts:
    ``benchmarks/delayed_ab/results/baseline_vs_candidate.json`` -- the per-round
        timings of both arms, the allocation figures, the per-arm and paired
        statistics, the gate checklist and the environment block.
    ``benchmarks/delayed_ab/results/report.md`` -- the environment summary, one
        table for the gated cases and a second for the informational sub-series,
        the note that the standard library exposes no peak block count, and a
        single closing ``OVERALL:`` line.

    Both hold the real measured numbers of the run that wrote them and are never
    hand-edited. The allocation gate is evaluated on ``tracemalloc`` peak bytes,
    the one figure the standard library reports exactly, while ``live_blocks_end``
    and ``max_observed_blocks`` are supporting evidence that is never called a
    peak.

Exit codes:
    ``0`` -- every gate item held. ``1`` -- the gate failed; the printed checklist
    names the item. ``2`` -- an equivalence mismatch, with no artefact written.
    ``3`` -- artefacts were requested inside a dirty working tree, so the runner
    refuses to write into ``results/`` and prints the offending paths.

Some conditions are reported rather than worked around, with no frozen file edited
and no expectation adjusted: the two arms failing to import side by side, an A/A
calibration run on unmodified code failing the equivalence assertions with
activation in place, a measurement appearing to require a change outside this
tree, or a canonical graph carrying a value that is not reproducible across
processes.
"""

from __future__ import annotations
