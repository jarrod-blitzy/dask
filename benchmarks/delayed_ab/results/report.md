# dask.delayed A/B performance report

Arm A is the frozen pre-refactor capture `benchmarks/delayed_ab/baseline_delayed.py`; arm B is the live `dask.delayed`. Both were measured in one interpreter, each under activation as `sys.modules["dask.delayed"]`, and every case was proven equivalent -- keys, canonical graphs and computed results, under both `pure=None` and `pure=True` -- before a single timing was recorded.

## Environment

- Generated (UTC): 2026-09-17T01:55:10.803353+00:00
- Schema version: 1
- Rounds: 3 warmup + 15 measured; each round is A,B,B,A with the starting arm alternating, and the paired ratio is (A1+A2)/(B1+B2), baseline over candidate
- Interpreter: cpython 3.14.6 (3.14.6 | packaged by conda-forge | (main, Jun 12 2026, 08:51:42) [GCC 14.3.0])
- Free-threaded build: Py_GIL_DISABLED=0, sys._is_gil_enabled()=True
- Platform: Linux-6.12.85+-x86_64-with-glibc2.42
- CPU: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz
- PYTHONHASHSEED: 0
- Active tokenize hasher: _hash_xxhash (it changes every pickle-sensitive token, so it is part of the result)
- Key packages: dask 0.0.post9826+gc9d1df34c, toolz 1.1.0, cloudpickle 3.1.2, numpy 2.4.6, pandas 3.0.3
- Arm A provenance: commit c9d1df34ccba182ddf43c2dbe4315c4d9c8c44e1 per the capture's own header (expected c9d1df34ccba182ddf43c2dbe4315c4d9c8c44e1, matches=True)
- Arm B provenance: git HEAD c1606857f3f80715c2ef8b8f8a77cf5d89709576, dirty=False, sha256(dask/delayed.py)=13cdd98a4339f63b32f7b597c7b9350f0596c3c14e298d0457038e990c3f3f94
- The arm B SHA identifies the commit whose `dask/delayed.py` was measured, not the commit that adds these artefacts: the runner refuses to write inside a dirty working tree, so the sources are committed first, the runner is executed from that clean commit, and these two files are committed afterwards.
- Rejected optimization on record: the `is_dask_collection(obj) or traverse` probe order in `delayed()` was not reordered -- for a non-Delayed object exposing the collection protocol the first probe calls x.expr / __dask_graph__() once before unpack_collections calls it again, so dropping it could change warning counts, mutation or exceptions in user wrappers. Measured cost of keeping it: 53.7 ns per probe against 9462.542 ns per `flat_loop` construction (0.57% of one construction), by timeit.timeit('is_dask_collection(target)', number=1000000) against a plain module-level function, in this process and this environment.
- Peak block count: The suite was asked to record a peak block count alongside peak bytes. The standard library exposes no such API: tracemalloc tracks peak bytes only (get_traced_memory()[1]) and a snapshot exposes the live block count, while the exact alternative would be a third-party allocation tracer, which the no-new-dependency directive forbids. The harder directive wins. The gate is evaluated on tracemalloc peak bytes, which the standard library supplies exactly, and two block figures are recorded as supporting evidence, each under its own definition and neither of them a peak: 'live_blocks_end' is the number of traced blocks still alive at the end of the region (end snapshot trace count minus the baseline count), and 'max_observed_blocks' is a sampled lower bound -- the maximum of sys.getallocatedblocks() read every 64th sys.setprofile event of one further untimed run, floored by the count taken immediately after the region while the constructed objects are still alive, minus the count taken before it.
- A/A calibration source: A/A calibration run, sha256 e8d162c68e9338dbb2004f92df014cca332285fd2013ec741a9f5c92a4d60401 (generated 2026-09-17T01:50:56.775227+00:00). Its path is not recorded: that run writes outside the checkout so the tree stays clean, which makes the location ephemeral and the digest the durable identifier. The A/A ratios are the noise floor of this machine; they are diagnostic, not a gate.
- A/A calibration (flat_loop): ratio median 1.000, 95% CI [0.996, 1.005]
- A/A calibration (linear_chain): ratio median 0.995, 95% CI [0.989, 1.005]
- A/A calibration (nested_containers): ratio median 0.994, 95% CI [0.977, 1.003]
- A/A calibration (wide_fan_in): ratio median 0.997, 95% CI [0.984, 1.027]
- A/A calibration (pure_vs_impure): ratio median 0.999, 95% CI [0.987, 1.011]
- A/A calibration (attr_and_operators): ratio median 1.002, 95% CI [0.997, 1.004]

## Gated cases

| case | baseline median (ms) | candidate median (ms) | paired ratio median | 95% CI | peak-allocation delta (%) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| flat_loop | 204.698 | 94.625 | 2.140 | [2.115, 2.162] | -0.0 | PASS |
| linear_chain | 103.124 | 79.769 | 1.306 | [1.286, 1.313] | -0.4 | PASS |
| nested_containers | 179.637 | 144.444 | 1.228 | [1.213, 1.238] | +0.0 | FAIL (ratio_median 1.228 < 1.25 (short by 0.022)) |
| wide_fan_in | 26.076 | 4.057 | 6.410 | [6.254, 6.494] | -36.5 | PASS |
| pure_vs_impure | 275.208 | 130.895 | 2.114 | [2.098, 2.140] | +0.2 | PASS |
| attr_and_operators | 845.963 | 287.692 | 2.938 | [2.927, 2.942] | -0.0 | PASS |

## Informational sub-series (not gated)

| case | baseline median (ms) | candidate median (ms) | paired ratio median | 95% CI | peak-allocation delta (%) |
| --- | --- | --- | --- | --- | --- |
| nested_containers_shallow | 99.459 | 77.853 | 1.267 | [1.257, 1.281] | -0.1 |
| pure_true | 165.742 | 91.699 | 1.798 | [1.777, 1.809] | +0.4 |
| pure_false | 109.437 | 36.976 | 2.950 | [2.923, 2.992] | -0.1 |

OVERALL: FAIL
