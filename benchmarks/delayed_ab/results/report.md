# dask.delayed A/B performance report

Arm A is the frozen pre-refactor capture `benchmarks/delayed_ab/baseline_delayed.py`; arm B is the live `dask.delayed`. Both were measured in one interpreter, each under activation as `sys.modules["dask.delayed"]`, and every case was proven equivalent -- keys, canonical graphs and computed results, under both `pure=None` and `pure=True` -- before a single timing was recorded.

## Environment

- Generated (UTC): 2026-09-16T22:01:09.205729+00:00
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
- Arm B provenance: git HEAD 04561b3bd75a2b1a3f76d26f2dc725827c07eb28, dirty=False, sha256(dask/delayed.py)=2793ce5f5feb044d7a14265a5456d4625bf2e885375f8c188ea74d8158bba44e
- The arm B SHA identifies the commit whose `dask/delayed.py` was measured, not the commit that adds these artefacts: the runner refuses to write inside a dirty working tree, so the sources are committed first, the runner is executed from that clean commit, and these two files are committed afterwards.
- Rejected optimization on record: the `is_dask_collection(obj) or traverse` probe order in `delayed()` was not reordered -- for a non-Delayed object exposing the collection protocol the first probe calls x.expr / __dask_graph__() once before unpack_collections calls it again, so dropping it could change warning counts, mutation or exceptions in user wrappers. Measured cost of keeping it: 51.7 ns per probe against 9509.065 ns per `flat_loop` construction (0.54% of one construction), by timeit.timeit('is_dask_collection(target)', number=1000000) against a plain module-level function, in this process and this environment.
- Peak block count: The suite was asked to record a peak block count alongside peak bytes. The standard library exposes no such API: tracemalloc tracks peak bytes only (get_traced_memory()[1]) and a snapshot exposes the live block count, while the exact alternative would be a third-party allocation tracer, which the no-new-dependency directive forbids. The harder directive wins. The gate is evaluated on tracemalloc peak bytes, which the standard library supplies exactly, and two block figures are recorded as supporting evidence, each under its own definition and neither of them a peak: 'live_blocks_end' is the number of traced blocks still alive at the end of the region (end snapshot trace count minus the baseline count), and 'max_observed_blocks' is a sampled lower bound -- the maximum of sys.getallocatedblocks() read every 64th sys.setprofile event of one further untimed run, floored by the count taken immediately after the region while the constructed objects are still alive, minus the count taken before it.
- A/A calibration source: /tmp/blitzy-clone-1/aa_calibration/baseline_vs_candidate.json (generated 2026-09-16T21:17:01.451641+00:00). The A/A ratios are the noise floor of this machine; they are diagnostic, not a gate.
- A/A calibration (attr_and_operators): ratio median 1.008, 95% CI [1.004, 1.015] -- arm/order bias to weigh: the interval excludes 1.0
- A/A calibration (flat_loop): ratio median 1.003, 95% CI [0.995, 1.010]
- A/A calibration (linear_chain): ratio median 1.006, 95% CI [0.997, 1.030]
- A/A calibration (nested_containers): ratio median 1.010, 95% CI [1.003, 1.018] -- arm/order bias to weigh: the interval excludes 1.0
- A/A calibration (pure_vs_impure): ratio median 0.999, 95% CI [0.988, 1.007]
- A/A calibration (wide_fan_in): ratio median 1.023, 95% CI [1.019, 1.035] -- arm/order bias to weigh: the interval excludes 1.0
- A/A calibration (nested_containers_shallow): ratio median 0.996, 95% CI [0.994, 1.014]
- A/A calibration (pure_false): ratio median 1.001, 95% CI [0.996, 1.012]
- A/A calibration (pure_true): ratio median 0.999, 95% CI [0.994, 1.009]

## Gate checklist

- PASS ratio_flat_loop [flat_loop]: measured 2.145, required >= 1.250
- PASS ci_lower_flat_loop [flat_loop]: measured 2.119, required > 1
- PASS ratio_nested_containers [nested_containers]: measured 1.276, required >= 1.250
- PASS ci_lower_nested_containers [nested_containers]: measured 1.271, required > 1
- PASS improved_case_count: measured 6, required >= 4
- PASS no_regression [nested_containers]: measured 1.282, required >= 0.980
- PASS peak_allocation [pure_vs_impure]: measured 1.002, required <= 1.050
- PASS equivalence: measured 9, required == 9

## Gated cases

| case | baseline median (ms) | candidate median (ms) | paired ratio median | 95% CI | peak-allocation delta (%) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| flat_loop | 205.224 | 95.091 | 2.145 | [2.119, 2.166] | -0.0 | PASS |
| linear_chain | 103.265 | 79.495 | 1.292 | [1.264, 1.303] | -0.4 | PASS |
| nested_containers | 179.975 | 140.626 | 1.276 | [1.271, 1.282] | +0.0 | PASS |
| wide_fan_in | 26.582 | 4.365 | 6.104 | [6.033, 6.130] | -5.7 | PASS |
| pure_vs_impure | 274.393 | 128.099 | 2.136 | [2.128, 2.161] | +0.2 | PASS |
| attr_and_operators | 847.113 | 276.563 | 3.049 | [3.024, 3.059] | -0.0 | PASS |

## Informational sub-series (not gated)

| case | baseline median (ms) | candidate median (ms) | paired ratio median | 95% CI | peak-allocation delta (%) |
| --- | --- | --- | --- | --- | --- |
| nested_containers_shallow | 98.397 | 75.249 | 1.315 | [1.301, 1.328] | -0.1 |
| pure_true | 164.351 | 91.471 | 1.791 | [1.774, 1.797] | +0.1 |
| pure_false | 109.668 | 36.587 | 2.997 | [2.988, 3.032] | -0.1 |

OVERALL: PASS
