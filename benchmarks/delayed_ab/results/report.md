# dask.delayed A/B performance report

Arm A is the frozen pre-refactor capture `benchmarks/delayed_ab/baseline_delayed.py`; arm B is the live `dask.delayed`. Both were measured in one interpreter, each under activation as `sys.modules["dask.delayed"]`, and every case was proven equivalent -- keys, canonical graphs and computed results, under both `pure=None` and `pure=True` -- before a single timing was recorded.

## Environment

- Generated (UTC): 2026-09-17T06:18:43.515096+00:00
- Schema version: 1
- Rounds: 3 warmup + 31 measured; each round is A,B,B,A with the starting arm alternating, and the paired ratio is (A1+A2)/(B1+B2), baseline over candidate
- Interpreter: cpython 3.14.6 (3.14.6 | packaged by conda-forge | (main, Jun 12 2026, 08:51:42) [GCC 14.3.0])
- Free-threaded build: Py_GIL_DISABLED=0, sys._is_gil_enabled()=True
- Platform: Linux-6.12.85+-x86_64-with-glibc2.42
- CPU: INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz
- PYTHONHASHSEED: 0
- Active tokenize hasher: _hash_xxhash (it changes every pickle-sensitive token, so it is part of the result)
- Key packages: dask 0.0.post9826+gc9d1df34c, toolz 1.1.0, cloudpickle 3.1.2, numpy 2.4.6, pandas 3.0.3
- Arm A provenance: commit c9d1df34ccba182ddf43c2dbe4315c4d9c8c44e1 per the capture's own header (expected c9d1df34ccba182ddf43c2dbe4315c4d9c8c44e1, matches=True)
- Arm B provenance: git HEAD 929fb7351627e2f5b3872ab438a86c6e1c7d1c1d, dirty=False, sha256(dask/delayed.py)=14398b9c956b7fa299335f2a27b8ea2a60381a956559701e4af62be5e4cc43b9
- The arm B SHA identifies the commit whose `dask/delayed.py` was measured, not the commit that adds these artefacts: the runner refuses to write inside a dirty working tree, so the sources are committed first, the runner is executed from that clean commit, and these two files are committed afterwards.
- Rejected optimization on record: the `is_dask_collection(obj) or traverse` probe order in `delayed()` was not reordered -- for a non-Delayed object exposing the collection protocol the first probe calls x.expr / __dask_graph__() once before unpack_collections calls it again, so dropping it could change warning counts, mutation or exceptions in user wrappers. Measured cost of keeping it: 51.0 ns per probe against 9312.452 ns per `flat_loop` construction (0.55% of one construction), by timeit.timeit('is_dask_collection(target)', number=1000000) against a plain module-level function, in this process and this environment.
- Peak block count: The suite was asked to record a peak block count alongside peak bytes. The standard library exposes no such API: tracemalloc tracks peak bytes only (get_traced_memory()[1]) and a snapshot exposes the live block count, while the exact alternative would be a third-party allocation tracer, which the no-new-dependency directive forbids. The harder directive wins. The gate is evaluated on tracemalloc peak bytes, which the standard library supplies exactly, and two block figures are recorded as supporting evidence, each under its own definition and neither of them a peak: 'live_blocks_end' is the number of traced blocks still alive at the end of the region (end snapshot trace count minus the baseline count), and 'max_observed_blocks' is a sampled lower bound -- the maximum of sys.getallocatedblocks() read every 64th sys.setprofile event of one further untimed run, floored by the count taken immediately after the region while the constructed objects are still alive, minus the count taken before it.
- A/A calibration source: A/A calibration run, sha256 35ff5985a54f95af63796d220caeefe182b27e40f5df38507ae59a3fdaf973e9 (generated 2026-09-17T06:13:11.691939+00:00). Its path is not recorded: that run writes outside the checkout so the tree stays clean, which makes the location ephemeral and the digest the durable identifier. The A/A ratios are the noise floor of this machine; they are diagnostic, not a gate.
- A/A calibration (flat_loop): ratio median 0.997, 95% CI [0.990, 1.002]
- A/A calibration (linear_chain): ratio median 1.001, 95% CI [0.993, 1.010]
- A/A calibration (nested_containers): ratio median 1.007, 95% CI [0.997, 1.011]
- A/A calibration (wide_fan_in): ratio median 1.001, 95% CI [0.992, 1.011]
- A/A calibration (pure_vs_impure): ratio median 1.002, 95% CI [0.998, 1.009]
- A/A calibration (attr_and_operators): ratio median 1.004, 95% CI [0.999, 1.006]

## Gated cases

| case | baseline median (ms) | candidate median (ms) | paired ratio median | 95% CI | peak-allocation delta (%) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| flat_loop | 205.264 | 93.125 | 2.196 | [2.187, 2.221] | -0.0 | PASS |
| linear_chain | 103.349 | 80.684 | 1.279 | [1.270, 1.288] | -0.4 | PASS |
| nested_containers | 179.052 | 142.385 | 1.258 | [1.249, 1.264] | +0.0 | PASS |
| wide_fan_in | 25.914 | 4.232 | 6.152 | [6.112, 6.207] | -36.5 | PASS |
| pure_vs_impure | 277.042 | 129.175 | 2.136 | [2.128, 2.151] | +0.2 | PASS |
| attr_and_operators | 849.866 | 288.726 | 2.939 | [2.930, 2.952] | -0.0 | PASS |

## Informational sub-series (not gated)

| case | baseline median (ms) | candidate median (ms) | paired ratio median | 95% CI | peak-allocation delta (%) |
| --- | --- | --- | --- | --- | --- |
| nested_containers_shallow | 98.483 | 76.191 | 1.296 | [1.290, 1.301] | -0.1 |
| pure_true | 165.781 | 92.814 | 1.783 | [1.763, 1.796] | +0.3 |
| pure_false | 110.711 | 36.757 | 3.017 | [3.003, 3.058] | -0.1 |

OVERALL: PASS
