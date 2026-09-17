"""Benchmark corpus for the ``dask.delayed`` A/B performance suite.

This module holds the six gated cases and the three informational sub-series measured
by ``benchmarks.delayed_ab.main``. Every case is parameterised by the module under
test, which arrives as the ``mod`` argument, so the identical Python code path runs
against both arms — the frozen ``benchmarks.delayed_ab.baseline_delayed`` (arm A) and
the live ``dask.delayed`` (arm B). That is why this module imports no ``dask`` at all:
importing either arm here would hard-wire one of them.

Each case is a ``Case`` record pairing an untimed ``setup`` with a ``build`` that is
the timed region and performs graph construction only. ``Case`` documents the full
contract both callables obey.

The case sizes — the module-level ``_*_N`` constants, so every number is auditable in
one place — and the argument shapes are fixed properties of the corpus: they were
chosen before any measurement and are never re-tuned against results, so a case that
misses its threshold stays in the corpus and is reported with its cause named.

``f``, ``add``, ``ident`` and ``Obj`` are defined here rather than in an arm module
because they are tokenized by pickle-by-reference: a corpus-local definition embeds
the same module path for both arms, which is what makes their deterministic keys
comparable. For the same reason reflected operators, iterator arguments and
namedtuples are absent from the corpus — each embeds the arm's own module path
(``_swap``, ``_reconstruct_namedtuple``, any ``Delayed`` reached *through* pickle), so
their tokens would differ legitimately between arms even on identical
implementations. ``dask/tests/test_delayed_equivalence.py`` covers those constructs
against the live module instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

# The two callable shapes a case is built from. ``mod`` is the module under test and is
# intentionally untyped (``Any``): the two arms are distinct module objects that expose
# the same duck-typed surface (``delayed``, ``Delayed``, ...), and neither may be
# imported here to type it more precisely.
SetupFn: TypeAlias = Callable[..., Any]
BuildFn: TypeAlias = Callable[..., list[Any]]


def f(x: int) -> int:
    """Return ``x + 1``.

    The single-argument workhorse of the corpus. Defined at module level so it pickles
    by reference and therefore tokenizes identically for both arms.

    Args:
        x: The integer to increment.
    """
    return x + 1


def add(x: int, y: int) -> int:
    """Return ``x + y``.

    Used by the chain and ``pure``-keying cases, which need a two-argument callable.

    Args:
        x: The first addend.
        y: The second addend.
    """
    return x + y


def ident(*args: Any) -> tuple[Any, ...]:
    """Return ``args`` unchanged.

    One uniform definition serves both the single nested-container argument of
    ``nested_containers`` and the 5,000-argument fan-in of ``wide_fan_in``, so it has
    to accept varargs and hand the tuple straight back.

    Args:
        *args: The positional arguments to return unchanged.
    """
    return args


class Obj:
    """A plain attribute-and-method carrier for the ``attr_and_operators`` case.

    Emphatically *not* a dataclass and *not* a namedtuple: ``unpack_collections`` has a
    dedicated dataclass branch and a dedicated namedtuple branch, and an instance that
    tripped either would take a different code path and would not wrap into the
    ``DelayedLeaf`` that ``attr_and_operators`` needs. A plain object falls through
    ``unpack_collections`` to the final ``return expr, ()``, so
    ``delayed(Obj(3), pure=True)`` yields a ``DelayedLeaf`` whose single graph layer is
    ``{key: DataNode(key, obj)}``.
    """

    def __init__(self, v: int) -> None:
        """Initialise the instance's ``v`` attribute and its ``items`` list.

        Args:
            v: The integer carried on ``v``; ``items`` becomes ``[v, v + 1, v + 2]``.
        """
        self.v = v
        self.items = [v, v + 1, v + 2]

    def meth(self, x: int) -> int:
        """Return ``self.v + x``; the target of the delayed method call.

        Args:
            x: The integer added to ``self.v``.
        """
        return self.v + x


@dataclass(frozen=True)
class Case:
    """One benchmark case: a name plus its untimed ``setup`` and timed ``build``.

    Attributes:
        name: The case's identifier. It is the key under which the runner records the
            case in both artefacts, and — for the two entries of ``RATIO_CASES`` — the
            name the gate looks up.
        setup: ``setup(mod: Any, *, pure: bool | None) -> Any``. **Untimed.** Builds the
            case's inputs using the arm's own module (``mod.delayed(...)``) and returns
            an opaque state object, or ``None`` when the case needs no inputs. It must
            be safe to call repeatedly: the runner calls it fresh before every timed
            region so that no region reuses another's warm state.
        build: ``build(mod: Any, state: Any, *, pure: bool | None) -> list[Any]``.
            **The timed region.** Construction only — it must not call ``.compute()``
            or ``dask.compute``, must not touch ``__dask_graph__()``, and must not
            perform I/O, print, read the clock or mutate module state. It returns the
            list of constructed ``Delayed`` objects for the runner's equivalence
            assertions.

    Both callables take the module under test as their first argument and perform every
    construction through ``mod.delayed``, never through an imported ``delayed``. The
    ``pure`` keyword is threaded straight into the ``mod.delayed(...)`` calls:
    ``pure=None`` is each case's native keying, and the runner drives every case a
    second time with ``pure=True`` for its exact-key equivalence check.
    """

    name: str
    setup: SetupFn
    build: BuildFn


# Fixed benchmark sizes.
_FLAT_LOOP_N = 10_000
_LINEAR_CHAIN_N = 1_000
_NESTED_CONTAINERS_N = 200
_WIDE_FAN_IN_N = 5_000
_PURE_VS_IMPURE_N = 2_000
_ATTR_AND_OPERATORS_N = 2_000


def _setup_none(mod: Any, *, pure: bool | None) -> None:
    """Return ``None``: the shared setup of every case that needs no untimed inputs.

    ``flat_loop``, ``linear_chain``, ``nested_containers``, ``pure_vs_impure`` and all
    three sub-series build everything they need inside their timed region, so their
    state is ``None``. The ``Case`` contract asks only for a callable of the setup
    signature; one shared no-op keeps the corpus free of seven identical stubs.

    Args:
        mod: The module under test. Unused: this setup constructs nothing.
        pure: Accepted for uniformity with the setup signature and ignored.
    """
    return None


def _build_flat_loop(mod: Any, state: Any, *, pure: bool | None) -> list[Any]:
    """Wrap and call ``f`` 10,000 times, independently.

    The whole loop is the timed region. ``mod.delayed(f, pure=pure)`` is deliberately
    re-evaluated on every iteration rather than hoisted out of the loop: per-iteration
    re-wrapping is part of this case's definition, and it is what exercises the wrap
    path (``delayed`` -> ``DelayedLeaf``) once per constructed node.

    Args:
        mod: The module under test; every wrap and call goes through ``mod.delayed``.
        state: Unused — this case's setup is ``_setup_none``.
        pure: Threaded into every ``mod.delayed`` call of the loop.
    """
    return [mod.delayed(f, pure=pure)(i) for i in range(_FLAT_LOOP_N)]


def _build_linear_chain(mod: Any, state: Any, *, pure: bool | None) -> list[Any]:
    """Build a 1,000-node chain, each node depending on the previous one.

    The whole chain is the timed region. Only the final node is returned: its graph
    transitively contains all 1,000 layers, so equivalence over it covers the whole
    chain and computing it evaluates every node.

    Args:
        mod: The module under test; every node is built through ``mod.delayed``.
        state: Unused — this case's setup is ``_setup_none``.
        pure: Threaded into every ``mod.delayed`` call of the chain.
    """
    x = mod.delayed(f, pure=pure)(0)
    for i in range(1, _LINEAR_CHAIN_N):
        x = mod.delayed(add, pure=pure)(x, i)
    return [x]


def _build_nested_containers(mod: Any, state: Any, *, pure: bool | None) -> list[Any]:
    """Make 200 calls whose single argument is a deep, mixed container tree.

    The three leaves are built *inside* the timed region on purpose: the case measures
    leaf construction together with the container recursion it drives.

    The argument builder is fixed and not tunable. Per call it is exactly 9 ``Delayed``
    references to 3 distinct leaves and 12 containers across six nesting levels, with
    ``lit`` reused four times, so every ``unpack_collections`` container branch — list,
    tuple, dict, literal-only and mixed — is exercised on every call. The shape was
    selected by that branch-coverage and recursion-depth criterion before any
    measurement was taken.

    Args:
        mod: The module under test; leaves and calls go through ``mod.delayed``.
        state: Unused — this case's setup is ``_setup_none``.
        pure: Threaded into the leaf wraps and the 200 container calls.
    """
    objs: list[Any] = []
    for i in range(_NESTED_CONTAINERS_N):
        a, b, c = (mod.delayed(f, pure=pure)(i + j) for j in range(3))
        lit = [i, 2.5, "s", None, (i, i + 1)]
        l3 = [lit, lit, a]
        d2 = {"list": l3, "tuple": (lit, b), "leaf": c}
        l1 = [d2, (a, b, c), {"k": lit}]
        arg = [a, b, c, l1]
        objs.append(mod.delayed(ident, pure=pure)(arg))
    return objs


def _setup_wide_fan_in(mod: Any, *, pure: bool | None) -> list[Any]:
    """Build the 5,000 leaves of ``wide_fan_in`` outside the timed region.

    Keeping the leaves in setup is what lets the timed region isolate a single call
    that carries 5,000 dependencies — the link where the graph container probes every
    dependency while merging it.

    Args:
        mod: The module under test; every leaf is built through ``mod.delayed``.
        pure: Threaded into every leaf's ``mod.delayed`` call.
    """
    return [mod.delayed(f, pure=pure)(i) for i in range(_WIDE_FAN_IN_N)]


def _build_wide_fan_in(mod: Any, state: Any, *, pure: bool | None) -> list[Any]:
    """Make the single 5,000-argument call; only this call is timed.

    Args:
        mod: The module under test; the call goes through ``mod.delayed``.
        state: The 5,000 leaves from ``_setup_wide_fan_in``, spread as the call's
            positional arguments.
        pure: Threaded into the single ``mod.delayed`` call.
    """
    return [mod.delayed(ident, pure=pure)(*state)]


def _pure_vs_impure_half(mod: Any, p: bool) -> list[Any]:
    """Run one half of the ``pure``-keying workload: 2,000 two-level constructions.

    Both halves of ``pure_vs_impure`` and both ``pure_true``/``pure_false`` sub-series
    go through this one helper, so the workloads they compare are provably identical
    apart from the ``p`` they pin.

    Args:
        mod: The module under test; both levels are built through ``mod.delayed``.
        p: The ``pure`` value this half pins — ``True`` for deterministic keys,
            ``False`` for UUID keys.
    """
    return [
        mod.delayed(add, pure=p)(mod.delayed(f, pure=p)(i), i)
        for i in range(_PURE_VS_IMPURE_N)
    ]


def _build_pure_vs_impure(mod: Any, state: Any, *, pure: bool | None) -> list[Any]:
    """Run the same workload twice — once deterministic, once UUID-keyed.

    Both halves sit inside the one timed region. This case pins ``p`` itself and so
    does **not** vary with the ``pure`` parameter, which it accepts only for uniformity
    with the rest of the corpus: contrasting the two keying modes over an identical
    workload is the entire point of the case.

    Args:
        mod: The module under test, passed to both halves.
        state: Unused — this case's setup is ``_setup_none``.
        pure: Accepted for uniformity and ignored; each half pins its own ``p``.
    """
    return _pure_vs_impure_half(mod, True) + _pure_vs_impure_half(mod, False)


def _setup_attr_and_operators(mod: Any, *, pure: bool | None) -> tuple[Any, Any]:
    """Wrap the plain ``Obj`` and one leaf call outside the timed region.

    Both are wrapped with ``pure=True`` regardless of the variant under test, so the
    timed region starts from a fixed pair: ``o`` is a ``DelayedLeaf`` over the object
    and ``a`` is an ordinary one-argument call.

    Args:
        mod: The module under test; both wraps go through ``mod.delayed``.
        pure: Accepted for uniformity and ignored; both wraps pin ``pure=True``.
    """
    return mod.delayed(Obj(3), pure=True), mod.delayed(f, pure=True)(1)


def _build_attr_and_operators(mod: Any, state: Any, *, pure: bool | None) -> list[Any]:
    """Construct five lazy attribute/operator expressions, 2,000 times over.

    The five constructions, in this exact order, exercise: a chained left-associative
    arithmetic expression ending in a comparison, a delayed method call, a unary
    operator, ``getitem``, and a second comparison. Attribute access (``o.v``,
    ``o.items``) goes through ``Delayed.__getattr__`` to a ``DelayedAttr``, whose graph
    layer carries the legacy ``(getattr, key, attr)`` node, and ``o.meth(...)`` goes
    through ``DelayedAttr.__call__``. Only left-associative binary operators, unary
    operators, comparisons and ``getitem`` appear: reflected operators are excluded
    from the corpus because their ``_swap`` partial tokenizes by its defining module.

    Args:
        mod: The module under test. Unused: every construction here starts from the
            two objects ``setup`` already wrapped with the arm's own module.
        state: The ``(o, a)`` pair from ``_setup_attr_and_operators``.
        pure: Threaded into the delayed method call ``o.meth(i, pure=pure)``.
    """
    o, a = state
    objs: list[Any] = []
    for i in range(_ATTR_AND_OPERATORS_N):
        objs.append(((o.v + a) * i - a) == a)
        objs.append(o.meth(i, pure=pure))
        objs.append(-o.v)
        objs.append(o.items[1])
        objs.append(a < o.v)
    return objs


#: Gated cases in gate order.
CASES: tuple[Case, ...] = (
    Case(name="flat_loop", setup=_setup_none, build=_build_flat_loop),
    Case(name="linear_chain", setup=_setup_none, build=_build_linear_chain),
    Case(name="nested_containers", setup=_setup_none, build=_build_nested_containers),
    Case(name="wide_fan_in", setup=_setup_wide_fan_in, build=_build_wide_fan_in),
    Case(name="pure_vs_impure", setup=_setup_none, build=_build_pure_vs_impure),
    Case(
        name="attr_and_operators",
        setup=_setup_attr_and_operators,
        build=_build_attr_and_operators,
    ),
)

#: The two cases that carry the >= 1.25 paired-ratio threshold of the gate.
RATIO_CASES: tuple[str, str] = ("flat_loop", "nested_containers")


def _build_nested_containers_shallow(
    mod: Any, state: Any, *, pure: bool | None
) -> list[Any]:
    """Make 200 calls whose argument is a single two-level list of ten literals.

    Three leaves per call as in ``nested_containers``, but the argument fails the
    depth/branch-coverage criterion the gated shape was selected by. It exists
    precisely to show what such a shape measures, so it is informational only and must
    never be promoted into ``CASES``.

    Args:
        mod: The module under test; leaves and calls go through ``mod.delayed``.
        state: Unused — this sub-series' setup is ``_setup_none``.
        pure: Threaded into the leaf wraps and the 200 container calls.
    """
    objs: list[Any] = []
    for i in range(_NESTED_CONTAINERS_N):
        a, b, c = (mod.delayed(f, pure=pure)(i + j) for j in range(3))
        arg = [
            a,
            b,
            c,
            [i, 2.5, "s", None, (i, i + 1), i + 2, "t", 3.5, None, (i + 3,)],
        ]
        objs.append(mod.delayed(ident, pure=pure)(arg))
    return objs


def _build_pure_true(mod: Any, state: Any, *, pure: bool | None) -> list[Any]:
    """Run the deterministic half of the ``pure``-keying workload on its own.

    Args:
        mod: The module under test, passed to the shared half helper.
        state: Unused — this sub-series' setup is ``_setup_none``.
        pure: Accepted for uniformity and ignored; the half pins ``p=True``.
    """
    return _pure_vs_impure_half(mod, True)


def _build_pure_false(mod: Any, state: Any, *, pure: bool | None) -> list[Any]:
    """Run the UUID-keyed half of the ``pure``-keying workload on its own.

    Args:
        mod: The module under test, passed to the shared half helper.
        state: Unused — this sub-series' setup is ``_setup_none``.
        pure: Accepted for uniformity and ignored; the half pins ``p=False``.
    """
    return _pure_vs_impure_half(mod, False)


#: The informational sub-series. They obey the same ``setup``/``build`` contract and
#: receive the same rounds, warmup, statistics, confidence interval, allocation figures
#: and equivalence assertions as the gated cases, but they are reported separately and
#: never enter the gate checklist or its four-of-six count. Membership in ``SUBSERIES``
#: rather than ``CASES`` is the only thing that distinguishes them.
SUBSERIES: tuple[Case, ...] = (
    Case(
        name="nested_containers_shallow",
        setup=_setup_none,
        build=_build_nested_containers_shallow,
    ),
    Case(name="pure_true", setup=_setup_none, build=_build_pure_true),
    Case(name="pure_false", setup=_setup_none, build=_build_pure_false),
)
