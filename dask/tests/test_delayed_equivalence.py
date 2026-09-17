"""Characterisation of every observable property of ``dask.delayed``.

This module is the equivalence evidence for a purely internal performance
refactor of ``dask/delayed.py``. It locks, for a corpus of delayed expressions
that reaches every branch of ``dask.delayed.unpack_collections``:

* the generated key strings -- character for character wherever they are
  deterministic and reproducible, and structurally (the token replaced by a
  placeholder) where the token is inherently random or, for the twelve entries
  of ``_ENVIRONMENT_DEPENDENT_KEY_TOKEN``, decided by the environment rather
  than by the module under test;
* the canonical, order-independent graph serialization, including the two
  order-sensitive fields ``layer_order`` and ``dependency_order`` -- layer
  insertion order is token-relevant, so it is part of the contract;
* ``__dask_keys__()`` and ``__dask_layers__()``;
* the computed result under the synchronous scheduler, plus equality of the
  results across the synchronous and threaded schedulers and across a pickle
  round trip;
* the exact side-effect counts (``__repr__``/``__hash__``/``__eq__`` calls) the
  construction path performs on user-supplied key, layer and type objects;
* every exception and warning the module raises, with its message.

Golden fixture:
    ``GOLDEN`` below was captured from the *pre-refactor* module and is frozen.
    Regenerating it to make a test pass is a failure of the refactoring run: the
    fixture is the only thing that can tell a behaviour-preserving change from a
    behaviour-changing one. The only legitimate edits are the initial capture and
    the addition of new expressions whose values are captured *before* the
    refactor begins.

    Capture command, run from the repository root::

        python -c "from dask.tests.test_delayed_equivalence import write_golden; write_golden()"

    It must never be run as ``__main__`` (``python
    dask/tests/test_delayed_equivalence.py``) because functions and classes
    defined in a ``__main__`` module pickle by value as ``__main__.*`` and would
    produce different pickle-sensitive tokens from the ones pytest sees.

Provenance of the golden block:
    The marker-delimited block was first captured, pre-refactor, in commit
    ``1cbdf00d3`` -- 81 entries, block sha256 ``6c8c6669d017d741``, unchanged
    through ``76a0d1ca3``. It was re-emitted once, in ``c771b566e`` -- 90
    entries, block sha256 ``3e5a9e5f9313ad4f``, unchanged since -- a commit that
    also modified ``dask/delayed.py``, so the block's git history alone no longer
    shows the capture predating the first production edit. What the re-emission
    changed, entry by entry: 9 expressions were added and none removed; of the 81
    entries the two blocks share, **0** changed their ``key`` and **0** changed
    their ``result_repr``; all 81 gained the then-new ``stable_graph`` field; and
    exactly 3 (``attr_v``, ``attr_of_attr``, ``attr_items_getitem``) changed their
    ``graph``, in one field only -- a legacy-tuple node's dependency list ``[]``
    became ``['obj']`` because ``canon.py``, in that same commit, began resolving
    a legacy tuple task's dependencies against the complete low-level graph
    instead of against one layer's nodes, which is where a ``DelayedAttr`` node's
    parent actually lives.

    The ordering half of the §0.5.1 proof is therefore permanently unavailable
    from this branch's history: the branch is published, rewriting it is
    forbidden, and the block cannot be re-captured into a corrected history. What
    is accepted in its place -- knowingly, as the resolution of that failure, and
    not as a claim this docstring makes -- is a re-derivation this module asserts
    on every run, in two tests::

        test_the_golden_re_derives_from_the_frozen_baseline_arm
        test_the_arm_dependent_entries_are_the_only_verbatim_exceptions

    The first rebuilds all 90 committed entries with the frozen baseline arm
    (``benchmarks/delayed_ab/baseline_delayed.py``: ``dask/delayed.py`` at the
    base commit, one commit in its git history, made before the first production
    edit) and compares every field of every entry against the block. The second
    restores the character-for-character comparison wherever the environment
    reproduces the capture: 88 of the 90 agree byte for byte, and the 2 that
    cannot -- ``arg_list_iterator_with_delayed`` and ``op_reflected_add`` --
    differ in nothing but their own key token, which embeds the arm's module path
    for the pickle-by-reference reason AAP §0.4.1 documents as inherently
    arm-dependent. That pair is what makes the substitution durable rather than
    historical: a golden carrying a value the pre-refactor module does not produce
    fails them, which is the whole of what the ordering requirement was there to
    prevent. The block is never regenerated again.

    The values are therefore a pre-refactor capture that was re-serialised, not a
    re-measurement of refactored behaviour, and that is checkable rather than
    asserted. Three further re-verifications were run by hand, each reproducible
    from this tree. The first two need a base-commit worktree, built from the
    repository root as::

        git worktree add <scratch>/base_wt c9d1df34ccba182ddf43c2dbe4315c4d9c8c44e1
        mkdir -p <scratch>/base_wt/benchmarks/delayed_ab
        cp dask/_version.py <scratch>/base_wt/dask/_version.py
        cp benchmarks/delayed_ab/__init__.py benchmarks/delayed_ab/canon.py <scratch>/base_wt/benchmarks/delayed_ab/
        cp benchmarks/delayed_ab/baseline_delayed.py <scratch>/base_wt/benchmarks/delayed_ab/
        cp dask/tests/test_delayed_equivalence.py <scratch>/base_wt/dask/tests/

    1. Re-capture against the pre-refactor module, from inside that worktree::

           python -c "from dask.tests.test_delayed_equivalence import write_golden; write_golden()"

       All 90 entries come back field for field identical to the block committed
       here -- no key, no graph, no ``stable_graph``, no ``result_repr`` differs.
    2. Run this module, unmodified, against the pre-refactor module, from inside
       that worktree::

           DASK_DELAYED_BASELINE_ORACLE=1 python -m pytest dask/tests/test_delayed_equivalence.py

       It passes -- 160 passed, 1 skipped, the skip being the gated test below.
       Every exact key and every raw canonical graph in the block is therefore
       reproduced *by the baseline module*, which is the property the git history
       was supposed to show.
    3. Run the evidence as it stood *before* the re-emission against the current
       module: restore the module and the canonicaliser of ``76a0d1ca3`` -- the
       81-entry block, the pre-``stable_graph`` field set -- over a worktree at
       this branch's head::

           git worktree add --detach <scratch>/head_wt HEAD
           cp dask/_version.py <scratch>/head_wt/dask/_version.py
           git -C <scratch>/head_wt checkout 76a0d1ca3 -- dask/tests/test_delayed_equivalence.py benchmarks/delayed_ab/canon.py
           cd <scratch>/head_wt && python -m pytest dask/tests/test_delayed_equivalence.py

       It passes (118 passed), so the older, provably-pre-refactor values still
       hold against the refactored module and the re-emission covers no
       behaviour change.

    A canonicaliser change lands before the first production edit next time, or
    the fixture is split so a serialization change cannot force a rewrite of
    behaviour-bearing values.

Portability:
    Deterministic tokens of anything tokenized through pickle depend on which
    optional hash library is installed, so the autouse fixture pins the hasher
    and the two relevant configuration keys. See ``_CONFIG_PINS``.

    The pin reaches every token this module's own builders produce, and two
    tokens it cannot reach, both recorded in ``_ENVIRONMENT_DEPENDENT_KEY_TOKEN``
    with the cause: a token frozen when ``dask.delayed`` was *imported* (the
    operator methods bind ``delayed(op, pure=True)`` at class-binding time), and
    a dataclass token whose input set is the running interpreter's. Those entries
    have their own key token -- and only that token, proven minimal by
    observation -- replaced by a placeholder on both the live and the golden side,
    exactly as an inherently random UUID token is; every other token, every node
    kind, every dependency edge, both insertion orders and the computed result
    stay under verbatim assertion, and the character-for-character comparison
    still runs wherever the environment reproduces the capture
    (``test_environment_dependent_key_tokens_match_the_golden_where_reproducible``).

Pre-refactor oracle run:
    Re-running this module against the pre-refactor module (re-verification 2
    above) needs one environment variable::

        DASK_DELAYED_BASELINE_ORACLE=1 pytest dask/tests/test_delayed_equivalence.py

    It skips the single test whose chain length is only affordable on the
    refactored construction path -- see
    ``test_chain_at_and_past_the_ancestor_bound_matches_the_generic_path`` -- and
    nothing else. Without it that test cannot terminate on the baseline, and
    because the project runs ``timeout_method = "thread"`` a 300 s hit takes the
    whole session down rather than the one test.

Notes:
    The canonicaliser is imported from ``benchmarks/delayed_ab/canon.py`` and is
    never re-implemented here: the A/B performance harness and this test have to
    share one definition of "equivalent graph" or the two bodies of evidence can
    drift apart.

"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import operator
import os
import pickle
import pprint
import subprocess
import sys
import traceback
import types
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import pytest

import dask
import dask.hashing
from dask._expr import _ExprSequence
from dask._task_spec import DataNode, Task, TaskRef
from dask.base import collections_to_expr
from dask.delayed import (
    Delayed,
    DelayedAttr,
    DelayedLeaf,
    delayed,
    finalize,
    to_task_dask,
    unpack_collections,
)
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as _threaded_get
from dask.tokenize import TokenizationError
from dask.utils_test import inc

# The single canonicaliser shared with the A/B harness: ``importlib`` names the
# canonical runtime module, so this module and ``benchmarks.delayed_ab.main`` bind
# the very same ``sys.modules["benchmarks.delayed_ab.canon"]`` entry and the two
# bodies of evidence cannot drift apart. The static absolute import is unavailable
# because ``benchmarks/`` is a namespace package, under which mypy maps ``canon.py``
# under a second module name and rejects the build. Attribute access on a module
# object is typed ``Any``, so each alias below is annotated with the signature
# ``canon.py`` declares for it and every call in this module is checked against it.
_canon = importlib.import_module("benchmarks.delayed_ab.canon")
canonical_graph: Callable[[object], dict[str, Any]] = _canon.canonical_graph
canonical_result: Callable[[Sequence[object]], tuple[Any, ...]] = (
    _canon.canonical_result
)
normalize_key: Callable[[object, dict[str, str]], object] = _canon.normalize_key

# Configuration pinned for every test *and* for ``write_golden()``, defined once
# so the capture and the assertions cannot drift:
#
# * ``tokenize.ensure-deterministic`` False -- the module default. Tests that
#   need strict mode enable it locally.
# * ``delayed_pure`` False -- the module default read by ``dask.delayed.tokenize``
#   whenever ``pure`` is omitted. Corpus entries that want the global setting
#   turn it on inside their own builder.
_CONFIG_PINS: dict[str, Any] = {
    "tokenize.ensure-deterministic": False,
    "delayed_pure": False,
}

#: The hash library ``dask.hashing`` selected in *this* process, read here at
#: import time and therefore before any fixture can pin it. It is the state
#: ``dask.delayed`` itself was imported under, which is what decides the two
#: tokens the pin cannot reach (see ``_ENVIRONMENT_DEPENDENT_KEY_TOKEN``).
_AMBIENT_HASHER: str = dask.hashing.hashers[0].__name__

#: The same reading, taken in the environment the golden block was captured in:
#: the locked ``pixi`` ``default`` environment, CPython 3.14.6 with
#: ``python-xxhash`` and ``mmh3`` installed and ``python-cityhash`` absent.
_CAPTURE_HASHER = "_hash_xxhash"

#: The interpreter the golden block was captured on.
_CAPTURE_PYTHON = (3, 14)

#: ``_DataclassParams.__slots__`` as the capture interpreter declared it.
#: ``dask.tokenize`` normalises a dataclass instance by reading every attribute
#: this tuple names, so its membership -- fixed by the interpreter, not by dask --
#: is an input to every dataclass token. CPython 3.10 declares the first six only.
_CAPTURE_DATACLASS_PARAM_SLOTS = (
    "init",
    "repr",
    "eq",
    "order",
    "unsafe_hash",
    "frozen",
    "match_args",
    "kw_only",
    "slots",
    "weakref_slot",
)

# Both predicates are declared ``bool`` rather than left as the comparison
# expressions: a bare ``sys.version_info`` comparison is folded by mypy against
# the configured 3.10 baseline, which would make every branch guarded by it
# unreachable and trip ``warn_unreachable``.
#: Whether this environment reproduces the import-time tokens of the capture.
_HASHER_REPRODUCES_CAPTURE: bool = _AMBIENT_HASHER == _CAPTURE_HASHER
#: Whether this environment reproduces the dataclass tokens of the capture.
_INTERPRETER_REPRODUCES_CAPTURE: bool = sys.version_info[:2] == _CAPTURE_PYTHON

#: Environment variable that switches this module into pre-refactor-oracle mode,
#: where it is run against the baseline ``dask/delayed.py`` in a base-commit
#: worktree. It gates exactly one test, whose cost on the pre-refactor
#: construction path is exponential in the chain length it needs; see the
#: "Pre-refactor oracle run" section of the module docstring.
_BASELINE_ORACLE_ENV = "DASK_DELAYED_BASELINE_ORACLE"

#: True when the variable is set to exactly ``"1"``, so an accidental ``"0"``,
#: ``"true"`` or empty value leaves the full module running.
_BASELINE_ORACLE: bool = os.environ.get(_BASELINE_ORACLE_ENV) == "1"


@pytest.fixture(autouse=True)
def _pin_tokenization(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the hasher and the configuration for every test in this module.

    The hasher pin is mandatory rather than cosmetic. ``dask.hashing.hashers`` is
    an ordered preference list built at import time (``cityhash`` > ``xxhash`` >
    ``mmh3`` > SHA-1, ``dask/hashing.py:6-70``) and the deterministic token of
    anything tokenized through pickle is the md5 of a digest produced by
    ``hashers[0]``. The project's own environments install ``python-xxhash``,
    ``mmh3`` and ``python-cityhash`` (``pixi.toml:82-88,118-121``), so without the
    pin the exact-key assertions would hold only on an installation that happens
    to match whichever library produced the golden fixture.

    Patching works -- and no edit to ``dask/hashing.py`` is needed -- because
    ``hash_buffer`` iterates the *module-level* ``hashers`` list at call time
    (``dask/hashing.py:86``), not at import time.

    What the pin does **not** reach is a token computed before it exists. The
    operator methods of ``Delayed`` are bound while the class is built, and each
    binding tokenizes its operator function under whichever library was ambient
    when ``dask.delayed`` was imported; a dataclass token, separately, is
    computed from inputs the interpreter decides. Those two are handled by
    ``_ENVIRONMENT_DEPENDENT_KEY_TOKEN`` rather than by this fixture, which
    cannot see them.

    Every mutation is reverted: ``monkeypatch`` restores the list and the
    ``dask.config.set`` context restores the configuration, so the module leaves
    no residue for the rest of the session.

    Args:
        monkeypatch: pytest's per-test patcher, used for the hasher pin because
            it restores ``dask.hashing.hashers`` when the test ends whatever the
            test does to it.

    Yields:
        None: with ``dask.hashing.hashers`` pinned to SHA-1 and ``_CONFIG_PINS``
        applied, for the duration of one test.

    """
    monkeypatch.setattr(dask.hashing, "hashers", [dask.hashing._hash_sha1])
    with dask.config.set(_CONFIG_PINS):
        yield


@contextlib.contextmanager
def _pinned() -> Iterator[None]:
    """Apply the same pins as ``_pin_tokenization`` without pytest.

    ``write_golden()`` runs outside pytest, where no fixture applies, so it needs
    a save/restore of its own. The pins are read from ``_CONFIG_PINS`` so the two
    code paths cannot diverge.

    Yields:
        None: with ``dask.hashing.hashers`` pinned to SHA-1 and ``_CONFIG_PINS``
        applied for the duration of the block.

    """
    saved = dask.hashing.hashers
    dask.hashing.hashers = [dask.hashing._hash_sha1]
    try:
        with dask.config.set(_CONFIG_PINS):
            yield
    finally:
        dask.hashing.hashers = saved


# ---------------------------------------------------------------------------
# Corpus callables and types.
#
# Every one of them is defined at module level so that it pickles *by
# reference*. Two things depend on that: the pickle round-trip test, and
# ``tokenize``'s pickle path, which produces the deterministic tokens the golden
# fixture records character for character. A callable or class defined inside a
# function (or in a ``__main__`` module) pickles by value and would tokenize
# differently in every process.
# ---------------------------------------------------------------------------


def ident(*args: Any) -> tuple[Any, ...]:
    return args


def collect(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return args, kwargs


def listof(*args: Any) -> list[Any]:
    return list(args)


def first(x: Any) -> Any:
    return x


def pair() -> tuple[int, int]:
    return (1, 2)


def single() -> tuple[int]:
    return (7,)


def nothing() -> tuple[Any, ...]:
    return ()


class Inner:
    def __init__(self, w: int) -> None:
        self.w = w


class Obj:
    def __init__(self, v: int) -> None:
        self.v = v
        self.items = [10, 11, 12]
        self.inner = Inner(9)

    def meth(self, x: int) -> int:
        return self.v + x

    def __repr__(self) -> str:
        """Return a value-based repr.

        The default ``object.__repr__`` embeds the instance's memory address,
        which would make the golden ``result_repr`` of the entry that computes to
        an :class:`Obj` unreproducible. Tokens are unaffected either way:
        ``tokenize`` normalises an instance through pickle, not through ``repr``.
        """
        return f"Obj(v={self.v!r})"


@dataclass
class Box:
    a: Any
    b: str


class Point(NamedTuple):
    x: Any
    y: int


class HandRolledCollection:
    """A non-``Delayed`` dask collection, modelled on ``test_delayed.py``'s ``Tuple``.

    This is what drives the ``base.is_dask_collection(expr)`` branch of
    ``unpack_collections`` without requiring NumPy, so the branch is covered in a
    minimal-dependency environment too.
    """

    __dask_scheduler__ = staticmethod(_threaded_get)
    __dask_optimize__ = None

    # ``dsk`` is annotated ``Any`` because a dask collection may hand back either
    # graph shape, and both are used: a plain ``dict`` low-level graph and a
    # ``HighLevelGraph`` (see ``_hlg_backed_collection``).
    def __init__(self, dsk: Any, keys: list[str]) -> None:
        self._dask = dsk
        self._keys = keys

    def __dask_tokenize__(self) -> list[str]:
        """Return the output keys as the collection's token.

        Returns:
            list: the output keys. ``normalize_object`` returns the value of this
            method as it stands, without recursive dispatch
            (``dask/tokenize.py:194-198``), so the keys alone decide the
            deterministic token of any expression that wraps this collection.

        """
        return self._keys

    def __dask_graph__(self) -> Any:
        return self._dask

    def __dask_keys__(self) -> list[str]:
        return self._keys

    def __dask_postcompute__(self) -> tuple[Any, tuple[Any, ...]]:
        return tuple, ()


class SubDelayed(Delayed):
    """A ``Delayed`` subclass with no overrides whatsoever.

    Its single distinguishing property is that ``type(obj) is Delayed`` is False,
    which is exactly what an exact-class-identity guard on the construction path
    must reject, sending the whole call down the generic
    ``HighLevelGraph.from_collections`` path.
    """

    __slots__ = ()


class SubDelayedLayers(Delayed):
    """A ``Delayed`` subclass that overrides ``__dask_layers__``.

    The override returns the same value the base class returns, so the graph
    content is unchanged; a public subclass is nonetheless free to return
    something else, which is why a merge shortcut may not assume the base
    implementation.
    """

    __slots__ = ()

    def __dask_layers__(self) -> Sequence[str]:
        return (self._layer,)


class SubDelayedLayerDict(Delayed):
    """A ``Delayed`` subclass that defines a method named ``_layer_dict``.

    This is the name-collision guard: a leaf-layer helper on the construction
    path must be a module-level function, not a method, or a public subclass
    could intercept it. Nothing may ever dispatch to this method, so it raises.
    """

    __slots__ = ()

    def _layer_dict(self) -> dict[Any, Any]:
        """Fail: the construction path must never dispatch to this name.

        Raises:
            AssertionError: always. Reaching this body would mean the graph-merge
                helper looked the name up on the object instead of calling the
                module-level function, letting a public subclass intercept it.

        """
        raise AssertionError(
            "a subclass method named _layer_dict must never be dispatched to"
        )


def _generate() -> Iterator[int]:
    """Yield one value; a generator object cannot be tokenized deterministically."""
    yield 1


def _leaf() -> Any:
    """Return a one-node, deterministically keyed ``Delayed`` used as a dependency.

    The return type is ``Any`` rather than ``Delayed`` on purpose: ``Delayed``
    binds its arithmetic, comparison and ``getitem`` operators dynamically at
    import time through ``Delayed._bind_operator``, so a static checker cannot
    see them and the operator corpus entries below would not type-check against a
    ``Delayed`` annotation.
    """
    return delayed(inc, name="inc", pure=True)(1)


def _other_leaf() -> Any:
    return delayed(inc, name="inc-other", pure=True)(2)


def _ident() -> Any:
    """Return ``delayed(ident)`` with an explicit ``name``.

    The explicit name keeps the resulting call key derived from strings, ints and
    dependency keys only -- ``call_function`` tokenizes the leaf's own key rather
    than the callable -- so the key does not depend on how the function object
    pickles. Entries that deliberately exercise the default naming forms are
    listed in ``_DEFAULT_NAMING``.
    """
    return delayed(ident, name="ident", pure=True)


def _collect() -> Any:
    return delayed(collect, name="collect", pure=True)


def _obj() -> Delayed:
    return delayed(Obj(3), name="obj", pure=True)


def _hand_rolled_collection() -> HandRolledCollection:
    return HandRolledCollection(
        {"ta": DataNode("ta", 1), "tb": DataNode("tb", 2)}, ["ta", "tb"]
    )


def _hlg_backed_collection() -> HandRolledCollection:
    """Return a hand-rolled dask collection whose graph is a ``HighLevelGraph``.

    The same protocol as :func:`_hand_rolled_collection` with the other graph
    shape, so that a claim about what ``finalize`` does to a collection can be
    made about both shapes rather than about one of them.
    """
    return HandRolledCollection(
        HighLevelGraph({"hc": {"hc": DataNode("hc", 11)}}, {"hc": set()}), ["hc"]
    )


class _OpaqueKey:
    """A valid graph key whose ``repr`` is the default, address-bearing one.

    Hashable and equality-comparable, so dask accepts it as a key; unencodable,
    because ``object.__repr__`` embeds the instance's memory address and
    committed evidence may not carry one.
    """

    def __hash__(self) -> int:
        return 11

    def __eq__(self, other: object) -> bool:
        return self is other


class _IdentityYieldingTuple(tuple):
    """A ``tuple`` subclass whose iteration yields its own identity.

    Nothing stops a graph key from being a tuple subclass, and a canonicaliser
    that accepts one on an ``isinstance`` test and rebuilds it as a plain tuple
    would carry whatever this ``__iter__`` produces into the canonical output --
    here a process-specific ``id()``. It is the reason the encodable-key check is
    by exact type.
    """

    def __iter__(self) -> Iterator[Any]:
        return iter([f"id-{id(self)}"])


class _ListSubclass(list):
    """A ``list`` subclass, for the single-list unwrapping boundary test."""


class _RaisingReprKey:
    """A valid graph key whose ``repr`` raises.

    The canonicaliser's refusal message must be built from type metadata alone:
    formatting the offending object would turn a clear diagnosis into this
    unrelated ``RuntimeError``.
    """

    def __hash__(self) -> int:
        return 13

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        raise RuntimeError("repr executed")


def _hlg_of(key: str) -> HighLevelGraph:
    return HighLevelGraph({key: {key: DataNode(key, 5)}}, {key: set()})


def _dict_graph_delayed() -> Delayed:
    """Return a ``Delayed`` backed by a plain ``dict`` low-level graph.

    ``Delayed`` legitimately accepts one (``dask.graph_manipulation`` builds them
    that way), and it selects the non-``HighLevelGraph`` branch of
    ``HighLevelGraph._from_collection`` (``dask/highlevelgraph.py:462-465``).
    """
    return Delayed("dgkey", {"dgkey": DataNode("dgkey", 5)})


def _mapping_proxy_delayed() -> Delayed:
    """Return a ``Delayed`` whose graph is a ``types.MappingProxyType``.

    It is neither exactly a ``HighLevelGraph`` nor exactly a ``dict``, and
    ``Delayed.__init__`` only validates the layer for a ``HighLevelGraph``, so
    construction succeeds and the object is a valid dependency that no
    exact-graph-type guard may take a shortcut for.
    """
    return Delayed("mpkey", types.MappingProxyType({"mpkey": DataNode("mpkey", 5)}))


def _sub_delayed() -> SubDelayed:
    return SubDelayed("subkey", _hlg_of("subkey"))


def _sub_delayed_dask_layers() -> SubDelayedLayers:
    return SubDelayedLayers("sublayerskey", _hlg_of("sublayerskey"))


def _sub_delayed_layer_dict() -> SubDelayedLayerDict:
    return SubDelayedLayerDict("sublayerdictkey", _hlg_of("sublayerdictkey"))


class _Expr(NamedTuple):
    """One corpus entry: a named builder plus the assertions it takes part in.

    Attributes:
        name: Registry key, ``GOLDEN`` key and pytest parameter id, all in one.
        build: Zero-argument builder returning the object to characterise. It is
            called afresh by every test, so the key and graph assertions are
            reproducibility checks rather than identity checks.
        deterministic: True when the generated key carries a deterministic token,
            so the golden key is asserted character for character. False when the
            token is a UUID, in which case the key is compared after
            ``normalize_key`` replaces the token with a structural placeholder.
        canonical: True when the canonical graph serialization is reproducible
            verbatim and is therefore recorded in the golden and compared as it
            stands.
        computable: True when the entry takes part in the result assertions.
        picklable: True when the entry takes part in the pickle round trip.
        volatile_tokens: True when the graph carries an inherently random 32-hex
            identifier -- a ``uuid.uuid4().hex`` fallback from ``dask._expr``, not
            a ``tokenize`` digest -- so the raw canonical dict differs between two
            builds of the same expression. Such an entry pairs
            ``canonical=False`` with this flag: its graph is still asserted in
            full, through ``_stable_graph``, which replaces *only* the tokens a
            double build proves random and leaves every deterministic token,
            every key, every node kind, every dependency and both insertion
            orders under assertion. No entry may opt out of graph comparison
            altogether; ``test_every_entry_asserts_its_graph`` enforces that.

    Every ``False`` flag is justified by name in ``_EXCLUSION_REASONS``, which
    ``test_flag_exclusions_are_documented`` keeps in step with this registry, and
    an entry whose computation raises is registered with its exact failure in
    ``_COMPUTE_FAILURES``.

    """

    name: str
    build: Callable[[], Any]
    deterministic: bool = True
    canonical: bool = True
    computable: bool = True
    picklable: bool = True
    volatile_tokens: bool = False


def _build_duplicate_positional() -> Delayed:
    a = _leaf()
    return _ident()(a, a)


def _build_duplicate_in_list() -> Delayed:
    a = _leaf()
    return _ident()([a, a, a])


def _build_global_delayed_pure() -> Delayed:
    """Take the deterministic key from the *global* ``delayed_pure`` setting.

    ``pure`` is omitted, so ``dask.delayed.tokenize`` reads
    ``config.get("delayed_pure", False)`` on every call.
    """
    with dask.config.set({"delayed_pure": True}):
        return delayed(inc, name="inc")(1)


def _build_finalize_collection() -> Delayed:
    """Wrap a non-``Delayed`` collection with ``dask.delayed.finalize``.

    This locks the ``"finalize-" + tokenize(collection)`` key form
    (``dask/delayed.py:57``). ``finalize`` reads the global ``delayed_pure``
    setting through the module-level ``tokenize`` wrapper, so the setting is what
    makes the key reproducible instead of a UUID.
    """
    with dask.config.set({"delayed_pure": True}):
        return finalize(_hand_rolled_collection())


def _build_nout_one_element() -> Delayed:
    (x,) = delayed(single, name="single", pure=True, nout=1)()
    return x


def _build_nout_two_first() -> Delayed:
    x, _y = delayed(pair, name="pair", pure=True, nout=2)()
    return x


def _build_nout_two_second() -> Delayed:
    _x, y = delayed(pair, name="pair", pure=True, nout=2)()
    return y


def _tuple_key_delayed() -> Delayed:
    """Return a dependency whose key -- and layer name -- is a ``tuple``.

    Tuple keys are what array, dataframe and bag collections use, and they are
    the reason a key guard on the construction path has to recurse: every
    element of the tuple has to be a builtin scalar before the guard may treat
    the key as one a merge shortcut can hash and ``repr`` without a user-visible
    side effect. This dependency is the accepting side of that recursion; the
    rejecting side is exercised by
    ``test_tuple_key_with_a_non_builtin_element_falls_back``, whose key cannot go
    into the golden because it is not JSON-expressible.
    """
    key = ("tk", 0)
    return Delayed(key, {key: DataNode(key, 5)})


def _build_list_with_dependent_task() -> Delayed:
    """``f([task, 1])``: a container holding a runnable node with a dependency.

    Nothing in the list is a dask collection, so the container branch of
    ``unpack_collections`` collects no collections at all -- and it still must
    not hand the list back unchanged, because the ``Task`` inside it references
    the key ``"dn"``. What decides is ``List(*args).dependencies``, and a
    construction path that reads it must reach the same verdict for a node whose
    ``dependencies`` are non-empty as for a bare ``TaskRef``.
    """
    return _ident()([Task("inner", ident, TaskRef("dn")), 1])


def _build_several_dependencies_with_leaf() -> Delayed:
    """``f(a, leaf)``: several dependencies, one of them a ``DelayedLeaf``.

    A leaf contributes a single-node layer rather than a graph of its own, which
    is a different arm of the several-dependency merge from the one an ordinary
    ``Delayed`` dependency takes.
    """
    return _ident()(_leaf(), delayed(3, name="three-leaf"))


def _build_several_dependencies_with_attr() -> Delayed:
    """``f(o.v, a)``: several dependencies, one of them a ``DelayedAttr``.

    Two things ride on this shape. The merge has to take the attribute's lazy
    ``dask`` -- which itself merges its parent -- alongside an unrelated
    dependency; and the resulting graph holds the legacy tuple task ``(getattr,
    parent_key, attr)`` in a layer *other* than the one holding its parent, which
    is the case a canonical serialization has to resolve against the whole graph
    rather than one layer.
    """
    return _ident()(_obj().v, _leaf())


def _build_delayed_call(pure: bool | None) -> Delayed:
    """Call a ``Delayed`` whose computed value is itself a callable.

    ``Delayed.__call__`` routes through ``delayed(apply, pure=pure)``, so with
    ``pure`` omitted both the ``apply`` leaf key and the call key carry UUID
    tokens.

    Args:
        pure: Forwarded to ``Delayed.__call__``; ``None`` omits the keyword.

    Returns:
        Delayed: the result of calling the delayed callable with ``1``.

    """
    callable_value = delayed(first, name="first", pure=True)(inc)
    if pure is None:
        return callable_value(1)
    return callable_value(1, pure=pure)


# ---------------------------------------------------------------------------
# The corpus.
#
# Hazard, handled deliberately below: ``Delayed.__hash__`` is ``hash(self.key)``
# over a ``str``, and ``str`` hashing is randomised by ``PYTHONHASHSEED``. A
# ``set`` holding more than one ``Delayed``, or elements that are not small ints,
# therefore iterates in a seed-dependent order, which would change both the
# ``List`` argument order and the generated token. Sets holding a ``Delayed`` are
# single-element, and literal sets hold small ints only (``hash(i) == i``, so
# iteration order is fixed).
# ---------------------------------------------------------------------------

CORPUS: tuple[_Expr, ...] = (
    # Atomic arguments: the branch cascade of ``unpack_collections`` falls
    # through to ``return expr, ()`` for each of them.
    _Expr("arg_int", lambda: _ident()(1)),
    _Expr("arg_float", lambda: _ident()(1.5)),
    _Expr("arg_str", lambda: _ident()("s")),
    _Expr("arg_none", lambda: _ident()(None)),
    _Expr("arg_bool", lambda: _ident()(True)),
    _Expr("arg_bytes", lambda: _ident()(b"xy")),
    _Expr("arg_complex", lambda: _ident()(complex(1, 2))),
    # Wrapped non-callables and the two default naming forms ``delayed`` builds
    # from ``obj.__name__`` and from ``type(obj).__name__``.
    _Expr("wrap_int_named", lambda: delayed(3, name="three")),
    _Expr("wrap_int_default", lambda: delayed(3), deterministic=False),
    _Expr("wrap_str_pure", lambda: delayed("s", pure=True)),
    _Expr("wrap_obj_pure", lambda: delayed(Obj(3), pure=True)),
    _Expr("wrap_func_pure", lambda: delayed(inc, pure=True), computable=False),
    _Expr("call_default_name", lambda: delayed(inc, pure=True)(1)),
    _Expr(
        "wrap_list_traverse_false_default",
        lambda: delayed([1, 2], traverse=False),
        deterministic=False,
    ),
    _Expr("wrap_taskref", lambda: delayed(TaskRef("dn")), computable=False),
    _Expr("wrap_datanode", lambda: delayed(DataNode("dn", 5), name="dn")),
    # Wrapping a *traversed* non-callable container that holds a ``Delayed``:
    # ``unpack_collections`` returns a task rather than the object, so the wrap in
    # ``delayed()`` takes its second branch -- the generated
    # ``type(obj).__name__-<token>`` key, the rewrite of the container node's own
    # key to that name, and a graph merged from the dependencies the traversal
    # found. The dict form carries two dependencies, so it also pins the
    # several-dependency insertion order of that branch.
    _Expr("wrap_list_with_delayed", lambda: delayed([_leaf(), 1], pure=True)),
    _Expr(
        "wrap_dict_with_delayed",
        lambda: delayed({"k": _leaf(), "j": _other_leaf()}, pure=True),
    ),
    # list/tuple/set branch of ``unpack_collections``.
    _Expr("arg_list_literal", lambda: _ident()([1, 2, 3])),
    _Expr("arg_list_with_delayed", lambda: _ident()([_leaf(), 1])),
    _Expr("arg_tuple_literal", lambda: _ident()((1, 2))),
    _Expr("arg_tuple_with_delayed", lambda: _ident()((_leaf(), 1))),
    _Expr("arg_set_literal", lambda: _ident()({1, 2, 3})),
    _Expr("arg_set_with_delayed", lambda: _ident()({_leaf()})),
    # Empty containers: each is returned verbatim and appears as-is in the task.
    _Expr("arg_empty_list", lambda: _ident()([])),
    _Expr("arg_empty_tuple", lambda: _ident()(())),
    _Expr("arg_empty_set", lambda: _ident()(set())),
    _Expr("arg_empty_dict", lambda: _ident()({})),
    # dict branch, and the kwargs dict that ``call_function`` unpacks -- the path
    # an empty-kwargs short-circuit must keep equivalent.
    _Expr("arg_dict_literal", lambda: _ident()({"k": 1})),
    _Expr("arg_dict_delayed_value", lambda: _ident()({"k": _leaf()})),
    _Expr("arg_dict_delayed_key", lambda: _ident()({_leaf(): 1})),
    _Expr("kwargs_literal", lambda: _collect()(x=1)),
    _Expr("kwargs_with_delayed", lambda: _collect()(x=_leaf())),
    # slice branch.
    _Expr("arg_slice_literal", lambda: _ident()(slice(1, 5, 2))),
    _Expr("arg_slice_with_delayed", lambda: _ident()(slice(_leaf(), 5, None))),
    # dataclass branch.
    _Expr("arg_dataclass_literal", lambda: _ident()(Box(a=1, b="s"))),
    _Expr("arg_dataclass_with_delayed", lambda: _ident()(Box(a=_leaf(), b="s"))),
    # namedtuple branch.
    _Expr("arg_namedtuple_literal", lambda: _ident()(Point(x=1, y=2))),
    _Expr("arg_namedtuple_with_delayed", lambda: _ident()(Point(x=_leaf(), y=2))),
    # Iterator coercion.
    _Expr("arg_list_iterator", lambda: _ident()(iter([1, 2]))),
    # The layer-order lock: ``call_function`` tokenizes the *raw* iterator before
    # ``unpack_collections`` coerces it, so ``tokenize`` pickles the underlying
    # list -- including the ``Delayed`` it holds. A ``Delayed``'s pickled slot
    # state contains its ``HighLevelGraph``, whose layer *insertion order*
    # therefore feeds this key's token.
    _Expr("arg_list_iterator_with_delayed", lambda: _ident()(iter([_leaf(), 1]))),
    _Expr("arg_tuple_iterator", lambda: _ident()(iter((1, 2)))),
    _Expr("arg_set_iterator", lambda: _ident()(iter({1}))),
    # Nesting and duplication.
    _Expr("arg_nested_mixed", lambda: _ident()([{"k": (_leaf(), 1)}, [2, _leaf()]])),
    _Expr("arg_nested_list_depth3", lambda: _ident()([[[_leaf()]]])),
    _Expr("arg_duplicate_positional", _build_duplicate_positional),
    _Expr("arg_duplicate_in_list", _build_duplicate_in_list),
    _Expr("arg_two_dependencies", lambda: _ident()(_leaf(), _other_leaf())),
    # Both arms of ``DelayedLeaf.dask`` as a dependency: a plain object becomes a
    # ``DataNode``, a ``GraphNode`` is used as it stands. Plus a dependency
    # carrying a plain ``dict`` graph.
    # A ``DelayedLeaf`` wrapping a callable computes to the function object, whose
    # repr embeds its address, so the value-wrapping variant beside it is the one
    # that takes part in the result assertions.
    _Expr(
        "arg_delayed_leaf",
        lambda: _ident()(delayed(inc, name="inc-leaf", pure=True)),
        computable=False,
    ),
    _Expr("arg_delayed_value_leaf", lambda: _ident()(delayed(3, name="three-leaf"))),
    _Expr("arg_datanode_leaf", lambda: _ident()(delayed(DataNode("dn", 5), name="dn"))),
    _Expr("arg_dict_graph_delayed", lambda: _ident()(_dict_graph_delayed())),
    # A dependency keyed by a ``tuple`` of builtins: the accepting side of a key
    # guard's recursion, and the only corpus entry whose keys are not strings.
    _Expr("arg_tuple_key_delayed", lambda: _ident()(_tuple_key_delayed())),
    # Several dependencies, one per arm of the merge: an ordinary ``Delayed``
    # beside a subclass that no exact-class guard may shortcut (so the *whole*
    # call has to fall back), beside a plain-``dict``-backed dependency, beside a
    # ``DelayedLeaf``, beside a ``DelayedAttr``.
    _Expr("arg_several_deps_guard_mixed", lambda: _ident()(_leaf(), _sub_delayed())),
    _Expr(
        "arg_several_deps_dict_graph",
        lambda: _ident()(_leaf(), _dict_graph_delayed()),
    ),
    _Expr("arg_several_deps_with_leaf", _build_several_dependencies_with_leaf),
    _Expr("arg_several_deps_with_attr", _build_several_dependencies_with_attr),
    # A container holding a task-spec node but no dask collection: the container
    # may not be handed back unchanged, because the node references a key. Both
    # entries reference the key ``"dn"``, which no layer of their graph holds, so
    # computing them raises -- the failure is asserted by name in
    # ``_COMPUTE_FAILURES`` rather than passed over.
    _Expr(
        "arg_list_with_taskref", lambda: _ident()([TaskRef("dn"), 1]), computable=False
    ),
    _Expr(
        "arg_list_with_dependent_task",
        _build_list_with_dependent_task,
        computable=False,
    ),
    # Non-``Delayed`` dask collection, the branch ``unpack_collections`` routes
    # through ``collections_to_expr``. Its graph acquires a
    # ``finalize-hlgfinalizecompute-<hex>-<hex>`` layer whose two hexes are
    # ``uuid4().hex`` fallbacks from ``dask._expr``, and the same hex suffixes the
    # node keys inside that layer, so the *raw* canonical dict differs between two
    # builds of the same expression. Everything else about the graph is fixed, and
    # that is what ``volatile_tokens`` asserts: the golden holds the canonical
    # form with only the observedly-random tokens replaced by ``<hexN>``, so every
    # key, node kind, dependency and insertion order stays compared.
    # ``finalize()`` stores an ``HLGFinalizeCompute`` expression rather than a
    # graph container, which is the same situation one volatile token further on.
    _Expr(
        "arg_hand_rolled_collection",
        lambda: _ident()(_hand_rolled_collection()),
        canonical=False,
        volatile_tokens=True,
    ),
    _Expr(
        "finalize_collection",
        _build_finalize_collection,
        canonical=False,
        computable=False,
        volatile_tokens=True,
    ),
    # ``pure`` semantics, as the module-local ``tokenize`` wrapper decides them.
    _Expr("call_pure_true", lambda: delayed(inc, name="inc", pure=True)(1)),
    _Expr(
        "call_pure_false",
        lambda: delayed(inc, name="inc", pure=False)(1),
        deterministic=False,
    ),
    _Expr("call_global_delayed_pure", _build_global_delayed_pure),
    # Explicit naming: ``dask_key_name`` overrides the generated call key.
    _Expr(
        "call_dask_key_name",
        lambda: delayed(inc, name="inc", pure=True)(
            1, dask_key_name="explicit-call-key"
        ),
    ),
    # ``nout``: validated by ``delayed``, stored as ``_length``, and passed to
    # the call node as ``length=nout``.
    _Expr("nout_none", lambda: delayed(pair, name="pair", pure=True)()),
    _Expr("nout_zero", lambda: delayed(nothing, name="nothing", pure=True, nout=0)()),
    _Expr("nout_one", lambda: delayed(single, name="single", pure=True, nout=1)()),
    _Expr("nout_one_element", _build_nout_one_element),
    _Expr("nout_two", lambda: delayed(pair, name="pair", pure=True, nout=2)()),
    _Expr("nout_two_unpacked_first", _build_nout_two_first),
    _Expr("nout_two_unpacked_second", _build_nout_two_second),
    _Expr(
        "nout_two_getitem_one",
        lambda: delayed(pair, name="pair", pure=True, nout=2)()[1],
    ),
    # ``traverse=False``: the object is quoted and no dependency is collected, so
    # the ``Delayed`` inside survives into the computed value as an object.
    _Expr(
        "traverse_false_with_delayed",
        lambda: delayed([_leaf(), 1], traverse=False, name="quoted"),
    ),
    # Lazy attribute access through ``Delayed.__getattr__`` and
    # ``DelayedAttr.dask``. These layers hold the legacy tuple task ``(getattr,
    # key, attr)``, which the graph-shape contract freezes; the canonicaliser
    # reports it as ``"legacy-tuple"``.
    _Expr("attr_v", lambda: _obj().v),
    _Expr("attr_of_attr", lambda: _obj().inner.w),
    _Expr("attr_items_getitem", lambda: _obj().items[1]),
    # Method calls through ``DelayedAttr.__call__``, which does not forward
    # ``pure``, so a method call is impure unless ``pure=True`` is passed in the
    # call kwargs, where ``call_function`` pops it.
    _Expr("method_pure", lambda: _obj().meth(2, pure=True)),
    _Expr("method_impure", lambda: _obj().meth(2), deterministic=False),
    # Operators, bound through ``Delayed._get_binary_operator`` and ``right``.
    _Expr("op_add", lambda: _leaf() + _other_leaf()),
    _Expr("op_reflected_add", lambda: 1 + _leaf()),
    _Expr("op_neg", lambda: -_leaf()),
    _Expr("op_lt", lambda: _leaf() < _other_leaf()),
    _Expr("op_getitem", lambda: delayed(listof, name="listof", pure=True)(1, 2, 3)[1]),
    # ``Delayed.__call__`` on a delayed callable.
    _Expr("delayed_call_pure", lambda: _build_delayed_call(True)),
    _Expr(
        "delayed_call_impure", lambda: _build_delayed_call(None), deterministic=False
    ),
    # Guard cases. Each dependency fails an exact-class-identity or
    # exact-graph-type guard, so a merge shortcut on the construction path must
    # fall back to ``HighLevelGraph.from_collections`` for the whole call.
    _Expr("guard_subclass", lambda: _ident()(_sub_delayed())),
    _Expr("guard_subclass_dask_layers", lambda: _ident()(_sub_delayed_dask_layers())),
    _Expr("guard_subclass_layer_dict", lambda: _ident()(_sub_delayed_layer_dict())),
    _Expr(
        "guard_mapping_proxy_graph",
        lambda: _ident()(_mapping_proxy_delayed()),
        picklable=False,
    ),
)

#: Why each corpus entry opts out of an assertion group. Kept in step with
#: ``CORPUS`` by ``test_flag_exclusions_are_documented``, which fails if an entry
#: carries a ``False`` flag without an entry here or vice versa.
_EXCLUSION_REASONS: dict[str, str] = {
    "wrap_func_pure": (
        "computable=False: the wrapped value is the function object itself, and its repr "
        "embeds its memory address, so no stable golden repr exists"
    ),
    "wrap_taskref": (
        "computable=False: a TaskRef is a pointer to a key, not a runnable node, so "
        "computing it raises KeyError('dn') -- asserted verbatim in _COMPUTE_FAILURES"
    ),
    "arg_delayed_leaf": (
        "computable=False: the dependency is a DelayedLeaf wrapping a function, so the "
        "result holds the function object, whose repr embeds its memory address; "
        "arg_delayed_value_leaf covers the same DataNode arm with a stable result"
    ),
    "arg_hand_rolled_collection": (
        "canonical=False with volatile_tokens=True: routing a non-Delayed collection through "
        "unpack_collections adds a finalize-hlgfinalizecompute-<hex>-<hex> layer whose hexes come "
        "from uuid4().hex (dask/_expr.py) and suffix the node keys inside it, so the raw "
        "canonical dict is not reproducible -- but the graph is still asserted in full, through "
        "the double-build normalisation of _stable_graph, alongside the deterministic output key "
        "and the computed result"
    ),
    "finalize_collection": (
        "canonical=False with volatile_tokens=True, and computable=False: finalize() returns a "
        "Delayed whose graph is an HLGFinalizeCompute expression, which carries one uuid4().hex "
        "token, so the raw canonical dict is not reproducible and the graph is asserted through "
        "_stable_graph instead; computing it raises AttributeError('HLGFinalizeCompute' object "
        "has no attribute 'copy') at this commit, which is asserted verbatim in "
        "_COMPUTE_FAILURES rather than passed over"
    ),
    "arg_list_with_taskref": (
        "computable=False: the list holds a bare TaskRef to the key 'dn', which no layer of the "
        "graph provides, so computing raises ValueError('Missing dependency dn for dependents "
        "...') -- asserted verbatim in _COMPUTE_FAILURES"
    ),
    "arg_list_with_dependent_task": (
        "computable=False: the list holds a Task referencing the key 'dn', which no layer of the "
        "graph provides, so computing raises ValueError('Missing dependency dn for dependents "
        "...') -- asserted verbatim in _COMPUTE_FAILURES"
    ),
    "guard_mapping_proxy_graph": (
        "picklable=False: the dependency's graph is a types.MappingProxyType, which pickle "
        "refuses (TypeError: cannot pickle 'mappingproxy' object)"
    ),
}

#: The corpus entries that cannot be computed, with the exception each one
#: raises on the pre-refactor module -- exact type and exact message, measured,
#: never a fragment. An entry lands here rather than simply carrying
#: ``computable=False``: a path whose result *is* an exception has that exception
#: asserted by ``test_non_runnable_entries_fail_exactly_as_they_did``, so a
#: change in how it fails is caught as readily as a change in a value. Two of the
#: messages embed the expression's own key, which is deterministic, so they are
#: recorded in full as well.
_COMPUTE_FAILURES: dict[str, tuple[type[BaseException], str]] = {
    "wrap_taskref": (KeyError, "'dn'"),
    "finalize_collection": (
        AttributeError,
        "'HLGFinalizeCompute' object has no attribute 'copy'\n\nThis often means "
        "that you are attempting to use an unsupported API function..",
    ),
    "arg_list_with_taskref": (
        ValueError,
        "Missing dependency dn for dependents "
        "{'ident-c79c565af7915db3336df040927eba3b'}",
    ),
    "arg_list_with_dependent_task": (
        ValueError,
        "Missing dependency dn for dependents "
        "{'ident-853f00e7b63921c6aa234b62a83f3f37'}",
    ),
}

#: Entries that deliberately keep ``delayed``'s default naming, so the
#: ``obj.__name__-<token>`` and ``type(obj).__name__-<token>`` key forms are
#: locked under the pinned hasher rather than derived from an explicit ``name=``.
_DEFAULT_NAMING: tuple[str, ...] = (
    "wrap_int_default",
    "wrap_str_pure",
    "wrap_obj_pure",
    "wrap_func_pure",
    "call_default_name",
    "wrap_list_traverse_false_default",
)

#: The guard/fallback entries, named so their presence is asserted rather than
#: assumed. They are the only corpus coverage of the generic
#: ``HighLevelGraph.from_collections`` path once a merge shortcut exists.
_GUARD_ENTRIES: tuple[str, ...] = (
    "guard_subclass",
    "guard_subclass_dask_layers",
    "guard_subclass_layer_dict",
    "guard_mapping_proxy_graph",
    "arg_several_deps_guard_mixed",
)

#: Every branch of ``unpack_collections`` mapped to one corpus entry that
#: exercises it, asserted by ``test_every_unpack_collections_branch_is_covered``.
#: The line numbers inside the branch labels below locate those branches in
#: ``dask/delayed.py`` at base commit ``c9d1df34``, the revision this corpus was
#: captured against, and are not current line numbers.
_BRANCH_COVERAGE: dict[str, str] = {
    "Delayed short-circuit (:166-172)": "arg_list_with_delayed",
    "non-Delayed dask collection (:177-195)": "arg_hand_rolled_collection",
    "list iterator coercion (:197-198)": "arg_list_iterator",
    "tuple iterator coercion (:199-200)": "arg_tuple_iterator",
    "set iterator coercion (:201-202)": "arg_set_iterator",
    "list/tuple/set literal short-circuit (:214-215)": "arg_list_literal",
    "list holding a dependent node, no collection (:206-221)": "arg_list_with_taskref",
    "list holding a collection (:206-217)": "arg_list_with_delayed",
    "tuple output-type restoration (:219-220)": "arg_tuple_with_delayed",
    "set holding a collection (:206-221)": "arg_set_with_delayed",
    "dict literal short-circuit (:232-233)": "arg_dict_literal",
    "dict value holding a collection (:223-236)": "arg_dict_delayed_value",
    "dict key holding a collection (:224-231)": "arg_dict_delayed_key",
    "kwargs dict (call_function:818)": "kwargs_with_delayed",
    "slice short-circuit (:242-243)": "arg_slice_literal",
    "slice holding a collection (:238-247)": "arg_slice_with_delayed",
    "dataclass short-circuit (:258-259)": "arg_dataclass_literal",
    "dataclass holding a collection (:249-281)": "arg_dataclass_with_delayed",
    "namedtuple short-circuit (:287-288)": "arg_namedtuple_literal",
    "namedtuple holding a collection (:283-291)": "arg_namedtuple_with_delayed",
    "fall-through, object returned unchanged (:293)": "arg_int",
}


class _TokenCause(NamedTuple):
    """One reason a deterministic key token is not comparable verbatim everywhere.

    Attributes:
        reproduces_capture: True when this environment produces the very token
            the golden recorded, so the character-for-character comparison is
            valid here and is still made.
        reason: What decides the token and why this environment does or does not
            reproduce it. Used verbatim as a skip reason, so it has to read on
            its own in a ``-rs`` summary.

    """

    reproduces_capture: bool
    reason: str


#: The two causes, each a property of the environment rather than of the module
#: under test. Neither is reachable by ``_pin_tokenization``: the first is fixed
#: before any fixture runs, the second by the interpreter the suite runs on.
_TOKEN_CAUSES: dict[str, _TokenCause] = {
    "import-time-hasher": _TokenCause(
        _HASHER_REPRODUCES_CAPTURE,
        f"dask.hashing selected {_AMBIENT_HASHER} in this process and the golden was "
        f"captured under {_CAPTURE_HASHER}; Delayed binds its operator methods to "
        "delayed(op, pure=True) at class-binding time, so the leaf token those keys "
        "derive from is fixed when dask.delayed is imported, before any fixture can "
        "pin the hasher",
    ),
    "interpreter-dataclass": _TokenCause(
        _INTERPRETER_REPRODUCES_CAPTURE,
        f"this is CPython {sys.version_info[0]}.{sys.version_info[1]} and the golden "
        f"was captured on {_CAPTURE_PYTHON[0]}.{_CAPTURE_PYTHON[1]}; dask tokenizes a "
        "dataclass instance from every attribute __dataclass_params__ declares, and "
        "the interpreter decides that set (six names on 3.10, ten on 3.14)",
    ),
}

#: Corpus entries whose *own* key token is decided by the environment, mapped to
#: the cause in ``_TOKEN_CAUSES``.
#:
#: Ten of them reach ``call_function`` through a class-bound operator --
#: ``Delayed._get_binary_operator`` evaluates ``delayed(op, pure=True)`` while the
#: class body is being built, so that leaf's key is tokenized under whichever hash
#: library ``dask.hashing`` had selected when ``dask.delayed`` was imported. The
#: leaf never enters the graph; it is passed as ``func_token`` and tokenized into
#: the call key
#: (``dask/delayed.py``), which is why the import-time state reaches exactly one
#: token and no other. ``op_getitem``, ``attr_items_getitem`` and the four
#: ``nout_*`` element entries are on that list because indexing and unpacking a
#: ``Delayed`` route through the class-bound ``operator.getitem`` too.
#:
#: The other two tokenize a dataclass instance, whose normalisation reads the
#: interpreter's own ``__dataclass_params__`` slot set.
#:
#: The masking these entries receive is one token wide, and that it is enough was
#: established by observation rather than by argument: built under the two
#: extremes this project supports -- CPython 3.14.6 with ``xxhash`` ambient, and
#: CPython 3.10.20 with SHA-1 ambient, ``mmh3`` ambient checked as a third -- each
#: of these twelve graphs differs in exactly one 32-hex token, its own output
#: key's, and replacing that one token makes the canonical dicts equal field for
#: field. Every other token in them, the leaf keys and the ``getattr-`` tokens
#: included, is produced under the fixture's pin and is compared verbatim.
_ENVIRONMENT_DEPENDENT_KEY_TOKEN: dict[str, str] = {
    "op_add": "import-time-hasher",
    "op_reflected_add": "import-time-hasher",
    "op_neg": "import-time-hasher",
    "op_lt": "import-time-hasher",
    "op_getitem": "import-time-hasher",
    "attr_items_getitem": "import-time-hasher",
    "nout_one_element": "import-time-hasher",
    "nout_two_unpacked_first": "import-time-hasher",
    "nout_two_unpacked_second": "import-time-hasher",
    "nout_two_getitem_one": "import-time-hasher",
    "arg_dataclass_literal": "interpreter-dataclass",
    "arg_dataclass_with_delayed": "interpreter-dataclass",
}

#: What an environment-dependent token is replaced by. It contains characters no
#: generated key can hold, so a masked form can never be mistaken for a real one,
#: and it is distinct from ``_stable_graph``'s ``<hex0>`` placeholders, which
#: stand for inherently *random* tokens rather than environment-dependent ones.
_ENV_TOKEN_PLACEHOLDER = "<env-token>"


def _by_name(name: str) -> _Expr:
    for entry in CORPUS:
        if entry.name == name:
            return entry
    raise KeyError(f"no corpus entry named {name!r}")


#: Marker comments delimiting the generated golden block. ``write_golden()``
#: locates them by whole-line equality, which is why the two assignments here --
#: whose lines merely *contain* the marker text -- are never mistaken for them.
_GOLDEN_BEGIN = (
    "# --- BEGIN GOLDEN (generated by write_golden(); do not edit by hand) ---"
)
_GOLDEN_END = "# --- END GOLDEN ---"


# --- BEGIN GOLDEN (generated by write_golden(); do not edit by hand) ---
GOLDEN: dict[str, dict[str, Any]] = {
    "arg_bool": {
        "graph": {
            "dask_keys": ["ident-479de180fadf35bef9b14377a02a2e3f"],
            "dask_layers": ["ident-479de180fadf35bef9b14377a02a2e3f"],
            "dependencies": [["ident-479de180fadf35bef9b14377a02a2e3f", []]],
            "dependency_order": ["ident-479de180fadf35bef9b14377a02a2e3f"],
            "key": "ident-479de180fadf35bef9b14377a02a2e3f",
            "layer_order": ["ident-479de180fadf35bef9b14377a02a2e3f"],
            "layers": [
                [
                    "ident-479de180fadf35bef9b14377a02a2e3f",
                    "MaterializedLayer",
                    [["ident-479de180fadf35bef9b14377a02a2e3f", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-479de180fadf35bef9b14377a02a2e3f",
        "result_repr": "(True,)",
        "stable_graph": None,
    },
    "arg_bytes": {
        "graph": {
            "dask_keys": ["ident-069e5d002edfbdd6b4fe85609c61f762"],
            "dask_layers": ["ident-069e5d002edfbdd6b4fe85609c61f762"],
            "dependencies": [["ident-069e5d002edfbdd6b4fe85609c61f762", []]],
            "dependency_order": ["ident-069e5d002edfbdd6b4fe85609c61f762"],
            "key": "ident-069e5d002edfbdd6b4fe85609c61f762",
            "layer_order": ["ident-069e5d002edfbdd6b4fe85609c61f762"],
            "layers": [
                [
                    "ident-069e5d002edfbdd6b4fe85609c61f762",
                    "MaterializedLayer",
                    [["ident-069e5d002edfbdd6b4fe85609c61f762", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-069e5d002edfbdd6b4fe85609c61f762",
        "result_repr": "(b'xy',)",
        "stable_graph": None,
    },
    "arg_complex": {
        "graph": {
            "dask_keys": ["ident-58b2c8b88ad440989db630826bb487b3"],
            "dask_layers": ["ident-58b2c8b88ad440989db630826bb487b3"],
            "dependencies": [["ident-58b2c8b88ad440989db630826bb487b3", []]],
            "dependency_order": ["ident-58b2c8b88ad440989db630826bb487b3"],
            "key": "ident-58b2c8b88ad440989db630826bb487b3",
            "layer_order": ["ident-58b2c8b88ad440989db630826bb487b3"],
            "layers": [
                [
                    "ident-58b2c8b88ad440989db630826bb487b3",
                    "MaterializedLayer",
                    [["ident-58b2c8b88ad440989db630826bb487b3", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-58b2c8b88ad440989db630826bb487b3",
        "result_repr": "((1+2j),)",
        "stable_graph": None,
    },
    "arg_dataclass_literal": {
        "graph": {
            "dask_keys": ["ident-2b8023b48c36384470ab10ebcc9619c3"],
            "dask_layers": ["ident-2b8023b48c36384470ab10ebcc9619c3"],
            "dependencies": [["ident-2b8023b48c36384470ab10ebcc9619c3", []]],
            "dependency_order": ["ident-2b8023b48c36384470ab10ebcc9619c3"],
            "key": "ident-2b8023b48c36384470ab10ebcc9619c3",
            "layer_order": ["ident-2b8023b48c36384470ab10ebcc9619c3"],
            "layers": [
                [
                    "ident-2b8023b48c36384470ab10ebcc9619c3",
                    "MaterializedLayer",
                    [["ident-2b8023b48c36384470ab10ebcc9619c3", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-2b8023b48c36384470ab10ebcc9619c3",
        "result_repr": "(Box(a=1, b='s'),)",
        "stable_graph": None,
    },
    "arg_dataclass_with_delayed": {
        "graph": {
            "dask_keys": ["ident-0e803bf67da18347a13dd41031ff7a89"],
            "dask_layers": ["ident-0e803bf67da18347a13dd41031ff7a89"],
            "dependencies": [
                [
                    "ident-0e803bf67da18347a13dd41031ff7a89",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-0e803bf67da18347a13dd41031ff7a89",
            ],
            "key": "ident-0e803bf67da18347a13dd41031ff7a89",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-0e803bf67da18347a13dd41031ff7a89",
            ],
            "layers": [
                [
                    "ident-0e803bf67da18347a13dd41031ff7a89",
                    "MaterializedLayer",
                    [
                        [
                            "ident-0e803bf67da18347a13dd41031ff7a89",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-0e803bf67da18347a13dd41031ff7a89",
        "result_repr": "(Box(a=2, b='s'),)",
        "stable_graph": None,
    },
    "arg_datanode_leaf": {
        "graph": {
            "dask_keys": ["ident-35c1fa32c0139a469f5604663da3d9f4"],
            "dask_layers": ["ident-35c1fa32c0139a469f5604663da3d9f4"],
            "dependencies": [
                ["dn", []],
                ["ident-35c1fa32c0139a469f5604663da3d9f4", ["dn"]],
            ],
            "dependency_order": ["dn", "ident-35c1fa32c0139a469f5604663da3d9f4"],
            "key": "ident-35c1fa32c0139a469f5604663da3d9f4",
            "layer_order": ["dn", "ident-35c1fa32c0139a469f5604663da3d9f4"],
            "layers": [
                ["dn", "MaterializedLayer", [["dn", "DataNode", [], None]]],
                [
                    "ident-35c1fa32c0139a469f5604663da3d9f4",
                    "MaterializedLayer",
                    [
                        [
                            "ident-35c1fa32c0139a469f5604663da3d9f4",
                            "Task",
                            ["dn"],
                            "ident",
                        ]
                    ],
                ],
            ],
        },
        "key": "ident-35c1fa32c0139a469f5604663da3d9f4",
        "result_repr": "(5,)",
        "stable_graph": None,
    },
    "arg_delayed_leaf": {
        "graph": {
            "dask_keys": ["ident-3fa65e5e2361d20664e44e110b0ee8d8"],
            "dask_layers": ["ident-3fa65e5e2361d20664e44e110b0ee8d8"],
            "dependencies": [
                ["ident-3fa65e5e2361d20664e44e110b0ee8d8", ["inc-leaf"]],
                ["inc-leaf", []],
            ],
            "dependency_order": ["inc-leaf", "ident-3fa65e5e2361d20664e44e110b0ee8d8"],
            "key": "ident-3fa65e5e2361d20664e44e110b0ee8d8",
            "layer_order": ["inc-leaf", "ident-3fa65e5e2361d20664e44e110b0ee8d8"],
            "layers": [
                [
                    "ident-3fa65e5e2361d20664e44e110b0ee8d8",
                    "MaterializedLayer",
                    [
                        [
                            "ident-3fa65e5e2361d20664e44e110b0ee8d8",
                            "Task",
                            ["inc-leaf"],
                            "ident",
                        ]
                    ],
                ],
                ["inc-leaf", "MaterializedLayer", [["inc-leaf", "DataNode", [], None]]],
            ],
        },
        "key": "ident-3fa65e5e2361d20664e44e110b0ee8d8",
        "result_repr": None,
        "stable_graph": None,
    },
    "arg_delayed_value_leaf": {
        "graph": {
            "dask_keys": ["ident-1598b4f5c1e157d4855e0da78b9743db"],
            "dask_layers": ["ident-1598b4f5c1e157d4855e0da78b9743db"],
            "dependencies": [
                ["ident-1598b4f5c1e157d4855e0da78b9743db", ["three-leaf"]],
                ["three-leaf", []],
            ],
            "dependency_order": [
                "three-leaf",
                "ident-1598b4f5c1e157d4855e0da78b9743db",
            ],
            "key": "ident-1598b4f5c1e157d4855e0da78b9743db",
            "layer_order": ["three-leaf", "ident-1598b4f5c1e157d4855e0da78b9743db"],
            "layers": [
                [
                    "ident-1598b4f5c1e157d4855e0da78b9743db",
                    "MaterializedLayer",
                    [
                        [
                            "ident-1598b4f5c1e157d4855e0da78b9743db",
                            "Task",
                            ["three-leaf"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "three-leaf",
                    "MaterializedLayer",
                    [["three-leaf", "DataNode", [], None]],
                ],
            ],
        },
        "key": "ident-1598b4f5c1e157d4855e0da78b9743db",
        "result_repr": "(3,)",
        "stable_graph": None,
    },
    "arg_dict_delayed_key": {
        "graph": {
            "dask_keys": ["ident-3d928f19640b869525079dea6b686d4a"],
            "dask_layers": ["ident-3d928f19640b869525079dea6b686d4a"],
            "dependencies": [
                [
                    "ident-3d928f19640b869525079dea6b686d4a",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-3d928f19640b869525079dea6b686d4a",
            ],
            "key": "ident-3d928f19640b869525079dea6b686d4a",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-3d928f19640b869525079dea6b686d4a",
            ],
            "layers": [
                [
                    "ident-3d928f19640b869525079dea6b686d4a",
                    "MaterializedLayer",
                    [
                        [
                            "ident-3d928f19640b869525079dea6b686d4a",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-3d928f19640b869525079dea6b686d4a",
        "result_repr": "({2: 1},)",
        "stable_graph": None,
    },
    "arg_dict_delayed_value": {
        "graph": {
            "dask_keys": ["ident-b8ff11051f6d686cfb732463bbdc2e72"],
            "dask_layers": ["ident-b8ff11051f6d686cfb732463bbdc2e72"],
            "dependencies": [
                [
                    "ident-b8ff11051f6d686cfb732463bbdc2e72",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-b8ff11051f6d686cfb732463bbdc2e72",
            ],
            "key": "ident-b8ff11051f6d686cfb732463bbdc2e72",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-b8ff11051f6d686cfb732463bbdc2e72",
            ],
            "layers": [
                [
                    "ident-b8ff11051f6d686cfb732463bbdc2e72",
                    "MaterializedLayer",
                    [
                        [
                            "ident-b8ff11051f6d686cfb732463bbdc2e72",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-b8ff11051f6d686cfb732463bbdc2e72",
        "result_repr": "({'k': 2},)",
        "stable_graph": None,
    },
    "arg_dict_graph_delayed": {
        "graph": {
            "dask_keys": ["ident-df5af5572037466a0e5bd86b9c97fa91"],
            "dask_layers": ["ident-df5af5572037466a0e5bd86b9c97fa91"],
            "dependencies": [
                ["dgkey", []],
                ["ident-df5af5572037466a0e5bd86b9c97fa91", ["dgkey"]],
            ],
            "dependency_order": ["ident-df5af5572037466a0e5bd86b9c97fa91", "dgkey"],
            "key": "ident-df5af5572037466a0e5bd86b9c97fa91",
            "layer_order": ["ident-df5af5572037466a0e5bd86b9c97fa91", "dgkey"],
            "layers": [
                ["dgkey", "MaterializedLayer", [["dgkey", "DataNode", [], None]]],
                [
                    "ident-df5af5572037466a0e5bd86b9c97fa91",
                    "MaterializedLayer",
                    [
                        [
                            "ident-df5af5572037466a0e5bd86b9c97fa91",
                            "Task",
                            ["dgkey"],
                            "ident",
                        ]
                    ],
                ],
            ],
        },
        "key": "ident-df5af5572037466a0e5bd86b9c97fa91",
        "result_repr": "(5,)",
        "stable_graph": None,
    },
    "arg_dict_literal": {
        "graph": {
            "dask_keys": ["ident-005d9b8feee2d6504f377842be0df79d"],
            "dask_layers": ["ident-005d9b8feee2d6504f377842be0df79d"],
            "dependencies": [["ident-005d9b8feee2d6504f377842be0df79d", []]],
            "dependency_order": ["ident-005d9b8feee2d6504f377842be0df79d"],
            "key": "ident-005d9b8feee2d6504f377842be0df79d",
            "layer_order": ["ident-005d9b8feee2d6504f377842be0df79d"],
            "layers": [
                [
                    "ident-005d9b8feee2d6504f377842be0df79d",
                    "MaterializedLayer",
                    [["ident-005d9b8feee2d6504f377842be0df79d", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-005d9b8feee2d6504f377842be0df79d",
        "result_repr": "({'k': 1},)",
        "stable_graph": None,
    },
    "arg_duplicate_in_list": {
        "graph": {
            "dask_keys": ["ident-5b9bf7a118115b19e651c954888eca1a"],
            "dask_layers": ["ident-5b9bf7a118115b19e651c954888eca1a"],
            "dependencies": [
                [
                    "ident-5b9bf7a118115b19e651c954888eca1a",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "ident-5b9bf7a118115b19e651c954888eca1a",
                "inc-5852f565112f1604bc52264ea5e287dc",
            ],
            "key": "ident-5b9bf7a118115b19e651c954888eca1a",
            "layer_order": [
                "ident-5b9bf7a118115b19e651c954888eca1a",
                "inc-5852f565112f1604bc52264ea5e287dc",
            ],
            "layers": [
                [
                    "ident-5b9bf7a118115b19e651c954888eca1a",
                    "MaterializedLayer",
                    [
                        [
                            "ident-5b9bf7a118115b19e651c954888eca1a",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-5b9bf7a118115b19e651c954888eca1a",
        "result_repr": "([2, 2, 2],)",
        "stable_graph": None,
    },
    "arg_duplicate_positional": {
        "graph": {
            "dask_keys": ["ident-2eb904727fd2af04e11a38ddaab9240e"],
            "dask_layers": ["ident-2eb904727fd2af04e11a38ddaab9240e"],
            "dependencies": [
                [
                    "ident-2eb904727fd2af04e11a38ddaab9240e",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "ident-2eb904727fd2af04e11a38ddaab9240e",
                "inc-5852f565112f1604bc52264ea5e287dc",
            ],
            "key": "ident-2eb904727fd2af04e11a38ddaab9240e",
            "layer_order": [
                "ident-2eb904727fd2af04e11a38ddaab9240e",
                "inc-5852f565112f1604bc52264ea5e287dc",
            ],
            "layers": [
                [
                    "ident-2eb904727fd2af04e11a38ddaab9240e",
                    "MaterializedLayer",
                    [
                        [
                            "ident-2eb904727fd2af04e11a38ddaab9240e",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-2eb904727fd2af04e11a38ddaab9240e",
        "result_repr": "(2, 2)",
        "stable_graph": None,
    },
    "arg_empty_dict": {
        "graph": {
            "dask_keys": ["ident-281fb13c7ae98bf73409f2defeecdf51"],
            "dask_layers": ["ident-281fb13c7ae98bf73409f2defeecdf51"],
            "dependencies": [["ident-281fb13c7ae98bf73409f2defeecdf51", []]],
            "dependency_order": ["ident-281fb13c7ae98bf73409f2defeecdf51"],
            "key": "ident-281fb13c7ae98bf73409f2defeecdf51",
            "layer_order": ["ident-281fb13c7ae98bf73409f2defeecdf51"],
            "layers": [
                [
                    "ident-281fb13c7ae98bf73409f2defeecdf51",
                    "MaterializedLayer",
                    [["ident-281fb13c7ae98bf73409f2defeecdf51", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-281fb13c7ae98bf73409f2defeecdf51",
        "result_repr": "({},)",
        "stable_graph": None,
    },
    "arg_empty_list": {
        "graph": {
            "dask_keys": ["ident-a0c2f3cfbfca5c11a6d69c20cce9c3b8"],
            "dask_layers": ["ident-a0c2f3cfbfca5c11a6d69c20cce9c3b8"],
            "dependencies": [["ident-a0c2f3cfbfca5c11a6d69c20cce9c3b8", []]],
            "dependency_order": ["ident-a0c2f3cfbfca5c11a6d69c20cce9c3b8"],
            "key": "ident-a0c2f3cfbfca5c11a6d69c20cce9c3b8",
            "layer_order": ["ident-a0c2f3cfbfca5c11a6d69c20cce9c3b8"],
            "layers": [
                [
                    "ident-a0c2f3cfbfca5c11a6d69c20cce9c3b8",
                    "MaterializedLayer",
                    [["ident-a0c2f3cfbfca5c11a6d69c20cce9c3b8", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-a0c2f3cfbfca5c11a6d69c20cce9c3b8",
        "result_repr": "([],)",
        "stable_graph": None,
    },
    "arg_empty_set": {
        "graph": {
            "dask_keys": ["ident-7df79d31aa263e6ef7f411886fa8e6c5"],
            "dask_layers": ["ident-7df79d31aa263e6ef7f411886fa8e6c5"],
            "dependencies": [["ident-7df79d31aa263e6ef7f411886fa8e6c5", []]],
            "dependency_order": ["ident-7df79d31aa263e6ef7f411886fa8e6c5"],
            "key": "ident-7df79d31aa263e6ef7f411886fa8e6c5",
            "layer_order": ["ident-7df79d31aa263e6ef7f411886fa8e6c5"],
            "layers": [
                [
                    "ident-7df79d31aa263e6ef7f411886fa8e6c5",
                    "MaterializedLayer",
                    [["ident-7df79d31aa263e6ef7f411886fa8e6c5", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-7df79d31aa263e6ef7f411886fa8e6c5",
        "result_repr": "(set(),)",
        "stable_graph": None,
    },
    "arg_empty_tuple": {
        "graph": {
            "dask_keys": ["ident-51d562c44b277dfd3c129022d175dab7"],
            "dask_layers": ["ident-51d562c44b277dfd3c129022d175dab7"],
            "dependencies": [["ident-51d562c44b277dfd3c129022d175dab7", []]],
            "dependency_order": ["ident-51d562c44b277dfd3c129022d175dab7"],
            "key": "ident-51d562c44b277dfd3c129022d175dab7",
            "layer_order": ["ident-51d562c44b277dfd3c129022d175dab7"],
            "layers": [
                [
                    "ident-51d562c44b277dfd3c129022d175dab7",
                    "MaterializedLayer",
                    [["ident-51d562c44b277dfd3c129022d175dab7", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-51d562c44b277dfd3c129022d175dab7",
        "result_repr": "((),)",
        "stable_graph": None,
    },
    "arg_float": {
        "graph": {
            "dask_keys": ["ident-e998c7aa085bf9349fa29aa9ece6f5f5"],
            "dask_layers": ["ident-e998c7aa085bf9349fa29aa9ece6f5f5"],
            "dependencies": [["ident-e998c7aa085bf9349fa29aa9ece6f5f5", []]],
            "dependency_order": ["ident-e998c7aa085bf9349fa29aa9ece6f5f5"],
            "key": "ident-e998c7aa085bf9349fa29aa9ece6f5f5",
            "layer_order": ["ident-e998c7aa085bf9349fa29aa9ece6f5f5"],
            "layers": [
                [
                    "ident-e998c7aa085bf9349fa29aa9ece6f5f5",
                    "MaterializedLayer",
                    [["ident-e998c7aa085bf9349fa29aa9ece6f5f5", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-e998c7aa085bf9349fa29aa9ece6f5f5",
        "result_repr": "(1.5,)",
        "stable_graph": None,
    },
    "arg_hand_rolled_collection": {
        "graph": None,
        "key": "ident-99645681c6f883e003c9b0070e583172",
        "result_repr": "((1, 2),)",
        "stable_graph": {
            "dask_keys": ["ident-99645681c6f883e003c9b0070e583172"],
            "dask_layers": ["ident-99645681c6f883e003c9b0070e583172"],
            "dependencies": [
                ["finalize-hlgfinalizecompute-<hex0>-<hex1>", []],
                [
                    "ident-99645681c6f883e003c9b0070e583172",
                    ["finalize-hlgfinalizecompute-<hex0>-<hex1>"],
                ],
            ],
            "dependency_order": [
                "ident-99645681c6f883e003c9b0070e583172",
                "finalize-hlgfinalizecompute-<hex0>-<hex1>",
            ],
            "key": "ident-99645681c6f883e003c9b0070e583172",
            "layer_order": [
                "ident-99645681c6f883e003c9b0070e583172",
                "finalize-hlgfinalizecompute-<hex0>-<hex1>",
            ],
            "layers": [
                [
                    "finalize-hlgfinalizecompute-<hex0>-<hex1>",
                    "MaterializedLayer",
                    [
                        [
                            "finalize-hlgfinalizecompute-<hex0>",
                            "Task",
                            ["ta", "tb"],
                            "tuple",
                        ],
                        [
                            "finalize-hlgfinalizecompute-<hex0>-<hex1>",
                            "Task",
                            ["ta-<hex1>", "tb-<hex1>"],
                            "_identity",
                        ],
                        ["ta", "DataNode", [], None],
                        ["ta-<hex1>", "Task", [], "_identity"],
                        ["tb", "DataNode", [], None],
                        ["tb-<hex1>", "Task", [], "_identity"],
                    ],
                ],
                [
                    "ident-99645681c6f883e003c9b0070e583172",
                    "MaterializedLayer",
                    [
                        [
                            "ident-99645681c6f883e003c9b0070e583172",
                            "Task",
                            ["finalize-hlgfinalizecompute-<hex0>-<hex1>"],
                            "ident",
                        ]
                    ],
                ],
            ],
        },
    },
    "arg_int": {
        "graph": {
            "dask_keys": ["ident-2c2468fc170d8d44e486152b85005245"],
            "dask_layers": ["ident-2c2468fc170d8d44e486152b85005245"],
            "dependencies": [["ident-2c2468fc170d8d44e486152b85005245", []]],
            "dependency_order": ["ident-2c2468fc170d8d44e486152b85005245"],
            "key": "ident-2c2468fc170d8d44e486152b85005245",
            "layer_order": ["ident-2c2468fc170d8d44e486152b85005245"],
            "layers": [
                [
                    "ident-2c2468fc170d8d44e486152b85005245",
                    "MaterializedLayer",
                    [["ident-2c2468fc170d8d44e486152b85005245", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-2c2468fc170d8d44e486152b85005245",
        "result_repr": "(1,)",
        "stable_graph": None,
    },
    "arg_list_iterator": {
        "graph": {
            "dask_keys": ["ident-36db7f0563b8b9aeaf638d565fa1f8c4"],
            "dask_layers": ["ident-36db7f0563b8b9aeaf638d565fa1f8c4"],
            "dependencies": [["ident-36db7f0563b8b9aeaf638d565fa1f8c4", []]],
            "dependency_order": ["ident-36db7f0563b8b9aeaf638d565fa1f8c4"],
            "key": "ident-36db7f0563b8b9aeaf638d565fa1f8c4",
            "layer_order": ["ident-36db7f0563b8b9aeaf638d565fa1f8c4"],
            "layers": [
                [
                    "ident-36db7f0563b8b9aeaf638d565fa1f8c4",
                    "MaterializedLayer",
                    [["ident-36db7f0563b8b9aeaf638d565fa1f8c4", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-36db7f0563b8b9aeaf638d565fa1f8c4",
        "result_repr": "([1, 2],)",
        "stable_graph": None,
    },
    "arg_list_iterator_with_delayed": {
        "graph": {
            "dask_keys": ["ident-3ce94eb4d5d85e5fd1bf2110e4be5f84"],
            "dask_layers": ["ident-3ce94eb4d5d85e5fd1bf2110e4be5f84"],
            "dependencies": [
                [
                    "ident-3ce94eb4d5d85e5fd1bf2110e4be5f84",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-3ce94eb4d5d85e5fd1bf2110e4be5f84",
            ],
            "key": "ident-3ce94eb4d5d85e5fd1bf2110e4be5f84",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-3ce94eb4d5d85e5fd1bf2110e4be5f84",
            ],
            "layers": [
                [
                    "ident-3ce94eb4d5d85e5fd1bf2110e4be5f84",
                    "MaterializedLayer",
                    [
                        [
                            "ident-3ce94eb4d5d85e5fd1bf2110e4be5f84",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-3ce94eb4d5d85e5fd1bf2110e4be5f84",
        "result_repr": "([2, 1],)",
        "stable_graph": None,
    },
    "arg_list_literal": {
        "graph": {
            "dask_keys": ["ident-95d449e68342442599e9a90ce127d03a"],
            "dask_layers": ["ident-95d449e68342442599e9a90ce127d03a"],
            "dependencies": [["ident-95d449e68342442599e9a90ce127d03a", []]],
            "dependency_order": ["ident-95d449e68342442599e9a90ce127d03a"],
            "key": "ident-95d449e68342442599e9a90ce127d03a",
            "layer_order": ["ident-95d449e68342442599e9a90ce127d03a"],
            "layers": [
                [
                    "ident-95d449e68342442599e9a90ce127d03a",
                    "MaterializedLayer",
                    [["ident-95d449e68342442599e9a90ce127d03a", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-95d449e68342442599e9a90ce127d03a",
        "result_repr": "([1, 2, 3],)",
        "stable_graph": None,
    },
    "arg_list_with_delayed": {
        "graph": {
            "dask_keys": ["ident-004d69c5aef967ffa674ba8fcbd84315"],
            "dask_layers": ["ident-004d69c5aef967ffa674ba8fcbd84315"],
            "dependencies": [
                [
                    "ident-004d69c5aef967ffa674ba8fcbd84315",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-004d69c5aef967ffa674ba8fcbd84315",
            ],
            "key": "ident-004d69c5aef967ffa674ba8fcbd84315",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-004d69c5aef967ffa674ba8fcbd84315",
            ],
            "layers": [
                [
                    "ident-004d69c5aef967ffa674ba8fcbd84315",
                    "MaterializedLayer",
                    [
                        [
                            "ident-004d69c5aef967ffa674ba8fcbd84315",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-004d69c5aef967ffa674ba8fcbd84315",
        "result_repr": "([2, 1],)",
        "stable_graph": None,
    },
    "arg_list_with_dependent_task": {
        "graph": {
            "dask_keys": ["ident-853f00e7b63921c6aa234b62a83f3f37"],
            "dask_layers": ["ident-853f00e7b63921c6aa234b62a83f3f37"],
            "dependencies": [["ident-853f00e7b63921c6aa234b62a83f3f37", []]],
            "dependency_order": ["ident-853f00e7b63921c6aa234b62a83f3f37"],
            "key": "ident-853f00e7b63921c6aa234b62a83f3f37",
            "layer_order": ["ident-853f00e7b63921c6aa234b62a83f3f37"],
            "layers": [
                [
                    "ident-853f00e7b63921c6aa234b62a83f3f37",
                    "MaterializedLayer",
                    [
                        [
                            "ident-853f00e7b63921c6aa234b62a83f3f37",
                            "Task",
                            ["dn"],
                            "ident",
                        ]
                    ],
                ]
            ],
        },
        "key": "ident-853f00e7b63921c6aa234b62a83f3f37",
        "result_repr": None,
        "stable_graph": None,
    },
    "arg_list_with_taskref": {
        "graph": {
            "dask_keys": ["ident-c79c565af7915db3336df040927eba3b"],
            "dask_layers": ["ident-c79c565af7915db3336df040927eba3b"],
            "dependencies": [["ident-c79c565af7915db3336df040927eba3b", []]],
            "dependency_order": ["ident-c79c565af7915db3336df040927eba3b"],
            "key": "ident-c79c565af7915db3336df040927eba3b",
            "layer_order": ["ident-c79c565af7915db3336df040927eba3b"],
            "layers": [
                [
                    "ident-c79c565af7915db3336df040927eba3b",
                    "MaterializedLayer",
                    [
                        [
                            "ident-c79c565af7915db3336df040927eba3b",
                            "Task",
                            ["dn"],
                            "ident",
                        ]
                    ],
                ]
            ],
        },
        "key": "ident-c79c565af7915db3336df040927eba3b",
        "result_repr": None,
        "stable_graph": None,
    },
    "arg_namedtuple_literal": {
        "graph": {
            "dask_keys": ["ident-9579ba3513aeed77b487aebb7575b121"],
            "dask_layers": ["ident-9579ba3513aeed77b487aebb7575b121"],
            "dependencies": [["ident-9579ba3513aeed77b487aebb7575b121", []]],
            "dependency_order": ["ident-9579ba3513aeed77b487aebb7575b121"],
            "key": "ident-9579ba3513aeed77b487aebb7575b121",
            "layer_order": ["ident-9579ba3513aeed77b487aebb7575b121"],
            "layers": [
                [
                    "ident-9579ba3513aeed77b487aebb7575b121",
                    "MaterializedLayer",
                    [["ident-9579ba3513aeed77b487aebb7575b121", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-9579ba3513aeed77b487aebb7575b121",
        "result_repr": "(Point(x=1, y=2),)",
        "stable_graph": None,
    },
    "arg_namedtuple_with_delayed": {
        "graph": {
            "dask_keys": ["ident-c3bc65bd4e077495ae8692e0745b4a9f"],
            "dask_layers": ["ident-c3bc65bd4e077495ae8692e0745b4a9f"],
            "dependencies": [
                [
                    "ident-c3bc65bd4e077495ae8692e0745b4a9f",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-c3bc65bd4e077495ae8692e0745b4a9f",
            ],
            "key": "ident-c3bc65bd4e077495ae8692e0745b4a9f",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-c3bc65bd4e077495ae8692e0745b4a9f",
            ],
            "layers": [
                [
                    "ident-c3bc65bd4e077495ae8692e0745b4a9f",
                    "MaterializedLayer",
                    [
                        [
                            "ident-c3bc65bd4e077495ae8692e0745b4a9f",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-c3bc65bd4e077495ae8692e0745b4a9f",
        "result_repr": "(Point(x=2, y=2),)",
        "stable_graph": None,
    },
    "arg_nested_list_depth3": {
        "graph": {
            "dask_keys": ["ident-4276245fe4734cde693c9f567e8db238"],
            "dask_layers": ["ident-4276245fe4734cde693c9f567e8db238"],
            "dependencies": [
                [
                    "ident-4276245fe4734cde693c9f567e8db238",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-4276245fe4734cde693c9f567e8db238",
            ],
            "key": "ident-4276245fe4734cde693c9f567e8db238",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-4276245fe4734cde693c9f567e8db238",
            ],
            "layers": [
                [
                    "ident-4276245fe4734cde693c9f567e8db238",
                    "MaterializedLayer",
                    [
                        [
                            "ident-4276245fe4734cde693c9f567e8db238",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-4276245fe4734cde693c9f567e8db238",
        "result_repr": "([[[2]]],)",
        "stable_graph": None,
    },
    "arg_nested_mixed": {
        "graph": {
            "dask_keys": ["ident-ad9104ee68e07579217ef0009c5d34a3"],
            "dask_layers": ["ident-ad9104ee68e07579217ef0009c5d34a3"],
            "dependencies": [
                [
                    "ident-ad9104ee68e07579217ef0009c5d34a3",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "ident-ad9104ee68e07579217ef0009c5d34a3",
                "inc-5852f565112f1604bc52264ea5e287dc",
            ],
            "key": "ident-ad9104ee68e07579217ef0009c5d34a3",
            "layer_order": [
                "ident-ad9104ee68e07579217ef0009c5d34a3",
                "inc-5852f565112f1604bc52264ea5e287dc",
            ],
            "layers": [
                [
                    "ident-ad9104ee68e07579217ef0009c5d34a3",
                    "MaterializedLayer",
                    [
                        [
                            "ident-ad9104ee68e07579217ef0009c5d34a3",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-ad9104ee68e07579217ef0009c5d34a3",
        "result_repr": "([{'k': (2, 1)}, [2, 2]],)",
        "stable_graph": None,
    },
    "arg_none": {
        "graph": {
            "dask_keys": ["ident-20e29962a34319a9f919fd1e3ef6801e"],
            "dask_layers": ["ident-20e29962a34319a9f919fd1e3ef6801e"],
            "dependencies": [["ident-20e29962a34319a9f919fd1e3ef6801e", []]],
            "dependency_order": ["ident-20e29962a34319a9f919fd1e3ef6801e"],
            "key": "ident-20e29962a34319a9f919fd1e3ef6801e",
            "layer_order": ["ident-20e29962a34319a9f919fd1e3ef6801e"],
            "layers": [
                [
                    "ident-20e29962a34319a9f919fd1e3ef6801e",
                    "MaterializedLayer",
                    [["ident-20e29962a34319a9f919fd1e3ef6801e", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-20e29962a34319a9f919fd1e3ef6801e",
        "result_repr": "(None,)",
        "stable_graph": None,
    },
    "arg_set_iterator": {
        "graph": {
            "dask_keys": ["ident-e13182a23ef98e7baee4c6fba8a7a8da"],
            "dask_layers": ["ident-e13182a23ef98e7baee4c6fba8a7a8da"],
            "dependencies": [["ident-e13182a23ef98e7baee4c6fba8a7a8da", []]],
            "dependency_order": ["ident-e13182a23ef98e7baee4c6fba8a7a8da"],
            "key": "ident-e13182a23ef98e7baee4c6fba8a7a8da",
            "layer_order": ["ident-e13182a23ef98e7baee4c6fba8a7a8da"],
            "layers": [
                [
                    "ident-e13182a23ef98e7baee4c6fba8a7a8da",
                    "MaterializedLayer",
                    [["ident-e13182a23ef98e7baee4c6fba8a7a8da", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-e13182a23ef98e7baee4c6fba8a7a8da",
        "result_repr": "({1},)",
        "stable_graph": None,
    },
    "arg_set_literal": {
        "graph": {
            "dask_keys": ["ident-accd85cbd80c042b687d0e8775f10505"],
            "dask_layers": ["ident-accd85cbd80c042b687d0e8775f10505"],
            "dependencies": [["ident-accd85cbd80c042b687d0e8775f10505", []]],
            "dependency_order": ["ident-accd85cbd80c042b687d0e8775f10505"],
            "key": "ident-accd85cbd80c042b687d0e8775f10505",
            "layer_order": ["ident-accd85cbd80c042b687d0e8775f10505"],
            "layers": [
                [
                    "ident-accd85cbd80c042b687d0e8775f10505",
                    "MaterializedLayer",
                    [["ident-accd85cbd80c042b687d0e8775f10505", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-accd85cbd80c042b687d0e8775f10505",
        "result_repr": "({1, 2, 3},)",
        "stable_graph": None,
    },
    "arg_set_with_delayed": {
        "graph": {
            "dask_keys": ["ident-2945caca037fdbaa9ac1525a08c99e5c"],
            "dask_layers": ["ident-2945caca037fdbaa9ac1525a08c99e5c"],
            "dependencies": [
                [
                    "ident-2945caca037fdbaa9ac1525a08c99e5c",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-2945caca037fdbaa9ac1525a08c99e5c",
            ],
            "key": "ident-2945caca037fdbaa9ac1525a08c99e5c",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-2945caca037fdbaa9ac1525a08c99e5c",
            ],
            "layers": [
                [
                    "ident-2945caca037fdbaa9ac1525a08c99e5c",
                    "MaterializedLayer",
                    [
                        [
                            "ident-2945caca037fdbaa9ac1525a08c99e5c",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-2945caca037fdbaa9ac1525a08c99e5c",
        "result_repr": "({2},)",
        "stable_graph": None,
    },
    "arg_several_deps_dict_graph": {
        "graph": {
            "dask_keys": ["ident-b807708bf3da31110e69cce30a60ca6f"],
            "dask_layers": ["ident-b807708bf3da31110e69cce30a60ca6f"],
            "dependencies": [
                ["dgkey", []],
                [
                    "ident-b807708bf3da31110e69cce30a60ca6f",
                    ["dgkey", "inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "ident-b807708bf3da31110e69cce30a60ca6f",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "dgkey",
            ],
            "key": "ident-b807708bf3da31110e69cce30a60ca6f",
            "layer_order": [
                "ident-b807708bf3da31110e69cce30a60ca6f",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "dgkey",
            ],
            "layers": [
                ["dgkey", "MaterializedLayer", [["dgkey", "DataNode", [], None]]],
                [
                    "ident-b807708bf3da31110e69cce30a60ca6f",
                    "MaterializedLayer",
                    [
                        [
                            "ident-b807708bf3da31110e69cce30a60ca6f",
                            "Task",
                            ["dgkey", "inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-b807708bf3da31110e69cce30a60ca6f",
        "result_repr": "(2, 5)",
        "stable_graph": None,
    },
    "arg_several_deps_guard_mixed": {
        "graph": {
            "dask_keys": ["ident-4d2fd37f644ecc9c79595b39881dc1ab"],
            "dask_layers": ["ident-4d2fd37f644ecc9c79595b39881dc1ab"],
            "dependencies": [
                [
                    "ident-4d2fd37f644ecc9c79595b39881dc1ab",
                    ["inc-5852f565112f1604bc52264ea5e287dc", "subkey"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
                ["subkey", []],
            ],
            "dependency_order": [
                "ident-4d2fd37f644ecc9c79595b39881dc1ab",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "subkey",
            ],
            "key": "ident-4d2fd37f644ecc9c79595b39881dc1ab",
            "layer_order": [
                "ident-4d2fd37f644ecc9c79595b39881dc1ab",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "subkey",
            ],
            "layers": [
                [
                    "ident-4d2fd37f644ecc9c79595b39881dc1ab",
                    "MaterializedLayer",
                    [
                        [
                            "ident-4d2fd37f644ecc9c79595b39881dc1ab",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc", "subkey"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
                ["subkey", "MaterializedLayer", [["subkey", "DataNode", [], None]]],
            ],
        },
        "key": "ident-4d2fd37f644ecc9c79595b39881dc1ab",
        "result_repr": "(2, 5)",
        "stable_graph": None,
    },
    "arg_several_deps_with_attr": {
        "graph": {
            "dask_keys": ["ident-a0ea2b61f30d75a4f638e5a22bb2b73a"],
            "dask_layers": ["ident-a0ea2b61f30d75a4f638e5a22bb2b73a"],
            "dependencies": [
                ["getattr-b73ec8674b3d2dc92415a8be924f4cbf", ["obj"]],
                [
                    "ident-a0ea2b61f30d75a4f638e5a22bb2b73a",
                    [
                        "getattr-b73ec8674b3d2dc92415a8be924f4cbf",
                        "inc-5852f565112f1604bc52264ea5e287dc",
                    ],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
                ["obj", []],
            ],
            "dependency_order": [
                "ident-a0ea2b61f30d75a4f638e5a22bb2b73a",
                "obj",
                "getattr-b73ec8674b3d2dc92415a8be924f4cbf",
                "inc-5852f565112f1604bc52264ea5e287dc",
            ],
            "key": "ident-a0ea2b61f30d75a4f638e5a22bb2b73a",
            "layer_order": [
                "ident-a0ea2b61f30d75a4f638e5a22bb2b73a",
                "obj",
                "getattr-b73ec8674b3d2dc92415a8be924f4cbf",
                "inc-5852f565112f1604bc52264ea5e287dc",
            ],
            "layers": [
                [
                    "getattr-b73ec8674b3d2dc92415a8be924f4cbf",
                    "MaterializedLayer",
                    [
                        [
                            "getattr-b73ec8674b3d2dc92415a8be924f4cbf",
                            "legacy-tuple",
                            ["obj"],
                            None,
                        ]
                    ],
                ],
                [
                    "ident-a0ea2b61f30d75a4f638e5a22bb2b73a",
                    "MaterializedLayer",
                    [
                        [
                            "ident-a0ea2b61f30d75a4f638e5a22bb2b73a",
                            "Task",
                            [
                                "getattr-b73ec8674b3d2dc92415a8be924f4cbf",
                                "inc-5852f565112f1604bc52264ea5e287dc",
                            ],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
                ["obj", "MaterializedLayer", [["obj", "DataNode", [], None]]],
            ],
        },
        "key": "ident-a0ea2b61f30d75a4f638e5a22bb2b73a",
        "result_repr": "(3, 2)",
        "stable_graph": None,
    },
    "arg_several_deps_with_leaf": {
        "graph": {
            "dask_keys": ["ident-c27f669c23db4071a9606a6d48bbce23"],
            "dask_layers": ["ident-c27f669c23db4071a9606a6d48bbce23"],
            "dependencies": [
                [
                    "ident-c27f669c23db4071a9606a6d48bbce23",
                    ["inc-5852f565112f1604bc52264ea5e287dc", "three-leaf"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
                ["three-leaf", []],
            ],
            "dependency_order": [
                "ident-c27f669c23db4071a9606a6d48bbce23",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "three-leaf",
            ],
            "key": "ident-c27f669c23db4071a9606a6d48bbce23",
            "layer_order": [
                "ident-c27f669c23db4071a9606a6d48bbce23",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "three-leaf",
            ],
            "layers": [
                [
                    "ident-c27f669c23db4071a9606a6d48bbce23",
                    "MaterializedLayer",
                    [
                        [
                            "ident-c27f669c23db4071a9606a6d48bbce23",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc", "three-leaf"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
                [
                    "three-leaf",
                    "MaterializedLayer",
                    [["three-leaf", "DataNode", [], None]],
                ],
            ],
        },
        "key": "ident-c27f669c23db4071a9606a6d48bbce23",
        "result_repr": "(2, 3)",
        "stable_graph": None,
    },
    "arg_slice_literal": {
        "graph": {
            "dask_keys": ["ident-7ab4c03942d58d373b077b8da246904c"],
            "dask_layers": ["ident-7ab4c03942d58d373b077b8da246904c"],
            "dependencies": [["ident-7ab4c03942d58d373b077b8da246904c", []]],
            "dependency_order": ["ident-7ab4c03942d58d373b077b8da246904c"],
            "key": "ident-7ab4c03942d58d373b077b8da246904c",
            "layer_order": ["ident-7ab4c03942d58d373b077b8da246904c"],
            "layers": [
                [
                    "ident-7ab4c03942d58d373b077b8da246904c",
                    "MaterializedLayer",
                    [["ident-7ab4c03942d58d373b077b8da246904c", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-7ab4c03942d58d373b077b8da246904c",
        "result_repr": "(slice(1, 5, 2),)",
        "stable_graph": None,
    },
    "arg_slice_with_delayed": {
        "graph": {
            "dask_keys": ["ident-644c86d459c4c086931a88aefe7b5119"],
            "dask_layers": ["ident-644c86d459c4c086931a88aefe7b5119"],
            "dependencies": [
                [
                    "ident-644c86d459c4c086931a88aefe7b5119",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-644c86d459c4c086931a88aefe7b5119",
            ],
            "key": "ident-644c86d459c4c086931a88aefe7b5119",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-644c86d459c4c086931a88aefe7b5119",
            ],
            "layers": [
                [
                    "ident-644c86d459c4c086931a88aefe7b5119",
                    "MaterializedLayer",
                    [
                        [
                            "ident-644c86d459c4c086931a88aefe7b5119",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-644c86d459c4c086931a88aefe7b5119",
        "result_repr": "(slice(2, 5, None),)",
        "stable_graph": None,
    },
    "arg_str": {
        "graph": {
            "dask_keys": ["ident-716f1034d088581f82f5d7ee76f565b9"],
            "dask_layers": ["ident-716f1034d088581f82f5d7ee76f565b9"],
            "dependencies": [["ident-716f1034d088581f82f5d7ee76f565b9", []]],
            "dependency_order": ["ident-716f1034d088581f82f5d7ee76f565b9"],
            "key": "ident-716f1034d088581f82f5d7ee76f565b9",
            "layer_order": ["ident-716f1034d088581f82f5d7ee76f565b9"],
            "layers": [
                [
                    "ident-716f1034d088581f82f5d7ee76f565b9",
                    "MaterializedLayer",
                    [["ident-716f1034d088581f82f5d7ee76f565b9", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-716f1034d088581f82f5d7ee76f565b9",
        "result_repr": "('s',)",
        "stable_graph": None,
    },
    "arg_tuple_iterator": {
        "graph": {
            "dask_keys": ["ident-752bf50d0900c4e6fecd809fe1e6caa2"],
            "dask_layers": ["ident-752bf50d0900c4e6fecd809fe1e6caa2"],
            "dependencies": [["ident-752bf50d0900c4e6fecd809fe1e6caa2", []]],
            "dependency_order": ["ident-752bf50d0900c4e6fecd809fe1e6caa2"],
            "key": "ident-752bf50d0900c4e6fecd809fe1e6caa2",
            "layer_order": ["ident-752bf50d0900c4e6fecd809fe1e6caa2"],
            "layers": [
                [
                    "ident-752bf50d0900c4e6fecd809fe1e6caa2",
                    "MaterializedLayer",
                    [["ident-752bf50d0900c4e6fecd809fe1e6caa2", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-752bf50d0900c4e6fecd809fe1e6caa2",
        "result_repr": "((1, 2),)",
        "stable_graph": None,
    },
    "arg_tuple_key_delayed": {
        "graph": {
            "dask_keys": ["ident-9eeed6a7f12006002ea644954aaebf76"],
            "dask_layers": ["ident-9eeed6a7f12006002ea644954aaebf76"],
            "dependencies": [
                ["('tk', 0)", []],
                ["ident-9eeed6a7f12006002ea644954aaebf76", ["('tk', 0)"]],
            ],
            "dependency_order": ["ident-9eeed6a7f12006002ea644954aaebf76", "('tk', 0)"],
            "key": "ident-9eeed6a7f12006002ea644954aaebf76",
            "layer_order": ["ident-9eeed6a7f12006002ea644954aaebf76", "('tk', 0)"],
            "layers": [
                [
                    "('tk', 0)",
                    "MaterializedLayer",
                    [["('tk', 0)", "DataNode", [], None]],
                ],
                [
                    "ident-9eeed6a7f12006002ea644954aaebf76",
                    "MaterializedLayer",
                    [
                        [
                            "ident-9eeed6a7f12006002ea644954aaebf76",
                            "Task",
                            ["('tk', 0)"],
                            "ident",
                        ]
                    ],
                ],
            ],
        },
        "key": "ident-9eeed6a7f12006002ea644954aaebf76",
        "result_repr": "(5,)",
        "stable_graph": None,
    },
    "arg_tuple_literal": {
        "graph": {
            "dask_keys": ["ident-29a987695b203e31c66ac7a157a66e8a"],
            "dask_layers": ["ident-29a987695b203e31c66ac7a157a66e8a"],
            "dependencies": [["ident-29a987695b203e31c66ac7a157a66e8a", []]],
            "dependency_order": ["ident-29a987695b203e31c66ac7a157a66e8a"],
            "key": "ident-29a987695b203e31c66ac7a157a66e8a",
            "layer_order": ["ident-29a987695b203e31c66ac7a157a66e8a"],
            "layers": [
                [
                    "ident-29a987695b203e31c66ac7a157a66e8a",
                    "MaterializedLayer",
                    [["ident-29a987695b203e31c66ac7a157a66e8a", "Task", [], "ident"]],
                ]
            ],
        },
        "key": "ident-29a987695b203e31c66ac7a157a66e8a",
        "result_repr": "((1, 2),)",
        "stable_graph": None,
    },
    "arg_tuple_with_delayed": {
        "graph": {
            "dask_keys": ["ident-abe2c00feb8e1d3007b82d33fbcad638"],
            "dask_layers": ["ident-abe2c00feb8e1d3007b82d33fbcad638"],
            "dependencies": [
                [
                    "ident-abe2c00feb8e1d3007b82d33fbcad638",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-abe2c00feb8e1d3007b82d33fbcad638",
            ],
            "key": "ident-abe2c00feb8e1d3007b82d33fbcad638",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "ident-abe2c00feb8e1d3007b82d33fbcad638",
            ],
            "layers": [
                [
                    "ident-abe2c00feb8e1d3007b82d33fbcad638",
                    "MaterializedLayer",
                    [
                        [
                            "ident-abe2c00feb8e1d3007b82d33fbcad638",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-abe2c00feb8e1d3007b82d33fbcad638",
        "result_repr": "((2, 1),)",
        "stable_graph": None,
    },
    "arg_two_dependencies": {
        "graph": {
            "dask_keys": ["ident-44dbd716a98aad621364346979df06de"],
            "dask_layers": ["ident-44dbd716a98aad621364346979df06de"],
            "dependencies": [
                [
                    "ident-44dbd716a98aad621364346979df06de",
                    [
                        "inc-2f16cb15cc39d4668d79bfb484510bbb",
                        "inc-5852f565112f1604bc52264ea5e287dc",
                    ],
                ],
                ["inc-2f16cb15cc39d4668d79bfb484510bbb", []],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "ident-44dbd716a98aad621364346979df06de",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "inc-2f16cb15cc39d4668d79bfb484510bbb",
            ],
            "key": "ident-44dbd716a98aad621364346979df06de",
            "layer_order": [
                "ident-44dbd716a98aad621364346979df06de",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "inc-2f16cb15cc39d4668d79bfb484510bbb",
            ],
            "layers": [
                [
                    "ident-44dbd716a98aad621364346979df06de",
                    "MaterializedLayer",
                    [
                        [
                            "ident-44dbd716a98aad621364346979df06de",
                            "Task",
                            [
                                "inc-2f16cb15cc39d4668d79bfb484510bbb",
                                "inc-5852f565112f1604bc52264ea5e287dc",
                            ],
                            "ident",
                        ]
                    ],
                ],
                [
                    "inc-2f16cb15cc39d4668d79bfb484510bbb",
                    "MaterializedLayer",
                    [["inc-2f16cb15cc39d4668d79bfb484510bbb", "Task", [], "inc"]],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "ident-44dbd716a98aad621364346979df06de",
        "result_repr": "(2, 3)",
        "stable_graph": None,
    },
    "attr_items_getitem": {
        "graph": {
            "dask_keys": ["getitem-e9440fd77ae78b0d4a6a58d9dcfbcfa3"],
            "dask_layers": ["getitem-e9440fd77ae78b0d4a6a58d9dcfbcfa3"],
            "dependencies": [
                ["getattr-f8efbec48aa4260ded710376183b7ba5", ["obj"]],
                [
                    "getitem-e9440fd77ae78b0d4a6a58d9dcfbcfa3",
                    ["getattr-f8efbec48aa4260ded710376183b7ba5"],
                ],
                ["obj", []],
            ],
            "dependency_order": [
                "obj",
                "getattr-f8efbec48aa4260ded710376183b7ba5",
                "getitem-e9440fd77ae78b0d4a6a58d9dcfbcfa3",
            ],
            "key": "getitem-e9440fd77ae78b0d4a6a58d9dcfbcfa3",
            "layer_order": [
                "obj",
                "getattr-f8efbec48aa4260ded710376183b7ba5",
                "getitem-e9440fd77ae78b0d4a6a58d9dcfbcfa3",
            ],
            "layers": [
                [
                    "getattr-f8efbec48aa4260ded710376183b7ba5",
                    "MaterializedLayer",
                    [
                        [
                            "getattr-f8efbec48aa4260ded710376183b7ba5",
                            "legacy-tuple",
                            ["obj"],
                            None,
                        ]
                    ],
                ],
                [
                    "getitem-e9440fd77ae78b0d4a6a58d9dcfbcfa3",
                    "MaterializedLayer",
                    [
                        [
                            "getitem-e9440fd77ae78b0d4a6a58d9dcfbcfa3",
                            "Task",
                            ["getattr-f8efbec48aa4260ded710376183b7ba5"],
                            "getitem",
                        ]
                    ],
                ],
                ["obj", "MaterializedLayer", [["obj", "DataNode", [], None]]],
            ],
        },
        "key": "getitem-e9440fd77ae78b0d4a6a58d9dcfbcfa3",
        "result_repr": "11",
        "stable_graph": None,
    },
    "attr_of_attr": {
        "graph": {
            "dask_keys": ["getattr-e6dab9e42a108e1d46bf06102a2d0a0b"],
            "dask_layers": ["getattr-e6dab9e42a108e1d46bf06102a2d0a0b"],
            "dependencies": [
                ["getattr-4d6d83f7536e13768743295ab3bd2775", ["obj"]],
                [
                    "getattr-e6dab9e42a108e1d46bf06102a2d0a0b",
                    ["getattr-4d6d83f7536e13768743295ab3bd2775"],
                ],
                ["obj", []],
            ],
            "dependency_order": [
                "obj",
                "getattr-4d6d83f7536e13768743295ab3bd2775",
                "getattr-e6dab9e42a108e1d46bf06102a2d0a0b",
            ],
            "key": "getattr-e6dab9e42a108e1d46bf06102a2d0a0b",
            "layer_order": [
                "obj",
                "getattr-4d6d83f7536e13768743295ab3bd2775",
                "getattr-e6dab9e42a108e1d46bf06102a2d0a0b",
            ],
            "layers": [
                [
                    "getattr-4d6d83f7536e13768743295ab3bd2775",
                    "MaterializedLayer",
                    [
                        [
                            "getattr-4d6d83f7536e13768743295ab3bd2775",
                            "legacy-tuple",
                            ["obj"],
                            None,
                        ]
                    ],
                ],
                [
                    "getattr-e6dab9e42a108e1d46bf06102a2d0a0b",
                    "MaterializedLayer",
                    [
                        [
                            "getattr-e6dab9e42a108e1d46bf06102a2d0a0b",
                            "legacy-tuple",
                            ["getattr-4d6d83f7536e13768743295ab3bd2775"],
                            None,
                        ]
                    ],
                ],
                ["obj", "MaterializedLayer", [["obj", "DataNode", [], None]]],
            ],
        },
        "key": "getattr-e6dab9e42a108e1d46bf06102a2d0a0b",
        "result_repr": "9",
        "stable_graph": None,
    },
    "attr_v": {
        "graph": {
            "dask_keys": ["getattr-b73ec8674b3d2dc92415a8be924f4cbf"],
            "dask_layers": ["getattr-b73ec8674b3d2dc92415a8be924f4cbf"],
            "dependencies": [
                ["getattr-b73ec8674b3d2dc92415a8be924f4cbf", ["obj"]],
                ["obj", []],
            ],
            "dependency_order": ["obj", "getattr-b73ec8674b3d2dc92415a8be924f4cbf"],
            "key": "getattr-b73ec8674b3d2dc92415a8be924f4cbf",
            "layer_order": ["obj", "getattr-b73ec8674b3d2dc92415a8be924f4cbf"],
            "layers": [
                [
                    "getattr-b73ec8674b3d2dc92415a8be924f4cbf",
                    "MaterializedLayer",
                    [
                        [
                            "getattr-b73ec8674b3d2dc92415a8be924f4cbf",
                            "legacy-tuple",
                            ["obj"],
                            None,
                        ]
                    ],
                ],
                ["obj", "MaterializedLayer", [["obj", "DataNode", [], None]]],
            ],
        },
        "key": "getattr-b73ec8674b3d2dc92415a8be924f4cbf",
        "result_repr": "3",
        "stable_graph": None,
    },
    "call_dask_key_name": {
        "graph": {
            "dask_keys": ["explicit-call-key"],
            "dask_layers": ["explicit-call-key"],
            "dependencies": [["explicit-call-key", []]],
            "dependency_order": ["explicit-call-key"],
            "key": "explicit-call-key",
            "layer_order": ["explicit-call-key"],
            "layers": [
                [
                    "explicit-call-key",
                    "MaterializedLayer",
                    [["explicit-call-key", "Task", [], "inc"]],
                ]
            ],
        },
        "key": "explicit-call-key",
        "result_repr": "2",
        "stable_graph": None,
    },
    "call_default_name": {
        "graph": {
            "dask_keys": ["inc-d16c77ed2f4f68bb5383a923c62fc22c"],
            "dask_layers": ["inc-d16c77ed2f4f68bb5383a923c62fc22c"],
            "dependencies": [["inc-d16c77ed2f4f68bb5383a923c62fc22c", []]],
            "dependency_order": ["inc-d16c77ed2f4f68bb5383a923c62fc22c"],
            "key": "inc-d16c77ed2f4f68bb5383a923c62fc22c",
            "layer_order": ["inc-d16c77ed2f4f68bb5383a923c62fc22c"],
            "layers": [
                [
                    "inc-d16c77ed2f4f68bb5383a923c62fc22c",
                    "MaterializedLayer",
                    [["inc-d16c77ed2f4f68bb5383a923c62fc22c", "Task", [], "inc"]],
                ]
            ],
        },
        "key": "inc-d16c77ed2f4f68bb5383a923c62fc22c",
        "result_repr": "2",
        "stable_graph": None,
    },
    "call_global_delayed_pure": {
        "graph": {
            "dask_keys": ["inc-5852f565112f1604bc52264ea5e287dc"],
            "dask_layers": ["inc-5852f565112f1604bc52264ea5e287dc"],
            "dependencies": [["inc-5852f565112f1604bc52264ea5e287dc", []]],
            "dependency_order": ["inc-5852f565112f1604bc52264ea5e287dc"],
            "key": "inc-5852f565112f1604bc52264ea5e287dc",
            "layer_order": ["inc-5852f565112f1604bc52264ea5e287dc"],
            "layers": [
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ]
            ],
        },
        "key": "inc-5852f565112f1604bc52264ea5e287dc",
        "result_repr": "2",
        "stable_graph": None,
    },
    "call_pure_false": {
        "graph": {
            "dask_keys": ["inc-#0"],
            "dask_layers": ["inc-#0"],
            "dependencies": [["inc-#0", []]],
            "dependency_order": ["inc-#0"],
            "key": "inc-#0",
            "layer_order": ["inc-#0"],
            "layers": [
                ["inc-#0", "MaterializedLayer", [["inc-#0", "Task", [], "inc"]]]
            ],
        },
        "key": "inc-#0",
        "result_repr": "2",
        "stable_graph": None,
    },
    "call_pure_true": {
        "graph": {
            "dask_keys": ["inc-5852f565112f1604bc52264ea5e287dc"],
            "dask_layers": ["inc-5852f565112f1604bc52264ea5e287dc"],
            "dependencies": [["inc-5852f565112f1604bc52264ea5e287dc", []]],
            "dependency_order": ["inc-5852f565112f1604bc52264ea5e287dc"],
            "key": "inc-5852f565112f1604bc52264ea5e287dc",
            "layer_order": ["inc-5852f565112f1604bc52264ea5e287dc"],
            "layers": [
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ]
            ],
        },
        "key": "inc-5852f565112f1604bc52264ea5e287dc",
        "result_repr": "2",
        "stable_graph": None,
    },
    "delayed_call_impure": {
        "graph": {
            "dask_keys": ["apply-#0"],
            "dask_layers": ["apply-#0"],
            "dependencies": [
                ["apply-#0", ["first-b1be2ceffdecc81a278c3b68c19d756e"]],
                ["first-b1be2ceffdecc81a278c3b68c19d756e", []],
            ],
            "dependency_order": ["first-b1be2ceffdecc81a278c3b68c19d756e", "apply-#0"],
            "key": "apply-#0",
            "layer_order": ["first-b1be2ceffdecc81a278c3b68c19d756e", "apply-#0"],
            "layers": [
                [
                    "apply-#0",
                    "MaterializedLayer",
                    [
                        [
                            "apply-#0",
                            "Task",
                            ["first-b1be2ceffdecc81a278c3b68c19d756e"],
                            "apply",
                        ]
                    ],
                ],
                [
                    "first-b1be2ceffdecc81a278c3b68c19d756e",
                    "MaterializedLayer",
                    [["first-b1be2ceffdecc81a278c3b68c19d756e", "Task", [], "first"]],
                ],
            ],
        },
        "key": "apply-#0",
        "result_repr": "2",
        "stable_graph": None,
    },
    "delayed_call_pure": {
        "graph": {
            "dask_keys": ["apply-48099599c4a4d25761240ca3decec089"],
            "dask_layers": ["apply-48099599c4a4d25761240ca3decec089"],
            "dependencies": [
                [
                    "apply-48099599c4a4d25761240ca3decec089",
                    ["first-b1be2ceffdecc81a278c3b68c19d756e"],
                ],
                ["first-b1be2ceffdecc81a278c3b68c19d756e", []],
            ],
            "dependency_order": [
                "first-b1be2ceffdecc81a278c3b68c19d756e",
                "apply-48099599c4a4d25761240ca3decec089",
            ],
            "key": "apply-48099599c4a4d25761240ca3decec089",
            "layer_order": [
                "first-b1be2ceffdecc81a278c3b68c19d756e",
                "apply-48099599c4a4d25761240ca3decec089",
            ],
            "layers": [
                [
                    "apply-48099599c4a4d25761240ca3decec089",
                    "MaterializedLayer",
                    [
                        [
                            "apply-48099599c4a4d25761240ca3decec089",
                            "Task",
                            ["first-b1be2ceffdecc81a278c3b68c19d756e"],
                            "apply",
                        ]
                    ],
                ],
                [
                    "first-b1be2ceffdecc81a278c3b68c19d756e",
                    "MaterializedLayer",
                    [["first-b1be2ceffdecc81a278c3b68c19d756e", "Task", [], "first"]],
                ],
            ],
        },
        "key": "apply-48099599c4a4d25761240ca3decec089",
        "result_repr": "2",
        "stable_graph": None,
    },
    "finalize_collection": {
        "graph": None,
        "key": "finalize-6bd795b2e9accac4e918747e25da4d38",
        "result_repr": None,
        "stable_graph": {
            "dask_keys": ["finalize-6bd795b2e9accac4e918747e25da4d38"],
            "dask_layers": ["finalize-6bd795b2e9accac4e918747e25da4d38"],
            "dependencies": [["finalize-6bd795b2e9accac4e918747e25da4d38", []]],
            "dependency_order": ["finalize-6bd795b2e9accac4e918747e25da4d38"],
            "key": "finalize-6bd795b2e9accac4e918747e25da4d38",
            "layer_order": ["finalize-6bd795b2e9accac4e918747e25da4d38"],
            "layers": [
                [
                    "finalize-6bd795b2e9accac4e918747e25da4d38",
                    "dict",
                    [
                        [
                            "finalize-hlgfinalizecompute-<hex0>",
                            "Task",
                            ["ta", "tb"],
                            "tuple",
                        ],
                        ["ta", "DataNode", [], None],
                        ["tb", "DataNode", [], None],
                    ],
                ]
            ],
        },
    },
    "guard_mapping_proxy_graph": {
        "graph": {
            "dask_keys": ["ident-1a05d5989fcd575082000c787d633dbd"],
            "dask_layers": ["ident-1a05d5989fcd575082000c787d633dbd"],
            "dependencies": [
                ["ident-1a05d5989fcd575082000c787d633dbd", ["mpkey"]],
                ["mpkey", []],
            ],
            "dependency_order": ["ident-1a05d5989fcd575082000c787d633dbd", "mpkey"],
            "key": "ident-1a05d5989fcd575082000c787d633dbd",
            "layer_order": ["ident-1a05d5989fcd575082000c787d633dbd", "mpkey"],
            "layers": [
                [
                    "ident-1a05d5989fcd575082000c787d633dbd",
                    "MaterializedLayer",
                    [
                        [
                            "ident-1a05d5989fcd575082000c787d633dbd",
                            "Task",
                            ["mpkey"],
                            "ident",
                        ]
                    ],
                ],
                ["mpkey", "MaterializedLayer", [["mpkey", "DataNode", [], None]]],
            ],
        },
        "key": "ident-1a05d5989fcd575082000c787d633dbd",
        "result_repr": "(5,)",
        "stable_graph": None,
    },
    "guard_subclass": {
        "graph": {
            "dask_keys": ["ident-11133fbff927db683bec9e0dde4c091d"],
            "dask_layers": ["ident-11133fbff927db683bec9e0dde4c091d"],
            "dependencies": [
                ["ident-11133fbff927db683bec9e0dde4c091d", ["subkey"]],
                ["subkey", []],
            ],
            "dependency_order": ["subkey", "ident-11133fbff927db683bec9e0dde4c091d"],
            "key": "ident-11133fbff927db683bec9e0dde4c091d",
            "layer_order": ["subkey", "ident-11133fbff927db683bec9e0dde4c091d"],
            "layers": [
                [
                    "ident-11133fbff927db683bec9e0dde4c091d",
                    "MaterializedLayer",
                    [
                        [
                            "ident-11133fbff927db683bec9e0dde4c091d",
                            "Task",
                            ["subkey"],
                            "ident",
                        ]
                    ],
                ],
                ["subkey", "MaterializedLayer", [["subkey", "DataNode", [], None]]],
            ],
        },
        "key": "ident-11133fbff927db683bec9e0dde4c091d",
        "result_repr": "(5,)",
        "stable_graph": None,
    },
    "guard_subclass_dask_layers": {
        "graph": {
            "dask_keys": ["ident-abd6a058945122515edd893ddfd4c571"],
            "dask_layers": ["ident-abd6a058945122515edd893ddfd4c571"],
            "dependencies": [
                ["ident-abd6a058945122515edd893ddfd4c571", ["sublayerskey"]],
                ["sublayerskey", []],
            ],
            "dependency_order": [
                "sublayerskey",
                "ident-abd6a058945122515edd893ddfd4c571",
            ],
            "key": "ident-abd6a058945122515edd893ddfd4c571",
            "layer_order": ["sublayerskey", "ident-abd6a058945122515edd893ddfd4c571"],
            "layers": [
                [
                    "ident-abd6a058945122515edd893ddfd4c571",
                    "MaterializedLayer",
                    [
                        [
                            "ident-abd6a058945122515edd893ddfd4c571",
                            "Task",
                            ["sublayerskey"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "sublayerskey",
                    "MaterializedLayer",
                    [["sublayerskey", "DataNode", [], None]],
                ],
            ],
        },
        "key": "ident-abd6a058945122515edd893ddfd4c571",
        "result_repr": "(5,)",
        "stable_graph": None,
    },
    "guard_subclass_layer_dict": {
        "graph": {
            "dask_keys": ["ident-b321f73872bbd12dc3802c7848bacc16"],
            "dask_layers": ["ident-b321f73872bbd12dc3802c7848bacc16"],
            "dependencies": [
                ["ident-b321f73872bbd12dc3802c7848bacc16", ["sublayerdictkey"]],
                ["sublayerdictkey", []],
            ],
            "dependency_order": [
                "sublayerdictkey",
                "ident-b321f73872bbd12dc3802c7848bacc16",
            ],
            "key": "ident-b321f73872bbd12dc3802c7848bacc16",
            "layer_order": [
                "sublayerdictkey",
                "ident-b321f73872bbd12dc3802c7848bacc16",
            ],
            "layers": [
                [
                    "ident-b321f73872bbd12dc3802c7848bacc16",
                    "MaterializedLayer",
                    [
                        [
                            "ident-b321f73872bbd12dc3802c7848bacc16",
                            "Task",
                            ["sublayerdictkey"],
                            "ident",
                        ]
                    ],
                ],
                [
                    "sublayerdictkey",
                    "MaterializedLayer",
                    [["sublayerdictkey", "DataNode", [], None]],
                ],
            ],
        },
        "key": "ident-b321f73872bbd12dc3802c7848bacc16",
        "result_repr": "(5,)",
        "stable_graph": None,
    },
    "kwargs_literal": {
        "graph": {
            "dask_keys": ["collect-bfcc9f334e410650ccc73099e6f11615"],
            "dask_layers": ["collect-bfcc9f334e410650ccc73099e6f11615"],
            "dependencies": [["collect-bfcc9f334e410650ccc73099e6f11615", []]],
            "dependency_order": ["collect-bfcc9f334e410650ccc73099e6f11615"],
            "key": "collect-bfcc9f334e410650ccc73099e6f11615",
            "layer_order": ["collect-bfcc9f334e410650ccc73099e6f11615"],
            "layers": [
                [
                    "collect-bfcc9f334e410650ccc73099e6f11615",
                    "MaterializedLayer",
                    [
                        [
                            "collect-bfcc9f334e410650ccc73099e6f11615",
                            "Task",
                            [],
                            "collect",
                        ]
                    ],
                ]
            ],
        },
        "key": "collect-bfcc9f334e410650ccc73099e6f11615",
        "result_repr": "((), {'x': 1})",
        "stable_graph": None,
    },
    "kwargs_with_delayed": {
        "graph": {
            "dask_keys": ["collect-a6ffaaa7e60d4c364fd17a173c9d6fee"],
            "dask_layers": ["collect-a6ffaaa7e60d4c364fd17a173c9d6fee"],
            "dependencies": [
                [
                    "collect-a6ffaaa7e60d4c364fd17a173c9d6fee",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "collect-a6ffaaa7e60d4c364fd17a173c9d6fee",
            ],
            "key": "collect-a6ffaaa7e60d4c364fd17a173c9d6fee",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "collect-a6ffaaa7e60d4c364fd17a173c9d6fee",
            ],
            "layers": [
                [
                    "collect-a6ffaaa7e60d4c364fd17a173c9d6fee",
                    "MaterializedLayer",
                    [
                        [
                            "collect-a6ffaaa7e60d4c364fd17a173c9d6fee",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "collect",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "collect-a6ffaaa7e60d4c364fd17a173c9d6fee",
        "result_repr": "((), {'x': 2})",
        "stable_graph": None,
    },
    "method_impure": {
        "graph": {
            "dask_keys": ["meth-#0"],
            "dask_layers": ["meth-#0"],
            "dependencies": [["meth-#0", ["obj"]], ["obj", []]],
            "dependency_order": ["obj", "meth-#0"],
            "key": "meth-#0",
            "layer_order": ["obj", "meth-#0"],
            "layers": [
                [
                    "meth-#0",
                    "MaterializedLayer",
                    [["meth-#0", "Task", ["obj"], "meth"]],
                ],
                ["obj", "MaterializedLayer", [["obj", "DataNode", [], None]]],
            ],
        },
        "key": "meth-#0",
        "result_repr": "5",
        "stable_graph": None,
    },
    "method_pure": {
        "graph": {
            "dask_keys": ["meth-6d9278d378c00467d3ba03aa418b83bb"],
            "dask_layers": ["meth-6d9278d378c00467d3ba03aa418b83bb"],
            "dependencies": [
                ["meth-6d9278d378c00467d3ba03aa418b83bb", ["obj"]],
                ["obj", []],
            ],
            "dependency_order": ["obj", "meth-6d9278d378c00467d3ba03aa418b83bb"],
            "key": "meth-6d9278d378c00467d3ba03aa418b83bb",
            "layer_order": ["obj", "meth-6d9278d378c00467d3ba03aa418b83bb"],
            "layers": [
                [
                    "meth-6d9278d378c00467d3ba03aa418b83bb",
                    "MaterializedLayer",
                    [
                        [
                            "meth-6d9278d378c00467d3ba03aa418b83bb",
                            "Task",
                            ["obj"],
                            "meth",
                        ]
                    ],
                ],
                ["obj", "MaterializedLayer", [["obj", "DataNode", [], None]]],
            ],
        },
        "key": "meth-6d9278d378c00467d3ba03aa418b83bb",
        "result_repr": "5",
        "stable_graph": None,
    },
    "nout_none": {
        "graph": {
            "dask_keys": ["pair-edf96df724e6b75e10d11c374d451b9d"],
            "dask_layers": ["pair-edf96df724e6b75e10d11c374d451b9d"],
            "dependencies": [["pair-edf96df724e6b75e10d11c374d451b9d", []]],
            "dependency_order": ["pair-edf96df724e6b75e10d11c374d451b9d"],
            "key": "pair-edf96df724e6b75e10d11c374d451b9d",
            "layer_order": ["pair-edf96df724e6b75e10d11c374d451b9d"],
            "layers": [
                [
                    "pair-edf96df724e6b75e10d11c374d451b9d",
                    "MaterializedLayer",
                    [["pair-edf96df724e6b75e10d11c374d451b9d", "Task", [], "pair"]],
                ]
            ],
        },
        "key": "pair-edf96df724e6b75e10d11c374d451b9d",
        "result_repr": "(1, 2)",
        "stable_graph": None,
    },
    "nout_one": {
        "graph": {
            "dask_keys": ["single-78e46f9ea9925b4405e7a575deb859be"],
            "dask_layers": ["single-78e46f9ea9925b4405e7a575deb859be"],
            "dependencies": [["single-78e46f9ea9925b4405e7a575deb859be", []]],
            "dependency_order": ["single-78e46f9ea9925b4405e7a575deb859be"],
            "key": "single-78e46f9ea9925b4405e7a575deb859be",
            "layer_order": ["single-78e46f9ea9925b4405e7a575deb859be"],
            "layers": [
                [
                    "single-78e46f9ea9925b4405e7a575deb859be",
                    "MaterializedLayer",
                    [["single-78e46f9ea9925b4405e7a575deb859be", "Task", [], "single"]],
                ]
            ],
        },
        "key": "single-78e46f9ea9925b4405e7a575deb859be",
        "result_repr": "(7,)",
        "stable_graph": None,
    },
    "nout_one_element": {
        "graph": {
            "dask_keys": ["getitem-fa6e2fe038607dc26095f08330271412"],
            "dask_layers": ["getitem-fa6e2fe038607dc26095f08330271412"],
            "dependencies": [
                [
                    "getitem-fa6e2fe038607dc26095f08330271412",
                    ["single-78e46f9ea9925b4405e7a575deb859be"],
                ],
                ["single-78e46f9ea9925b4405e7a575deb859be", []],
            ],
            "dependency_order": [
                "single-78e46f9ea9925b4405e7a575deb859be",
                "getitem-fa6e2fe038607dc26095f08330271412",
            ],
            "key": "getitem-fa6e2fe038607dc26095f08330271412",
            "layer_order": [
                "single-78e46f9ea9925b4405e7a575deb859be",
                "getitem-fa6e2fe038607dc26095f08330271412",
            ],
            "layers": [
                [
                    "getitem-fa6e2fe038607dc26095f08330271412",
                    "MaterializedLayer",
                    [
                        [
                            "getitem-fa6e2fe038607dc26095f08330271412",
                            "Task",
                            ["single-78e46f9ea9925b4405e7a575deb859be"],
                            "getitem",
                        ]
                    ],
                ],
                [
                    "single-78e46f9ea9925b4405e7a575deb859be",
                    "MaterializedLayer",
                    [["single-78e46f9ea9925b4405e7a575deb859be", "Task", [], "single"]],
                ],
            ],
        },
        "key": "getitem-fa6e2fe038607dc26095f08330271412",
        "result_repr": "7",
        "stable_graph": None,
    },
    "nout_two": {
        "graph": {
            "dask_keys": ["pair-edf96df724e6b75e10d11c374d451b9d"],
            "dask_layers": ["pair-edf96df724e6b75e10d11c374d451b9d"],
            "dependencies": [["pair-edf96df724e6b75e10d11c374d451b9d", []]],
            "dependency_order": ["pair-edf96df724e6b75e10d11c374d451b9d"],
            "key": "pair-edf96df724e6b75e10d11c374d451b9d",
            "layer_order": ["pair-edf96df724e6b75e10d11c374d451b9d"],
            "layers": [
                [
                    "pair-edf96df724e6b75e10d11c374d451b9d",
                    "MaterializedLayer",
                    [["pair-edf96df724e6b75e10d11c374d451b9d", "Task", [], "pair"]],
                ]
            ],
        },
        "key": "pair-edf96df724e6b75e10d11c374d451b9d",
        "result_repr": "(1, 2)",
        "stable_graph": None,
    },
    "nout_two_getitem_one": {
        "graph": {
            "dask_keys": ["getitem-ffab24e2d54dfb79b8c0089b214c60db"],
            "dask_layers": ["getitem-ffab24e2d54dfb79b8c0089b214c60db"],
            "dependencies": [
                [
                    "getitem-ffab24e2d54dfb79b8c0089b214c60db",
                    ["pair-edf96df724e6b75e10d11c374d451b9d"],
                ],
                ["pair-edf96df724e6b75e10d11c374d451b9d", []],
            ],
            "dependency_order": [
                "pair-edf96df724e6b75e10d11c374d451b9d",
                "getitem-ffab24e2d54dfb79b8c0089b214c60db",
            ],
            "key": "getitem-ffab24e2d54dfb79b8c0089b214c60db",
            "layer_order": [
                "pair-edf96df724e6b75e10d11c374d451b9d",
                "getitem-ffab24e2d54dfb79b8c0089b214c60db",
            ],
            "layers": [
                [
                    "getitem-ffab24e2d54dfb79b8c0089b214c60db",
                    "MaterializedLayer",
                    [
                        [
                            "getitem-ffab24e2d54dfb79b8c0089b214c60db",
                            "Task",
                            ["pair-edf96df724e6b75e10d11c374d451b9d"],
                            "getitem",
                        ]
                    ],
                ],
                [
                    "pair-edf96df724e6b75e10d11c374d451b9d",
                    "MaterializedLayer",
                    [["pair-edf96df724e6b75e10d11c374d451b9d", "Task", [], "pair"]],
                ],
            ],
        },
        "key": "getitem-ffab24e2d54dfb79b8c0089b214c60db",
        "result_repr": "2",
        "stable_graph": None,
    },
    "nout_two_unpacked_first": {
        "graph": {
            "dask_keys": ["getitem-df5a93d04a111fdac11aa82ac1ccc98e"],
            "dask_layers": ["getitem-df5a93d04a111fdac11aa82ac1ccc98e"],
            "dependencies": [
                [
                    "getitem-df5a93d04a111fdac11aa82ac1ccc98e",
                    ["pair-edf96df724e6b75e10d11c374d451b9d"],
                ],
                ["pair-edf96df724e6b75e10d11c374d451b9d", []],
            ],
            "dependency_order": [
                "pair-edf96df724e6b75e10d11c374d451b9d",
                "getitem-df5a93d04a111fdac11aa82ac1ccc98e",
            ],
            "key": "getitem-df5a93d04a111fdac11aa82ac1ccc98e",
            "layer_order": [
                "pair-edf96df724e6b75e10d11c374d451b9d",
                "getitem-df5a93d04a111fdac11aa82ac1ccc98e",
            ],
            "layers": [
                [
                    "getitem-df5a93d04a111fdac11aa82ac1ccc98e",
                    "MaterializedLayer",
                    [
                        [
                            "getitem-df5a93d04a111fdac11aa82ac1ccc98e",
                            "Task",
                            ["pair-edf96df724e6b75e10d11c374d451b9d"],
                            "getitem",
                        ]
                    ],
                ],
                [
                    "pair-edf96df724e6b75e10d11c374d451b9d",
                    "MaterializedLayer",
                    [["pair-edf96df724e6b75e10d11c374d451b9d", "Task", [], "pair"]],
                ],
            ],
        },
        "key": "getitem-df5a93d04a111fdac11aa82ac1ccc98e",
        "result_repr": "1",
        "stable_graph": None,
    },
    "nout_two_unpacked_second": {
        "graph": {
            "dask_keys": ["getitem-ffab24e2d54dfb79b8c0089b214c60db"],
            "dask_layers": ["getitem-ffab24e2d54dfb79b8c0089b214c60db"],
            "dependencies": [
                [
                    "getitem-ffab24e2d54dfb79b8c0089b214c60db",
                    ["pair-edf96df724e6b75e10d11c374d451b9d"],
                ],
                ["pair-edf96df724e6b75e10d11c374d451b9d", []],
            ],
            "dependency_order": [
                "pair-edf96df724e6b75e10d11c374d451b9d",
                "getitem-ffab24e2d54dfb79b8c0089b214c60db",
            ],
            "key": "getitem-ffab24e2d54dfb79b8c0089b214c60db",
            "layer_order": [
                "pair-edf96df724e6b75e10d11c374d451b9d",
                "getitem-ffab24e2d54dfb79b8c0089b214c60db",
            ],
            "layers": [
                [
                    "getitem-ffab24e2d54dfb79b8c0089b214c60db",
                    "MaterializedLayer",
                    [
                        [
                            "getitem-ffab24e2d54dfb79b8c0089b214c60db",
                            "Task",
                            ["pair-edf96df724e6b75e10d11c374d451b9d"],
                            "getitem",
                        ]
                    ],
                ],
                [
                    "pair-edf96df724e6b75e10d11c374d451b9d",
                    "MaterializedLayer",
                    [["pair-edf96df724e6b75e10d11c374d451b9d", "Task", [], "pair"]],
                ],
            ],
        },
        "key": "getitem-ffab24e2d54dfb79b8c0089b214c60db",
        "result_repr": "2",
        "stable_graph": None,
    },
    "nout_zero": {
        "graph": {
            "dask_keys": ["nothing-16d1ed9214c24df845518c080dc5dba0"],
            "dask_layers": ["nothing-16d1ed9214c24df845518c080dc5dba0"],
            "dependencies": [["nothing-16d1ed9214c24df845518c080dc5dba0", []]],
            "dependency_order": ["nothing-16d1ed9214c24df845518c080dc5dba0"],
            "key": "nothing-16d1ed9214c24df845518c080dc5dba0",
            "layer_order": ["nothing-16d1ed9214c24df845518c080dc5dba0"],
            "layers": [
                [
                    "nothing-16d1ed9214c24df845518c080dc5dba0",
                    "MaterializedLayer",
                    [
                        [
                            "nothing-16d1ed9214c24df845518c080dc5dba0",
                            "Task",
                            [],
                            "nothing",
                        ]
                    ],
                ]
            ],
        },
        "key": "nothing-16d1ed9214c24df845518c080dc5dba0",
        "result_repr": "()",
        "stable_graph": None,
    },
    "op_add": {
        "graph": {
            "dask_keys": ["add-35cee928b5a34cf2c76c77e3106dd858"],
            "dask_layers": ["add-35cee928b5a34cf2c76c77e3106dd858"],
            "dependencies": [
                [
                    "add-35cee928b5a34cf2c76c77e3106dd858",
                    [
                        "inc-2f16cb15cc39d4668d79bfb484510bbb",
                        "inc-5852f565112f1604bc52264ea5e287dc",
                    ],
                ],
                ["inc-2f16cb15cc39d4668d79bfb484510bbb", []],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "add-35cee928b5a34cf2c76c77e3106dd858",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "inc-2f16cb15cc39d4668d79bfb484510bbb",
            ],
            "key": "add-35cee928b5a34cf2c76c77e3106dd858",
            "layer_order": [
                "add-35cee928b5a34cf2c76c77e3106dd858",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "inc-2f16cb15cc39d4668d79bfb484510bbb",
            ],
            "layers": [
                [
                    "add-35cee928b5a34cf2c76c77e3106dd858",
                    "MaterializedLayer",
                    [
                        [
                            "add-35cee928b5a34cf2c76c77e3106dd858",
                            "Task",
                            [
                                "inc-2f16cb15cc39d4668d79bfb484510bbb",
                                "inc-5852f565112f1604bc52264ea5e287dc",
                            ],
                            "add",
                        ]
                    ],
                ],
                [
                    "inc-2f16cb15cc39d4668d79bfb484510bbb",
                    "MaterializedLayer",
                    [["inc-2f16cb15cc39d4668d79bfb484510bbb", "Task", [], "inc"]],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "add-35cee928b5a34cf2c76c77e3106dd858",
        "result_repr": "5",
        "stable_graph": None,
    },
    "op_getitem": {
        "graph": {
            "dask_keys": ["getitem-5de6b0a227b247a33d30ed9e054db1c6"],
            "dask_layers": ["getitem-5de6b0a227b247a33d30ed9e054db1c6"],
            "dependencies": [
                [
                    "getitem-5de6b0a227b247a33d30ed9e054db1c6",
                    ["listof-820c86bc186a713b9489ff050ac57426"],
                ],
                ["listof-820c86bc186a713b9489ff050ac57426", []],
            ],
            "dependency_order": [
                "listof-820c86bc186a713b9489ff050ac57426",
                "getitem-5de6b0a227b247a33d30ed9e054db1c6",
            ],
            "key": "getitem-5de6b0a227b247a33d30ed9e054db1c6",
            "layer_order": [
                "listof-820c86bc186a713b9489ff050ac57426",
                "getitem-5de6b0a227b247a33d30ed9e054db1c6",
            ],
            "layers": [
                [
                    "getitem-5de6b0a227b247a33d30ed9e054db1c6",
                    "MaterializedLayer",
                    [
                        [
                            "getitem-5de6b0a227b247a33d30ed9e054db1c6",
                            "Task",
                            ["listof-820c86bc186a713b9489ff050ac57426"],
                            "getitem",
                        ]
                    ],
                ],
                [
                    "listof-820c86bc186a713b9489ff050ac57426",
                    "MaterializedLayer",
                    [["listof-820c86bc186a713b9489ff050ac57426", "Task", [], "listof"]],
                ],
            ],
        },
        "key": "getitem-5de6b0a227b247a33d30ed9e054db1c6",
        "result_repr": "2",
        "stable_graph": None,
    },
    "op_lt": {
        "graph": {
            "dask_keys": ["lt-f5c6fcf3a5a1622c4b88dfc936751cb3"],
            "dask_layers": ["lt-f5c6fcf3a5a1622c4b88dfc936751cb3"],
            "dependencies": [
                ["inc-2f16cb15cc39d4668d79bfb484510bbb", []],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
                [
                    "lt-f5c6fcf3a5a1622c4b88dfc936751cb3",
                    [
                        "inc-2f16cb15cc39d4668d79bfb484510bbb",
                        "inc-5852f565112f1604bc52264ea5e287dc",
                    ],
                ],
            ],
            "dependency_order": [
                "lt-f5c6fcf3a5a1622c4b88dfc936751cb3",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "inc-2f16cb15cc39d4668d79bfb484510bbb",
            ],
            "key": "lt-f5c6fcf3a5a1622c4b88dfc936751cb3",
            "layer_order": [
                "lt-f5c6fcf3a5a1622c4b88dfc936751cb3",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "inc-2f16cb15cc39d4668d79bfb484510bbb",
            ],
            "layers": [
                [
                    "inc-2f16cb15cc39d4668d79bfb484510bbb",
                    "MaterializedLayer",
                    [["inc-2f16cb15cc39d4668d79bfb484510bbb", "Task", [], "inc"]],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
                [
                    "lt-f5c6fcf3a5a1622c4b88dfc936751cb3",
                    "MaterializedLayer",
                    [
                        [
                            "lt-f5c6fcf3a5a1622c4b88dfc936751cb3",
                            "Task",
                            [
                                "inc-2f16cb15cc39d4668d79bfb484510bbb",
                                "inc-5852f565112f1604bc52264ea5e287dc",
                            ],
                            "lt",
                        ]
                    ],
                ],
            ],
        },
        "key": "lt-f5c6fcf3a5a1622c4b88dfc936751cb3",
        "result_repr": "True",
        "stable_graph": None,
    },
    "op_neg": {
        "graph": {
            "dask_keys": ["neg-df7b95d9f5b8b840ed79dc3297c50e01"],
            "dask_layers": ["neg-df7b95d9f5b8b840ed79dc3297c50e01"],
            "dependencies": [
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
                [
                    "neg-df7b95d9f5b8b840ed79dc3297c50e01",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "neg-df7b95d9f5b8b840ed79dc3297c50e01",
            ],
            "key": "neg-df7b95d9f5b8b840ed79dc3297c50e01",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "neg-df7b95d9f5b8b840ed79dc3297c50e01",
            ],
            "layers": [
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
                [
                    "neg-df7b95d9f5b8b840ed79dc3297c50e01",
                    "MaterializedLayer",
                    [
                        [
                            "neg-df7b95d9f5b8b840ed79dc3297c50e01",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "neg",
                        ]
                    ],
                ],
            ],
        },
        "key": "neg-df7b95d9f5b8b840ed79dc3297c50e01",
        "result_repr": "-2",
        "stable_graph": None,
    },
    "op_reflected_add": {
        "graph": {
            "dask_keys": ["_swap-fe9fc34b1482c4119fa8e0dedc089afc"],
            "dask_layers": ["_swap-fe9fc34b1482c4119fa8e0dedc089afc"],
            "dependencies": [
                [
                    "_swap-fe9fc34b1482c4119fa8e0dedc089afc",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "_swap-fe9fc34b1482c4119fa8e0dedc089afc",
            ],
            "key": "_swap-fe9fc34b1482c4119fa8e0dedc089afc",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "_swap-fe9fc34b1482c4119fa8e0dedc089afc",
            ],
            "layers": [
                [
                    "_swap-fe9fc34b1482c4119fa8e0dedc089afc",
                    "MaterializedLayer",
                    [
                        [
                            "_swap-fe9fc34b1482c4119fa8e0dedc089afc",
                            "Task",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "_swap",
                        ]
                    ],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "_swap-fe9fc34b1482c4119fa8e0dedc089afc",
        "result_repr": "3",
        "stable_graph": None,
    },
    "traverse_false_with_delayed": {
        "graph": {
            "dask_keys": ["quoted"],
            "dask_layers": ["quoted"],
            "dependencies": [["quoted", []]],
            "dependency_order": ["quoted"],
            "key": "quoted",
            "layer_order": ["quoted"],
            "layers": [
                ["quoted", "MaterializedLayer", [["quoted", "legacy-tuple", [], None]]]
            ],
        },
        "key": "quoted",
        "result_repr": "[Delayed('inc-5852f565112f1604bc52264ea5e287dc'), 1]",
        "stable_graph": None,
    },
    "wrap_datanode": {
        "graph": {
            "dask_keys": ["dn"],
            "dask_layers": ["dn"],
            "dependencies": [["dn", []]],
            "dependency_order": ["dn"],
            "key": "dn",
            "layer_order": ["dn"],
            "layers": [["dn", "MaterializedLayer", [["dn", "DataNode", [], None]]]],
        },
        "key": "dn",
        "result_repr": "5",
        "stable_graph": None,
    },
    "wrap_dict_with_delayed": {
        "graph": {
            "dask_keys": ["dict-060816f9d1dbe6ff4ba5f8a57d4c94ab"],
            "dask_layers": ["dict-060816f9d1dbe6ff4ba5f8a57d4c94ab"],
            "dependencies": [
                [
                    "dict-060816f9d1dbe6ff4ba5f8a57d4c94ab",
                    [
                        "inc-2f16cb15cc39d4668d79bfb484510bbb",
                        "inc-5852f565112f1604bc52264ea5e287dc",
                    ],
                ],
                ["inc-2f16cb15cc39d4668d79bfb484510bbb", []],
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
            ],
            "dependency_order": [
                "dict-060816f9d1dbe6ff4ba5f8a57d4c94ab",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "inc-2f16cb15cc39d4668d79bfb484510bbb",
            ],
            "key": "dict-060816f9d1dbe6ff4ba5f8a57d4c94ab",
            "layer_order": [
                "dict-060816f9d1dbe6ff4ba5f8a57d4c94ab",
                "inc-5852f565112f1604bc52264ea5e287dc",
                "inc-2f16cb15cc39d4668d79bfb484510bbb",
            ],
            "layers": [
                [
                    "dict-060816f9d1dbe6ff4ba5f8a57d4c94ab",
                    "MaterializedLayer",
                    [
                        [
                            "dict-060816f9d1dbe6ff4ba5f8a57d4c94ab",
                            "Dict",
                            [
                                "inc-2f16cb15cc39d4668d79bfb484510bbb",
                                "inc-5852f565112f1604bc52264ea5e287dc",
                            ],
                            "to_container",
                        ]
                    ],
                ],
                [
                    "inc-2f16cb15cc39d4668d79bfb484510bbb",
                    "MaterializedLayer",
                    [["inc-2f16cb15cc39d4668d79bfb484510bbb", "Task", [], "inc"]],
                ],
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
            ],
        },
        "key": "dict-060816f9d1dbe6ff4ba5f8a57d4c94ab",
        "result_repr": "{'k': 2, 'j': 3}",
        "stable_graph": None,
    },
    "wrap_func_pure": {
        "graph": {
            "dask_keys": ["inc-303ca154fe489de9eadb7a14143fd6fd"],
            "dask_layers": ["inc-303ca154fe489de9eadb7a14143fd6fd"],
            "dependencies": [["inc-303ca154fe489de9eadb7a14143fd6fd", []]],
            "dependency_order": ["inc-303ca154fe489de9eadb7a14143fd6fd"],
            "key": "inc-303ca154fe489de9eadb7a14143fd6fd",
            "layer_order": ["inc-303ca154fe489de9eadb7a14143fd6fd"],
            "layers": [
                [
                    "inc-303ca154fe489de9eadb7a14143fd6fd",
                    "MaterializedLayer",
                    [["inc-303ca154fe489de9eadb7a14143fd6fd", "DataNode", [], None]],
                ]
            ],
        },
        "key": "inc-303ca154fe489de9eadb7a14143fd6fd",
        "result_repr": None,
        "stable_graph": None,
    },
    "wrap_int_default": {
        "graph": {
            "dask_keys": ["int-#0"],
            "dask_layers": ["int-#0"],
            "dependencies": [["int-#0", []]],
            "dependency_order": ["int-#0"],
            "key": "int-#0",
            "layer_order": ["int-#0"],
            "layers": [
                ["int-#0", "MaterializedLayer", [["int-#0", "DataNode", [], None]]]
            ],
        },
        "key": "int-#0",
        "result_repr": "3",
        "stable_graph": None,
    },
    "wrap_int_named": {
        "graph": {
            "dask_keys": ["three"],
            "dask_layers": ["three"],
            "dependencies": [["three", []]],
            "dependency_order": ["three"],
            "key": "three",
            "layer_order": ["three"],
            "layers": [
                ["three", "MaterializedLayer", [["three", "DataNode", [], None]]]
            ],
        },
        "key": "three",
        "result_repr": "3",
        "stable_graph": None,
    },
    "wrap_list_traverse_false_default": {
        "graph": {
            "dask_keys": ["list-#0"],
            "dask_layers": ["list-#0"],
            "dependencies": [["list-#0", []]],
            "dependency_order": ["list-#0"],
            "key": "list-#0",
            "layer_order": ["list-#0"],
            "layers": [
                [
                    "list-#0",
                    "MaterializedLayer",
                    [["list-#0", "legacy-tuple", [], None]],
                ]
            ],
        },
        "key": "list-#0",
        "result_repr": "[1, 2]",
        "stable_graph": None,
    },
    "wrap_list_with_delayed": {
        "graph": {
            "dask_keys": ["list-22dc33baac1c077dd729a2a207ff908e"],
            "dask_layers": ["list-22dc33baac1c077dd729a2a207ff908e"],
            "dependencies": [
                ["inc-5852f565112f1604bc52264ea5e287dc", []],
                [
                    "list-22dc33baac1c077dd729a2a207ff908e",
                    ["inc-5852f565112f1604bc52264ea5e287dc"],
                ],
            ],
            "dependency_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "list-22dc33baac1c077dd729a2a207ff908e",
            ],
            "key": "list-22dc33baac1c077dd729a2a207ff908e",
            "layer_order": [
                "inc-5852f565112f1604bc52264ea5e287dc",
                "list-22dc33baac1c077dd729a2a207ff908e",
            ],
            "layers": [
                [
                    "inc-5852f565112f1604bc52264ea5e287dc",
                    "MaterializedLayer",
                    [["inc-5852f565112f1604bc52264ea5e287dc", "Task", [], "inc"]],
                ],
                [
                    "list-22dc33baac1c077dd729a2a207ff908e",
                    "MaterializedLayer",
                    [
                        [
                            "list-22dc33baac1c077dd729a2a207ff908e",
                            "List",
                            ["inc-5852f565112f1604bc52264ea5e287dc"],
                            "to_container",
                        ]
                    ],
                ],
            ],
        },
        "key": "list-22dc33baac1c077dd729a2a207ff908e",
        "result_repr": "[2, 1]",
        "stable_graph": None,
    },
    "wrap_obj_pure": {
        "graph": {
            "dask_keys": ["Obj-7fce9a01b4683ffac192d67366158c60"],
            "dask_layers": ["Obj-7fce9a01b4683ffac192d67366158c60"],
            "dependencies": [["Obj-7fce9a01b4683ffac192d67366158c60", []]],
            "dependency_order": ["Obj-7fce9a01b4683ffac192d67366158c60"],
            "key": "Obj-7fce9a01b4683ffac192d67366158c60",
            "layer_order": ["Obj-7fce9a01b4683ffac192d67366158c60"],
            "layers": [
                [
                    "Obj-7fce9a01b4683ffac192d67366158c60",
                    "MaterializedLayer",
                    [["Obj-7fce9a01b4683ffac192d67366158c60", "DataNode", [], None]],
                ]
            ],
        },
        "key": "Obj-7fce9a01b4683ffac192d67366158c60",
        "result_repr": "Obj(v=3)",
        "stable_graph": None,
    },
    "wrap_str_pure": {
        "graph": {
            "dask_keys": ["str-2f63bc91734514c0b208c7332139c210"],
            "dask_layers": ["str-2f63bc91734514c0b208c7332139c210"],
            "dependencies": [["str-2f63bc91734514c0b208c7332139c210", []]],
            "dependency_order": ["str-2f63bc91734514c0b208c7332139c210"],
            "key": "str-2f63bc91734514c0b208c7332139c210",
            "layer_order": ["str-2f63bc91734514c0b208c7332139c210"],
            "layers": [
                [
                    "str-2f63bc91734514c0b208c7332139c210",
                    "MaterializedLayer",
                    [["str-2f63bc91734514c0b208c7332139c210", "DataNode", [], None]],
                ]
            ],
        },
        "key": "str-2f63bc91734514c0b208c7332139c210",
        "result_repr": "'s'",
        "stable_graph": None,
    },
    "wrap_taskref": {
        "graph": {
            "dask_keys": ["dn"],
            "dask_layers": ["dn"],
            "dependencies": [["dn", []]],
            "dependency_order": ["dn"],
            "key": "dn",
            "layer_order": ["dn"],
            "layers": [["dn", "MaterializedLayer", [["dn", "TaskRef", ["dn"], None]]]],
        },
        "key": "dn",
        "result_repr": None,
        "stable_graph": None,
    },
}
# --- END GOLDEN ---


#: Length of a bare dask token: 32 lowercase hex characters, which is both what
#: ``tokenize`` produces and what ``uuid.uuid4().hex`` produces. The two cannot be
#: told apart by shape, which is why the random ones are identified by observing
#: two builds rather than by matching a pattern.
_HEX_TOKEN_LEN = 32
_HEX_DIGITS = frozenset("0123456789abcdef")


def _walk_strings(value: Any) -> Iterator[str]:
    """Yield every string of a canonical form, in a deterministic order.

    Args:
        value: A canonical-form fragment: a ``dict``, a ``list`` or a leaf.

    Yields:
        str: each string leaf, dicts in insertion order and lists in order, so
        that two canonical forms of the same shape yield their strings in
        corresponding positions.

    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)


def _hex_tokens(graph: dict[str, Any]) -> list[str]:
    """Return the 32-hex tokens of a canonical form, first appearance first.

    Args:
        graph: A canonical graph dict.

    Returns:
        list[str]: every distinct hyphen-delimited run of exactly 32 hex digits,
        in the order the deterministic walk first reaches it. Both a ``tokenize``
        digest and a ``uuid4().hex`` fallback have that shape; which is which is
        decided by comparing two builds, not here.

    """
    tokens: list[str] = []
    for text in _walk_strings(graph):
        for part in text.split("-"):
            if (
                len(part) == _HEX_TOKEN_LEN
                and _HEX_DIGITS.issuperset(part)
                and part not in tokens
            ):
                tokens.append(part)
    return tokens


def _substitute(value: Any, replacements: dict[str, str]) -> Any:
    """Replace token substrings throughout a canonical form.

    Args:
        value: A canonical-form fragment.
        replacements: Token to placeholder.

    Returns:
        Any: the same structure with every occurrence of every token replaced,
        and nothing else touched.

    """
    if isinstance(value, str):
        for token, placeholder in replacements.items():
            value = value.replace(token, placeholder)
        return value
    if isinstance(value, list):
        return [_substitute(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _substitute(item, replacements) for key, item in value.items()}
    return value


def _stable_graph(entry: _Expr) -> dict[str, Any]:
    """Canonicalise a ``volatile_tokens`` entry into a reproducible form.

    Some graphs carry an identifier that is random by construction:
    ``dask._expr.HLGExpr.deterministic_token`` and
    ``dask._expr.ProhibitReuse._suffix`` fall back to ``uuid.uuid4().hex``, and an
    expression reached through a non-``Delayed`` collection picks one up. A bare
    32-hex string cannot be recognised as random by shape -- a deterministic
    token looks exactly the same -- so this function establishes which tokens are
    random by *observation*: it builds the expression twice and replaces only the
    tokens that actually changed, leaving every deterministic token, key, node
    kind, dependency and insertion order exactly as canonicalised. That keeps the
    whole graph under assertion for a path whose raw canonical dict could
    otherwise not be compared at all.

    Args:
        entry: A corpus entry whose ``volatile_tokens`` flag is set.

    Returns:
        dict: the canonical graph with the ``i``-th random token replaced by
        ``f"<hex{i}>"``.

    """
    first = canonical_graph(entry.build())
    second = canonical_graph(entry.build())
    first_tokens = _hex_tokens(first)
    second_tokens = _hex_tokens(second)
    assert len(first_tokens) == len(second_tokens), (
        f"{entry.name}: two builds produced different numbers of 32-hex tokens "
        f"({len(first_tokens)} and {len(second_tokens)}), so the graph differs by more "
        "than its random identifiers"
    )
    volatile = [
        (before, after)
        for before, after in zip(first_tokens, second_tokens)
        if before != after
    ]
    assert volatile, (
        f"{entry.name} is flagged volatile_tokens but two builds produced identical "
        "tokens; if the graph has become reproducible, drop the flag and record the "
        "raw canonical form instead"
    )
    normalised = _substitute(
        first, {before: f"<hex{i}>" for i, (before, _) in enumerate(volatile)}
    )
    assert normalised == _substitute(
        second, {after: f"<hex{i}>" for i, (_, after) in enumerate(volatile)}
    ), (
        f"{entry.name}: the two builds still differ once their random tokens are "
        "replaced, so something other than a random identifier moved"
    )
    return normalised


def _hex_key_token(name: str, key: object) -> str:
    """Return the deterministic 32-hex token of a generated key.

    Args:
        name: The corpus entry the key belongs to, for the failure message.
        key: The key to split at its last ``"-"``.

    Returns:
        str: the token following that hyphen.

    Raises:
        AssertionError: if the key is not a ``str`` carrying a prefix and a
            32-hex token. Refusing is the point: an environment-dependent entry
            whose token has stopped being a deterministic digest -- a UUID
            fallback above all -- must fail here rather than be masked into
            agreement with the golden.

    """
    assert isinstance(key, str), f"{name}: expected a str key, got {type(key).__name__}"
    prefix, _, token = key.rpartition("-")
    assert prefix, f"{name}: key {key!r} carries no prefix"
    assert len(token) == _HEX_TOKEN_LEN and _HEX_DIGITS.issuperset(
        token
    ), f"{name}: key {key!r} does not end in a deterministic 32-hex token"
    return token


def _token_masker(name: str, key: object) -> Callable[[Any], Any]:
    """Return the transform that makes one entry's canonical forms portable.

    For all but the entries registered in ``_ENVIRONMENT_DEPENDENT_KEY_TOKEN``
    this is the identity, so their keys and graphs stay under verbatim
    comparison. For a registered entry it replaces that entry's own key token --
    one token, and only where it occurs -- with ``_ENV_TOKEN_PLACEHOLDER``.
    Applying it to the live form with the live key and to the golden form with
    the golden key leaves the two comparable on any interpreter and under any
    installed hash library, while every other token in both forms is still
    compared as it stands.

    Args:
        name: Corpus entry name, looked up in the registry.
        key: The key whose token is to be masked -- the live key for a live
            form, the golden key for a golden form.

    Returns:
        Callable: a transform over a canonical form (a ``dict``, a ``list`` or a
        bare key), returning the same structure with nothing but that token
        replaced.

    """
    if name not in _ENVIRONMENT_DEPENDENT_KEY_TOKEN:

        def keep(form: Any) -> Any:
            return form

        return keep

    replacements = {_hex_key_token(name, key): _ENV_TOKEN_PLACEHOLDER}

    def mask(form: Any) -> Any:
        return _substitute(form, replacements)

    return mask


def _golden_names(names: Sequence[Any]) -> list[Any]:
    """Live graph names in the form the canonical golden fields record them.

    ``canonical_graph`` records a key JSON can express as itself and encodes any
    other key -- a ``tuple`` above all -- as its ``repr``
    (``benchmarks/delayed_ab/canon.py``). The live ``HighLevelGraph`` holds the
    keys themselves, so the same rule is applied before the two are compared. For
    the string keys that make up most of the corpus it changes nothing, and for
    the ``tuple``-keyed entry it is what lets the live insertion order be checked
    against the golden at all.

    Args:
        names: Layer names or graph keys, in their live order.

    Returns:
        list: the same sequence with every non-JSON-expressible name replaced by
        its ``repr``.

    """
    coerced: list[Any] = []
    for name in names:
        typ = type(name)
        if name is None or typ is bool or typ is int or typ is float or typ is str:
            coerced.append(name)
        else:
            coerced.append(repr(name))
    return coerced


def _golden_entry(entry: _Expr) -> dict[str, Any]:
    """Build one corpus entry and return the four fields the golden records.

    Args:
        entry: The corpus entry to characterise.

    Returns:
        dict: ``{"key": ..., "graph": ..., "stable_graph": ..., "result_repr":
        ...}``. ``graph`` is the full ``canonical_graph`` dict, or ``None`` for an
        entry whose raw canonical form is not reproducible -- for which
        ``stable_graph`` carries the same dict with only its observedly-random
        32-hex tokens replaced, so that the graph is asserted either way.
        ``result_repr`` is the ``repr`` of the synchronously computed result, or
        ``None`` for an entry that cannot be computed, whose failure is asserted
        from ``_COMPUTE_FAILURES`` instead. Every exclusion is justified by name
        in ``_EXCLUSION_REASONS``.

    ``repr`` rather than the value itself, because a module-level dict literal
    can hold only source-representable values. The ordinary ``==`` comparison of
    results is done by the cross-scheduler and pickle tests through
    ``canonical_result``.

    """
    obj = entry.build()
    table: dict[str, str] = {}
    return {
        "key": normalize_key(obj.key, table),
        "graph": canonical_graph(obj) if entry.canonical else None,
        "stable_graph": _stable_graph(entry) if entry.volatile_tokens else None,
        "result_repr": repr(canonical_result([obj])[0]) if entry.computable else None,
    }


def _golden_block_bounds(lines: list[str]) -> tuple[int, int]:
    """Locate the golden block in this module's own source.

    Args:
        lines: The module source, split with line endings kept.

    Returns:
        tuple: ``(begin, end)``, the indices of the two marker lines.

    Raises:
        RuntimeError: if either marker is missing or they appear out of order.
            Refusing to write is the point: a golden fixture appended to the
            wrong place, or silently dropped, is worse than no capture at all.

    """
    begin: int | None = None
    end: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == _GOLDEN_BEGIN:
            begin = index
        elif stripped == _GOLDEN_END:
            end = index
    if begin is None or end is None or end <= begin:
        raise RuntimeError(
            "golden markers not found in "
            f"{Path(__file__).name} (begin={begin}, end={end}); refusing to write"
        )
    return begin, end


def write_golden() -> None:
    """Capture the golden fixture from the currently importable ``dask.delayed``.

    Builds every corpus entry under the same pins the autouse fixture applies,
    records the four golden fields per entry, and rewrites *only* the text
    between the two marker comments in this module's own source, leaving every
    other byte untouched.

    Run it from the repository root as::

        python -c "from dask.tests.test_delayed_equivalence import write_golden; write_golden()"

    Never as ``__main__``: functions and classes defined in a ``__main__`` module
    pickle by value as ``__main__.*``, so the pickle-sensitive tokens captured
    that way would not be the ones pytest sees.

    Raises:
        RuntimeError: if the golden markers cannot be located.

    """
    with _pinned():
        golden = {entry.name: _golden_entry(entry) for entry in CORPUS}
    path = Path(__file__)
    lines = path.read_text().splitlines(keepends=True)
    begin, end = _golden_block_bounds(lines)
    # A width far beyond any value's length on purpose: ``pprint`` splits a
    # string that would overflow the width into implicitly concatenated literals,
    # which ``ruff``'s ISC rules reject. Emitting wide lines and letting ``black``
    # wrap the structure -- it never splits a string -- keeps the generated block
    # lint-clean. ``sort_dicts`` makes the output stable and reviewable.
    literal = pprint.pformat(
        dict(sorted(golden.items())), indent=4, width=10_000, sort_dicts=True
    )
    block = [
        lines[begin],
        f"GOLDEN: dict[str, dict[str, Any]] = {literal}\n",
        lines[end],
    ]
    path.write_text("".join(lines[:begin] + block + lines[end + 1 :]))
    print(f"captured {len(golden)} golden entries into {path}")


# ---------------------------------------------------------------------------
# Provenance of the golden fixture, re-derived rather than read out of git.
#
# AAP §0.5.1 asks two things of the block above: that its values came from the
# pre-refactor module, and that the capture happened before the first edit to
# ``dask/delayed.py``. The second was to be read off this file's git history, and
# on this branch it cannot be -- the block was re-emitted after that edit, the
# branch is published, and rewriting its history is forbidden. What that
# re-emission did and did not change is set out in the module docstring under
# "Provenance of the golden block"; what stands in place of the missing history
# read is here, and it is re-run by every suite run rather than inspected once.
#
# The frozen baseline arm -- ``dask/delayed.py`` as it stood at the base commit,
# copied into ``benchmarks/delayed_ab/baseline_delayed.py`` in a single commit
# made before the first production edit -- rebuilds the whole corpus, and every
# committed golden value is compared against what that module produces. A golden
# regenerated from the refactored module would disagree, which is the guarantee
# the ordering requirement existed to give.
# ---------------------------------------------------------------------------

#: Repository root: ``dask/tests/`` sits two directories below it. The
#: re-derivation subprocess runs there because ``benchmarks`` is a PEP 420
#: namespace package, importable from the root and nowhere else.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Arm A of the A/B performance suite, used here as the pre-refactor oracle.
_BASELINE_ARM_MODULE = "benchmarks.delayed_ab.baseline_delayed"
_BASELINE_ARM_PATH = _REPO_ROOT / "benchmarks" / "delayed_ab" / "baseline_delayed.py"

#: The base commit the arm was captured from, and the ``sha256`` of
#: ``dask/delayed.py`` there. Recorded here independently of the arm's own header
#: comment, so that an edited oracle fails this module rather than lending its
#: credibility to the re-derivation below. Re-derive it from the repository root
#: with ``git show c9d1df34ccba182ddf43c2dbe4315c4d9c8c44e1:dask/delayed.py |
#: sha256sum``.
_BASE_COMMIT = "c9d1df34ccba182ddf43c2dbe4315c4d9c8c44e1"
_BASELINE_ARM_BODY_SHA256 = (
    "4c0000e204ea5b701cbef0879f6af74e3b547249edede4a1003ef4d78493d6f1"
)

#: The two golden entries the frozen arm cannot reproduce character for
#: character, each with the construct that makes it arm-dependent. Both are the
#: pickle-by-reference case AAP §0.4.1 documents: a token computed by pickling
#: something by reference embeds the defining module's name, which is
#: ``benchmarks.delayed_ab.baseline_delayed`` on one arm and ``dask.delayed`` on
#: the other, so the two arms necessarily produce different deterministic tokens
#: from identical code. The difference is one token wide -- their own key's -- and
#: that it is no wider is asserted rather than assumed
#: (``test_the_golden_re_derives_from_the_frozen_baseline_arm`` compares them with
#: that one token set aside and finds no other difference, including in the value
#: each expression computes to).
_ARM_DEPENDENT_GOLDEN_ENTRIES: dict[str, str] = {
    "arg_list_iterator_with_delayed": (
        "call_function tokenizes the raw iterator before unpack_collections coerces "
        "it, so tokenize pickles the underlying list -- including the Delayed it "
        "holds, whose class pickles by reference to the module that defined it"
    ),
    "op_reflected_add": (
        "a reflected operator tokenizes right(op) -- partial(_swap, op) -- and a "
        "module-level function pickles by reference as <module>._swap, so the key "
        "token embeds the arm's own module path while the _swap prefix does not"
    ),
}

#: What an arm-dependent key token is replaced by while the two arms are
#: compared. It holds characters no generated key can, so a masked form can never
#: be mistaken for a real one, and it is distinct from ``_ENV_TOKEN_PLACEHOLDER``
#: and from ``_stable_graph``'s ``<hexN>`` so that the three relaxations stay
#: distinguishable in a failure message.
_ARM_TOKEN_PLACEHOLDER = "<arm-token>"

#: Whether this environment reproduces both kinds of token the capture
#: environment decided, and therefore whether the verbatim half of the
#: re-derivation can be asserted here at all.
_CAPTURE_ENVIRONMENT_REPRODUCES: bool = (
    _HASHER_REPRODUCES_CAPTURE and _INTERPRETER_REPRODUCES_CAPTURE
)

#: Skip reason for the verbatim half, naming the causes this environment does not
#: reproduce. Read on its own in a ``-rs`` summary, like every other skip here.
_NON_CAPTURE_ENVIRONMENT_REASON = (
    "the character-for-character half of the baseline-arm re-derivation needs the "
    "environment the golden was captured in, because twelve entries carry a key "
    "token the environment rather than the module decides: "
    + "; ".join(
        cause.reason for cause in _TOKEN_CAUSES.values() if not cause.reproduces_capture
    )
)

#: How long the re-derivation subprocess may take. It builds ninety small
#: expressions in a fresh interpreter -- under a second in the environment of
#: record -- so this is a fail-fast bound rather than a budget, and it stays well
#: inside the project's 300 s per-test limit.
_REDERIVATION_TIMEOUT = 120

#: The program the subprocess runs. The arm has to be installed as
#: ``sys.modules["dask.delayed"]`` -- AAP §0.4.1's arm activation -- *before* this
#: module is imported, because the ``from dask.delayed import ...`` at the top of
#: this file is what binds the classes under test; hence a fresh interpreter
#: rather than a context manager. This module is imported under its own canonical
#: name so that its corpus callables still pickle by reference to the very paths
#: the capture saw, exactly as ``write_golden()`` requires of the capture.
_REDERIVATION_BOOTSTRAP = (
    "import importlib, sys; "
    f"sys.modules['dask.delayed'] = importlib.import_module({_BASELINE_ARM_MODULE!r}); "
    f"importlib.import_module({__name__!r})._emit_golden_rederivation()"
)


def _baseline_arm_body_sha256() -> str:
    """Return the ``sha256`` of the frozen arm's captured body.

    The arm is ``dask/delayed.py`` at ``_BASE_COMMIT`` with a leading comment
    header prepended, so the digest is taken over the text from the first line
    that is neither blank nor a comment -- the file's own ``from __future__ import
    annotations`` -- to the end, which is the captured source and nothing else.

    Returns:
        str: the hex digest, comparable with ``_BASELINE_ARM_BODY_SHA256``.

    """
    lines = _BASELINE_ARM_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    start = 0
    while start < len(lines) and (
        not lines[start].strip() or lines[start].startswith("#")
    ):
        start += 1
    body = "".join(lines[start:])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _rederivation_masker(name: str, key: object) -> Callable[[Any], Any]:
    """Return the transform that makes one entry comparable across the two arms.

    ``_token_masker`` sets aside the key token of the twelve entries whose token
    the *environment* decides. This sets aside those and, in addition, the key
    token of the two entries whose token the *arm* decides
    (``_ARM_DEPENDENT_GOLDEN_ENTRIES``). For every other entry it is
    ``_token_masker``'s verdict, which is the identity, so their keys, graphs and
    results stay under verbatim comparison.

    Args:
        name: Corpus entry name, looked up in both registries.
        key: The key whose token is to be set aside -- the golden key for a golden
            form, the re-derived key for a re-derived one.

    Returns:
        Callable: a transform over a canonical form (a ``dict``, a ``list`` or a
        bare key), returning the same structure with nothing but that one token
        replaced.

    """
    if name not in _ARM_DEPENDENT_GOLDEN_ENTRIES:
        return _token_masker(name, key)

    replacements = {_hex_key_token(name, key): _ARM_TOKEN_PLACEHOLDER}

    def mask(form: Any) -> Any:
        return _substitute(form, replacements)

    return mask


def _rederive_golden() -> dict[str, Any]:
    """Rebuild every corpus entry with whatever module ``dask.delayed`` now names.

    Run in the subprocess of ``_REDERIVATION_BOOTSTRAP``, that module is the
    frozen baseline arm, so this rebuilds the corpus on the pre-refactor
    construction path -- under the same hasher and configuration pins the capture
    and every test here use -- and compares each entry against the committed
    golden twice: verbatim, and with the key tokens ``_rederivation_masker`` sets
    aside.

    The comparison happens here rather than in the parent so that no graph has to
    travel: a canonical graph is JSON-expressible, but a round trip through JSON
    would turn its tuples into lists and compare something weaker than what was
    captured. Only names, field names and keys go into the payload.

    An entry that raises is recorded with its traceback instead of aborting the
    run, because a payload naming the one broken entry is worth more to whoever
    reads the failure than a subprocess that died on it.

    Returns:
        dict: ``delayed_module`` (the ``__module__`` of the ``Delayed`` this
        module bound, which is how the parent tells an activated arm from an
        unactivated one), ``arm_file``, ``ambient_hasher``, ``corpus`` and
        ``golden`` (the two name lists, so a partial run cannot pass as a
        complete one), ``raw`` and ``masked`` (each ``{"identical": [names],
        "differing": {name: {"fields", "golden_key", "derived_key"}}}``), and
        ``errors``.

    """
    raw: dict[str, Any] = {"identical": [], "differing": {}}
    masked: dict[str, Any] = {"identical": [], "differing": {}}
    errors: dict[str, str] = {}
    with _pinned():
        for entry in CORPUS:
            golden = GOLDEN[entry.name]
            try:
                derived = _golden_entry(entry)
                mask_derived = _rederivation_masker(entry.name, derived["key"])
                mask_golden = _rederivation_masker(entry.name, golden["key"])
                comparisons = (
                    (
                        raw,
                        sorted(
                            field for field in golden if derived[field] != golden[field]
                        ),
                    ),
                    (
                        masked,
                        sorted(
                            field
                            for field in golden
                            if mask_derived(derived[field])
                            != mask_golden(golden[field])
                        ),
                    ),
                )
            except Exception:
                errors[entry.name] = traceback.format_exc(limit=6)
                continue
            for group, fields in comparisons:
                if fields:
                    group["differing"][entry.name] = {
                        "fields": fields,
                        "golden_key": golden["key"],
                        "derived_key": derived["key"],
                    }
                else:
                    group["identical"].append(entry.name)
    arm = sys.modules.get(_BASELINE_ARM_MODULE)
    return {
        "delayed_module": Delayed.__module__,
        "arm_file": None if arm is None else arm.__file__,
        "ambient_hasher": _AMBIENT_HASHER,
        "corpus": [entry.name for entry in CORPUS],
        "golden": sorted(GOLDEN),
        "raw": raw,
        "masked": masked,
        "errors": errors,
    }


def _emit_golden_rederivation() -> None:
    """Print the re-derivation payload as JSON on stdout.

    Entry point of the subprocess ``_REDERIVATION_BOOTSTRAP`` starts. Stdout
    carries the payload and nothing else, so the parent can parse it whatever the
    interpreter writes to stderr.
    """
    json.dump(_rederive_golden(), sys.stdout)


@pytest.fixture(scope="module")
def _golden_rederivation() -> dict[str, Any]:
    """Re-derive the golden fixture from the frozen baseline arm, once per module.

    Returns:
        dict: the payload ``_rederive_golden`` built in the subprocess, parsed
        from its stdout and shared by the two tests below.

    Raises:
        AssertionError: if the arm is missing, if the subprocess fails, or if it
            prints something other than the payload -- with the command, the
            working directory and both streams in the message. A re-derivation
            that cannot run leaves the golden unproven, which is not a pass.

    """
    assert _BASELINE_ARM_PATH.is_file(), (
        f"the frozen baseline arm {_BASELINE_ARM_PATH} is missing, so the golden's "
        "provenance cannot be re-derived. It is a committed deliverable of this work "
        "(AAP §0.4.1); if this module was copied into another tree, copy the arm with "
        "it -- see the module docstring"
    )
    completed = subprocess.run(
        [sys.executable, "-c", _REDERIVATION_BOOTSTRAP],
        capture_output=True,
        check=False,
        cwd=_REPO_ROOT,
        text=True,
        timeout=_REDERIVATION_TIMEOUT,
    )
    context = (
        f"command: {sys.executable} -c {_REDERIVATION_BOOTSTRAP}\n"
        f"cwd: {_REPO_ROOT}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert (
        completed.returncode == 0
    ), f"the baseline-arm re-derivation exited {completed.returncode}\n{context}"
    try:
        payload: dict[str, Any] = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"the baseline-arm re-derivation printed no parsable payload: {error}\n"
            f"{context}"
        ) from error
    return payload


def test_the_golden_re_derives_from_the_frozen_baseline_arm(
    _golden_rederivation: dict[str, Any],
) -> None:
    """Every committed golden value is one the pre-refactor module produces.

    This is the standing substitute for the half of AAP §0.5.1 that this branch's
    git history cannot show -- that the block was captured before the first edit
    to ``dask/delayed.py`` -- and it is the assertion the substitution rests on
    rather than a claim about it. The oracle is checked first, because a
    re-derivation is evidence only if the module that produced it is the
    pre-refactor one: the arm's captured body must still hash to
    ``dask/delayed.py`` at ``_BASE_COMMIT``, and the subprocess must really have
    bound the arm's classes rather than the live module's.

    All ninety entries then agree field for field -- key, canonical graph, stable
    graph and computed result -- with only the two arm-dependent key tokens of
    ``_ARM_DEPENDENT_GOLDEN_ENTRIES`` set aside. That comparison is portable: the
    twelve environment-dependent tokens are set aside here too, so the property
    holds on any interpreter and under any installed hash library. Wherever the
    environment reproduces the capture, the character-for-character comparison is
    restored by the test below::

        test_the_arm_dependent_entries_are_the_only_verbatim_exceptions

    which also pins that those two tokens are the only relaxation the arms need.

    Args:
        _golden_rederivation: The payload of the re-derivation subprocess.

    """
    payload = _golden_rederivation
    assert _baseline_arm_body_sha256() == _BASELINE_ARM_BODY_SHA256, (
        f"{_BASELINE_ARM_PATH} is no longer the capture of dask/delayed.py at "
        f"{_BASE_COMMIT}, so it cannot serve as the pre-refactor oracle"
    )
    assert payload["delayed_module"] == _BASELINE_ARM_MODULE, (
        "the re-derivation ran against "
        f"{payload['delayed_module']!r} rather than the frozen arm, so it says "
        "nothing about pre-refactor behaviour"
    )
    assert payload["arm_file"] == str(
        _BASELINE_ARM_PATH
    ), f"the arm imported was {payload['arm_file']!r}, not {_BASELINE_ARM_PATH}"
    assert payload["ambient_hasher"] == _AMBIENT_HASHER, (
        f"the subprocess selected {payload['ambient_hasher']!r} where this process "
        f"selected {_AMBIENT_HASHER!r}, so the two are not the same environment"
    )
    assert payload["errors"] == {}, (
        f"the frozen arm could not rebuild {sorted(payload['errors'])}: "
        f"{payload['errors']}"
    )
    assert payload["corpus"] == [entry.name for entry in CORPUS]
    assert payload["golden"] == sorted(GOLDEN)

    for name, reason in sorted(_ARM_DEPENDENT_GOLDEN_ENTRIES.items()):
        # A registry entry has to name a real, deterministically keyed corpus
        # entry whose golden key the masker can split -- otherwise the relaxation
        # it grants would be granted to nothing, or to a token that is not one.
        assert _by_name(name).deterministic, f"{name}: only a deterministic key token"
        assert _hex_key_token(name, GOLDEN[name]["key"])
        assert reason, f"{name}: an arm-dependent entry needs its cause recorded"

    assert payload["masked"]["differing"] == {}, (
        "the frozen pre-refactor arm does not reproduce "
        f"{sorted(payload['masked']['differing'])}, so those golden values are not a "
        "pre-refactor capture. The golden is never regenerated to make this pass -- "
        "see the module docstring, 'Provenance of the golden block'"
    )
    assert sorted(payload["masked"]["identical"]) == sorted(GOLDEN)


@pytest.mark.skipif(
    not _CAPTURE_ENVIRONMENT_REPRODUCES, reason=_NON_CAPTURE_ENVIRONMENT_REASON
)
def test_the_arm_dependent_entries_are_the_only_verbatim_exceptions(
    _golden_rederivation: dict[str, Any],
) -> None:
    """Character for character, the frozen arm reproduces all but two entries.

    The portable comparison above sets aside fourteen key tokens: twelve the
    environment decides and two the arm does. In the environment the golden was
    captured in, the first twelve need no relaxation at all, so this restores the
    verbatim comparison and pins the exact figure the acknowledgement of the
    §0.5.1 ordering failure rests on: eighty-eight of the ninety committed entries
    re-derive byte for byte from the pre-refactor module, and the two that cannot
    are the ones ``_ARM_DEPENDENT_GOLDEN_ENTRIES`` names.

    It also pins how narrowly they may differ, which is what keeps the registry
    from becoming a licence: the difference has to be in the key, it may reach no
    field but the graph serializations that embed that key, it may never reach the
    computed result, and both sides must still be deterministic 32-hex digests
    under one shared prefix -- a UUID fallback or a changed prefix fails here.

    Args:
        _golden_rederivation: The payload of the re-derivation subprocess.

    """
    payload = _golden_rederivation
    differing = payload["raw"]["differing"]
    assert sorted(differing) == sorted(_ARM_DEPENDENT_GOLDEN_ENTRIES), (
        "the entries the frozen arm cannot reproduce verbatim have changed: "
        f"expected {sorted(_ARM_DEPENDENT_GOLDEN_ENTRIES)}, got {sorted(differing)}"
    )
    assert len(payload["raw"]["identical"]) == len(GOLDEN) - len(
        _ARM_DEPENDENT_GOLDEN_ENTRIES
    )

    for name, record in sorted(differing.items()):
        fields = record["fields"]
        golden_key = record["golden_key"]
        derived_key = record["derived_key"]
        assert (
            "key" in fields
        ), f"{name}: differs somewhere other than its key: {fields}"
        assert "result_repr" not in fields, (
            f"{name}: the value the expression computes to differs between the arms, "
            "which no module path can explain"
        )
        assert set(fields) <= {"key", "graph", "stable_graph"}, (
            f"{name}: the arm difference reaches {sorted(set(fields) - {'key', 'graph', 'stable_graph'})}, "
            "beyond the key and the graph serializations that embed it"
        )
        assert golden_key.rpartition("-")[0] == derived_key.rpartition("-")[0], (
            f"{name}: the key prefix changed, {golden_key!r} against {derived_key!r}; "
            "only the token may differ between the arms"
        )
        assert _hex_key_token(name, golden_key) != _hex_key_token(name, derived_key), (
            f"{name}: both arms produced the same token, so the entry is not "
            "arm-dependent and does not belong in _ARM_DEPENDENT_GOLDEN_ENTRIES"
        )


@pytest.mark.parametrize("entry", CORPUS, ids=[entry.name for entry in CORPUS])
def test_characterisation(entry: _Expr) -> None:
    """Compare one corpus entry against its frozen golden record.

    Args:
        entry: The corpus entry to build. Its generated key, its canonical graph
            -- or its stable graph, for an entry carrying inherently random
            tokens -- and its computed result are compared against
            ``GOLDEN[entry.name]``, which was captured before the refactor and is
            never regenerated to make this test pass.

    An entry listed in ``_ENVIRONMENT_DEPENDENT_KEY_TOKEN`` is compared with its
    own key token masked on both sides, because that one token is decided by the
    environment rather than by the module under test; everything else about it,
    this test's own graph and result comparisons included, is compared as it
    stands, and the unmasked key is asserted by
    ``test_environment_dependent_key_tokens_match_the_golden_where_reproducible``
    wherever the environment can reproduce it.

    """
    golden = GOLDEN[entry.name]
    obj = entry.build()
    table: dict[str, str] = {}
    normalised_key = normalize_key(obj.key, table)
    # The identity transform for every entry but the twelve whose own key token
    # the environment decides rather than the module under test. For those, the
    # live form is masked with the token this environment produced and the golden
    # form with the token the capture recorded, so the two are compared in the
    # only form that is portable -- see ``_ENVIRONMENT_DEPENDENT_KEY_TOKEN``.
    mask_live = _token_masker(entry.name, obj.key)
    mask_golden = _token_masker(entry.name, golden["key"])

    if entry.deterministic:
        # Character for character, 32-hex token included: a deterministic token is
        # part of the contract because ``pure=True`` shares work by key identity,
        # so a changed token silently changes which nodes are deduplicated.
        assert mask_live(obj.key) == mask_golden(golden["key"])
        if entry.name in _ENVIRONMENT_DEPENDENT_KEY_TOKEN:
            # Two properties of the masked token that the golden can no longer
            # pin here, both asserted instead: it is still a deterministic digest
            # rather than a UUID fallback (``_hex_key_token``, inside the masker,
            # refuses anything else), and it reproduces across builds, which is
            # what sharing by key identity under ``pure=True`` rests on. Its
            # exact value is asserted against the golden wherever the environment
            # can reproduce it, by
            # ``test_environment_dependent_key_tokens_match_the_golden_where_reproducible``.
            assert entry.build().key == obj.key
    else:
        assert obj.key != golden["key"], "a UUID token must not equal its placeholder"
    assert mask_live(normalised_key) == mask_golden(golden["key"])

    if entry.canonical:
        graph = mask_live(canonical_graph(obj))
        expected = mask_golden(golden["graph"])
        # Field by field first, so a failure names the property that moved, then
        # as a whole, so nothing outside those fields can drift unnoticed.
        assert graph["key"] == expected["key"]
        assert graph["dask_keys"] == expected["dask_keys"]
        assert graph["dask_layers"] == expected["dask_layers"]
        assert graph["layer_order"] == expected["layer_order"]
        assert graph["dependency_order"] == expected["dependency_order"]
        assert graph["layers"] == expected["layers"]
        assert graph["dependencies"] == expected["dependencies"]
        assert graph == expected

        materialised = obj.__dask_graph__()
        assert isinstance(materialised, HighLevelGraph)
        if entry.deterministic:
            # Spelled out against the live container rather than only against the
            # canonical dict. Layer insertion order is the single most
            # refactor-sensitive property of the graph: ``tokenize`` hashes a
            # ``Delayed``'s pickled slot state, ``HighLevelGraph`` included, so two
            # graphs with identical content in a different order tokenize
            # differently wherever a ``Delayed`` is reached through pickle.
            assert (
                mask_live(_golden_names(list(materialised.layers)))
                == expected["layer_order"]
            )
            assert (
                mask_live(_golden_names(list(materialised.dependencies)))
                == expected["dependency_order"]
            )
            assert (
                mask_live(_golden_names(list(obj.__dask_keys__())))
                == expected["dask_keys"]
            )
            assert (
                mask_live(_golden_names(list(obj.__dask_layers__())))
                == expected["dask_layers"]
            )
    else:
        assert golden["graph"] is None
        assert entry.name in _EXCLUSION_REASONS

    if entry.volatile_tokens:
        # The graph of an entry whose raw canonical form is not reproducible is
        # asserted here in full, with only the identifiers a double build proves
        # random replaced. Field by field first, then as a whole, exactly as
        # above: every key, node kind, dependency and both insertion orders are
        # compared, so a graph-shape regression on this path fails the suite.
        stable = _stable_graph(entry)
        golden_stable = golden["stable_graph"]
        assert golden_stable is not None
        assert stable["key"] == golden_stable["key"]
        assert stable["dask_keys"] == golden_stable["dask_keys"]
        assert stable["dask_layers"] == golden_stable["dask_layers"]
        assert stable["layer_order"] == golden_stable["layer_order"]
        assert stable["dependency_order"] == golden_stable["dependency_order"]
        assert stable["layers"] == golden_stable["layers"]
        assert stable["dependencies"] == golden_stable["dependencies"]
        assert stable == golden_stable
    else:
        assert golden["stable_graph"] is None

    if entry.computable:
        (result,) = canonical_result([obj])
        assert repr(result) == golden["result_repr"]
    else:
        assert golden["result_repr"] is None
        assert entry.name in _EXCLUSION_REASONS


@pytest.mark.parametrize("name", sorted(_ENVIRONMENT_DEPENDENT_KEY_TOKEN))
def test_environment_dependent_key_tokens_match_the_golden_where_reproducible(
    name: str,
) -> None:
    """Assert the verbatim golden for a masked entry wherever that is possible.

    ``test_characterisation`` compares these twelve entries with their own key
    token masked, because nothing this module can pin decides that token. The
    relaxation is not a licence: wherever the environment reproduces the capture
    -- the same ambient hash library for the import-time cause, the same
    interpreter for the dataclass cause -- the exact key string and the raw
    canonical graph are asserted here, so the character-for-character evidence
    AAP §0.5.1 asks for still runs, and an entry masked without cause fails here
    instead of passing quietly.

    Args:
        name: A corpus entry registered in ``_ENVIRONMENT_DEPENDENT_KEY_TOKEN``.

    """
    cause = _TOKEN_CAUSES[_ENVIRONMENT_DEPENDENT_KEY_TOKEN[name]]
    if not cause.reproduces_capture:
        pytest.skip(cause.reason)
    entry = _by_name(name)
    obj = entry.build()
    assert obj.key == GOLDEN[name]["key"]
    assert canonical_graph(obj) == GOLDEN[name]["graph"]


def test_a_token_the_masker_cannot_split_is_refused_rather_than_masked() -> None:
    """The masker refuses anything but a deterministic 32-hex token.

    This is the property that keeps the relaxation from swallowing a regression.
    A masked entry whose token stopped being a deterministic digest -- a UUID
    fallback from a lost ``pure=True``, a key with no token at all, a
    non-``str`` key -- would otherwise be masked into agreement with the golden
    and pass. Each of those inputs fails instead.
    """
    with pytest.raises(AssertionError, match="expected a str key"):
        _hex_key_token("op_add", ("tk", 0))
    with pytest.raises(AssertionError, match="carries no prefix"):
        _hex_key_token("op_add", "nohyphen")
    with pytest.raises(AssertionError, match="does not end in a deterministic"):
        _hex_key_token("op_add", "add-2f4c1e1a-9b2e-4d0a-8f7c-1a2b3c4d5e6f")
    with pytest.raises(AssertionError, match="does not end in a deterministic"):
        _hex_key_token("op_add", "add-notahexdigestatall")
    # And the accepting side, so the guard is not merely strict: the golden key
    # of a masked entry splits into its prefix and its 32-hex token.
    golden_key = GOLDEN["op_add"]["key"]
    assert golden_key == f"add-{_hex_key_token('op_add', golden_key)}"


def test_environment_dependent_entries_are_justified_corpus_entries() -> None:
    """The masking registry names real entries, with a real cause, and masks one token.

    Three properties, each of which a careless addition to the registry would
    break: a registered name is a deterministically-keyed corpus entry with a
    golden key the masker can split; every cause named is one of the two defined;
    and masking actually changes the form it is applied to -- a mask that matched
    nothing would compare two unmasked forms and quietly reintroduce the
    portability failure it exists to fix.
    """
    assert set(_TOKEN_CAUSES) == set(_ENVIRONMENT_DEPENDENT_KEY_TOKEN.values()), (
        "every defined cause must be in use and every cause in use must be defined: "
        f"defined={sorted(_TOKEN_CAUSES)}, "
        f"used={sorted(set(_ENVIRONMENT_DEPENDENT_KEY_TOKEN.values()))}"
    )
    for name, cause in sorted(_ENVIRONMENT_DEPENDENT_KEY_TOKEN.items()):
        # ``_by_name`` raises if the registry names something the corpus does not.
        entry = _by_name(name)
        assert cause in _TOKEN_CAUSES, name
        assert entry.deterministic, f"{name}: only a deterministic token is masked"
        assert not entry.volatile_tokens, f"{name}: a random token is handled already"
        golden_key = GOLDEN[name]["key"]
        masked = _token_masker(name, golden_key)(golden_key)
        prefix = golden_key.rpartition("-")[0]
        assert masked == f"{prefix}-{_ENV_TOKEN_PLACEHOLDER}", name
        assert masked != golden_key, name


def test_operator_method_leaf_keys_are_fixed_when_dask_delayed_is_imported() -> None:
    """The hasher pin cannot reach an operator method's leaf key.

    ``Delayed._get_binary_operator`` evaluates ``delayed(op, pure=True)`` while
    the class is being built, so that leaf's key -- which tokenizes the operator
    function through pickle, and therefore through ``dask.hashing.hashers[0]`` --
    is fixed when ``dask.delayed`` is imported, before any fixture exists to pin
    the hasher. The leaf itself never enters a graph: it is handed to
    ``call_function`` as ``func_token`` and tokenized into the call key, which is
    why the import-time state reaches exactly one token of the resulting graph
    and no other.

    That is the whole of the ``"import-time-hasher"`` cause, asserted rather than
    described: the leaf the class holds agrees with one rebuilt under the pin
    exactly when the ambient hasher already was the pinned one.
    """
    # ``_bind_operator`` installs the method with ``setattr``, so the class's own
    # ``__dict__`` is where it lives; subscripting avoids both a dynamic-attribute
    # type error and a constant ``getattr``.
    bound = Delayed.__dict__["__add__"]
    closure = bound.__closure__
    assert closure is not None, "Delayed.__add__ no longer closes over its leaf"
    assert len(closure) == 1, f"expected one closure cell, got {len(closure)}"
    leaf = closure[0].cell_contents
    assert type(leaf) is DelayedLeaf

    rebuilt = delayed(operator.add, pure=True)
    assert type(rebuilt) is DelayedLeaf
    assert (leaf.key == rebuilt.key) == (
        _AMBIENT_HASHER == dask.hashing._hash_sha1.__name__
    ), (
        f"the class-bound operator leaf {leaf.key!r} and a leaf rebuilt under the "
        f"SHA-1 pin {rebuilt.key!r} may agree only where the ambient hasher "
        f"({_AMBIENT_HASHER}) is the pinned one"
    )


def test_a_dataclass_key_token_takes_one_input_per_declared_dataclass_param() -> None:
    """A dataclass token's input set is the interpreter's, not dask's.

    ``dask.tokenize`` normalises a dataclass instance by reading every attribute
    ``__dataclass_params__`` declares (``dask/tokenize.py``'s
    ``_normalize_dataclass``), so the identity and number of those attributes are
    inputs to the token -- and CPython decides them, and has changed them: six
    names on 3.10, ten on 3.14. That is the whole of the
    ``"interpreter-dataclass"`` cause, and it is independent of the hasher: under
    the same SHA-1 pin the ``arg_dataclass_*`` entries reproduce their captured
    tokens on the capture interpreter and cannot on another.
    """
    # The class attribute dask reads, taken from the class ``__dict__`` because
    # typeshed's dataclass protocol declares only ``__dataclass_fields__``.
    params = Box.__dict__["__dataclass_params__"]
    slots = tuple(params.__slots__)
    # The six every supported interpreter declares, in the order dask reads them.
    assert slots[:6] == ("init", "repr", "eq", "order", "unsafe_hash", "frozen")
    if _INTERPRETER_REPRODUCES_CAPTURE:
        assert slots == _CAPTURE_DATACLASS_PARAM_SLOTS


def test_delayed_of_a_delayed_returns_the_same_object() -> None:
    existing = _leaf()
    assert delayed(existing) is existing
    assert delayed(existing, name="ignored", pure=True, nout=3) is existing


def test_corpus_floor_and_golden_cover_each_other() -> None:
    names = [entry.name for entry in CORPUS]
    assert len(names) == len(set(names)), "corpus names must be unique"
    assert (
        len(CORPUS) >= 40
    ), f"corpus floor is 40 NumPy-free expressions, got {len(CORPUS)}"
    assert set(names) == set(GOLDEN), (
        "every corpus entry needs a golden entry captured before the refactor: "
        f"missing={sorted(set(names) - set(GOLDEN))}, stale={sorted(set(GOLDEN) - set(names))}"
    )


def test_default_naming_key_forms_are_locked() -> None:
    """At least five entries keep default naming, locking both generated forms.

    ``obj.__name__-<token>`` and ``type(obj).__name__-<token>`` are only
    exercised when no explicit ``name=`` is passed, and their tokens depend on
    the pinned hasher.
    """
    names = {entry.name for entry in CORPUS}
    assert len(_DEFAULT_NAMING) >= 5
    assert set(_DEFAULT_NAMING) <= names
    assert GOLDEN["wrap_int_default"]["key"].startswith("int-")
    assert GOLDEN["wrap_str_pure"]["key"].startswith("str-")
    assert GOLDEN["wrap_obj_pure"]["key"].startswith("Obj-")
    assert GOLDEN["wrap_list_traverse_false_default"]["key"].startswith("list-")
    assert GOLDEN["wrap_func_pure"]["key"].startswith("inc-")
    assert GOLDEN["call_default_name"]["key"].startswith("inc-")


def test_every_unpack_collections_branch_is_covered() -> None:
    names = {entry.name for entry in CORPUS}
    missing = {
        branch: name for branch, name in _BRANCH_COVERAGE.items() if name not in names
    }
    assert not missing, f"branch table points at entries that do not exist: {missing}"
    assert len(_BRANCH_COVERAGE) >= 20


def test_guard_entries_defeat_every_merge_shortcut_guard() -> None:
    """The guard dependencies really fail the guards a merge shortcut must apply.

    These four are the corpus's only coverage of the generic
    ``HighLevelGraph.from_collections`` path once the construction path grows a
    shortcut, so their *preconditions* are asserted here and their graphs are
    asserted against the golden by ``test_characterisation``.
    """
    names = {entry.name for entry in CORPUS}
    assert set(_GUARD_ENTRIES) <= names

    for dependency in (
        _sub_delayed(),
        _sub_delayed_dask_layers(),
        _sub_delayed_layer_dict(),
    ):
        assert isinstance(dependency, Delayed)
        # Exact-class identity fails, which is what sends the call down the
        # generic path even though the object is a perfectly valid dependency.
        assert type(dependency) is not Delayed
        assert isinstance(dependency.dask, HighLevelGraph)
    # A subclass is free to define the very name a leaf-layer helper might use,
    # which is why such a helper has to be a module-level function.
    assert callable(SubDelayedLayerDict._layer_dict)
    assert SubDelayedLayers.__dask_layers__ is not Delayed.__dask_layers__

    proxy = _mapping_proxy_delayed()
    assert type(proxy) is Delayed
    # Neither exactly a ``HighLevelGraph`` nor exactly a ``dict``, so a
    # graph-type guard must reject it; ``Delayed.__init__`` accepts it because it
    # only validates the layer for a ``HighLevelGraph``.
    # Compared by exact type rather than ``isinstance``, and with the positive
    # check last: ``MappingProxyType`` is ``final``, so once a static checker has
    # narrowed the expression to it, any further class comparison is reported as
    # an impossible subclass relation.
    graph = proxy.dask
    assert type(graph) is not HighLevelGraph
    assert type(graph) is not dict
    assert type(graph) is types.MappingProxyType


def test_flag_exclusions_are_documented() -> None:
    flagged = {
        entry.name
        for entry in CORPUS
        if not (entry.canonical and entry.computable and entry.picklable)
    }
    assert flagged == set(_EXCLUSION_REASONS), (
        "every corpus exclusion must be justified in _EXCLUSION_REASONS: "
        f"undocumented={sorted(flagged - set(_EXCLUSION_REASONS))}, "
        f"stale={sorted(set(_EXCLUSION_REASONS) - flagged)}"
    )
    for name, reason in _EXCLUSION_REASONS.items():
        assert reason.strip(), f"{name} has an empty reason"


def test_numpy_collection_argument_is_characterised() -> None:
    """Characterise a real non-``Delayed`` dask collection argument.

    Guarded in the body rather than at module level so that an environment
    without NumPy skips only this function and the >= 40 NumPy-free corpus still
    runs. Its expectations live here rather than in ``GOLDEN`` for the same
    reason -- a golden entry would be missing wherever NumPy is absent -- and they
    are the portable ones: the key prefix and shape, the single output key, the
    graph container and the computed value, never a pickle-derived token.
    """
    np = pytest.importorskip("numpy")
    da = pytest.importorskip("dask.array")

    array = da.from_array(np.arange(4), chunks=2)
    obj = delayed(ident, name="ident", pure=True)(array)
    rebuilt = delayed(ident, name="ident", pure=True)(
        da.from_array(np.arange(4), chunks=2)
    )

    assert obj.key == rebuilt.key
    assert obj.key.startswith("ident-")
    assert len(obj.key) == len("ident-") + 32
    assert obj.__dask_keys__() == [obj.key]
    assert tuple(obj.__dask_layers__()) == (obj.key,)

    graph = obj.__dask_graph__()
    assert isinstance(graph, HighLevelGraph)
    # The finalized collection carries a plain low-level graph, so
    # ``_from_collection`` takes its non-``HighLevelGraph`` branch
    # (``dask/highlevelgraph.py:462-465``) and orders the layers new-then-existing.
    layers = list(graph.layers)
    assert len(layers) == 2
    assert layers[0] == obj.key
    assert graph.dependencies[obj.key] == {layers[1]}
    assert graph.dependencies[layers[1]] == set()

    (result,) = canonical_result([obj])
    assert len(result) == 1
    assert list(result[0]) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Side-effect and count characterisations.
#
# These inputs are neither orderable nor JSON-serialisable, so they stay out of
# ``CORPUS``/``GOLDEN`` and assert counts, types and messages instead. Every
# expected count below was *measured* on the pre-refactor module; none is a
# guess, and none may be adjusted to make a test pass.
#
# The counters live in one module-level dict so that reading a count cannot
# itself trigger the ``__eq__`` the metaclass case is counting.
# ---------------------------------------------------------------------------

_COUNTS: dict[str, int] = {"repr": 0, "hash": 0, "eq": 0}


class _CountingKey:
    """A valid graph key that counts the ``repr()`` calls made on it."""

    def __repr__(self) -> str:
        _COUNTS["repr"] += 1
        return "_CountingKey()"

    def __hash__(self) -> int:
        return 4242

    def __eq__(self, other: object) -> bool:
        return self is other


class _CountingLayer:
    """A layer name object that counts the ``hash()`` calls made on it."""

    def __repr__(self) -> str:
        return "_CountingLayer()"

    def __hash__(self) -> int:
        _COUNTS["hash"] += 1
        return 99

    def __eq__(self, other: object) -> bool:
        return self is other


class _CountingMeta(type):
    """Metaclass that counts ``==`` against the class and forbids ``hash()``."""

    def __eq__(cls, other: object) -> bool:
        _COUNTS["eq"] += 1
        return cls is other

    def __hash__(cls) -> int:
        """Fail: the construction path must never hash a type object.

        A metaclass defining ``__eq__`` has to define ``__hash__`` too, and
        raising here is what turns "the branch cascade hashed a type" into a
        failure instead of a silent behaviour change.

        Raises:
            AssertionError: always.

        """
        raise AssertionError("the construction path must never hash a type object")


class _WithCountingMeta(metaclass=_CountingMeta):
    pass


def test_dependency_key_is_repred_once_for_an_impure_call() -> None:
    """An impure call ``repr()``s a user-defined dependency key exactly once.

    Mechanism, all of it observable: ``HighLevelGraph.from_collections`` probes
    each dependency with ``is_dask_collection``, which reads ``c.expr``
    (``dask/base.py:247-253``); ``Delayed.__getattr__`` turns that into a
    throw-away ``DelayedAttr(c, "expr")`` whose ``__init__`` tokenizes. Inside
    ``tokenize``, ``normalize_object`` sees ``__dask_tokenize__`` and returns the
    dependency's key *as is*, without recursive dispatch
    (``dask/tokenize.py:194-198``), and ``_tokenize`` then calls ``str()`` on the
    normalised tuple (``dask/tokenize.py:33-39``) -- which invokes the key's
    ``__repr__``.

    The count is therefore a behavioural contract for user-defined keys, not an
    implementation detail: a construction path that drops the probe for such a
    key would be observably different.
    """
    key = _CountingKey()
    dependency = Delayed(key, {key: DataNode(key, 5)})

    _COUNTS["repr"] = 0
    result = delayed(ident, name="ident")(dependency)

    assert _COUNTS["repr"] == 1
    # The key object itself -- not a copy, not its repr -- is still what names the
    # dependency's layer and its node.
    layer_name = next(name for name in result.dask.layers if name is key)
    assert any(node_key is key for node_key in result.dask.layers[layer_name])


def test_dependency_key_is_repred_twice_for_a_pure_call() -> None:
    """A ``pure=True`` call ``repr()``s the dependency key exactly twice.

    One ``repr`` comes from the discarded ``DelayedAttr(c, "expr")`` probe
    described in the impure test; the second comes from the call key's own
    ``tokenize``, which normalises the dependency through
    ``__dask_tokenize__`` -> its key -> ``str()``.
    """
    key = _CountingKey()
    dependency = Delayed(key, {key: DataNode(key, 5)})

    _COUNTS["repr"] = 0
    delayed(ident, name="ident", pure=True)(dependency)

    assert _COUNTS["repr"] == 2


def test_dependency_layer_name_is_hashed_twice_for_a_high_level_graph() -> None:
    """A ``HighLevelGraph``-backed dependency hashes its layer name exactly twice.

    The two sites are ``deps[name] = set(collection.__dask_layers__())``
    (``dask/highlevelgraph.py:461``) and the ``isinstance`` re-wrap comprehension
    in ``HighLevelGraph.__init__`` (``:446``). The ``ensure_dict(..., copy=True)``
    calls do not add any: copying a ``dict`` reuses the stored hashes.

    The hash of the layer name during ``Delayed.__init__``'s membership check
    happens before the counter is reset, so it is deliberately not part of the
    count.
    """
    layer_name = _CountingLayer()
    # Annotated ``Any`` because ``HighLevelGraph`` declares its layer names as
    # ``str`` while the runtime accepts -- and this test requires -- an arbitrary
    # hashable object as a layer name.
    layers: dict[Any, Any] = {layer_name: {"dkey": DataNode("dkey", 5)}}
    layer_dependencies: dict[Any, Any] = {layer_name: set()}
    dependency = Delayed(
        "dkey",
        HighLevelGraph(layers, layer_dependencies),
        layer=layer_name,
    )

    _COUNTS["hash"] = 0
    delayed(ident, name="ident")(dependency)

    assert _COUNTS["hash"] == 2


def test_dependency_layer_name_is_hashed_four_times_for_a_dict_graph() -> None:
    """A ``dict``-backed dependency hashes its layer name exactly four times.

    ``HighLevelGraph._from_collection`` takes its non-``HighLevelGraph`` branch
    here, and the four sites are ``layers = {name: layer, key: graph}``
    (``dask/highlevelgraph.py:464``), the two insertions in
    ``deps = {name: {key}, key: set()}`` (``:465``), and the re-wrap comprehension
    in ``HighLevelGraph.__init__`` (``:446``).

    Both graph shapes are asserted because they differ: the same dependency
    carrying a ``HighLevelGraph`` is hashed twice, as the test above records. A
    construction-path shortcut that skips the probe for a dependency with a
    primitive layer name leaves both counts untouched, which is exactly what
    these two tests are here to prove.
    """
    layer_name = _CountingLayer()
    dependency = Delayed("dkey", {"dkey": DataNode("dkey", 5)}, layer=layer_name)

    _COUNTS["hash"] = 0
    delayed(ident, name="ident")(dependency)

    assert _COUNTS["hash"] == 4


def test_attribute_chain_over_an_unsafe_ancestor_reprs_its_key_as_today() -> None:
    """A ``DelayedAttr`` chain ``repr()``s an unsafe ancestor's key 2**depth times.

    ``DelayedAttr.dask`` merges its parent's graph into its own, so the counts of
    an *ancestor* are decided by how often the attribute's own graph is built.
    ``HighLevelGraph._from_collection`` builds it twice -- once for the
    ``is_dask_collection`` probe, once for the merge -- and each level of the
    chain doubles the level below it, so a user-defined key at the root of the
    chain is ``repr()``-ed 1, 2, 4 and 8 times for chains of length 0, 1, 2 and 3
    (all four measured on the pre-refactor module).

    The counts are therefore a behavioural contract for attribute chains, not an
    implementation detail: a construction path that fast-paths the attribute on
    its own primitive key and layer name while an ancestor is unsafe would report
    1 at every depth. Building the chain itself ``repr()``s the key too -- every
    ``DelayedAttr.__init__`` tokenizes its parent -- which is why the counter is
    reset once the chain stands.
    """
    for depth, expected in ((0, 1), (1, 2), (2, 4), (3, 8)):
        key = _CountingKey()
        dependency: Delayed = Delayed(key, {key: DataNode(key, 5)})
        for _ in range(depth):
            dependency = dependency.attr

        _COUNTS["repr"] = 0
        delayed(ident, name="ident")(dependency)

        assert _COUNTS["repr"] == expected, f"attribute chain of length {depth}"


def test_attribute_chain_over_an_unsafe_ancestor_hashes_its_layer_as_today() -> None:
    """The same chain hashes an unsafe ancestor's layer name exactly as today.

    The hash sites are the ones the two graph-shape tests above enumerate, and
    the doubling of the previous test applies to them as well -- with one extra
    hash per level, because the attribute's own merged graph re-inserts the
    ancestor's layer name into the layer and dependency dicts it copies. Measured
    on the pre-refactor module: 2, 5 and 11 hashes for chains of length 0, 1
    and 2.
    """
    for depth, expected in ((0, 2), (1, 5), (2, 11)):
        layer_name = _CountingLayer()
        layers: dict[Any, Any] = {layer_name: {"dkey": DataNode("dkey", 5)}}
        layer_dependencies: dict[Any, Any] = {layer_name: set()}
        dependency: Delayed = Delayed(
            "dkey",
            HighLevelGraph(layers, layer_dependencies),
            layer=layer_name,
        )
        for _ in range(depth):
            dependency = dependency.attr

        _COUNTS["hash"] = 0
        delayed(ident, name="ident")(dependency)

        assert _COUNTS["hash"] == expected, f"attribute chain of length {depth}"


def test_unpack_collections_compares_types_three_times_and_never_hashes() -> None:
    """``unpack_collections`` compares the value's type three times, hashing none.

    The three comparisons are the tuple-membership test ``typ in (list, tuple,
    set)`` inside ``unpack_collections``, which uses ``==`` element-wise and never
    hashes. ``_CountingMeta.__hash__`` raises, so any fast path that reached for a
    ``set``/``frozenset`` membership test -- or that dispatched on the type object
    in a dict -- would fail this test rather than quietly change the side effects
    a user metaclass observes.
    """
    literal = _WithCountingMeta()

    _COUNTS["eq"] = 0
    task, collections = unpack_collections(literal)

    assert _COUNTS["eq"] == 3
    assert task is literal
    assert collections == ()


def test_nominal_immutability_asymmetry_is_preserved() -> None:
    """``Delayed`` is *nominally* immutable, exactly as it stands today.

    Declared slot names can be assigned and deleted; everything else raises. This
    test asserts the asymmetry rather than the stricter immutability one might
    expect, because the asymmetry is what ``Delayed.__setattr__`` and
    ``Delayed.__setitem__`` do, and no part of this work strengthens it.
    """
    obj = _leaf()

    obj._length = 5
    assert obj._length == 5
    assert len(obj) == 5

    with pytest.raises(TypeError, match="Delayed objects are immutable"):
        obj.foo = 1
    with pytest.raises(TypeError, match="Delayed objects are immutable"):
        obj[0] = 1

    # There is no ``__delattr__`` override, so slot deletion has its default
    # semantics; afterwards the read routes through ``Delayed.__getattr__``.
    del obj._length
    with pytest.raises(AttributeError, match="Attribute _length not found"):
        obj._length
    with pytest.raises(AttributeError, match="Attribute _length not found"):
        len(obj)

    # ``_key`` is probed only as far as the analogous ``AttributeError``: reading
    # the public ``key`` property afterwards recurses through ``__getattr__`` to a
    # ``RecursionError`` whose depth is interpreter-dependent.
    other = _leaf()
    del other._key
    with pytest.raises(AttributeError, match="Attribute _key not found"):
        other._key


def test_every_entry_asserts_its_graph() -> None:
    """No corpus entry may opt out of graph comparison altogether.

    An entry whose raw canonical dict is reproducible is compared verbatim
    (``canonical=True``); one that carries an inherently random identifier is
    compared after that identifier is normalised out (``volatile_tokens=True``).
    There is no third option: a graph nobody compares is a graph in which a shape
    regression passes unnoticed, which is precisely what this corpus exists to
    prevent.
    """
    unasserted = [
        entry.name for entry in CORPUS if not (entry.canonical or entry.volatile_tokens)
    ]
    assert not unasserted, f"these entries assert no graph at all: {unasserted}"

    volatile = {entry.name for entry in CORPUS if entry.volatile_tokens}
    assert volatile, "the volatile-token path must stay covered by at least one entry"
    for name in volatile:
        entry = _by_name(name)
        # The flag exists because the raw dict is *not* reproducible; an entry
        # that sets both would be recording two versions of the same thing.
        assert not entry.canonical, name
        assert GOLDEN[name]["graph"] is None, name
        assert GOLDEN[name]["stable_graph"] is not None, name


def test_non_runnable_entries_fail_exactly_as_they_did() -> None:
    """Every entry that cannot be computed fails with its recorded exception.

    ``computable=False`` is not permission to ignore what the expression does: a
    path whose result is an exception has the exception asserted -- exact type,
    exact message -- so that a change of failure mode is caught as readily as a
    change of value. All four are dangling-reference or unsupported-graph cases
    in the *current* module, measured on the pre-refactor code.
    """
    for name, (expected_type, message) in _COMPUTE_FAILURES.items():
        entry = _by_name(name)
        assert not entry.computable, f"{name} computes; drop it from _COMPUTE_FAILURES"
        assert name in _EXCLUSION_REASONS, name
        obj = entry.build()
        with pytest.raises(expected_type) as excinfo:
            canonical_result([obj])
        # Exact type, not a subclass, and the whole message rather than a
        # fragment of it.
        assert type(excinfo.value) is expected_type, name
        assert str(excinfo.value) == message, name


def _assert_finalize_is_not_computable(label: str, collection: Any) -> None:
    """Assert that ``finalize`` wraps a collection into an uncomputable ``Delayed``.

    Args:
        label: Name of the collection shape, attached to every assertion so a
            failure names the shape whose behaviour moved.
        collection: A dask collection to wrap with ``dask.delayed.finalize``.

    Raises:
        AssertionError: if the wrapper is not a plain ``Delayed`` keyed
            ``finalize-<token>`` over an ``HLGFinalizeCompute`` expression, or if
            computing it does not raise the exact exception recorded for
            ``finalize_collection`` in ``_COMPUTE_FAILURES``.

    """
    expected_type, message = _COMPUTE_FAILURES["finalize_collection"]
    with dask.config.set({"delayed_pure": True}):
        wrapped = finalize(collection)
        assert type(wrapped) is Delayed, label
        assert wrapped.key.startswith("finalize-"), label
        assert type(wrapped.__dask_graph__()).__name__ == "HLGFinalizeCompute", label
        with pytest.raises(expected_type) as excinfo:
            canonical_result([wrapped])
        assert type(excinfo.value) is expected_type, label
        assert str(excinfo.value) == message, label


def test_no_finalize_result_exists_to_characterise() -> None:
    """``finalize()`` has no computable result for any NumPy-free corpus shape.

    ``finalize`` stores ``collections_to_expr(collection).finalize_compute()`` --
    an ``HLGFinalizeCompute`` expression -- in the ``Delayed``, and computing one
    reaches ``collection.dask.copy()`` in ``dask._expr``, which an expression
    does not provide. The three shapes asserted here are the ones reachable
    without an optional dependency, and between them they cover the graph shapes
    a collection can carry: a hand-rolled collection with a plain ``dict``
    low-level graph, a hand-rolled collection with a ``HighLevelGraph``, and a
    ``Delayed``. ``test_finalize_of_a_dask_array_is_not_computable_either``
    asserts the same of a ``dask.array`` wherever NumPy is installed.

    The ``finalize_collection`` corpus entry therefore records no result: its
    graph is asserted in full through ``stable_graph`` and its failure verbatim
    through ``_COMPUTE_FAILURES``.
    """
    for label, collection in (
        ("dict-graph collection", _hand_rolled_collection()),
        ("HighLevelGraph collection", _hlg_backed_collection()),
        ("Delayed", _leaf()),
    ):
        _assert_finalize_is_not_computable(label, collection)


def test_finalize_of_a_dask_array_is_not_computable_either() -> None:
    """A ``dask.array`` collection wrapped with ``finalize`` fails the same way.

    Guarded in the body rather than at module level, so an environment without
    NumPy skips only this function while the NumPy-free shapes above still run.
    """
    np = pytest.importorskip("numpy")
    da = pytest.importorskip("dask.array")

    _assert_finalize_is_not_computable(
        "dask.array", da.from_array(np.arange(4), chunks=2)
    )


def test_canonicaliser_refuses_a_key_it_cannot_encode_reproducibly() -> None:
    """A key outside the encodable types is refused, never repr-ed into evidence.

    The canonical form is committed evidence, so a key whose textual form carries
    object identity -- a default ``object.__repr__`` embeds a memory address --
    must not reach it. Three shapes are refused, and the third is the reason the
    refusal is built from type metadata rather than from the object: a key whose
    ``__repr__`` raises would otherwise replace the diagnosis with an unrelated
    exception.
    """
    for label, key in (
        ("opaque object", _OpaqueKey()),
        ("tuple subclass with custom iteration", _IdentityYieldingTuple(("tk", 0))),
        ("key whose repr raises", _RaisingReprKey()),
    ):
        dependency = Delayed(key, {key: DataNode(key, 5)})
        with pytest.raises(TypeError) as excinfo:
            canonical_graph(dependency)
        assert "cannot canonicalise a graph key of type" in str(excinfo.value), label
        assert type(key).__qualname__ in str(excinfo.value), label

    # The encodable shapes still go through, byte for byte as before.
    plain: Any = ("tk", 0)
    encodable = Delayed(plain, {plain: DataNode(plain, 5)})
    assert canonical_graph(encodable)["key"] == "('tk', 0)"


def test_impure_fan_in_canonicalises_identically_on_two_builds() -> None:
    """Placeholder numbering does not depend on the random tokens themselves.

    A node with several impure dependencies sharing one key prefix is the case
    that decides how a node's dependencies may be ordered while they are being
    numbered: ordering them by their raw keys orders them by their UUID tokens,
    so two builds of the same expression -- and therefore the two arms of the A/B
    suite -- would produce different placeholder numbers and different
    ``layer_order`` fields for identical graphs. This asserts the invariant the
    canonicaliser's ordering exists to provide.
    """

    def build() -> Any:
        leaves = [delayed(inc, pure=False)(i) for i in range(6)]
        return delayed(ident, pure=False)(*leaves)

    first = canonical_graph(build())
    second = canonical_graph(build())
    assert first == second
    assert first["key"].endswith("-#0")
    assert len(first["layer_order"]) == 7


def test_tuple_key_with_a_non_builtin_element_falls_back() -> None:
    """A tuple key holding a non-builtin element keeps the generic merge path.

    A key guard that treats a key as "builtin" has to recurse into a ``tuple``
    and reject the whole key on the first element that is not a builtin scalar.
    What rides on it is observable: the generic
    ``HighLevelGraph.from_collections`` probes each dependency with
    ``is_dask_collection``, which builds a throw-away ``DelayedAttr(c, "expr")``
    whose ``tokenize`` calls ``repr()`` on the dependency's key
    (``dask/tokenize.py:33-39``). One ``repr`` call is therefore the pre-refactor
    behaviour for such a key -- measured -- and a shortcut that accepted the
    tuple would make that call disappear.

    The key cannot go into ``GOLDEN``: it holds an object, so it is neither
    JSON-expressible nor safe to serialise (the canonicaliser refuses a key it
    cannot encode reproducibly). The graph is therefore asserted here, directly.
    """
    key: Any = ("tk", _CountingKey())
    dependency = Delayed(key, {key: DataNode(key, 5)})

    _COUNTS["repr"] = 0
    result = delayed(ident, name="ident")(dependency)

    assert _COUNTS["repr"] == 1

    graph = result.dask
    assert isinstance(graph, HighLevelGraph)
    # Insertion order: new-then-existing, the order a single dependency carrying
    # a plain ``dict`` low-level graph produces
    # (``dask/highlevelgraph.py:462-465``).
    layer_names = list(graph.layers)
    assert len(layer_names) == 2
    assert layer_names[0] == result.key
    assert layer_names[1] is key
    assert list(graph.dependencies) == [result.key, key]
    assert graph.dependencies[result.key] == {key}
    assert graph.dependencies[key] == set()
    assert [node_key is key for node_key in dict(graph.layers[key])] == [True]
    assert list(result.__dask_keys__()) == [result.key]
    assert tuple(result.__dask_layers__()) == (result.key,)
    assert canonical_result([result]) == ((5,),)


def test_sequence_without_a_collection_detects_dependent_nodes() -> None:
    """A container holding a task-spec node is not handed back unchanged.

    The container branch of ``unpack_collections`` short-circuits -- returning the
    object itself with no collections -- only when nothing inside it contributes a
    graph dependency. A bare ``TaskRef`` contributes one, and so does a runnable
    node whose own ``dependencies`` are non-empty; a node without dependencies
    does not. All three verdicts are asserted together, because they are the
    three arms of the same decision.
    """
    task, collections = unpack_collections([TaskRef("dn"), 1])
    assert type(task).__name__ == "List"
    assert task.dependencies == {"dn"}
    assert collections == ()

    task, collections = unpack_collections([Task("inner", ident, TaskRef("dn")), 1])
    assert type(task).__name__ == "List"
    assert task.dependencies == {"dn"}
    assert collections == ()

    # A ``GraphNode`` with no dependencies of its own leaves the short-circuit
    # intact: the list is the task.
    literal_node = [DataNode("dn3", 7), 1]
    task, collections = unpack_collections(literal_node)
    assert task is literal_node
    assert collections == ()

    literal = [1, 2]
    task, collections = unpack_collections(literal)
    assert task is literal
    assert collections == ()


def test_single_element_container_unwraps_a_list_subclass_element() -> None:
    """A lone ``list``-subclass element is unwrapped by the container node itself.

    ``NestedContainer.__init__`` (``dask/_task_spec.py:851-853``) replaces its
    arguments with the single argument it was given whenever that argument is a
    ``list`` instance, and the test is an ``isinstance`` one, so a ``list``
    *subclass* is unwrapped too. The exact-type dispatch of
    ``unpack_collections`` leaves such a subclass atomic, so a ``TaskRef`` inside
    it is invisible to the traversal and becomes a dependency only because the
    node was built: the container branch has to keep constructing ``List(*args)``
    and read the verdict off the node, instead of concluding from the recursion's
    own verdicts that the container can be handed back unchanged.

    The boundary is the argument count. With one argument the element's contents
    are hoisted and the dependency appears; with two there is no unwrapping, the
    subclass stays an opaque element, the ``List`` carries no dependency and the
    branch short-circuits to the input object. Both sides are asserted here,
    along with the two shapes that could be mistaken for them: a plain inner
    ``list``, which takes the sequence branch itself so that the outer lone
    argument is a ``List`` node and nothing is unwrapped, and the subclass at
    top level, which takes the fallthrough and is returned as itself.
    """
    single_element = [_ListSubclass([TaskRef("k")])]
    task, collections = unpack_collections(single_element)
    assert type(task).__name__ == "List"
    assert task.dependencies == {"k"}
    # The lone element was unwrapped, so its ``TaskRef`` is a direct argument of
    # the node -- the only way the dependency can be derived at all.
    assert task.args == (TaskRef("k"),)
    assert collections == ()
    assert task is not single_element

    # Two arguments: no unwrapping, hence no dependency, and the branch's own
    # short-circuit hands back the input list object.
    two_elements = [_ListSubclass([TaskRef("k")]), 1]
    task, collections = unpack_collections(two_elements)
    assert task is two_elements
    assert collections == ()

    # A plain inner ``list`` is not an opaque element: it takes the sequence
    # branch, so the outer container's lone argument is already a ``List`` node
    # and ``isinstance(args[0], list)`` is false -- nothing is unwrapped, and the
    # dependency comes from the nested node.
    task, collections = unpack_collections([[TaskRef("k")]])
    assert type(task).__name__ == "List"
    assert task.dependencies == {"k"}
    assert len(task.args) == 1
    assert type(task.args[0]).__name__ == "List"
    assert collections == ()

    # The same subclass at top level: exact-type membership excludes it from the
    # container branch entirely.
    top_level = _ListSubclass([TaskRef("k")])
    task, collections = unpack_collections(top_level)
    assert task is top_level
    assert collections == ()

    # Unwrapping a subclass that holds no node contributes nothing, so the
    # constructed ``List`` is discarded in favour of the input object.
    without_a_node = [_ListSubclass([1, 2])]
    task, collections = unpack_collections(without_a_node)
    assert task is without_a_node
    assert collections == ()


def test_traversed_container_wrap_rewrites_the_container_node_key() -> None:
    """``delayed([a])`` keys the container node after the wrapper itself.

    Wrapping a traversed non-callable whose traversal produced a task takes the
    second branch of ``delayed()``: the node built by ``unpack_collections``
    carries no key of its own, so the wrapper's generated name is written onto it
    and becomes the single key of the new layer. ``GOLDEN`` compares the whole
    graph of both entries; this test names the property that makes the graph
    valid at all, on the live node object.
    """
    for name, kind, dependency_count in (
        ("wrap_list_with_delayed", "List", 1),
        ("wrap_dict_with_delayed", "Dict", 2),
    ):
        obj = _by_name(name).build()
        graph = obj.__dask_graph__()
        assert isinstance(graph, HighLevelGraph)
        node = graph.layers[obj.key][obj.key]
        assert type(node).__name__ == kind
        assert node.key == obj.key
        assert graph.dependencies[obj.key] == set(node.dependencies)
        assert len(node.dependencies) == dependency_count
        assert len(graph.layers) == dependency_count + 1


# ---------------------------------------------------------------------------
# Errors and warnings.
# ---------------------------------------------------------------------------


class _MultiKeyCollection:
    """A dask collection whose expression finalizes to more than one key.

    ``is_dask_collection`` accepts it because its ``expr`` is a real ``Expr``
    (``dask/base.py:243-254``), and ``collections_to_expr`` then uses that
    expression as it stands, so ``finalize_compute()`` keeps both outputs. The
    wrapper is necessary: handing the ``_ExprSequence`` to ``unpack_collections``
    directly fails earlier, on ``__dask_postcompute__``.
    """

    def __init__(self, expr: Any) -> None:
        self.expr = expr

    def __dask_graph__(self) -> Any:
        return self.expr.__dask_graph__()

    def __dask_keys__(self) -> Any:
        return self.expr.__dask_keys__()


#: Deterministically keyed corpus entries re-built under
#: ``tokenize.ensure-deterministic``. Their keys must be unchanged: strict mode
#: may not turn a deterministic token into a UUID, and nothing may turn a UUID
#: into a deterministic token.
_STRICT_DETERMINISTIC: tuple[str, ...] = (
    "arg_int",
    "arg_list_with_delayed",
    "arg_dict_delayed_key",
    "arg_dataclass_with_delayed",
    "arg_namedtuple_with_delayed",
    "call_default_name",
    "attr_v",
    "op_add",
)


@pytest.mark.parametrize("bad", [-1, "x", 1.5])
def test_nout_must_be_none_or_a_non_negative_int(bad: object) -> None:
    with pytest.raises(ValueError) as excinfo:
        delayed(inc, name="inc", nout=bad)
    assert (
        str(excinfo.value) == f"nout must be None or a non-negative integer, got {bad}"
    )


def test_nout_zero_is_a_length_of_zero_not_an_absent_length() -> None:
    """``nout=0`` keeps a zero length rather than an absent one.

    ``nout`` is validated as "None or a non-negative int" and stored verbatim in
    ``_length``, so zero and ``None`` are two different states: zero makes the
    ``Delayed`` a sized, iterable object with no elements, ``None`` makes it
    neither. Coercing one to the other -- the natural mistake for a falsy value on
    a construction path being rewritten -- would leave the golden key, graph and
    computed result of the ``nout_zero`` corpus entry untouched, so it is asserted
    here instead.
    """
    obj = delayed(nothing, name="nothing", pure=True, nout=0)()

    assert obj._length == 0
    assert obj._length is not None, "zero must not be stored, or read back, as None"
    assert len(obj) == 0
    assert list(obj) == []
    (result,) = canonical_result([obj])
    assert result == ()

    # The contrast that gives the assertions above their meaning: the same
    # callable without ``nout`` has no length at all and raises on both.
    absent = delayed(nothing, name="nothing", pure=True)()
    assert absent._length is None
    with pytest.raises(TypeError) as excinfo:
        len(absent)
    assert str(excinfo.value) == "Delayed objects of unspecified length have no len()"


def test_delayed_rejects_a_layer_absent_from_its_high_level_graph() -> None:
    graph = _hlg_of("present")
    with pytest.raises(ValueError) as excinfo:
        Delayed("present", graph, layer="absent")
    assert (
        str(excinfo.value)
        == "Layer absent not in the HighLevelGraph's layers: ['present']"
    )


def test_truth_iteration_and_length_raise_without_nout() -> None:
    obj = _leaf()
    with pytest.raises(TypeError) as truth:
        bool(obj)
    assert str(truth.value) == "Truth of Delayed objects is not supported"
    # ``__iter__`` is a generator function, so the body -- and the raise -- only
    # runs once the iterator is advanced.
    with pytest.raises(TypeError) as iteration:
        list(obj)
    assert (
        str(iteration.value) == "Delayed objects of unspecified length are not iterable"
    )
    with pytest.raises(TypeError) as length:
        len(obj)
    assert str(length.value) == "Delayed objects of unspecified length have no len()"


def test_dataclass_with_a_set_init_false_field_raises_value_error() -> None:
    @dataclass
    class ADataClass:
        a: Any
        b: int = field(init=False)

    def prepare(a: Any) -> ADataClass:
        data = ADataClass(a=a)
        data.b = 4
        return data

    with pytest.raises(ValueError) as excinfo:
        delayed(prepare(_leaf()))

    assert str(excinfo.value) == (
        f"Failed to unpack {ADataClass} instance. "
        "Note that using fields with `init=False` are not supported."
    )
    # The code chains the original ``replace()`` failure, whichever of the two
    # types it raised.
    assert isinstance(excinfo.value.__cause__, (ValueError, TypeError))


def test_dataclass_with_a_custom_init_raises_type_error() -> None:
    @dataclass
    class ADataClass:
        a: Any

        def __init__(self, b: Any) -> None:
            self.a = b

    with pytest.raises(TypeError) as excinfo:
        delayed({"data": ADataClass(b=_leaf())})

    assert str(excinfo.value) == (
        f"Failed to unpack {ADataClass} instance. "
        "Note that using a custom __init__ is not supported."
    )
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_private_attribute_access_raises_attribute_error() -> None:
    obj = _leaf()
    with pytest.raises(AttributeError) as excinfo:
        obj._foo
    assert str(excinfo.value) == "Attribute _foo not found"


def test_visualise_typo_warns_and_still_returns_a_delayed_attr() -> None:
    """The ``visualise`` spelling guard warns *and* keeps working.

    Both halves of the guard are asserted: the complete message -- including the
    suggested spelling, which is the entire point of the warning -- and the
    ``DelayedAttr`` that access still returns afterwards. The origin is checked
    too, and deliberately against ``dask/delayed.py`` rather than against this
    file: this ``warnings.warn`` call passes no ``stacklevel``, so the warning is
    attributed to the module that raises it, the opposite of the ``to_task_dask``
    warning below. Only the file is asserted, never a line number in a module
    this run is refactoring.
    """
    obj = _leaf()
    with pytest.warns(UserWarning) as record:
        attribute = obj.visualise
    assert len(record) == 1
    assert record[0].category is UserWarning
    assert str(record[0].message) == (
        "dask.delayed objects have no `visualise` method. "
        "Perhaps you meant `visualize`?"
    )
    assert (
        Path(record[0].filename).resolve()
        == Path(Delayed.__getattr__.__code__.co_filename).resolve()
    )
    assert isinstance(attribute, DelayedAttr)
    assert attribute._attr == "visualise"


def test_to_task_dask_still_warns_and_still_works() -> None:
    """The deprecated shim keeps its behaviour and its warning.

    ``pytest.warns`` both asserts and consumes the warning, so the test stays
    clean under the project's warnings-as-errors configuration.

    The ``stacklevel=2`` the shim passes is part of the frozen contract, and it
    is the one property of a warning that no message comparison can see, so it is
    asserted through the recorded origin: the user-facing warning must be
    attributed to *this file* and to the line that calls ``to_task_dask``. Level 1
    would point at ``dask/delayed.py`` and level 3 at pytest's own call frame, so
    both directions of drift fail. The expected line is read back out of this
    module's source rather than hard-coded, so inserting a line above cannot
    break it.
    """
    expected = (
        "The dask.delayed.to_dask_dask function has been "
        "Deprecated in favor of unpack_collections"
    )
    a = delayed(1, name="a")
    with pytest.warns(UserWarning) as record:
        task, graph = to_task_dask([a, 3])
    warnings_raised = list(record)
    # One warning per invocation, measured: the shim recurses into each element of
    # the list and warns again on every recursive call, so a two-element list
    # yields three warnings rather than one.
    assert len(warnings_raised) == 3
    assert {warning.category for warning in warnings_raised} == {UserWarning}
    assert {str(warning.message) for warning in warnings_raised} == {expected}
    # The user's own call is the first one, and ``stacklevel=2`` attributes it to
    # the caller's frame -- this file, at the calling line.
    assert Path(warnings_raised[0].filename).resolve() == Path(__file__).resolve()
    source = Path(__file__).read_text().splitlines()
    assert "to_task_dask([a, 3])" in source[warnings_raised[0].lineno - 1]
    # The recursive warnings are attributed to the shim's own frame, because for a
    # call made from inside the module that is what level 2 points at.
    for warning in warnings_raised[1:]:
        assert (
            Path(warning.filename).resolve()
            == Path(to_task_dask.__code__.co_filename).resolve()
        )
    assert task == ["a", 3]
    assert dict(graph) == dict(a.dask)


def test_a_task_used_as_a_task_callable_raises() -> None:
    with pytest.raises(TypeError) as excinfo:
        delayed(Task("t", inc, 1), name="t")(2)
    assert str(excinfo.value) == "Cannot nest tasks"


def test_collection_that_does_not_finalize_to_one_key_raises() -> None:
    a = delayed(1, name="a")
    b = delayed(2, name="b")
    collection = _MultiKeyCollection(
        _ExprSequence(collections_to_expr(a), collections_to_expr(b))
    )
    with pytest.raises(RuntimeError) as excinfo:
        unpack_collections(collection)
    assert str(excinfo.value) == (
        "Cannot unpack dask collections which don't finalize to a "
        f"single key. Got {_MultiKeyCollection} with keys=['a', 'b']"
    )


def test_strict_mode_raises_tokenization_error_for_a_generator() -> None:
    """Strict tokenization propagates unchanged through ``delayed``.

    A generator falls through ``unpack_collections`` to ``return expr, ()`` and is
    then tokenized by pickle, which fails and routes to
    ``_maybe_raise_nondeterministic`` (``dask/tokenize.py:83-89``), reached from
    the module-local ``dask.delayed.tokenize`` wrapper. The strict flag is read
    from the ``_ENSURE_DETERMINISTIC`` ContextVar first and the configuration key
    second, so ``dask.config.set`` is the right trigger.

    The message names the object that could not be hashed, so the expectation is
    built from that object's own ``repr`` -- the one dynamic value in it -- and
    the complete string is then compared. A fragment match would pass even if the
    offending object were no longer reported.
    """
    generator = _generate()
    expected = (
        f"Object {generator!r} cannot be deterministically hashed. This likely "
        "indicates that the object cannot be serialized deterministically."
    )
    with (
        dask.config.set({"tokenize.ensure-deterministic": True}),
        pytest.raises(TokenizationError) as excinfo,
    ):
        delayed(ident, name="ident", pure=True)(generator)
    assert str(excinfo.value) == expected


@pytest.mark.parametrize("name", _STRICT_DETERMINISTIC)
def test_no_silent_flip_deterministic_stays_deterministic(name: str) -> None:
    entry = _by_name(name)
    assert entry.deterministic
    with dask.config.set({"tokenize.ensure-deterministic": True}):
        obj = entry.build()
    # Strict mode may not change the key at all, which is asserted twice over:
    # against the same expression built outside strict mode -- portable, and the
    # property itself -- and against the golden, masked for the two entries whose
    # own token the environment decides rather than the module under test.
    assert obj.key == entry.build().key
    mask_live = _token_masker(name, obj.key)
    mask_golden = _token_masker(name, GOLDEN[name]["key"])
    assert mask_live(obj.key) == mask_golden(GOLDEN[name]["key"])


def test_no_silent_flip_nondeterministic_stays_nondeterministic() -> None:
    """A non-deterministically tokenizable input keeps failing, and keeps varying.

    The other half of the property: outside strict mode the generator is
    tokenized non-deterministically -- an md5-shaped token that differs per call,
    not a UUID and not a stable digest -- and inside strict mode it raises. No
    change may turn either outcome into the other.
    """
    with (
        dask.config.set({"tokenize.ensure-deterministic": True}),
        pytest.raises(TokenizationError),
    ):
        delayed(ident, name="ident", pure=True)(_generate())

    first = delayed(ident, name="ident", pure=True)(_generate())
    second = delayed(ident, name="ident", pure=True)(_generate())
    assert first.key.startswith("ident-")
    assert len(first.key) == len("ident-") + 32
    assert first.key != second.key


# ---------------------------------------------------------------------------
# Cross-scheduler equality and serializability.
#
# ``computable`` says only "this entry has a stable golden ``result_repr``". It
# is not a statement that the entry has no result worth checking, and it has
# nothing to do with whether the object pickles. The two tables below give the
# entries without a golden repr the assertion that *is* stable about them --
# a function identity, or an exception -- so the whole corpus takes part in the
# cross-scheduler and round-trip evidence, and the pickle round trip is gated on
# ``picklable`` alone.
# ---------------------------------------------------------------------------


#: Entries whose computed value is, or contains, a function object. Their value
#: is perfectly stable -- the same module-level function, by identity -- but its
#: ``repr`` embeds a memory address, which is the only reason they carry no
#: golden ``result_repr``. They are therefore compared by the module-qualified
#: name ``_result_signature`` produces, which is stable across schedulers, across
#: a pickle round trip and across processes.
_FUNCTION_VALUED_RESULTS: dict[str, Any] = {
    "wrap_func_pure": "dask.utils_test.inc",
    "arg_delayed_leaf": ("dask.utils_test.inc",),
}

#: Entries that are deliberately not runnable, with the exception type and the
#: message prefix that computing one raises. ``wrap_taskref`` wraps a ``TaskRef``,
#: a pointer to a key rather than a runnable node, so the scheduler cannot find
#: ``'dn'``; ``finalize_collection``'s graph is an ``HLGFinalizeCompute``
#: expression rather than a container, which the scheduler cannot copy;
#: ``arg_list_with_taskref`` and ``arg_list_with_dependent_task`` place a node
#: depending on ``'dn'`` inside a list argument, so the graph is rejected for the
#: missing dependency before anything runs. Raising is their observable
#: behaviour, so it is asserted rather than stepped around. Only the prefix is
#: pinned here -- the remainder of each message is dask's own generic advice or
#: the expression's own key, both recorded in full in ``_COMPUTE_FAILURES`` --
#: and equality between the two schedulers and across the pickle round trip is
#: asserted separately.
_NON_RUNNABLE_RESULTS: dict[str, tuple[type[BaseException], str]] = {
    "wrap_taskref": (KeyError, "'dn'"),
    "finalize_collection": (
        AttributeError,
        "'HLGFinalizeCompute' object has no attribute 'copy'",
    ),
    "arg_list_with_taskref": (
        ValueError,
        "Missing dependency dn for dependents ",
    ),
    "arg_list_with_dependent_task": (
        ValueError,
        "Missing dependency dn for dependents ",
    ),
}


def _result_signature(value: Any) -> Any:
    """Return a representation-independent signature of a computed value.

    Args:
        value: A computed result, possibly a list or tuple holding callables.

    Returns:
        Any: ``"<module>.<qualname>"`` for a callable, the same container kind
        holding the signatures of its elements for a list or a tuple, and the
        value itself for anything else. Only callables need this treatment: they
        compare equal by identity but their ``repr`` varies between processes.

    """
    if isinstance(value, tuple):
        return tuple(_result_signature(item) for item in value)
    if isinstance(value, list):
        return [_result_signature(item) for item in value]
    qualname = getattr(value, "__qualname__", None)
    if callable(value) and isinstance(qualname, str):
        return f"{getattr(value, '__module__', None)}.{qualname}"
    return value


def _computed_exception(obj: Any, scheduler: str) -> BaseException:
    """Compute an expression that is expected to fail and return its exception.

    Args:
        obj: The ``Delayed`` to compute.
        scheduler: Scheduler name passed to ``dask.compute``.

    Returns:
        BaseException: the exception the computation raised.

    Raises:
        AssertionError: if the computation succeeded. A deliberately non-runnable
            path that quietly became runnable is a behaviour change, so it fails
            here rather than being reported as a pass.

    """
    try:
        dask.compute(obj, scheduler=scheduler)
    except Exception as error:
        return error
    raise AssertionError(f"{obj.key!r} computed successfully under {scheduler!r}")


def test_every_corpus_entry_has_a_result_assertion() -> None:
    """No entry escapes the result evidence: every flag is backed by a table.

    ``test_flag_exclusions_are_documented`` proves each ``False`` flag carries a
    written reason; this proves each entry without a golden ``result_repr`` still
    has something asserted about its result. A new ``computable=False`` entry
    fails here until it is added to one of the two tables.

    The contract of ``_result_signature`` is locked here as well, since the
    strength of every function-valued comparison rests on it: a callable collapses
    to its module-qualified name, a container keeps its kind and maps its elements
    recursively, and every other value is returned untouched so it is still
    compared by value.
    """
    without_golden_result = {entry.name for entry in CORPUS if not entry.computable}
    covered = set(_FUNCTION_VALUED_RESULTS) | set(_NON_RUNNABLE_RESULTS)
    assert covered == without_golden_result, (
        "every entry without a golden result_repr must be covered by "
        "_FUNCTION_VALUED_RESULTS or _NON_RUNNABLE_RESULTS: "
        f"uncovered={sorted(without_golden_result - covered)}, "
        f"stale={sorted(covered - without_golden_result)}"
    )
    assert not set(_FUNCTION_VALUED_RESULTS) & set(_NON_RUNNABLE_RESULTS)

    assert _result_signature(inc) == "dask.utils_test.inc"
    assert _result_signature([(inc,), 3]) == [("dask.utils_test.inc",), 3]
    assert _result_signature((1, "s", None)) == (1, "s", None)


def test_results_match_across_schedulers() -> None:
    """The whole corpus behaves identically under ``sync`` and ``threads``.

    Three result shapes, all asserted and none skipped: entries with a golden
    ``result_repr`` are compared value for value; the two whose value is a
    function object are compared by module-qualified name, because only their
    ``repr`` is unstable; and the deliberately non-runnable entries are
    compared by the exception they raise, which for them *is* the result. The
    synchronous leg of the first group goes through ``canonical_result`` so that
    this test and the A/B harness share one definition of "the result of an
    expression".
    """
    entries = [entry for entry in CORPUS if entry.computable]
    objects = [entry.build() for entry in entries]

    synchronous = canonical_result(objects)
    threaded = dask.compute(*objects, scheduler="threads")

    assert len(synchronous) == len(entries)
    assert len(threaded) == len(entries)
    for entry, sync_value, threaded_value in zip(entries, synchronous, threaded):
        assert sync_value == threaded_value, f"{entry.name} differs across schedulers"

    for name, signature in _FUNCTION_VALUED_RESULTS.items():
        obj = _by_name(name).build()
        (sync_value,) = canonical_result([obj])
        (threaded_value,) = dask.compute(obj, scheduler="threads")
        assert _result_signature(sync_value) == signature, name
        assert _result_signature(threaded_value) == signature, name
        # Function objects compare by identity, so the plain equality the rest of
        # the corpus gets applies to these values too.
        assert sync_value == threaded_value, f"{name} differs across schedulers"

    for name, (exception_type, message) in _NON_RUNNABLE_RESULTS.items():
        obj = _by_name(name).build()
        synchronous_error = _computed_exception(obj, "sync")
        threaded_error = _computed_exception(obj, "threads")
        assert type(synchronous_error) is exception_type, name
        assert type(threaded_error) is exception_type, name
        assert str(synchronous_error).startswith(message), name
        assert str(threaded_error) == str(
            synchronous_error
        ), f"{name} fails differently across schedulers"


def _assert_round_tripped_result(entry: _Expr, obj: Any, restored: Any) -> None:
    """Assert an unpickled object produces the same result as the original.

    The three result shapes of the corpus are handled in the same order as in
    ``test_results_match_across_schedulers``, so an entry is covered by whichever
    of them applies rather than by being skipped.

    Args:
        entry: The corpus entry being round-tripped.
        obj: The object as built.
        restored: The same object after a pickle round trip.

    Raises:
        AssertionError: if the restored object's result -- its value, its function
            identity or the exception it raises -- differs from the original's.

    """
    if entry.computable:
        assert (
            repr(canonical_result([restored])[0]) == GOLDEN[entry.name]["result_repr"]
        ), entry.name
        return
    if entry.name in _FUNCTION_VALUED_RESULTS:
        (value,) = canonical_result([restored])
        assert (
            _result_signature(value) == _FUNCTION_VALUED_RESULTS[entry.name]
        ), entry.name
        return
    exception_type, message = _NON_RUNNABLE_RESULTS[entry.name]
    original_error = _computed_exception(obj, "sync")
    restored_error = _computed_exception(restored, "sync")
    assert type(original_error) is exception_type, entry.name
    assert type(restored_error) is exception_type, entry.name
    assert str(restored_error).startswith(message), entry.name
    assert str(restored_error) == str(original_error), entry.name


def test_objects_and_graphs_round_trip_through_pickle() -> None:
    """Every picklable object and its graph survive ``pickle`` at protocol 5.

    This is the serializability property non-synchronous schedulers depend on:
    the wrapped values, the tasks and the graph container all have to cross a
    process boundary. Corpus callables and types are module-level, so they pickle
    by reference.

    Eligibility is picklability and nothing else. Whether an entry has a stable
    golden ``result_repr`` says nothing about whether it survives ``pickle``, so
    the entries without one -- two that compute to a function object, four
    that are deliberately not runnable -- are round-tripped here like every other
    entry, and their results are asserted by ``_assert_round_tripped_result``
    through the property that is stable about them. The one entry that genuinely
    cannot be pickled is a ``types.MappingProxyType`` graph, which ``pickle``
    refuses outright; that exclusion is justified by name in
    ``_EXCLUSION_REASONS``, which ``test_flag_exclusions_are_documented`` keeps
    honest.

    Computable results are compared against the golden ``result_repr`` rather than
    with ``==`` because one entry's computed value contains a ``Delayed`` -- with
    ``traverse=False`` the object is quoted, not traversed -- and
    ``Delayed.__eq__`` is a lazy operator that builds a new ``Delayed`` instead of
    returning a boolean.
    """
    eligible = [entry for entry in CORPUS if entry.picklable]
    excluded = {entry.name for entry in CORPUS if not entry.picklable}
    assert len(eligible) >= 40, "the pickle round trip must cover the corpus floor"
    assert excluded <= set(_EXCLUSION_REASONS), (
        "an entry skipped by the pickle round trip must carry a reason: "
        f"{sorted(excluded - set(_EXCLUSION_REASONS))}"
    )

    for entry in eligible:
        obj = entry.build()
        graph = obj.__dask_graph__()

        restored = pickle.loads(pickle.dumps(obj, protocol=5))
        restored_graph = pickle.loads(pickle.dumps(graph, protocol=5))

        assert type(restored) is type(obj), entry.name
        assert restored.key == obj.key, entry.name
        assert list(restored.__dask_keys__()) == list(obj.__dask_keys__()), entry.name
        assert tuple(restored.__dask_layers__()) == tuple(obj.__dask_layers__())
        assert type(restored_graph) is type(graph), entry.name
        if isinstance(graph, HighLevelGraph):
            assert list(restored_graph.layers) == list(graph.layers), entry.name
            assert list(restored_graph.dependencies) == list(
                graph.dependencies
            ), entry.name
        else:
            # ``finalize()`` hands back an expression (``HLGFinalizeCompute``)
            # rather than a layer container, and an expression's identity is its
            # ``_name``; there is no ``layers`` mapping to compare.
            assert restored_graph._name == graph._name, entry.name
        if entry.canonical:
            # ``restored.key == obj.key`` is asserted just above, so the live
            # token is the right one to mask on the restored side too.
            mask_live = _token_masker(entry.name, obj.key)
            mask_golden = _token_masker(entry.name, GOLDEN[entry.name]["key"])
            assert mask_live(canonical_graph(restored)) == mask_golden(
                GOLDEN[entry.name]["graph"]
            ), entry.name
        _assert_round_tripped_result(entry, obj, restored)


# Longest chain of lazy attribute accesses whose ancestors the construction path
# still inspects before merging a dependency's graph; the 65th link sends the
# whole call to ``HighLevelGraph.from_collections`` (``dask/delayed.py:290``).
# Mirrored here rather than imported, because the bound is a property of the
# production path that these tests pin from the outside.
_ANCESTOR_BOUND = 64


class _SelfAttribute:
    """A value whose ``a`` attribute is the value itself.

    It makes an attribute chain of any length runnable: every ``getattr(x, "a")``
    task in the graph hands back the same object, so a chain past
    ``_ANCESTOR_BOUND`` links computes to that object instead of failing on a
    missing attribute.
    """

    @property
    def a(self) -> _SelfAttribute:
        """The value itself, so ``delayed_value.a.a.a`` stays computable."""
        return self


def _attribute_chain(links: int, value: Any) -> Delayed:
    """Stack ``links`` lazy attribute accesses over a wrapped value.

    Args:
        links: Number of ``DelayedAttr`` links to place above the root node.
        value: The value the root node holds.

    Returns:
        Delayed: The topmost link of the chain, or the root itself for zero
        links. The root carries a plain ``dict`` graph and a ``str`` key and
        layer name, so nothing but the chain's length can decide how the
        construction path treats it.

    """
    chain: Delayed = Delayed(
        "chain-root", {"chain-root": DataNode("chain-root", value)}
    )
    for _ in range(links):
        chain = chain.a
    return chain


def test_a_cyclic_attribute_chain_is_bounded_rather_than_walked_forever() -> None:
    """A cyclic ``DelayedAttr`` chain leaves the merge shortcut, never loops.

    ``_obj`` is a declared slot, so ``Delayed.__setattr__`` lets a caller assign
    it -- the asymmetry ``test_nominal_immutability_asymmetry_is_preserved``
    pins -- and build a chain that never reaches a root. The construction path
    inspects a dependency's ancestors before it merges the dependency's graph, so
    an unbounded inspection would spin here until the 300 s timeout. Bounded at
    ``_ANCESTOR_BOUND`` ancestors, it hands the cycle to the generic
    ``HighLevelGraph.from_collections`` instead, which resolves a cycle by
    recursion and so raises ``RecursionError``.

    That is the pre-refactor outcome too: the same construction against the
    frozen baseline arm (``benchmarks/delayed_ab/baseline_delayed.py`` installed
    as ``sys.modules["dask.delayed"]``), where the generic path is the only path,
    raises ``RecursionError`` as well. The raised exception is itself the evidence
    that the inspection terminated -- an unbounded walk never returns to raise
    anything.
    """
    root = Delayed("cyclic-root", {"cyclic-root": DataNode("cyclic-root", 1)})
    cyclic = root.a
    assert type(cyclic) is DelayedAttr
    cyclic._obj = cyclic
    assert cyclic._obj is cyclic

    with pytest.raises(RecursionError):
        delayed(ident, name="ident")(cyclic)

    # Nothing leaks from the failed construction: the tokenizer releases its lock
    # and its cycle registry on the way out, so the next construction in the same
    # process behaves exactly as it would have without the cycle.
    assert canonical_result([delayed(ident, name="ident")(1, 2)]) == ((1, 2),)


@pytest.mark.skipif(
    _BASELINE_ORACLE,
    reason=(
        f"{_BASELINE_ORACLE_ENV}=1: the pre-refactor module builds a dependency's "
        "graph twice per level, so this test's chain costs 2**links graph builds "
        "there and cannot terminate"
    ),
)
def test_chain_at_and_past_the_ancestor_bound_matches_the_generic_path() -> None:
    """Either side of the ancestor bound builds the generic path's graph.

    A chain of ``_ANCESTOR_BOUND`` links is merged directly; one link more sends
    the whole call to ``HighLevelGraph.from_collections``. The two have to be
    indistinguishable from the outside, so each is compared against
    ``from_collections`` given the same name, layer and dependency: the ordered
    ``layers``, the ordered ``dependencies``, every layer's dependency set and
    every layer's contents, plus the value the graph computes to.

    An arm-to-arm comparison is unavailable at this depth and always will be,
    which is a property of the pre-refactor code rather than a gap here: the
    generic path builds a dependency's graph twice per level, so an attribute
    chain costs 2**links graph builds there -- measured on the frozen baseline
    arm at 8.2 s for 18 links and over 45 s for 22. The same shape at 0 to 3
    links is what ``attr_v``/``attr_of_attr`` compare against the golden, and
    ``test_attribute_chain_over_an_unsafe_ancestor_reprs_its_key_as_today`` pins
    the side-effect counts those levels perform.

    For the same reason this is the one test that cannot run in the pre-refactor
    oracle run, where the whole module is executed against the baseline module in
    a base-commit worktree: it would hold the interpreter until the 300 s
    ``timeout_method = "thread"`` limit, which takes the whole session down
    rather than the one test. Setting ``DASK_DELAYED_BASELINE_ORACLE=1`` skips it
    and nothing else, which makes that run mechanical; see the module docstring.
    """
    for links in (_ANCESTOR_BOUND, _ANCESTOR_BOUND + 1):
        value = _SelfAttribute()
        chain = _attribute_chain(links, value)
        node = delayed(ident, name="ident")(chain)

        graph = node.__dask_graph__()
        assert isinstance(graph, HighLevelGraph)
        # The new node's layer, one layer per link, and the root's.
        assert len(graph.layers) == links + 2

        generic = HighLevelGraph.from_collections(
            node.key, dict(graph.layers[node.key]), dependencies=[chain]
        )
        assert list(generic.layers) == list(graph.layers), f"{links} links"
        assert list(generic.dependencies) == list(graph.dependencies), f"{links} links"
        for layer_name in graph.layers:
            assert set(generic.dependencies[layer_name]) == set(
                graph.dependencies[layer_name]
            ), f"{links} links, layer {layer_name}"
            assert dict(generic.layers[layer_name]) == dict(
                graph.layers[layer_name]
            ), f"{links} links, layer {layer_name}"

        assert canonical_result([node]) == ((value,),)
