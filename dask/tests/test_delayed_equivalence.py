"""Characterisation of every observable property of ``dask.delayed``.

This module is the equivalence evidence for a purely internal performance
refactor of ``dask/delayed.py``. It locks, for a corpus of delayed expressions
that reaches every branch of ``dask.delayed.unpack_collections``:

* the generated key strings -- character for character wherever they are
  deterministic, and structurally (UUID tokens replaced by placeholders) where
  they are inherently random;
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

Portability:
    Deterministic tokens of anything tokenized through pickle depend on which
    optional hash library is installed, so the autouse fixture pins the hasher
    and the two relevant configuration keys. See ``_CONFIG_PINS``.

Notes:
    The canonicaliser is imported from ``benchmarks/delayed_ab/canon.py`` and is
    never re-implemented here: the A/B performance harness and this test have to
    share one definition of "equivalent graph" or the two bodies of evidence can
    drift apart.
"""

from __future__ import annotations

import contextlib
import pickle
import pprint
import types
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import pytest

import dask
import dask.hashing
from benchmarks.delayed_ab.canon import canonical_graph, canonical_result, normalize_key
from dask._expr import _ExprSequence
from dask._task_spec import DataNode, Task, TaskRef
from dask.base import collections_to_expr
from dask.delayed import (
    Delayed,
    DelayedAttr,
    delayed,
    finalize,
    to_task_dask,
    unpack_collections,
)
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as _threaded_get
from dask.tokenize import TokenizationError
from dask.utils_test import inc

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

    Every mutation is reverted: ``monkeypatch`` restores the list and the
    ``dask.config.set`` context restores the configuration, so the module leaves
    no residue for the rest of the session.
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
    """Return the positional arguments unchanged, as a tuple."""
    return args


def collect(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return both argument groups, so keyword arguments reach the result."""
    return args, kwargs


def listof(*args: Any) -> list[Any]:
    """Return the positional arguments as a list, a ``getitem`` target."""
    return list(args)


def first(x: Any) -> Any:
    """Return the single argument unchanged; used to wrap a callable value."""
    return x


def pair() -> tuple[int, int]:
    """Return a two-element tuple, for ``nout=2``."""
    return (1, 2)


def single() -> tuple[int]:
    """Return a one-element tuple, for ``nout=1``."""
    return (7,)


def nothing() -> tuple[Any, ...]:
    """Return an empty tuple, for ``nout=0``."""
    return ()


class Inner:
    """Attribute target for the attribute-of-attribute corpus entry."""

    def __init__(self, w: int) -> None:
        self.w = w


class Obj:
    """Plain object wrapped by ``delayed`` for the attribute and method entries."""

    def __init__(self, v: int) -> None:
        self.v = v
        self.items = [10, 11, 12]
        self.inner = Inner(9)

    def meth(self, x: int) -> int:
        """Return ``self.v + x``."""
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
    """Dataclass used both with literal fields and with a field holding a ``Delayed``."""

    a: Any
    b: str


class Point(NamedTuple):
    """Namedtuple used both with literal fields and with a field holding a ``Delayed``."""

    x: Any
    y: int


class HandRolledCollection:
    """A non-``Delayed`` dask collection, modelled on ``test_delayed.py``'s ``Tuple``.

    This is what drives the ``base.is_dask_collection(expr)`` branch of
    ``unpack_collections`` (``dask/delayed.py:177-195``) without requiring NumPy,
    so the branch is covered in a minimal-dependency environment too.
    """

    __dask_scheduler__ = staticmethod(_threaded_get)
    __dask_optimize__ = None

    def __init__(self, dsk: dict[str, Any], keys: list[str]) -> None:
        self._dask = dsk
        self._keys = keys

    def __dask_tokenize__(self) -> list[str]:
        return self._keys

    def __dask_graph__(self) -> dict[str, Any]:
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
    import time (``Delayed._bind_operator``, ``dask/delayed.py:896-927``), so a
    static checker cannot see them and the operator corpus entries below would
    not type-check against a ``Delayed`` annotation.
    """
    return delayed(inc, name="inc", pure=True)(1)


def _other_leaf() -> Any:
    """Return a second dependency with a different key, for two-dependency shapes."""
    return delayed(inc, name="inc-other", pure=True)(2)


def _ident() -> Any:
    """Return ``delayed(ident)`` with an explicit ``name``.

    The explicit name keeps the resulting call key derived from strings, ints and
    dependency keys only -- ``call_function`` tokenizes ``self._key`` rather than
    the callable (``dask/delayed.py:811``) -- so the key does not depend on how
    the function object pickles. Entries that deliberately exercise the default
    naming forms are listed in ``_DEFAULT_NAMING``.
    """
    return delayed(ident, name="ident", pure=True)


def _collect() -> Any:
    """Return ``delayed(collect)`` with an explicit name, for keyword arguments."""
    return delayed(collect, name="collect", pure=True)


def _obj() -> Delayed:
    """Return a ``delayed``-wrapped :class:`Obj` with an explicit name."""
    return delayed(Obj(3), name="obj", pure=True)


def _hand_rolled_collection() -> HandRolledCollection:
    """Return a two-key hand-rolled dask collection with a low-level graph."""
    return HandRolledCollection(
        {"ta": DataNode("ta", 1), "tb": DataNode("tb", 2)}, ["ta", "tb"]
    )


def _hlg_of(key: str) -> HighLevelGraph:
    """Return a single-layer ``HighLevelGraph`` holding one ``DataNode``."""
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
    ``Delayed.__init__`` only validates the layer for a ``HighLevelGraph``
    (``dask/delayed.py:688``), so construction succeeds and the object is a valid
    dependency that no exact-graph-type guard may take a shortcut for.
    """
    return Delayed("mpkey", types.MappingProxyType({"mpkey": DataNode("mpkey", 5)}))


def _sub_delayed() -> SubDelayed:
    """Return a :class:`SubDelayed` dependency backed by a ``HighLevelGraph``."""
    return SubDelayed("subkey", _hlg_of("subkey"))


def _sub_delayed_dask_layers() -> SubDelayedLayers:
    """Return a :class:`SubDelayedLayers` dependency backed by a ``HighLevelGraph``."""
    return SubDelayedLayers("sublayerskey", _hlg_of("sublayerskey"))


def _sub_delayed_layer_dict() -> SubDelayedLayerDict:
    """Return a :class:`SubDelayedLayerDict` dependency backed by a ``HighLevelGraph``."""
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
            and is therefore recorded in the golden and compared.
        computable: True when the entry takes part in the result assertions.
        picklable: True when the entry takes part in the pickle round trip.

    Every ``False`` flag is justified by name in ``_EXCLUSION_REASONS``, which
    ``test_flag_exclusions_are_documented`` keeps in step with this registry.
    """

    name: str
    build: Callable[[], Any]
    deterministic: bool = True
    canonical: bool = True
    computable: bool = True
    picklable: bool = True


def _build_duplicate_positional() -> Delayed:
    """``f(a, a)``: one dependency passed twice, de-duplicated by ``id``."""
    a = _leaf()
    return _ident()(a, a)


def _build_duplicate_in_list() -> Delayed:
    """``f([a, a, a])``: one dependency three times inside a container."""
    a = _leaf()
    return _ident()([a, a, a])


def _build_global_delayed_pure() -> Delayed:
    """Take the deterministic key from the *global* ``delayed_pure`` setting.

    ``pure`` is omitted, so ``dask.delayed.tokenize`` reads
    ``config.get("delayed_pure", False)`` on every call (``dask/delayed.py:405``).
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
    """Unpack the single element of an ``nout=1`` call."""
    (x,) = delayed(single, name="single", pure=True, nout=1)()
    return x


def _build_nout_two_first() -> Delayed:
    """Unpack the first element of an ``nout=2`` call."""
    x, _y = delayed(pair, name="pair", pure=True, nout=2)()
    return x


def _build_nout_two_second() -> Delayed:
    """Unpack the second element of an ``nout=2`` call."""
    _x, y = delayed(pair, name="pair", pure=True, nout=2)()
    return y


def _build_delayed_call(pure: bool | None) -> Delayed:
    """Call a ``Delayed`` whose computed value is itself a callable.

    ``Delayed.__call__`` routes through ``delayed(apply, pure=pure)``
    (``dask/delayed.py:782-786``), so with ``pure`` omitted both the ``apply``
    leaf key and the call key carry UUID tokens.

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
    # Atomic arguments: the branch cascade falls through to ``return expr, ()``
    # (``dask/delayed.py:293``) for each of them.
    _Expr("arg_int", lambda: _ident()(1)),
    _Expr("arg_float", lambda: _ident()(1.5)),
    _Expr("arg_str", lambda: _ident()("s")),
    _Expr("arg_none", lambda: _ident()(None)),
    _Expr("arg_bool", lambda: _ident()(True)),
    _Expr("arg_bytes", lambda: _ident()(b"xy")),
    _Expr("arg_complex", lambda: _ident()(complex(1, 2))),
    # Wrapped non-callables and the two default naming forms
    # (``dask/delayed.py:632-638`` and ``:642``).
    _Expr("wrap_int_named", lambda: delayed(3, name="three")),
    _Expr("wrap_int_default", lambda: delayed(3), deterministic=False),
    _Expr("wrap_str_pure", lambda: delayed("s", pure=True)),
    _Expr("wrap_obj_pure", lambda: delayed(Obj(3), pure=True)),
    # ``delayed(inc, pure=True)`` computes to the function object itself, whose
    # repr embeds its address, so it is excluded from the result assertions.
    _Expr("wrap_func_pure", lambda: delayed(inc, pure=True), computable=False),
    _Expr("call_default_name", lambda: delayed(inc, pure=True)(1)),
    _Expr(
        "wrap_list_traverse_false_default",
        lambda: delayed([1, 2], traverse=False),
        deterministic=False,
    ),
    _Expr("wrap_taskref", lambda: delayed(TaskRef("dn")), computable=False),
    _Expr("wrap_datanode", lambda: delayed(DataNode("dn", 5), name="dn")),
    # list/tuple/set branch (``dask/delayed.py:206-221``).
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
    # dict branch (``dask/delayed.py:223-236``) and the kwargs dict that
    # ``call_function`` unpacks (``:818``) -- the path an empty-kwargs
    # short-circuit must keep equivalent.
    _Expr("arg_dict_literal", lambda: _ident()({"k": 1})),
    _Expr("arg_dict_delayed_value", lambda: _ident()({"k": _leaf()})),
    _Expr("arg_dict_delayed_key", lambda: _ident()({_leaf(): 1})),
    _Expr("kwargs_literal", lambda: _collect()(x=1)),
    _Expr("kwargs_with_delayed", lambda: _collect()(x=_leaf())),
    # slice branch (``dask/delayed.py:238-247``).
    _Expr("arg_slice_literal", lambda: _ident()(slice(1, 5, 2))),
    _Expr("arg_slice_with_delayed", lambda: _ident()(slice(_leaf(), 5, None))),
    # dataclass branch (``dask/delayed.py:249-281``).
    _Expr("arg_dataclass_literal", lambda: _ident()(Box(a=1, b="s"))),
    _Expr("arg_dataclass_with_delayed", lambda: _ident()(Box(a=_leaf(), b="s"))),
    # namedtuple branch (``dask/delayed.py:283-291``).
    _Expr("arg_namedtuple_literal", lambda: _ident()(Point(x=1, y=2))),
    _Expr("arg_namedtuple_with_delayed", lambda: _ident()(Point(x=_leaf(), y=2))),
    # Iterator coercion (``dask/delayed.py:197-202``).
    _Expr("arg_list_iterator", lambda: _ident()(iter([1, 2]))),
    # The layer-order lock: ``call_function`` tokenizes the *raw* iterator
    # (``dask/delayed.py:811``) before ``unpack_collections`` coerces it, so
    # ``tokenize`` pickles the underlying list -- including the ``Delayed`` it
    # holds. A ``Delayed``'s pickled slot state contains its ``HighLevelGraph``,
    # whose layer *insertion order* therefore feeds this key's token.
    _Expr("arg_list_iterator_with_delayed", lambda: _ident()(iter([_leaf(), 1]))),
    _Expr("arg_tuple_iterator", lambda: _ident()(iter((1, 2)))),
    _Expr("arg_set_iterator", lambda: _ident()(iter({1}))),
    # Nesting and duplication.
    _Expr("arg_nested_mixed", lambda: _ident()([{"k": (_leaf(), 1)}, [2, _leaf()]])),
    _Expr("arg_nested_list_depth3", lambda: _ident()([[[_leaf()]]])),
    _Expr("arg_duplicate_positional", _build_duplicate_positional),
    _Expr("arg_duplicate_in_list", _build_duplicate_in_list),
    _Expr("arg_two_dependencies", lambda: _ident()(_leaf(), _other_leaf())),
    # Both arms of ``DelayedLeaf.dask`` (``dask/delayed.py:838-844``) as a
    # dependency: a plain object becomes a ``DataNode``, a ``GraphNode`` is used
    # as it stands. Plus a dependency carrying a plain ``dict`` graph.
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
    # Non-``Delayed`` dask collection (``dask/delayed.py:177-195``). Its graph
    # acquires a ``finalize-hlgfinalizecompute-<hex>-<hex>`` layer whose hex is a
    # ``uuid4().hex`` fallback, so no canonical form is reproducible; the output
    # key and the computed result are (see ``_EXCLUSION_REASONS``).
    _Expr(
        "arg_hand_rolled_collection",
        lambda: _ident()(_hand_rolled_collection()),
        canonical=False,
    ),
    _Expr(
        "finalize_collection",
        _build_finalize_collection,
        canonical=False,
        computable=False,
    ),
    # ``pure`` semantics (``dask/delayed.py:392-410``).
    _Expr("call_pure_true", lambda: delayed(inc, name="inc", pure=True)(1)),
    _Expr(
        "call_pure_false",
        lambda: delayed(inc, name="inc", pure=False)(1),
        deterministic=False,
    ),
    _Expr("call_global_delayed_pure", _build_global_delayed_pure),
    # Explicit naming (``dask/delayed.py:810-813``).
    _Expr(
        "call_dask_key_name",
        lambda: delayed(inc, name="inc", pure=True)(
            1, dask_key_name="explicit-call-key"
        ),
    ),
    # ``nout`` (``dask/delayed.py:627-628``, ``:771-780``, ``:826``).
    _Expr("nout_none", lambda: delayed(pair, name="pair", pure=True)()),
    _Expr("nout_zero", lambda: delayed(nothing, name="nothing", pure=True, nout=0)()),
    _Expr("nout_one", lambda: delayed(single, name="single", pure=True, nout=1)()),
    _Expr("nout_one_element", _build_nout_one_element),
    _Expr("nout_two", lambda: delayed(pair, name="pair", pure=True, nout=2)()),
    _Expr("nout_two_unpacked_first", _build_nout_two_first),
    _Expr("nout_two_unpacked_second", _build_nout_two_second),
    # Unpacking routes through the same bound ``getitem`` operator, so this entry
    # differs from ``nout_two_unpacked_second`` only in how it is written.
    _Expr(
        "nout_two_getitem_one",
        lambda: delayed(pair, name="pair", pure=True, nout=2)()[1],
    ),
    # ``traverse=False`` (``dask/delayed.py:621-625``): the object is quoted and
    # no dependency is collected, so the ``Delayed`` inside survives into the
    # computed value as an object.
    _Expr(
        "traverse_false_with_delayed",
        lambda: delayed([_leaf(), 1], traverse=False, name="quoted"),
    ),
    # Lazy attribute access (``dask/delayed.py:743-755``, ``:883-888``). These
    # layers hold the legacy tuple task ``(getattr, key, attr)``, which the
    # graph-shape contract freezes; the canonicaliser reports it as
    # ``"legacy-tuple"``.
    _Expr("attr_v", lambda: _obj().v),
    _Expr("attr_of_attr", lambda: _obj().inner.w),
    _Expr("attr_items_getitem", lambda: _obj().items[1]),
    # Method calls (``dask/delayed.py:890-893``). ``DelayedAttr.__call__`` does
    # not forward ``pure``, so a method call is impure unless ``pure=True`` is
    # passed in the call kwargs, where ``call_function`` pops it (``:808``).
    _Expr("method_pure", lambda: _obj().meth(2, pure=True)),
    _Expr("method_impure", lambda: _obj().meth(2), deterministic=False),
    # Operators (``dask/delayed.py:798-803``, ``:896-927``).
    _Expr("op_add", lambda: _leaf() + _other_leaf()),
    _Expr("op_reflected_add", lambda: 1 + _leaf()),
    _Expr("op_neg", lambda: -_leaf()),
    _Expr("op_lt", lambda: _leaf() < _other_leaf()),
    _Expr("op_getitem", lambda: delayed(listof, name="listof", pure=True)(1, 2, 3)[1]),
    # ``Delayed.__call__`` on a delayed callable (``dask/delayed.py:782-786``).
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
        "computing it raises KeyError('dn')"
    ),
    "arg_delayed_leaf": (
        "computable=False: the dependency is a DelayedLeaf wrapping a function, so the "
        "result holds the function object, whose repr embeds its memory address; "
        "arg_delayed_value_leaf covers the same DataNode arm with a stable result"
    ),
    "arg_hand_rolled_collection": (
        "canonical=False: routing a non-Delayed collection through unpack_collections adds a "
        "finalize-hlgfinalizecompute-<hex>-<hex> layer whose hex comes from uuid4().hex "
        "(dask/_expr.py), so the layer name -- and with it the canonical graph -- is not "
        "reproducible; the deterministic output key and the computed result are asserted "
        "instead, as benchmarks/delayed_ab/canon.py prescribes"
    ),
    "finalize_collection": (
        "canonical=False and computable=False: finalize() returns a Delayed whose graph is an "
        "HLGFinalizeCompute expression rather than a HighLevelGraph or a dict, so "
        "canonical_graph raises TypeError('HLGFinalizeCompute' object is not iterable) and "
        "computing raises AttributeError('HLGFinalizeCompute' object has no attribute 'copy'); "
        "the entry exists to lock the 'finalize-<token>' key form"
    ),
    "guard_mapping_proxy_graph": (
        "picklable=False: the dependency's graph is a types.MappingProxyType, which pickle "
        "refuses (TypeError: cannot pickle 'mappingproxy' object)"
    ),
}

#: Entries that deliberately keep ``delayed``'s default naming, so the
#: ``obj.__name__-<token>`` (``dask/delayed.py:632-638``) and
#: ``type(obj).__name__-<token>`` (``:642``) key forms are locked under the
#: pinned hasher rather than derived from an explicit ``name=``.
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
)

#: Every branch of ``unpack_collections`` (``dask/delayed.py:115-293``) mapped to
#: one corpus entry that exercises it, asserted by
#: ``test_every_unpack_collections_branch_is_covered``.
_BRANCH_COVERAGE: dict[str, str] = {
    "Delayed short-circuit (:166-172)": "arg_list_with_delayed",
    "non-Delayed dask collection (:177-195)": "arg_hand_rolled_collection",
    "list iterator coercion (:197-198)": "arg_list_iterator",
    "tuple iterator coercion (:199-200)": "arg_tuple_iterator",
    "set iterator coercion (:201-202)": "arg_set_iterator",
    "list/tuple/set literal short-circuit (:214-215)": "arg_list_literal",
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


def _by_name(name: str) -> _Expr:
    """Return the corpus entry registered under ``name``.

    Args:
        name: Registry key of the wanted entry.

    Returns:
        _Expr: the matching entry.

    Raises:
        KeyError: if no entry is registered under that name.
    """
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
    },
    "arg_hand_rolled_collection": {
        "graph": None,
        "key": "ident-99645681c6f883e003c9b0070e583172",
        "result_repr": "((1, 2),)",
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
                            [],
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
                            [],
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
                            [],
                            None,
                        ]
                    ],
                ],
                ["obj", "MaterializedLayer", [["obj", "DataNode", [], None]]],
            ],
        },
        "key": "getattr-e6dab9e42a108e1d46bf06102a2d0a0b",
        "result_repr": "9",
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
                            [],
                            None,
                        ]
                    ],
                ],
                ["obj", "MaterializedLayer", [["obj", "DataNode", [], None]]],
            ],
        },
        "key": "getattr-b73ec8674b3d2dc92415a8be924f4cbf",
        "result_repr": "3",
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
    },
    "finalize_collection": {
        "graph": None,
        "key": "finalize-6bd795b2e9accac4e918747e25da4d38",
        "result_repr": None,
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
    },
}
# --- END GOLDEN ---


def _golden_entry(entry: _Expr) -> dict[str, Any]:
    """Build one corpus entry and return the three fields the golden records.

    Args:
        entry: The corpus entry to characterise.

    Returns:
        dict: ``{"key": ..., "graph": ..., "result_repr": ...}``. ``graph`` is the
        full ``canonical_graph`` dict, or ``None`` for an entry whose canonical
        form is not reproducible. ``result_repr`` is the ``repr`` of the
        synchronously computed result, or ``None`` for an entry excluded from the
        result assertions. Both exclusions are justified by name in
        ``_EXCLUSION_REASONS``.

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
    records the three golden fields per entry, and rewrites *only* the text
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


@pytest.mark.parametrize("entry", CORPUS, ids=[entry.name for entry in CORPUS])
def test_characterisation(entry: _Expr) -> None:
    """Assert the golden key, canonical graph and computed result of one entry.

    This is the assertion the whole refactor is measured against: every field
    compared here is an observable property of ``dask.delayed`` that a purely
    internal performance change must leave untouched.
    """
    golden = GOLDEN[entry.name]
    obj = entry.build()
    table: dict[str, str] = {}
    normalised_key = normalize_key(obj.key, table)

    if entry.deterministic:
        # Character for character, 32-hex token included: a deterministic token is
        # part of the contract because ``pure=True`` shares work by key identity,
        # so a changed token silently changes which nodes are deduplicated.
        assert obj.key == golden["key"]
    else:
        assert obj.key != golden["key"], "a UUID token must not equal its placeholder"
    assert normalised_key == golden["key"]

    if entry.canonical:
        graph = canonical_graph(obj)
        # Field by field first, so a failure names the property that moved, then
        # as a whole, so nothing outside those fields can drift unnoticed.
        assert graph["key"] == golden["graph"]["key"]
        assert graph["dask_keys"] == golden["graph"]["dask_keys"]
        assert graph["dask_layers"] == golden["graph"]["dask_layers"]
        assert graph["layer_order"] == golden["graph"]["layer_order"]
        assert graph["dependency_order"] == golden["graph"]["dependency_order"]
        assert graph["layers"] == golden["graph"]["layers"]
        assert graph["dependencies"] == golden["graph"]["dependencies"]
        assert graph == golden["graph"]

        materialised = obj.__dask_graph__()
        assert isinstance(materialised, HighLevelGraph)
        if entry.deterministic:
            # Spelled out against the live container rather than only against the
            # canonical dict. Layer insertion order is the single most
            # refactor-sensitive property of the graph: ``tokenize`` hashes a
            # ``Delayed``'s pickled slot state, ``HighLevelGraph`` included, so two
            # graphs with identical content in a different order tokenize
            # differently wherever a ``Delayed`` is reached through pickle.
            assert list(materialised.layers) == golden["graph"]["layer_order"]
            assert (
                list(materialised.dependencies) == golden["graph"]["dependency_order"]
            )
            assert list(obj.__dask_keys__()) == golden["graph"]["dask_keys"]
            assert list(obj.__dask_layers__()) == golden["graph"]["dask_layers"]
    else:
        assert golden["graph"] is None
        assert entry.name in _EXCLUSION_REASONS

    if entry.computable:
        (result,) = canonical_result([obj])
        assert repr(result) == golden["result_repr"]
    else:
        assert golden["result_repr"] is None
        assert entry.name in _EXCLUSION_REASONS


def test_delayed_of_a_delayed_returns_the_same_object() -> None:
    """``delayed`` short-circuits on a ``Delayed`` (``dask/delayed.py:618-619``)."""
    existing = _leaf()
    assert delayed(existing) is existing
    assert delayed(existing, name="ignored", pure=True, nout=3) is existing


def test_corpus_floor_and_golden_cover_each_other() -> None:
    """The corpus meets its floor, has unique names and matches the golden exactly."""
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

    ``obj.__name__-<token>`` (``dask/delayed.py:632-638``) and
    ``type(obj).__name__-<token>`` (``:642``) are only exercised when no explicit
    ``name=`` is passed, and their tokens depend on the pinned hasher.
    """
    names = {entry.name for entry in CORPUS}
    assert len(_DEFAULT_NAMING) >= 5
    assert set(_DEFAULT_NAMING) <= names
    # ``type(obj).__name__`` forms.
    assert GOLDEN["wrap_int_default"]["key"].startswith("int-")
    assert GOLDEN["wrap_str_pure"]["key"].startswith("str-")
    assert GOLDEN["wrap_obj_pure"]["key"].startswith("Obj-")
    assert GOLDEN["wrap_list_traverse_false_default"]["key"].startswith("list-")
    # ``obj.__name__`` forms, for the leaf and for the call it keys.
    assert GOLDEN["wrap_func_pure"]["key"].startswith("inc-")
    assert GOLDEN["call_default_name"]["key"].startswith("inc-")


def test_every_unpack_collections_branch_is_covered() -> None:
    """Each branch of ``unpack_collections`` maps to a corpus entry that reaches it."""
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
    # only validates the layer for a ``HighLevelGraph`` (``dask/delayed.py:688``).
    # Compared by exact type rather than ``isinstance``, and with the positive
    # check last: ``MappingProxyType`` is ``final``, so once a static checker has
    # narrowed the expression to it, any further class comparison is reported as
    # an impossible subclass relation.
    graph = proxy.dask
    assert type(graph) is not HighLevelGraph
    assert type(graph) is not dict
    assert type(graph) is types.MappingProxyType


def test_flag_exclusions_are_documented() -> None:
    """Every ``False`` flag in the corpus is justified by name, and none is stale."""
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
        raise AssertionError("the construction path must never hash a type object")


class _WithCountingMeta(metaclass=_CountingMeta):
    """An ordinary value whose *type* observes equality comparisons."""


def test_dependency_key_is_repred_once_for_an_impure_call() -> None:
    """An impure call ``repr()``s a user-defined dependency key exactly once.

    Mechanism, all of it observable: ``HighLevelGraph.from_collections`` probes
    each dependency with ``is_dask_collection``, which reads ``c.expr``
    (``dask/base.py:247-253``); ``Delayed.__getattr__`` turns that into a
    throw-away ``DelayedAttr(c, "expr")`` whose ``__init__`` tokenizes
    (``dask/delayed.py:743-755``, ``:867-868``). Inside ``tokenize``,
    ``normalize_object`` sees ``__dask_tokenize__`` and returns the dependency's
    key *as is*, without recursive dispatch (``dask/tokenize.py:194-198``), and
    ``_tokenize`` then calls ``str()`` on the normalised tuple
    (``dask/tokenize.py:33-39``) -- which invokes the key's ``__repr__``.

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
    (``dask/delayed.py:688``) happens before the counter is reset, so it is
    deliberately not part of the count.
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


def test_unpack_collections_compares_types_three_times_and_never_hashes() -> None:
    """``unpack_collections`` compares the value's type three times, hashing none.

    The three comparisons are the tuple-membership test ``typ in (list, tuple,
    set)`` (``dask/delayed.py:206``), which uses ``==`` element-wise and never
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
    expect, because the asymmetry is the current observable behaviour
    (``dask/delayed.py:757-770``) and no part of this work strengthens it.
    """
    obj = _leaf()

    # A declared slot name succeeds and reads back.
    obj._length = 5
    assert obj._length == 5
    assert len(obj) == 5

    with pytest.raises(TypeError, match="Delayed objects are immutable"):
        obj.foo = 1
    with pytest.raises(TypeError, match="Delayed objects are immutable"):
        obj[0] = 1

    # There is no ``__delattr__`` override, so slot deletion has its default
    # semantics; afterwards the read routes through ``__getattr__``
    # (``dask/delayed.py:743-746``).
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
    """``nout`` validation (``dask/delayed.py:627-628``)."""
    with pytest.raises(
        ValueError, match="nout must be None or a non-negative integer, got"
    ):
        delayed(inc, name="inc", nout=bad)


def test_delayed_rejects_a_layer_absent_from_its_high_level_graph() -> None:
    """``Delayed.__init__`` validates ``layer`` against the HLG (``:688-691``)."""
    graph = _hlg_of("present")
    with pytest.raises(ValueError, match="not in the HighLevelGraph's layers"):
        Delayed("present", graph, layer="absent")


def test_truth_iteration_and_length_raise_without_nout() -> None:
    """``bool``/iteration/``len`` on a length-less ``Delayed`` (``:771-789``)."""
    obj = _leaf()
    with pytest.raises(TypeError, match="Truth of Delayed objects is not supported"):
        bool(obj)
    # ``__iter__`` is a generator function, so the body -- and the raise -- only
    # runs once the iterator is advanced.
    with pytest.raises(
        TypeError, match="Delayed objects of unspecified length are not iterable"
    ):
        list(obj)
    with pytest.raises(
        TypeError, match="Delayed objects of unspecified length have no len"
    ):
        len(obj)


def test_dataclass_with_a_set_init_false_field_raises_value_error() -> None:
    """A set ``init=False`` field cannot be reconstructed (``:270-275``)."""

    @dataclass
    class ADataClass:
        a: Any
        b: int = field(init=False)

    def prepare(a: Any) -> ADataClass:
        data = ADataClass(a=a)
        data.b = 4
        return data

    with pytest.raises(ValueError, match="`init=False` are not supported") as excinfo:
        delayed(prepare(_leaf()))

    assert excinfo.match("ADataClass")
    # The code chains the original ``replace()`` failure, whichever of the two
    # types it raised.
    assert isinstance(excinfo.value.__cause__, (ValueError, TypeError))


def test_dataclass_with_a_custom_init_raises_type_error() -> None:
    """A custom ``__init__`` cannot be reconstructed (``:276-280``)."""

    @dataclass
    class ADataClass:
        a: Any

        def __init__(self, b: Any) -> None:
            self.a = b

    with pytest.raises(TypeError, match="custom __init__ is not supported") as excinfo:
        delayed({"data": ADataClass(b=_leaf())})

    assert excinfo.match("ADataClass")
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_private_attribute_access_raises_attribute_error() -> None:
    """Underscore-prefixed attributes are not lazily wrapped (``:744-745``)."""
    obj = _leaf()
    with pytest.raises(AttributeError, match="Attribute _foo not found"):
        obj._foo


def test_visualise_typo_warns_and_still_returns_a_delayed_attr() -> None:
    """The ``visualise`` spelling guard warns *and* keeps working (``:747-755``)."""
    obj = _leaf()
    with pytest.warns(UserWarning, match="Perhaps you meant"):
        attribute = obj.visualise
    assert isinstance(attribute, DelayedAttr)
    assert attribute._attr == "visualise"


def test_to_task_dask_still_warns_and_still_works() -> None:
    """The deprecated shim keeps its behaviour and its warning (``:335-339``).

    ``pytest.warns`` both asserts and consumes the warning, so the test stays
    clean under the project's warnings-as-errors configuration.
    """
    a = delayed(1, name="a")
    with pytest.warns(UserWarning, match="has been Deprecated"):
        task, graph = to_task_dask([a, 3])
    assert task == ["a", 3]
    assert dict(graph) == dict(a.dask)


def test_a_task_used_as_a_task_callable_raises() -> None:
    """Nested task callables are rejected by the task-spec layer.

    ``Task.__init__`` raises when its ``func`` is itself a ``Task``
    (``dask/_task_spec.py:657-659``), and the error surfaces unchanged through
    ``call_function``.
    """
    with pytest.raises(TypeError, match="Cannot nest tasks"):
        delayed(Task("t", inc, 1), name="t")(2)


def test_collection_that_does_not_finalize_to_one_key_raises() -> None:
    """A multi-output collection is refused with the documented message (``:184-189``)."""
    a = delayed(1, name="a")
    b = delayed(2, name="b")
    collection = _MultiKeyCollection(
        _ExprSequence(collections_to_expr(a), collections_to_expr(b))
    )
    with pytest.raises(
        RuntimeError,
        match="Cannot unpack dask collections which don't finalize to a single key",
    ):
        unpack_collections(collection)


def test_strict_mode_raises_tokenization_error_for_a_generator() -> None:
    """Strict tokenization propagates unchanged through ``delayed``.

    A generator falls through ``unpack_collections`` to ``return expr, ()`` and is
    then tokenized by pickle, which fails and routes to
    ``_maybe_raise_nondeterministic`` (``dask/tokenize.py:83-89``), reached from
    the module-local ``tokenize`` wrapper (``dask/delayed.py:408``). The strict
    flag is read from the ``_ENSURE_DETERMINISTIC`` ContextVar first and the
    configuration key second, so ``dask.config.set`` is the right trigger.
    """
    with (
        dask.config.set({"tokenize.ensure-deterministic": True}),
        pytest.raises(TokenizationError, match="cannot be deterministically hashed"),
    ):
        delayed(ident, name="ident", pure=True)(_generate())


@pytest.mark.parametrize("name", _STRICT_DETERMINISTIC)
def test_no_silent_flip_deterministic_stays_deterministic(name: str) -> None:
    """A deterministically keyed expression keeps its golden key in strict mode.

    Half of the "no silent flip" property: strict mode must not change a key, and
    in particular must not push a deterministic token onto the UUID fallback.
    """
    entry = _by_name(name)
    assert entry.deterministic
    with dask.config.set({"tokenize.ensure-deterministic": True}):
        obj = entry.build()
    assert obj.key == GOLDEN[name]["key"]


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
# ---------------------------------------------------------------------------


def test_results_match_across_schedulers() -> None:
    """The computable corpus computes identically under ``sync`` and ``threads``.

    The synchronous leg goes through ``canonical_result`` so that this test and
    the A/B harness share one definition of "the result of an expression".
    """
    entries = [entry for entry in CORPUS if entry.computable]
    objects = [entry.build() for entry in entries]

    synchronous = canonical_result(objects)
    threaded = dask.compute(*objects, scheduler="threads")

    assert len(synchronous) == len(entries)
    assert len(threaded) == len(entries)
    for entry, sync_value, threaded_value in zip(entries, synchronous, threaded):
        assert sync_value == threaded_value, f"{entry.name} differs across schedulers"


def test_objects_and_graphs_round_trip_through_pickle() -> None:
    """Every eligible object and its graph survive ``pickle`` at protocol 5.

    This is the serializability property non-synchronous schedulers depend on:
    the wrapped values, the tasks and the graph container all have to cross a
    process boundary. Corpus callables and types are module-level, so they pickle
    by reference.

    Results are compared against the golden ``result_repr`` rather than with
    ``==`` because one entry's computed value contains a ``Delayed`` -- with
    ``traverse=False`` the object is quoted, not traversed -- and
    ``Delayed.__eq__`` is a lazy operator that builds a new ``Delayed`` instead of
    returning a boolean.

    Entries excluded by a flag are skipped by name, and every skip is justified in
    ``_EXCLUSION_REASONS``; ``test_flag_exclusions_are_documented`` keeps that
    table honest.
    """
    eligible = [entry for entry in CORPUS if entry.computable and entry.picklable]
    excluded = {
        entry.name for entry in CORPUS if not (entry.computable and entry.picklable)
    }
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

        assert restored.key == obj.key, entry.name
        assert list(restored.__dask_keys__()) == list(obj.__dask_keys__()), entry.name
        assert tuple(restored.__dask_layers__()) == tuple(obj.__dask_layers__())
        assert list(restored_graph.layers) == list(graph.layers), entry.name
        assert list(restored_graph.dependencies) == list(graph.dependencies), entry.name
        if entry.canonical:
            assert canonical_graph(restored) == GOLDEN[entry.name]["graph"], entry.name
        assert (
            repr(canonical_result([restored])[0]) == GOLDEN[entry.name]["result_repr"]
        ), entry.name
