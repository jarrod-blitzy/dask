"""Canonical, order-independent serialization of ``dask.delayed`` graphs.

This module holds the *single* definition of the canonical form that the
``delayed`` A/B refactoring run uses as equivalence evidence. It has exactly two
consumers and they deliberately share one implementation so that they cannot
drift apart:

* ``benchmarks/delayed_ab/main.py`` canonicalises every object built by both arms
  of the A/B suite -- the frozen pre-refactor baseline and the live
  ``dask.delayed`` -- and refuses to record a single timing until the canonical
  forms, the generated keys and the computed results all match.
* ``dask/tests/test_delayed_equivalence.py`` captures a committed golden fixture
  through these functions on the pre-refactor code and re-asserts that fixture,
  unmodified, after the refactor.

Because the golden fixture was captured through the functions below, the field
set, the field names, the sort orders and the key-normalisation rules here are
**frozen**. Changing any of them silently invalidates the committed evidence, so
nothing may be added, renamed or reordered.

Canonical form
--------------
``canonical_graph`` returns a JSON-serialisable dict with exactly seven fields:

``key``
    The normalised output key of the collection.
``dask_keys``
    ``__dask_keys__()`` normalised, preserving its list nesting.
``dask_layers``
    ``__dask_layers__()`` normalised, emitted as a list.
``layer_order``
    ``list(graph.layers)`` normalised. **Order-sensitive; never sorted.**
``dependency_order``
    ``list(graph.dependencies)`` normalised. **Order-sensitive; never sorted.**
``layers``
    Sorted list of ``[layer_name, layer_type_name, sorted_nodes]``, where each
    node is ``[key, node_kind, sorted_dependencies, func_name]``.
``dependencies``
    Sorted list of ``[layer_name, sorted_dependency_layer_names]``.

Order independence comes from sorting every list except ``layer_order`` and
``dependency_order``. Those two stay unsorted on purpose: layer insertion order
is token-relevant. Two ``Delayed`` objects with identical graph content but
different layer insertion order tokenize identically under ``tokenize(d)`` and
*differently* once the object is reached through pickle (an iterator argument, a
user object holding a ``Delayed``), because ``tokenize`` hashes the pickled slot
state. These two fields are the only ones that can catch an insertion-order
regression. The orders they lock, as built by ``dask.highlevelgraph``, are
existing-then-new for a single dependency (``_from_collection``) and
new-then-existing for several dependencies (``from_collections``).

Sorting and JSON-serialisability
--------------------------------
Graph keys mix ``str``, ``int`` and ``tuple``, so ``sorted()`` on raw values can
raise ``TypeError``. Every sort in this module therefore uses ``_sort_key``,
which orders values by their ``repr`` -- deterministic, total and type-tolerant.

Every leaf of the returned dict is a ``str``, ``int``, ``float``, ``bool``,
``None`` or a list of those, because both consumers serialise the dict to JSON
(the harness writes it into its artefact on mismatch; the test compares it with a
golden dict literal). Normalised tuple keys -- and any other key type that JSON
cannot express -- enter the output as their ``repr``. That rule is applied once,
in ``_canonical_key``.

Halt-and-report signals
-----------------------
``dask.highlevelgraph._get_some_layer_name`` falls back to ``str(id(collection))``
when a collection exposes no usable ``__dask_layers__()``. ``Delayed`` always
returns ``(self._layer,)``, so no expression in the run's corpus should ever
produce an ``id()``-derived layer name. Such a name is not reproducible across
processes and would poison both the golden fixture and the A/B comparison. This
module does **not** normalise it away: a purely numeric layer name is left in the
output verbatim so that the comparison fails loudly, and it is a halt-and-report
signal for the operator -- it means a graph shape the plan did not anticipate was
reached. The response is to stop and report, never to special-case it here and
never to edit a frozen ``dask`` module.

A second signal has the same shape and the same response. A key or layer name
whose 32-hex token changes between two otherwise identical builds does not come
from ``tokenize``: ``dask._expr.HLGExpr.deterministic_token`` and
``dask._expr.ProhibitReuse._suffix`` fall back to ``uuid.uuid4().hex``, which is
32 lowercase hex characters and therefore indistinguishable from a deterministic
md5 digest. Expressions that route a non-``Delayed`` dask collection through
``unpack_collections`` pick one up -- their dependency layer is named
``finalize-hlgfinalizecompute-<hex>-<hex>`` -- so they have no stable canonical
form, in any process or either arm. Normalising bare hex tokens is not the fix:
it would destroy the exact comparison of the deterministic tokens that the
characterisation test exists to assert. Such an expression is reported, and its
equivalence is established through its computed result and its stable output key
rather than through a canonical graph.

Notes
-----
The module is stateless. The only mutable state involved is the caller-supplied
``table`` memo threaded through ``normalize_key``, which keeps the functions
usable under the harness's arm activation (where ``sys.modules["dask.delayed"]``
is swapped between arms).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from dask import compute
from dask._task_spec import GraphNode, Task, TaskRef
from dask.core import get_dependencies
from dask.highlevelgraph import HighLevelGraph
from dask.utils import funcname

#: Pattern of a UUID4 token as produced by ``str(uuid.uuid4())``, which
#: ``dask.delayed.tokenize`` returns for impure (``pure=False``) expressions.
_UUID4_RE: re.Pattern[str] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

#: Length of ``str(uuid.uuid4())``. The key that carries it is one character
#: longer still, for the ``"-"`` that separates the prefix from the token.
_UUID4_TOKEN_LEN: int = 36


def normalize_key(key: object, table: dict[str, str]) -> object:
    """Replace inherently random key tokens with stable placeholders.

    Deterministic keys are returned verbatim so that they can be compared
    exactly -- that is the point of the characterisation test's exact-key
    assertions. ``dask``'s deterministic tokens are 32 lowercase hex characters
    (``hashlib.md5(...).hexdigest()``); a UUID4 token is 36 characters including
    hyphens, which is what lets the two be told apart by shape alone.

    Parameters
    ----------
    key
        A graph key, layer name or dependency name of any type.
    table
        First-appearance memo owned by the caller and mutated in place, mapping a
        raw key with a UUID4 token to its placeholder. The caller must pass the
        same ``table`` for every key belonging to one object, otherwise the
        placeholder numbering is not consistent across that object's fields.
        ``canonical_graph`` creates and threads exactly one ``table`` per object.

    Returns
    -------
    object
        ``f"{prefix}-#{n}"`` for a ``str`` key whose token is a UUID4, where
        ``n`` counts the order of first appearance of that whole raw key and is
        **0-based**; a ``tuple`` with every element normalised recursively for a
        ``tuple`` key; the key itself, unchanged and unreformatted, for anything
        else.
    """
    if isinstance(key, tuple):
        # Collection keys are ``(name, i, j)`` shaped; normalise element-wise and
        # recursively so that nested tuple keys work too.
        return tuple(normalize_key(element, table) for element in key)
    if not isinstance(key, str):
        return key
    # A UUID4 token contains hyphens itself, so ``rsplit("-", 1)`` would split
    # inside the token and leave a random fragment behind. Match the fixed-width
    # suffix instead: 36 UUID4 characters preceded by the "-" that separates the
    # prefix from the token.
    if (
        len(key) > _UUID4_TOKEN_LEN + 1
        and key[-(_UUID4_TOKEN_LEN + 1)] == "-"
        and _UUID4_RE.match(key[-_UUID4_TOKEN_LEN:])
    ):
        placeholder = table.get(key)
        if placeholder is None:
            prefix = key[: -(_UUID4_TOKEN_LEN + 1)]
            placeholder = f"{prefix}-#{len(table)}"
            table[key] = placeholder
        return placeholder
    return key


def canonical_graph(obj: object) -> dict[str, Any]:
    """Serialise a collection's graph into the canonical, comparable form.

    Two objects are equivalent for the purposes of this run exactly when their
    canonical dicts compare equal. The traversal that fixes the placeholder
    numbering is frozen: the output key first, then the layer names in
    ``list(graph.layers)`` order, each immediately followed by the keys of its
    nodes in layer-iteration order. Node dependencies are **not** numbered while
    that walk runs: a dependency is a key of some node in the graph, so the walk
    itself reaches it -- in insertion order, which the arms are required to share.
    Numbering dependencies inline would have to order them first, and ordering raw
    keys whose tokens are UUIDs orders them randomly, so two identical graphs with
    a node holding several impure dependencies would canonicalise differently.
    Only after the walk are stragglers numbered: dangling dependencies (references
    to keys no layer holds, which ``delayed`` never produces) in node order, each
    node's own in ``_sort_key`` order, and finally the names in
    ``graph.dependencies`` in mapping order, so the numbering never depends on set
    iteration order.

    Parameters
    ----------
    obj
        A ``Delayed`` -- or any dask collection exposing ``key``,
        ``__dask_graph__()``, ``__dask_keys__()`` and ``__dask_layers__()``. Both
        graph shapes ``Delayed`` supports are handled: a ``HighLevelGraph`` and a
        plain ``dict`` low-level graph, the latter treated as a single layer named
        by ``__dask_layers__()[0]`` with no layer dependencies.

    Returns
    -------
    dict
        The seven-field, JSON-serialisable canonical form documented in the module
        docstring.
    """
    # ``Delayed`` is consumed duck-typed here: the collection protocol is accessed
    # dynamically so that the public signature can stay ``object``.
    collection: Any = obj
    table: dict[str, str] = {}
    layers, dependencies = _layer_views(collection, collection.__dask_graph__())

    # Pass 1 -- walk the frozen traversal order purely to assign placeholders, and
    # collect the raw node data so that nothing has to be recomputed in pass 2.
    # Layer names and node keys are numbered in insertion order, which both arms
    # must share; node dependencies are deferred (see the docstring) because
    # sorting raw UUID-token keys would make the numbering random.
    normalize_key(collection.key, table)
    raw_layers: list[Any] = []
    for layer_name, layer in layers.items():
        normalize_key(layer_name, table)
        # ``dict(layer)`` materialises a ``HighLevelGraph`` layer (normally a
        # ``MaterializedLayer``) while preserving its iteration order, and doubles
        # as the ``dsk`` argument used for legacy tuple tasks below.
        nodes = dict(layer)
        entries: list[Any] = []
        for node_key, node in nodes.items():
            normalize_key(node_key, table)
            kind, node_dependencies = _node_kind_and_dependencies(node, nodes)
            entries.append((node_key, kind, node_dependencies, _node_func_name(node)))
        raw_layers.append((layer_name, type(layer).__name__, entries))
    # Stragglers: a dependency on a key that no layer holds. ``delayed`` graphs
    # never contain one, so this is a memo hit for every well-formed graph.
    for _layer_name, _layer_type, entries in raw_layers:
        for _node_key, _kind, node_dependencies, _func_name in entries:
            for dependency in sorted(node_dependencies, key=_sort_key):
                normalize_key(dependency, table)
    for layer_name, layer_dependencies in dependencies.items():
        normalize_key(layer_name, table)
        for dependency in sorted(layer_dependencies, key=_sort_key):
            normalize_key(dependency, table)

    # Pass 2 -- build the fields. Every ``normalize_key`` call below is a memo hit.
    canonical_layers: list[Any] = []
    for layer_name, layer_type, entries in raw_layers:
        canonical_nodes: list[Any] = [
            [
                _canonical_key(node_key, table),
                kind,
                sorted(
                    (_canonical_key(d, table) for d in node_dependencies),
                    key=_sort_key,
                ),
                func_name,
            ]
            for node_key, kind, node_dependencies, func_name in entries
        ]
        canonical_nodes.sort(key=_sort_key)
        canonical_layers.append(
            [_canonical_key(layer_name, table), layer_type, canonical_nodes]
        )
    canonical_layers.sort(key=_sort_key)

    canonical_dependencies: list[Any] = [
        [
            _canonical_key(layer_name, table),
            sorted(
                (_canonical_key(d, table) for d in layer_dependencies), key=_sort_key
            ),
        ]
        for layer_name, layer_dependencies in dependencies.items()
    ]
    canonical_dependencies.sort(key=_sort_key)

    return {
        "key": _canonical_key(collection.key, table),
        "dask_keys": _canonical_nested_keys(collection.__dask_keys__(), table),
        "dask_layers": [
            _canonical_key(name, table) for name in collection.__dask_layers__()
        ],
        "layer_order": [_canonical_key(name, table) for name in layers],
        "dependency_order": [_canonical_key(name, table) for name in dependencies],
        "layers": canonical_layers,
        "dependencies": canonical_dependencies,
    }


def canonical_result(objs: Sequence[object]) -> tuple[Any, ...]:
    """Compute collections under the synchronous scheduler for comparison.

    The synchronous scheduler is mandatory: it keeps the equivalence check free of
    scheduler-specific behaviour and of any thread of its own. Callers compare the
    returned values with ``==``; the run's corpus produces plain Python values, so
    no comparison helper belongs here.

    This must never be called from inside a timed region -- computing is not part
    of what the A/B suite measures. Keeping it out of the timed region is the
    caller's responsibility (``benchmarks/delayed_ab/main.py``).

    Parameters
    ----------
    objs
        The collections to compute, in the order their results are returned.

    Returns
    -------
    tuple
        The computed results, one per input, or an empty tuple for empty input.
    """
    materialised = tuple(objs)
    if not materialised:
        return ()
    return compute(*materialised, scheduler="sync")


def _sort_key(value: object) -> str:
    """Total, type-tolerant sort key: order values by their ``repr``."""
    return repr(value)


def _json_safe(value: object) -> str | int | float | bool | None:
    """Coerce a normalised key to a JSON-expressible leaf.

    ``str``, ``int``, ``float``, ``bool`` and ``None`` pass through untouched.
    Everything else -- a normalised tuple key above all -- becomes its ``repr``,
    which is stable across processes for the key types dask permits and identical
    for both arms of the suite.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _canonical_key(
    key: object, table: dict[str, str]
) -> str | int | float | bool | None:
    """Normalise a key and coerce the result to a JSON-expressible leaf."""
    return _json_safe(normalize_key(key, table))


def _canonical_nested_keys(keys: Sequence[Any], table: dict[str, str]) -> list[Any]:
    """Normalise a nested-key structure, preserving its list nesting.

    ``__dask_keys__()`` returns a list whose elements are either keys or further
    lists of keys. Only ``list`` nesting is descended into; a ``tuple`` is a key,
    not a level of nesting, and is handled by ``normalize_key``.
    """
    canonical: list[Any] = []
    for key in keys:
        if isinstance(key, list):
            canonical.append(_canonical_nested_keys(key, table))
        else:
            canonical.append(_canonical_key(key, table))
    return canonical


def _layer_views(collection: Any, graph: Any) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Return ``(layers, dependencies)`` as plain dicts in their original order.

    A ``HighLevelGraph`` supplies both mappings directly. A plain ``dict``
    low-level graph -- which ``Delayed`` legitimately accepts, for instance from
    ``dask.graph_manipulation`` -- is presented as a single layer named by
    ``__dask_layers__()[0]`` with an empty dependency set, so that its actual
    mapping type is what gets recorded as the layer type.
    """
    if isinstance(graph, HighLevelGraph):
        return dict(graph.layers), dict(graph.dependencies)
    layer_name = list(collection.__dask_layers__())[0]
    return {layer_name: graph}, {layer_name: set()}


def _node_kind_and_dependencies(
    node: object, nodes: dict[Any, Any]
) -> tuple[str, set[Any] | frozenset[Any]]:
    """Classify one graph node and extract the keys it depends on.

    Task-spec nodes report their own dependencies. A raw ``tuple`` is the legacy
    task form -- ``DelayedAttr.dask`` emits ``{key: (getattr, obj_key, attr)}``
    and that node form is frozen by the equivalence contract -- so it is labelled
    ``"legacy-tuple"`` and its dependencies are resolved with
    ``get_dependencies``, whose keyword ``task=`` form is required (a positional
    second argument would bind to ``key``). ``get_dependencies`` only reports
    references that are keys of the mapping it is given, and that mapping is this
    layer's own materialised nodes: cross-layer references are carried by the
    ``dependencies`` field instead.

    An unanticipated node shape is reported as data -- its type name with no
    dependencies -- rather than raised, so that a surprise shows up in the
    comparison legibly instead of aborting canonicalisation.
    """
    if isinstance(node, GraphNode):
        return type(node).__name__, node.dependencies
    if isinstance(node, TaskRef):
        # ``TaskRef`` is not a ``GraphNode``; it points at exactly one key.
        return type(node).__name__, {node.key}
    if type(node) is tuple:
        return "legacy-tuple", get_dependencies(nodes, task=node)
    return type(node).__name__, frozenset()


def _node_func_name(node: object) -> str | None:
    """Return the callable's name for ``Task`` nodes, ``None`` for other nodes.

    The ``Task`` check deliberately covers ``NestedContainer`` and its
    ``List``/``Tuple``/``Set``/``Dict`` subclasses, which carry a bound
    ``to_container`` as ``.func``. ``dask.utils.funcname`` already unwraps
    ``functools.partial`` and maps ``methodcaller`` to its method name.
    """
    if isinstance(node, Task):
        return funcname(node.func)
    return None
