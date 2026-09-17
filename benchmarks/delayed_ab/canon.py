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
golden dict literal). Only the key leaf types dask itself permits are encoded:
``str``, ``bytes``, ``int``, ``float``, ``bool``, ``None`` and ``tuple``\\ s of
those, recursively. A ``bytes`` or ``tuple`` key enters the output as its
``repr``, which is deterministic, carries no object identity and is identical in
both arms. Any other object is refused with ``_UnsupportedKeyError`` rather than
serialised, because a default ``object.__repr__`` embeds a process-specific
address: it would make the golden fixture unreproducible and the A/B comparison
fail for a reason that is not a behaviour change. That rule is applied once, in
``_canonical_key``, and an expression that trips it is a halt-and-report signal
for the operator.

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

A second signal has the same shape and a narrower response. A key or layer name
whose 32-hex token changes between two otherwise identical builds does not come
from ``tokenize``: ``dask._expr.HLGExpr.deterministic_token`` and
``dask._expr.ProhibitReuse._suffix`` fall back to ``uuid.uuid4().hex``, which is
32 lowercase hex characters and therefore indistinguishable *by shape* from a
deterministic md5 digest. Expressions that route a non-``Delayed`` dask
collection through ``unpack_collections`` pick one up -- their dependency layer
is named ``finalize-hlgfinalizecompute-<hex>-<hex>`` -- and so does the
``HLGFinalizeCompute`` expression ``dask.delayed.finalize`` stores. This module
does not normalise bare hex tokens: it cannot tell a random one from a
deterministic one, and normalising both would destroy the exact comparison of
deterministic tokens that the characterisation test exists to assert. A consumer
that has to compare such an expression identifies the random tokens by
*observation* -- it builds the expression twice and substitutes only the tokens
that actually changed, leaving every deterministic token verbatim -- which keeps
every structural field of the canonical form under assertion. The canonicaliser
stays honest about what it saw; the consumer supplies the proof of which token
was random.

Requirement conflict, decided
----------------------------
Two requirements of this run meet in the placeholder numbering and cannot both
be taken literally.

* The canonical traversal names "each node's sorted dependencies" as the last
  stage of the walk that fixes the numbering.
* The A/B suite must find the two arms' canonical forms *identical* for every
  case, including the impure variants, whose keys differ in every token between
  one build and the next.

Sorting a node's dependencies by their raw keys satisfies the first and breaks
the second. Measured, with the raw ordering in place: the ``wide_fan_in`` case --
one call taking 5,000 impure leaves, so one node with 5,000 unnumbered
``f-<uuid4>`` dependencies -- canonicalises *differently* on two builds of the
same expression, because the random tokens decide which dependency is numbered
first and ``layer_order`` then carries the permutation. The suite would report an
equivalence mismatch between two arms that built identical graphs.

The decision is to sort a node's dependencies by ``_sort_key`` of the
**normalised** key, breaking ties by the position at which the walk first reached
the key (``_numbering_key``). Three things make this the narrower deviation: it
is the same order in which the canonical form sorts the dependency list it
actually emits; it is identical to raw ``_sort_key`` order for every dependency
set whose keys are deterministic, which is the regime the exact-key assertions
care about; and the tie-break is layer insertion order, which both arms are
required to share anyway. Nothing about the *content* of the canonical form
changes -- only which random token is called ``#0``.

This is a deviation from the letter of the traversal specification and is
reported as one, not absorbed: it is stated here, in ``_numbering_key``, and in
the closing report of the run that made it, so that it can be overruled. The
alternative -- taking the traversal literally and accepting an A/B suite that
cannot certify equivalence for an impure fan-in -- was rejected because it would
leave the performance evidence unprovable.

Notes
-----
The module is stateless. The only mutable state involved is the caller-supplied
``table`` memo threaded through ``normalize_key``, which keeps the functions
usable under the harness's arm activation (where ``sys.modules["dask.delayed"]``
is swapped between arms).
"""

# The future feature is aliased so that it does not bind a public name either:
# the compiler recognises a future statement by the feature's name, so
# postponed evaluation of annotations is enabled exactly as it is by the
# unaliased form, and the module's non-underscore namespace stays confined to
# the three functions the contract permits.
from __future__ import annotations as _annotations

# Every support import is bound under a leading underscore, and ``__all__``
# names the three functions the module contract permits, so that the public
# surface of this module is exactly those three and nothing else.
import re as _re
from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from typing import Any as _Any
from typing import cast as _cast

from dask import compute as _compute
from dask._task_spec import GraphNode as _GraphNode
from dask._task_spec import Task as _Task
from dask._task_spec import TaskRef as _TaskRef
from dask.core import get_dependencies as _get_dependencies
from dask.highlevelgraph import HighLevelGraph as _HighLevelGraph
from dask.utils import funcname as _funcname

__all__ = ["canonical_graph", "canonical_result", "normalize_key"]

#: Pattern of a UUID4 token as produced by ``str(uuid.uuid4())``, which
#: ``dask.delayed.tokenize`` returns for impure (``pure=False``) expressions.
_UUID4_RE: _re.Pattern[str] = _re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

#: Length of ``str(uuid.uuid4())``. The key that carries it is one character
#: longer still, for the ``"-"`` that separates the prefix from the token.
_UUID4_TOKEN_LEN: int = 36


class _UnsupportedKeyError(TypeError):
    """A graph key that cannot be canonicalised reproducibly.

    Raised instead of falling back to ``repr`` for an object outside the key leaf
    types dask permits. It derives from ``TypeError`` so that it reads as the
    encoding failure it is, and it is deliberately private: the module contract
    permits three public names, and no consumer catches this -- it propagates as
    a halt-and-report signal.
    """


def normalize_key(key: object, table: dict[str, str]) -> object:
    """Replace inherently random key tokens with stable placeholders.

    Deterministic keys are returned verbatim so that they can be compared
    exactly -- that is the point of the characterisation test's exact-key
    assertions. ``dask``'s deterministic tokens are 32 lowercase hex characters
    (``hashlib.md5(...).hexdigest()``); a UUID4 token is 36 characters including
    hyphens, which is what lets the two be told apart by shape alone.

    Args:
        key: A graph key, layer name or dependency name of any type.
        table: First-appearance memo owned by the caller and mutated in place,
            mapping a raw key with a UUID4 token to its placeholder. The caller
            must pass the same ``table`` for every key belonging to one object,
            otherwise the placeholder numbering is not consistent across that
            object's fields. ``canonical_graph`` creates and threads exactly one
            ``table`` per object.

    Returns:
        object: ``f"{prefix}-#{n}"`` for a ``str`` key whose token is a UUID4,
        where ``n`` counts the order of first appearance of that whole raw key
        and is **0-based**; a ``tuple`` with every element normalised recursively
        for a ``tuple`` key; the key itself, unchanged and unreformatted, for
        anything else.
    """
    typ = type(key)
    if typ is tuple:
        # Collection keys are ``(name, i, j)`` shaped; normalise element-wise and
        # recursively so that nested tuple keys work too. Exactly a ``tuple``, not
        # ``isinstance``: a subclass may define ``__iter__`` or ``__repr__`` of its
        # own, and rebuilding it as a plain tuple would erase the very type that
        # ``_require_encodable_key`` has to see in order to refuse it.
        return tuple(
            normalize_key(element, table)
            for element in _cast("tuple[object, ...]", key)
        )
    if typ is not str:
        return key
    # Exactly a ``str`` from here on; the cast carries that to the type checker,
    # which cannot read an identity comparison against ``str`` as a narrowing.
    text = _cast(str, key)
    # A UUID4 token contains hyphens itself, so ``rsplit("-", 1)`` would split
    # inside the token and leave a random fragment behind. Match the fixed-width
    # suffix instead: 36 UUID4 characters preceded by the "-" that separates the
    # prefix from the token.
    if (
        len(text) > _UUID4_TOKEN_LEN + 1
        and text[-(_UUID4_TOKEN_LEN + 1)] == "-"
        and _UUID4_RE.match(text[-_UUID4_TOKEN_LEN:])
    ):
        placeholder = table.get(text)
        if placeholder is None:
            prefix = text[: -(_UUID4_TOKEN_LEN + 1)]
            placeholder = f"{prefix}-#{len(table)}"
            table[text] = placeholder
        return placeholder
    return text


def canonical_graph(obj: object) -> dict[str, _Any]:
    """Serialise a collection's graph into the canonical, comparable form.

    Two objects are equivalent for the purposes of this run exactly when their
    canonical dicts compare equal. The traversal that fixes the placeholder
    numbering is frozen: the output key, then the layer names in
    ``list(graph.layers)`` order, then the nodes of each layer in layer-iteration
    order, and each node's sorted dependencies immediately after that node. The
    names in ``graph.dependencies`` are walked last, in mapping order, so that a
    name no layer holds is still numbered deterministically rather than by set
    iteration order.

    The sort applied to a node's dependencies is ``_numbering_key``, which is
    ``_sort_key`` order over the normalised key -- the same order in which the
    emitted dependency list itself is sorted -- with the graph-walk position as
    tie-break. It coincides with ``sorted(dependencies, key=_sort_key)`` for
    every dependency set whose keys are deterministic. Where it does not, it is a
    reported deviation rather than a silent one: see "Requirement conflict,
    decided" in the module docstring.

    Args:
        obj: A ``Delayed`` -- or any dask collection exposing ``key``,
            ``__dask_graph__()``, ``__dask_keys__()`` and ``__dask_layers__()``.
            Every graph shape ``Delayed`` accepts is handled: a
            ``HighLevelGraph``, a plain ``dict`` low-level graph, and the
            expression ``dask.delayed.finalize`` stores; the latter two are
            treated as a single layer named by ``__dask_layers__()[0]`` with no
            layer dependencies.

    Returns:
        dict: The seven-field, JSON-serialisable canonical form documented in the
        module docstring.

    Raises:
        _UnsupportedKeyError: if a key, layer name or dependency name is outside
            the key leaf types the module can encode reproducibly.
    """
    # ``Delayed`` is consumed duck-typed here: the collection protocol is accessed
    # dynamically so that the public signature can stay ``object``.
    collection: _Any = obj
    table: dict[str, str] = {}
    layers, dependencies = _layer_views(collection, collection.__dask_graph__())

    # Pass 0 -- materialise every layer once, in order, and with it the complete
    # low-level mapping of the whole graph. ``dict(layer)`` materialises a
    # ``HighLevelGraph`` layer (normally a ``MaterializedLayer``) while preserving
    # its iteration order. The complete mapping is what a legacy tuple task has to
    # be resolved against: ``DelayedAttr`` emits ``{key: (getattr, parent_key,
    # attr)}`` and its parent lives in another layer, so a per-layer mapping would
    # report no dependency at all. ``walk_position`` records where the walk first
    # reaches each key, which is the tie-break the dependency ordering uses.
    #
    # Every raw name is checked here, before anything normalises it, sorts it or
    # puts it in a message: ``_numbering_key`` and ``_sort_key`` reach a key's own
    # ``repr``, so a key this module cannot encode has to be refused before that
    # happens rather than after.
    raw_layers: list[_Any] = []
    low_level: dict[_Any, _Any] = {}
    walk_position: dict[_Any, int] = {}
    _require_encodable_key(collection.key)
    for layer_name, layer in layers.items():
        _require_encodable_key(layer_name)
        nodes = dict(layer)
        for node_key in nodes:
            _require_encodable_key(node_key)
            if node_key not in walk_position:
                walk_position[node_key] = len(walk_position)
        low_level.update(nodes)
        raw_layers.append((layer_name, type(layer).__name__, nodes))

    # Pass 1 -- walk the frozen traversal order to assign placeholders, and
    # collect the node data so that nothing has to be recomputed in pass 2.
    normalize_key(collection.key, table)
    node_entries: list[_Any] = []
    for layer_name, layer_type, nodes in raw_layers:
        normalize_key(layer_name, table)
        entries: list[_Any] = []
        for node_key, node in nodes.items():
            normalize_key(node_key, table)
            kind, node_dependencies = _node_kind_and_dependencies(node, low_level)
            for dependency in node_dependencies:
                _require_encodable_key(dependency)
            for dependency in sorted(
                node_dependencies, key=lambda d: _numbering_key(d, walk_position)
            ):
                normalize_key(dependency, table)
            entries.append((node_key, kind, node_dependencies, _node_func_name(node)))
        node_entries.append((layer_name, layer_type, entries))
    for layer_name, layer_dependencies in dependencies.items():
        _require_encodable_key(layer_name)
        normalize_key(layer_name, table)
        for dependency in layer_dependencies:
            _require_encodable_key(dependency)
        for dependency in sorted(
            layer_dependencies, key=lambda d: _numbering_key(d, walk_position)
        ):
            normalize_key(dependency, table)

    # Pass 2 -- build the fields. Every ``normalize_key`` call below is a memo hit.
    canonical_layers: list[_Any] = []
    for layer_name, layer_type, entries in node_entries:
        canonical_nodes: list[_Any] = [
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

    canonical_dependencies: list[_Any] = [
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


def canonical_result(objs: _Sequence[object]) -> tuple[_Any, ...]:
    """Compute collections under the synchronous scheduler for comparison.

    The synchronous scheduler is mandatory: it keeps the equivalence check free of
    scheduler-specific behaviour and of any thread of its own. Callers compare the
    returned values with ``==``; the run's corpus produces plain Python values, so
    no comparison helper belongs here.

    This must never be called from inside a timed region -- computing is not part
    of what the A/B suite measures. Keeping it out of the timed region is the
    caller's responsibility (``benchmarks/delayed_ab/main.py``).

    Args:
        objs: The collections to compute, in the order their results are
            returned.

    Returns:
        tuple: The computed results, one per input, or an empty tuple for empty
        input.
    """
    materialised = tuple(objs)
    if not materialised:
        return ()
    return _compute(*materialised, scheduler="sync")


def _sort_key(value: object) -> str:
    """Total, type-tolerant sort key: order values by their ``repr``."""
    return repr(value)


def _require_encodable_key(value: object) -> None:
    """Check that a key -- raw or normalised -- is one this module can encode.

    Args:
        value: A key, layer name, dependency name, or one element of a ``tuple``
            key. Raw keys are checked before anything normalises or sorts them,
            and the normalised result is checked again in ``_json_safe``.

    Raises:
        _UnsupportedKeyError: if ``value`` -- or, recursively, any element of a
            ``tuple`` -- is not exactly ``str``, ``bytes``, ``int``, ``float``,
            ``bool``, ``None`` or ``tuple``. The check is by exact type because a
            subclass is free to define ``__repr__`` or ``__iter__`` of its own:
            what has to be ruled out is any ``repr`` that can carry object
            identity, and a ``tuple`` subclass whose iteration yields something
            other than its elements would slip identity in that way.
    """
    typ = type(value)
    if (
        value is None
        or typ is bool
        or typ is int
        or typ is float
        or typ is str
        or typ is bytes
    ):
        return
    if typ is tuple:
        for element in value:  # type: ignore[attr-defined]
            _require_encodable_key(element)
        return
    # The message is built from the type's own metadata and never from the
    # object: an unsupported key's ``repr`` is exactly what must not be trusted
    # here -- it may carry a memory address, and it may raise, which would
    # replace this diagnosis with an unrelated exception.
    raise _UnsupportedKeyError(
        f"cannot canonicalise a graph key of type {typ.__module__}.{typ.__qualname__}: "
        "only str, bytes, int, float, bool, None and tuples of those are encoded, and "
        "by exact type. Any other object would enter the canonical form as its repr, "
        "and a default object repr embeds a process-specific address, which would make "
        "the golden fixture unreproducible and the A/B comparison fail for a reason "
        "that is not a behaviour change. Halt and report the expression that produced "
        "this key instead of relaxing the rule"
    )


def _json_safe(value: object) -> str | int | float | bool | None:
    """Coerce a normalised key to a JSON-expressible leaf.

    Args:
        value: A key already passed through :func:`normalize_key`.

    Returns:
        str | int | float | bool | None: ``value`` itself for the scalar key
        types JSON can express, and the ``repr`` of a ``bytes`` or ``tuple`` key
        -- deterministic, free of object identity and identical in both arms of
        the suite -- once every leaf of it has been checked.

    Raises:
        _UnsupportedKeyError: for any other object, rather than serialising a
            ``repr`` that may carry a memory address.
    """
    typ = type(value)
    if value is None or typ is bool or typ is int or typ is float or typ is str:
        # The exact-type test above is what decides; the cast only tells the type
        # checker what those five identity comparisons already established.
        return _cast("str | int | float | bool | None", value)
    _require_encodable_key(value)
    return repr(value)


def _canonical_key(
    key: object, table: dict[str, str]
) -> str | int | float | bool | None:
    """Normalise a key and coerce the result to a JSON-expressible leaf."""
    return _json_safe(normalize_key(key, table))


def _numbering_key(
    dependency: object, walk_position: dict[object, int]
) -> tuple[str, int, str]:
    """Order one node's dependencies for placeholder numbering.

    ``_sort_key`` of the dependency comes first, taken over the key with any
    UUID4 token masked to ``"<prefix>-#0"`` by a throw-away memo -- that is, over
    the *normalised* key, which is also what the emitted dependency list is
    sorted by. For a dependency set whose keys are all deterministic the mask
    changes nothing, so the order is exactly ``sorted(dependencies,
    key=_sort_key)``. For keys carrying a UUID4 token the mask collapses the
    random part and the position at which the graph walk first reached the key
    breaks the tie -- layer insertion order, then node order, both of which the
    two arms are required to share.

    This is the deviation recorded under "Requirement conflict, decided" in the
    module docstring, and the reason for it is measurable rather than stylistic:
    ordering by the raw key orders same-prefix impure dependencies by their
    random tokens, so a node with several of them numbers them differently in two
    builds of the same expression and two identical graphs canonicalise
    differently.

    Args:
        dependency: The dependency key being ordered.
        walk_position: Key to the index at which the graph walk first reached it.

    Returns:
        tuple: A total, type-tolerant sort key. A dangling dependency -- a
        reference to a key no layer holds, which ``delayed`` never produces --
        sorts after every key the graph holds, and among its own kind by ``repr``.
    """
    masked = _sort_key(normalize_key(dependency, {}))
    position = walk_position.get(dependency)
    if position is None:
        return (masked, len(walk_position), _sort_key(dependency))
    return (masked, position, "")


def _canonical_nested_keys(keys: _Sequence[_Any], table: dict[str, str]) -> list[_Any]:
    """Normalise a nested-key structure, preserving its list nesting.

    ``__dask_keys__()`` returns a list whose elements are either keys or further
    lists of keys. Only ``list`` nesting is descended into; a ``tuple`` is a key,
    not a level of nesting, and is handled by ``normalize_key``.
    """
    canonical: list[_Any] = []
    for key in keys:
        if isinstance(key, list):
            canonical.append(_canonical_nested_keys(key, table))
        else:
            canonical.append(_canonical_key(key, table))
    return canonical


def _layer_views(
    collection: _Any, graph: _Any
) -> tuple[dict[_Any, _Any], dict[_Any, _Any]]:
    """Return ``(layers, dependencies)`` as plain dicts in their original order.

    A ``HighLevelGraph`` supplies both mappings directly. Any other graph is
    presented as a single layer named by ``__dask_layers__()[0]`` with an empty
    dependency set, so that its actual mapping type is what gets recorded as the
    layer type. Two shapes reach that branch, both of which ``Delayed``
    legitimately holds: a plain ``dict`` low-level graph (``dask.graph_manipulation``
    builds them that way, and so does a caller passing one straight to
    ``Delayed``), and an expression -- ``dask.delayed.finalize`` stores the
    ``HLGFinalizeCompute`` returned by ``collections_to_expr(...).finalize_compute()``.
    An expression is not a mapping at all, so it is materialised through its own
    ``__dask_graph__()``; without that the canonical form of a ``finalize()``
    result could not be compared at all, and a graph-shape regression on that path
    would go unnoticed.
    """
    if isinstance(graph, _HighLevelGraph):
        return dict(graph.layers), dict(graph.dependencies)
    layer_name = list(collection.__dask_layers__())[0]
    if not isinstance(graph, _Mapping):
        graph = graph.__dask_graph__()
    return {layer_name: graph}, {layer_name: set()}


def _node_kind_and_dependencies(
    node: object, low_level: dict[_Any, _Any]
) -> tuple[str, set[_Any] | frozenset[_Any]]:
    """Classify one graph node and extract the keys it depends on.

    Task-spec nodes report their own dependencies. A raw ``tuple`` is the legacy
    task form -- ``DelayedAttr.dask`` emits ``{key: (getattr, obj_key, attr)}``
    and that node form is frozen by the equivalence contract -- so it is labelled
    ``"legacy-tuple"`` and its dependencies are resolved with
    ``get_dependencies``, whose keyword ``task=`` form is required (a positional
    second argument would bind to ``key``). ``get_dependencies`` only reports
    references that are keys of the mapping it is given, which is why the mapping
    passed here is the complete low-level graph and not one layer of it: the
    parent a ``DelayedAttr`` node refers to lives in another layer, and a
    per-layer mapping would record that node as depending on nothing.

    An unanticipated node shape is reported as data -- its type name with no
    dependencies -- rather than raised, so that a surprise shows up in the
    comparison legibly instead of aborting canonicalisation.

    Args:
        node: One value of a materialised graph layer.
        low_level: The complete materialised mapping of every layer of the graph,
            used to resolve the references a legacy tuple task carries.

    Returns:
        tuple: The node kind -- its type name, or ``"legacy-tuple"`` for a raw
        tuple task -- and the set of keys it depends on.
    """
    if isinstance(node, _GraphNode):
        return type(node).__name__, node.dependencies
    if isinstance(node, _TaskRef):
        # ``TaskRef`` is not a ``GraphNode``; it points at exactly one key.
        return type(node).__name__, {node.key}
    if type(node) is tuple:
        return "legacy-tuple", _get_dependencies(low_level, task=node)
    return type(node).__name__, frozenset()


def _node_func_name(node: object) -> str | None:
    """Return the callable's name for ``Task`` nodes, ``None`` for other nodes.

    The ``Task`` check deliberately covers ``NestedContainer`` and its
    ``List``/``Tuple``/``Set``/``Dict`` subclasses, which carry a bound
    ``to_container`` as ``.func``. ``dask.utils.funcname`` already unwraps
    ``functools.partial`` and maps ``methodcaller`` to its method name.
    """
    if isinstance(node, _Task):
        return _funcname(node.func)
    return None
