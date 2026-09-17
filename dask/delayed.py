from __future__ import annotations

import operator
import types
import uuid
import warnings
from collections.abc import Sequence
from dataclasses import fields, is_dataclass, replace
from functools import partial

from tlz import curry, merge

from dask import base, config, utils
from dask._expr import FinalizeCompute, ProhibitReuse, _ExprSequence
from dask._task_spec import (
    DataNode,
    Dict,
    GraphNode,
    List,
    Task,
    TaskRef,
    convert_legacy_graph,
    cull,
    fuse_linear_task_spec,
)
from dask.base import (
    DaskMethodsMixin,
    collections_to_expr,
    is_dask_collection,
    named_schedulers,
    replace_name_in_key,
)
from dask.base import tokenize as _tokenize
from dask.context import globalmethod
from dask.core import flatten, quote
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.typing import Graph, NestedKeys
from dask.utils import (
    OperatorMethodMixin,
    apply,
    ensure_dict,
    funcname,
    is_namedtuple_instance,
    methodcaller,
    unzip,
)

__all__ = ["Delayed", "delayed"]


DEFAULT_GET = named_schedulers.get("threads", named_schedulers["sync"])


def finalize(collection):
    assert is_dask_collection(collection)

    name = "finalize-" + tokenize(collection)
    expr = collections_to_expr(collection).finalize_compute()
    return Delayed(name, expr)


def _convert_dask_keys(keys: NestedKeys) -> List:
    assert isinstance(keys, list)
    new_keys: list[List | TaskRef] = []
    for key in keys:
        if isinstance(key, list):
            new_keys.append(_convert_dask_keys(key))
        else:
            new_keys.append(TaskRef(key))
    return List(*new_keys)


def _get_partial(key, dct, default):
    return dct.get(key, default)


def _finalize_args_collections(args, collections):
    old_keys = [c.__dask_keys__()[0] for c in collections]
    collections = _ExprSequence(*collections).optimize()
    new_keys = collections.__dask_keys__()
    dsk = convert_legacy_graph(collections.__dask_graph__())
    annots = collections.__dask_annotations__()
    outcollections = []
    for k in new_keys:
        # Annotations are defined per HLG Layer but after this transformation
        # these no longer properly exist which is why __dask_annotations__
        # returns a fully materialized dictionary {annot: {key: value}}
        # Introducing a tombstone with a callable is the only way I found how we
        # could revert this transformation (not necessarily efficient but
        # well...)
        layer_annotations = {
            annot: partial(
                _get_partial, dct=key_val, default=collections._annotations_tombstone()
            )
            for annot, key_val in annots.items()
        }
        hlg = HighLevelGraph(
            {
                k[0]: MaterializedLayer(
                    cull(dsk, [k[0]]),
                    annotations=layer_annotations,
                )
            },
            dependencies={k[0]: set()},
        )
        outcollections.append(Delayed(k[0], hlg))
    collections = tuple(outcollections)
    subs = {old: new[0] for old, new in zip(old_keys, new_keys) if old != new}
    args = args.substitute(subs)
    return args, collections


# Iterator types and the container types ``unpack_collections`` dispatches on are
# fixed properties of the interpreter, so they are computed once here instead of
# on every call. They are immutable and never rebound after import, which is what
# makes them safe to share between threads on a free-threaded build.
_LIST_ITER_TYPE: type = type(iter([]))
_TUPLE_ITER_TYPE: type = type(iter(()))
_SET_ITER_TYPE: type = type(iter(set()))
# Deliberately a tuple used with ``in``, not a set: membership on a tuple compares
# element-wise with ``==`` and never hashes, which is the behaviour a user-defined
# metaclass observes today (see ``unpack_collections``).
_SEQUENCE_TYPES: tuple[type, ...] = (list, tuple, set)


def _leaf_layer_dict(leaf: DelayedLeaf) -> Graph:
    """Return the one-entry layer mapping of a ``DelayedLeaf``.

    A module-level function rather than a method on purpose: ``Delayed`` and its
    subclasses are public, so a subclass is free to define a method of any name,
    and dispatching through one would let it intercept the construction path.

    Args:
        leaf: The leaf whose wrapped object is the layer's single node.

    Returns:
        A mapping of exactly one entry, keyed by ``leaf._key``. The value is part
        of the graph shape: a wrapped object that is already a ``TaskRef`` or a
        ``GraphNode`` is the value as it is, and any other value is wrapped in
        ``DataNode(leaf._key, leaf._obj)``.
    """
    obj = leaf._obj
    if isinstance(obj, (TaskRef, GraphNode)):
        return {leaf._key: obj}
    return {leaf._key: DataNode(leaf._key, obj)}


def _graph_from_collections(
    name: object, layer: Graph, collections: Sequence | set
) -> HighLevelGraph:
    """Build the ``HighLevelGraph`` of a new node from its dependencies.

    Produces what ``HighLevelGraph.from_collections(name, layer, collections)``
    produces for the ``Delayed`` family: the same class, the same layers, the
    same dependencies and the same dict *insertion order* - existing-then-new for
    a single ``HighLevelGraph``-backed dependency, new-then-existing for a single
    ``dict``-backed one and for several dependencies. Insertion order is part of
    the contract because ``tokenize`` hashes the pickled state of a ``Delayed``.
    The public ``HighLevelGraph`` constructor always has the last word, so the
    container keeps ownership of its own invariants.

    A dependency is merged directly only when it is exactly a ``Delayed``,
    ``DelayedLeaf`` or ``DelayedAttr`` (never a subclass); when an exact
    ``Delayed`` carries exactly a ``HighLevelGraph`` or exactly a ``dict`` (any
    other ``Mapping``, such as ``types.MappingProxyType``, does not qualify); and
    when its key *and* its layer name are recursively builtin - exactly ``str``,
    ``bytes``, ``int``, ``float``, ``bool`` or ``None``, or exactly a ``tuple`` of
    such values. A ``DelayedAttr`` merges its parent's graph into its own,
    recursively, so the same test applies to every ancestor reachable through
    ``_obj``. One dependency, or one ancestor, that fails the test sends the whole
    call to ``HighLevelGraph.from_collections``, whose side effects a caller can
    count: it ``repr()``s a key that is not builtin, through the ``tokenize`` of
    its ``is_dask_collection`` probe, and hashes a layer name that is not builtin
    while merging. The guard therefore completes before any output dictionary is
    built and before any lazy ``dask`` property is read, so a call that falls back
    reproduces the generic path exactly.

    Args:
        name: Key and layer name of the new node - any key object, exactly as
            ``HighLevelGraph.from_collections`` accepts (its own annotation says
            ``str``, which is why the fallback call below carries an ignore).
        layer: The new node's layer mapping, ``{name: task}``.
        collections: The dependencies, as any sized iterable:
            ``unpack_collections`` passes a tuple, ``call_function`` a list,
            ``DelayedAttr.dask`` a one-tuple and the ``traverse=False`` path of
            ``delayed`` an empty ``set``. Duplicates are allowed and are merged
            once, in order of first appearance.

    Returns:
        The new node's ``HighLevelGraph``: every dependency's layers and
        dependencies merged with ``layer`` under ``name``.
    """
    # Both dicts are keyed by layer name and hold whatever layer mapping their
    # source carries - a raw mapping here, an already-wrapped ``Layer`` when it
    # comes from another graph - so they stay deliberately unparameterised; the
    # ``HighLevelGraph`` constructor is what normalises the values.
    layers: dict
    deps: dict

    ndependencies = len(collections)
    if not ndependencies:
        layers = {name: layer}
        deps = {name: set()}
        return HighLevelGraph(layers, deps)

    dependencies: Sequence | set
    if ndependencies == 1:
        # ``from_collections`` dispatches on the length of the collection it is
        # handed, before any de-duplication, and a single dependency takes the
        # ``_from_collection`` branches - whose insertion order differs from the
        # several-dependency one below.
        dependencies = collections
    else:
        # De-duplicate by identity, first occurrence winning and the order being
        # that of first appearance - exactly ``toolz.unique(..., key=id)``. A
        # dict keyed by ``id`` says that in one pass: a repeated key keeps its
        # first position, and the object it maps to is the same either way.
        dependencies = list({id(c): c for c in collections}.values())

    # Guard pass. It completes before either output dictionary - ``layers`` or
    # ``deps`` - is built, so a dependency that fails late cannot leave
    # half-merged state behind, and no lazy ``dask`` property is evaluated for a
    # call that ends up falling back. ``graphs`` holds, per dependency, the graph
    # read here and reused by the merge, or ``None`` where the merge produces the
    # layer itself (a leaf) or has to build it lazily (an attribute).
    graphs: list = []
    shortcut = True
    for collection in dependencies:
        candidate = collection
        stored = None
        depth = 0
        while True:
            typ = type(candidate)
            if typ is Delayed:
                # Read through the public property; on the exact class that is a
                # plain slot read, so it builds nothing and cannot be observed.
                candidate_graph = candidate.dask
                graph_typ = type(candidate_graph)
                if graph_typ is not HighLevelGraph and graph_typ is not dict:
                    shortcut = False
                    break
                if candidate is collection:
                    stored = candidate_graph
            elif typ is not DelayedLeaf and typ is not DelayedAttr:
                shortcut = False
                break
            key = candidate._key
            layer_name = candidate._layer
            if type(key) is not str or type(layer_name) is not str:
                # A plain string is the overwhelmingly common key and layer name
                # and needs no walk. Otherwise both have to be *recursively
                # builtin*: exactly ``str``, ``bytes``, ``int``, ``float``,
                # ``bool`` or ``None``, or exactly a ``tuple`` of such values -
                # the tuple form being what collections such as arrays and
                # dataframes use for their keys. Subclasses are rejected on
                # purpose: only for an exactly builtin key and layer name is it
                # unobservable that the shortcut skips the
                # ``is_dask_collection`` probe and the hashing that
                # ``HighLevelGraph.from_collections`` would otherwise perform -
                # a user-defined key is ``repr()``-ed by the probe's
                # ``tokenize`` and a user-defined layer name is hashed by the
                # merge, and both are side effects callers can count. The walk
                # uses an explicit stack and exact-type identity tests
                # throughout, so the guard itself can never run user code (no
                # ``__eq__``, ``__hash__`` or ``__instancecheck__`` dispatch).
                values = [key, layer_name]
                while values:
                    value = values.pop()
                    value_typ = type(value)
                    if value_typ is tuple:
                        values += value
                    elif not (
                        value_typ is str
                        or value_typ is bytes
                        or value_typ is int
                        or value_typ is float
                        or value_typ is bool
                        or value is None
                    ):
                        shortcut = False
                        break
                if not shortcut:
                    break
            if typ is not DelayedAttr:
                break
            # An attribute's ``dask`` builds its parent's graph into its own, so
            # the parent - and the parent's own ancestors - must be safe too.
            candidate = candidate._obj
            depth += 1
            if depth > 64:
                # ``_obj`` is a declared slot, so a caller can assign to it and
                # build a cyclic chain. Bounding the walk sends such a chain down
                # the generic path rather than looping here.
                shortcut = False
                break
        if not shortcut:
            # One dependency failing sends the whole call down the generic path,
            # with the dependencies exactly as they were handed in.
            # ``from_collections`` declares ``name: str`` and
            # ``dependencies: Sequence[DaskCollection]``; this helper is
            # deliberately as broad as the runtime is - a key may be any
            # hashable object and ``delayed`` passes a ``set`` - so those two
            # arguments are silenced for the static checker only.
            return HighLevelGraph.from_collections(
                name,  # type: ignore[arg-type]
                layer,
                dependencies=collections,  # type: ignore[arg-type]
            )
        graphs.append(stored)

    if ndependencies == 1:
        (collection,) = dependencies
        (graph,) = graphs
        if graph is None:
            if type(collection) is DelayedLeaf:
                layers = {collection._key: _leaf_layer_dict(collection), name: layer}
                deps = {collection._key: set(), name: {collection._layer}}
                return HighLevelGraph(layers, deps)
            graph = collection.dask
        if type(graph) is HighLevelGraph:
            layers = ensure_dict(graph.layers, copy=True)
            layers[name] = layer
            deps = ensure_dict(graph.dependencies, copy=True)
            deps[name] = {collection._layer}
            return HighLevelGraph(layers, deps)
        # ``_get_some_layer_name`` unpacks ``__dask_layers__()``, which for the
        # ``Delayed`` family is ``(_layer,)``. Note that the new layer goes in
        # first on this branch only.
        layers = {name: layer, collection._layer: graph}
        deps = {name: {collection._layer}, collection._layer: set()}
        return HighLevelGraph(layers, deps)

    layers = {name: layer}
    name_dep: set = set()
    deps = {name: name_dep}
    for collection, graph in zip(dependencies, graphs):
        # ``name_dep`` is grown exactly the way ``from_collections`` grows it:
        # in-place union with a one-element set where it merges a
        # ``HighLevelGraph`` and ``add`` where it takes a low-level graph. The
        # two differ in how the set resizes, and a set's resize history decides
        # its iteration order and therefore the bytes it pickles to - which feed
        # the token of any expression that reaches a ``Delayed`` through pickle.
        if graph is None:
            if type(collection) is DelayedLeaf:
                layers[collection._key] = _leaf_layer_dict(collection)
                deps[collection._key] = set()
                name_dep |= {collection._layer}
                continue
            graph = collection.dask
        if type(graph) is HighLevelGraph:
            layers.update(graph.layers)
            deps.update(graph.dependencies)
            name_dep |= {collection._layer}
        else:
            layers[collection._layer] = graph
            name_dep.add(collection._layer)
            deps[collection._layer] = set()
    return HighLevelGraph(layers, deps)


def unpack_collections(expr, _return_collections=True):
    """Normalize a python object and merge all sub-graphs.

    - Replace ``Delayed`` with their keys
    - Convert literals to things the schedulers can handle
    - Extract dask graphs from all enclosed values.

    Note, that the returned _task_ is not necessarily runnable and the caller is
    responsible to deal with the output types accordingly.

    The task is one of
    - `TaskRef` as a pointer to the collection returned in collections. This is
      not callable and should not be a top-level member of a dask task graph.
    - A runnable task (i.e. subclass `GraphNode`) which can be embedded
      directly into a task graph. This indicates that a dask collection was
      encountered on a deeper nesting level and this runnable task restores the
      input nesting with the computed dask collection replaced.
    - The unaltered object as provided if no dask collections are found.

    Parameters
    ----------
    expr : object
        The object to be normalized. This function knows how to handle
        dask collections, as well as most builtin python types.

    _optimize_collections: bool, optional
        Internal use only!


    Returns
    -------
    task : object
    collections : a tuple of collections

    Examples
    --------
    >>> import dask
    >>> a = delayed(1, 'a')
    >>> b = delayed(2, 'b')
    >>> task, collections = unpack_collections([a, b, 3])
    >>> task
    List((TaskRef('a'), TaskRef('b'), 3))
    >>> collections
    (Delayed('a'), Delayed('b'))

    >>> task, collections = unpack_collections({a: 1, b: 2})
    >>> task
    Dict(a: 1, b: 2)
    >>> collections
    (Delayed('a'), Delayed('b'))
    """
    typ = type(expr)

    # Fast path for the builtin scalars that can match none of the branches below
    # and therefore always fall through to ``return expr, ()``. The dispatch is by
    # exact-type identity only - never ``isinstance``, never a set or dict
    # membership test - so it runs no user code at all and leaves ``typ in
    # _SEQUENCE_TYPES`` below as the first and only place the traversal compares a
    # type object. An exact type also settles the two probes it skips: a value of
    # exactly one of these types is never a ``Delayed``, and it can never be a dask
    # collection either, because a built-in type cannot acquire a
    # ``__dask_graph__`` attribute. A subclass of, say, ``int`` matches no identity
    # test here and takes the full cascade.
    if (
        expr is None
        or typ is int
        or typ is str
        or typ is float
        or typ is bool
        or typ is bytes
        or typ is complex
    ):
        return expr, ()

    if isinstance(expr, Delayed):
        if _return_collections:
            return TaskRef(expr._key), (expr,)
        else:
            expr = collections_to_expr(expr).finalize_compute()
            (name,) = expr.__dask_keys__()
            return TaskRef(name), (expr,)

    # FIXME: Make this not trigger materialization
    # Currently this is checking with hasattr for __dask_graph__ which triggers
    # a materialization
    if base.is_dask_collection(expr):
        if _return_collections:
            expr2 = ProhibitReuse(collections_to_expr(expr).finalize_compute())
            finalized = expr2.optimize()
            # FIXME: Make this also go away
            dsk = finalized.__dask_graph__()
            keys = list(flatten(finalized.__dask_keys__()))
            if len(keys) > 1:
                # `finalize_compute` _should_ guarantee that we only have one key
                raise RuntimeError(
                    "Cannot unpack dask collections which don't finalize to a "
                    f"single key. Got {type(expr)} with {keys=}",
                )

            return unpack_collections(Delayed(keys[0], dsk))
        else:
            expr = collections_to_expr(expr).finalize_compute()
            (name,) = expr.__dask_keys__()
            return TaskRef(name), (expr,)

    # Iterators are materialized into the container they iterate over. ``typ``
    # is read once above, so each coercion has to update it too.
    if typ is _LIST_ITER_TYPE:
        expr = list(expr)
        typ = list
    elif typ is _TUPLE_ITER_TYPE:
        expr = tuple(expr)
        typ = tuple
    elif typ is _SET_ITER_TYPE:
        expr = set(expr)
        typ = set

    if typ in _SEQUENCE_TYPES:
        args: list = []
        collections: list = []
        # Bound once rather than looked up per element: this loop runs for every
        # element of every container the traversal reaches.
        append_arg = args.append
        extend_collections = collections.extend
        # ``verbatim`` stays true while every element comes back from the
        # recursion as the object that went in and is itself no node, which is
        # the condition under which the ``List`` built below carries no
        # ``dependencies``: they can only come from a ``TaskRef`` or a
        # ``GraphNode`` among its arguments. The branch then returns ``expr``
        # itself, so the pair is the same whether or not the node was built
        # first. ``len(args) != 1`` is what keeps that reasoning sound: a lone
        # argument is the one case ``NestedContainer.__init__`` unwraps when it
        # is a ``list`` instance, and unwrapping scans the *contents* of an
        # element the exact-type dispatch left atomic - a ``list`` subclass
        # holding a ``TaskRef`` does carry a dependency that way. Every
        # container with a single element therefore still goes through
        # ``List(*args)``, and so does every container that reaches a node.
        verbatim = True
        for e in expr:
            arg, subcollections = unpack_collections(e, _return_collections=False)
            if arg is not e or isinstance(e, (TaskRef, GraphNode)):
                verbatim = False
            append_arg(arg)
            extend_collections(subcollections)
        if verbatim and not collections and len(args) != 1:
            return expr, ()
        if len(collections) > 1:
            # De-duplicate by identity, first occurrence winning and the order
            # being that of first appearance - exactly ``unique(..., key=id)``.
            # A dict keyed by ``id`` says precisely that: a repeated key keeps
            # its first position, and the value it keeps is the same object
            # either way. Every element is kept alive by ``expr`` for the
            # duration of the call, so an id can neither be reused nor go stale
            # here.
            collections = list({id(c): c for c in collections}.values())
        # Every branch hands back a tuple of collections, including the
        # short-circuits below, so that callers can concatenate them.
        collections = tuple(collections)
        # The List constructor also checks for futures
        args = List(*args)
        if not collections and not args.dependencies:
            return expr, ()
        if _return_collections:
            args, collections = _finalize_args_collections(args, collections)
        # Ensure output type matches input type
        if typ is not list:
            args = Task(None, typ, args)
        return args, collections

    if typ is dict:
        if not expr:
            return expr, ()
        keyargs, kcollections = unpack_collections(
            list(expr.keys()), _return_collections=False
        )
        valargs, valcollections = unpack_collections(
            list(expr.values()), _return_collections=False
        )
        collections = kcollections + valcollections
        args = Dict([[k, v] for k, v in zip(keyargs, valargs)])
        if not collections and not args.dependencies:
            return expr, ()
        if _return_collections:
            args, collections = _finalize_args_collections(args, collections)
        return args, collections

    if typ is slice:
        args, collections = unpack_collections(
            [expr.start, expr.stop, expr.step], _return_collections=False
        )
        if not collections and not isinstance(args, GraphNode):
            return expr, ()

        if _return_collections:
            args, collections = _finalize_args_collections(args, collections)
        return Task(None, apply, slice, args), collections

    if is_dataclass(expr):
        args, collections = unpack_collections(
            [
                [f.name, getattr(expr, f.name)]
                for f in fields(expr)
                if hasattr(expr, f.name)  # if init=False, field might not exist
            ],
            _return_collections=False,
        )
        if not collections and not isinstance(args, GraphNode):
            return expr, ()

        if _return_collections:
            args, collections = _finalize_args_collections(args, collections)
        try:
            _fields = {
                f.name: getattr(expr, f.name)
                for f in fields(expr)
                if hasattr(expr, f.name)
            }
            replace(expr, **_fields)
        except (TypeError, ValueError) as e:
            if isinstance(e, ValueError) or "is declared with init=False" in str(e):
                raise ValueError(
                    f"Failed to unpack {typ} instance. "
                    "Note that using fields with `init=False` are not supported."
                ) from e
            else:
                raise TypeError(
                    f"Failed to unpack {typ} instance. "
                    "Note that using a custom __init__ is not supported."
                ) from e
        return Task(None, apply, typ, (), Task(None, dict, args)), collections

    if utils.is_namedtuple_instance(expr):
        args, collections = unpack_collections(
            tuple(v for v in expr), _return_collections=False
        )
        if not collections:
            return expr, ()
        if _return_collections:
            args, collections = _finalize_args_collections(args, collections)
        return Task(None, _reconstruct_namedtuple, typ, args), collections

    return expr, ()


def _reconstruct_namedtuple(typ, fields):
    return typ(*fields)


def to_task_dask(expr):
    """Normalize a python object and merge all sub-graphs.

    - Replace ``Delayed`` with their keys
    - Convert literals to things the schedulers can handle
    - Extract dask graphs from all enclosed values

    Parameters
    ----------
    expr : object
        The object to be normalized. This function knows how to handle
        ``Delayed``s, as well as most builtin python types.

    Returns
    -------
    task : normalized task to be run
    dask : a merged dask graph that forms the dag for this task

    Examples
    --------
    >>> import dask
    >>> a = delayed(1, 'a')
    >>> b = delayed(2, 'b')
    >>> task, dask = to_task_dask([a, b, 3])  # doctest: +SKIP
    >>> task  # doctest: +SKIP
    ['a', 'b', 3]
    >>> dict(dask)  # doctest: +SKIP
    {'a': 1, 'b': 2}

    >>> task, dasks = to_task_dask({a: 1, b: 2})  # doctest: +SKIP
    >>> task  # doctest: +SKIP
    (dict, [['a', 1], ['b', 2]])
    >>> dict(dask)  # doctest: +SKIP
    {'a': 1, 'b': 2}
    """
    warnings.warn(
        "The dask.delayed.to_dask_dask function has been "
        "Deprecated in favor of unpack_collections",
        stacklevel=2,
    )

    if isinstance(expr, Delayed):
        return expr.key, expr.dask

    if is_dask_collection(expr):
        expr = collections_to_expr(expr)
        expr = FinalizeCompute(expr)
        expr = expr.optimize()
        (name,) = expr.__dask_keys__()
        return name, expr.__dask_graph__()

    if type(expr) is type(iter(list())):
        expr = list(expr)
    elif type(expr) is type(iter(tuple())):
        expr = tuple(expr)
    elif type(expr) is type(iter(set())):
        expr = set(expr)
    typ = type(expr)

    if typ in (list, tuple, set):
        args, dasks = unzip((to_task_dask(e) for e in expr), 2)
        args = list(args)
        dsk = merge(dasks)
        # Ensure output type matches input type
        return (args, dsk) if typ is list else ((typ, args), dsk)

    if typ is dict:
        args, dsk = to_task_dask([[k, v] for k, v in expr.items()])
        return (dict, args), dsk

    if is_dataclass(expr):
        args, dsk = to_task_dask(
            [
                [f.name, getattr(expr, f.name)]
                for f in fields(expr)
                if hasattr(expr, f.name)  # if init=False, field might not exist
            ]
        )

        return (apply, typ, (), (dict, args)), dsk

    if is_namedtuple_instance(expr):
        args, dsk = to_task_dask([v for v in expr])
        return (typ, *args), dsk

    if typ is slice:
        args, dsk = to_task_dask([expr.start, expr.stop, expr.step])
        return (slice,) + tuple(args), dsk

    return expr, {}


def tokenize(*args, pure=None, **kwargs):
    """Mapping function from task -> consistent name.

    Parameters
    ----------
    args : object
        Python objects that summarize the task.
    pure : boolean, optional
        If True, a consistent hash function is tried on the input. If this
        fails, then a unique identifier is used. If False (default), then a
        unique identifier is always used.
    """
    if pure is None:
        pure = config.get("delayed_pure", False)

    if pure:
        return _tokenize(*args, **kwargs)
    else:
        return str(uuid.uuid4())


@curry
def delayed(obj, name=None, pure=None, nout=None, traverse=True):
    """Wraps a function or object to produce a ``Delayed``.

    ``Delayed`` objects act as proxies for the object they wrap, but all
    operations on them are done lazily by building up a dask graph internally.

    Parameters
    ----------
    obj : object
        The function or object to wrap
    name : Dask key, optional
        The key to use in the underlying graph for the wrapped object. Defaults
        to hashing content. Note that this only affects the name of the object
        wrapped by this call to delayed, and *not* the output of delayed
        function calls - for that use ``dask_key_name=`` as described below.

        .. note::

           Because this ``name`` is used as the key in task graphs, you should
           ensure that it uniquely identifies ``obj``. If you'd like to provide
           a descriptive name that is still unique, combine the descriptive name
           with :func:`dask.base.tokenize` of the ``array_like``. See
           :ref:`graphs` for more.

    pure : bool, optional
        Indicates whether calling the resulting ``Delayed`` object is a pure
        operation. If True, arguments to the call are hashed to produce
        deterministic keys. If not provided, the default is to check the global
        ``delayed_pure`` setting, and fallback to ``False`` if unset.
    nout : int, optional
        The number of outputs returned from calling the resulting ``Delayed``
        object. If provided, the ``Delayed`` output of the call can be iterated
        into ``nout`` objects, allowing for unpacking of results. By default
        iteration over ``Delayed`` objects will error. Note, that ``nout=1``
        expects ``obj`` to return a tuple of length 1, and consequently for
        ``nout=0``, ``obj`` should return an empty tuple.
    traverse : bool, optional
        By default dask traverses builtin python collections looking for dask
        objects passed to ``delayed``. For large collections this can be
        expensive. If ``obj`` doesn't contain any dask objects, set
        ``traverse=False`` to avoid doing this traversal.

    Examples
    --------
    Apply to functions to delay execution:

    >>> from dask import delayed
    >>> def inc(x):
    ...     return x + 1

    >>> inc(10)
    11

    >>> x = delayed(inc, pure=True)(10)
    >>> type(x) == Delayed
    True
    >>> x.compute()
    11

    Can be used as a decorator:

    >>> @delayed(pure=True)
    ... def add(a, b):
    ...     return a + b
    >>> add(1, 2).compute()
    3

    ``delayed`` also accepts an optional keyword ``pure``. If False, then
    subsequent calls will always produce a different ``Delayed``. This is
    useful for non-pure functions (such as ``time`` or ``random``).

    >>> from random import random
    >>> out1 = delayed(random, pure=False)()
    >>> out2 = delayed(random, pure=False)()
    >>> out1.key == out2.key
    False

    If you know a function is pure (output only depends on the input, with no
    global state), then you can set ``pure=True``. This will attempt to apply a
    consistent name to the output, but will fallback on the same behavior of
    ``pure=False`` if this fails.

    >>> @delayed(pure=True)
    ... def add(a, b):
    ...     return a + b
    >>> out1 = add(1, 2)
    >>> out2 = add(1, 2)
    >>> out1.key == out2.key
    True

    Instead of setting ``pure`` as a property of the callable, you can also set
    it contextually using the ``delayed_pure`` setting. Note that this
    influences the *call* and not the *creation* of the callable:

    >>> @delayed
    ... def mul(a, b):
    ...     return a * b
    >>> import dask
    >>> with dask.config.set(delayed_pure=True):
    ...     print(mul(1, 2).key == mul(1, 2).key)
    True
    >>> with dask.config.set(delayed_pure=False):
    ...     print(mul(1, 2).key == mul(1, 2).key)
    False

    The key name of the result of calling a delayed object is determined by
    hashing the arguments by default. To explicitly set the name, you can use
    the ``dask_key_name`` keyword when calling the function:

    >>> add(1, 2)   # doctest: +SKIP
    Delayed('add-3dce7c56edd1ac2614add714086e950f')
    >>> add(1, 2, dask_key_name='three')
    Delayed('three')

    Note that objects with the same key name are assumed to have the same
    result. If you set the names explicitly you should make sure your key names
    are different for different results.

    >>> add(1, 2, dask_key_name='three')
    Delayed('three')
    >>> add(2, 1, dask_key_name='three')
    Delayed('three')
    >>> add(2, 2, dask_key_name='four')
    Delayed('four')

    ``delayed`` can also be applied to objects to make operations on them lazy:

    >>> a = delayed([1, 2, 3])
    >>> isinstance(a, Delayed)
    True
    >>> a.compute()
    [1, 2, 3]

    The key name of a delayed object is hashed by default if ``pure=True`` or
    is generated randomly if ``pure=False`` (default).  To explicitly set the
    name, you can use the ``name`` keyword. To ensure that the key is unique
    you should include the tokenized value as well, or otherwise ensure that
    it's unique:

    >>> from dask.base import tokenize
    >>> data = [1, 2, 3]
    >>> a = delayed(data, name='mylist-' + tokenize(data))
    >>> a  # doctest: +SKIP
    Delayed('mylist-55af65871cb378a4fa6de1660c3e8fb7')

    Delayed results act as a proxy to the underlying object. Many operators
    are supported:

    >>> (a + [1, 2]).compute()
    [1, 2, 3, 1, 2]
    >>> a[1].compute()
    2

    Method and attribute access also works:

    >>> a.count(2).compute()
    1

    Note that if a method doesn't exist, no error will be thrown until runtime:

    >>> res = a.not_a_real_method() # doctest: +SKIP
    >>> res.compute()  # doctest: +SKIP
    AttributeError("'list' object has no attribute 'not_a_real_method'")

    "Magic" methods (e.g. operators and attribute access) are assumed to be
    pure, meaning that subsequent calls must return the same results. This
    behavior is not overridable through the ``delayed`` call, but can be
    modified using other ways as described below.

    To invoke an impure attribute or operator, you'd need to use it in a
    delayed function with ``pure=False``:

    >>> class Incrementer:
    ...     def __init__(self):
    ...         self._n = 0
    ...     @property
    ...     def n(self):
    ...         self._n += 1
    ...         return self._n
    ...
    >>> x = delayed(Incrementer())
    >>> x.n.key == x.n.key
    True
    >>> get_n = delayed(lambda x: x.n, pure=False)
    >>> get_n(x).key == get_n(x).key
    False

    In contrast, methods are assumed to be impure by default, meaning that
    subsequent calls may return different results. To assume purity, set
    ``pure=True``. This allows sharing of any intermediate values.

    >>> a.count(2, pure=True).key == a.count(2, pure=True).key
    True

    As with function calls, method calls also respect the global
    ``delayed_pure`` setting and support the ``dask_key_name`` keyword:

    >>> a.count(2, dask_key_name="count_2")
    Delayed('count_2')
    >>> import dask
    >>> with dask.config.set(delayed_pure=True):
    ...     print(a.count(2).key == a.count(2).key)
    True
    """
    if isinstance(obj, Delayed):
        return obj

    if is_dask_collection(obj) or traverse:
        task, collections = unpack_collections(obj)
    else:
        task = quote(obj)
        collections = set()

    if not (nout is None or (type(nout) is int and nout >= 0)):
        raise ValueError(f"nout must be None or a non-negative integer, got {nout}")
    if task is obj:
        if isinstance(obj, TaskRef):
            name = obj.key
        elif not name:
            try:
                prefix = obj.__name__
            except AttributeError:
                prefix = type(obj).__name__
            token = tokenize(obj, nout, pure=pure)
            name = f"{prefix}-{token}"
        return DelayedLeaf(obj, name, pure=pure, nout=nout)
    else:
        if not name:
            name = f"{type(obj).__name__}-{tokenize(task, pure=pure)}"
        layer = {name: task}
        if isinstance(task, GraphNode):
            task.key = name
        graph = _graph_from_collections(name, layer, collections)
        return Delayed(name, graph, nout)


def _swap(method, self, other):
    return method(other, self)


def right(method):
    """Wrapper to create 'right' version of operator given left version"""
    return partial(_swap, method)


def optimize(dsk, keys, **kwargs):
    if not isinstance(keys, (list, set)):
        keys = [keys]

    if config.get("optimization.fuse.delayed"):
        dsk = ensure_dict(dsk)
        dsk = fuse_linear_task_spec(dsk, keys, **kwargs)

    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())
    dsk = dsk.cull(set(flatten(keys)))
    return dsk


class Delayed(DaskMethodsMixin, OperatorMethodMixin):
    """Represents a value to be computed by dask.

    Equivalent to the output from a single key in a dask graph.
    """

    __slots__ = ("_key", "_dask", "_length", "_layer")

    def __init__(self, key, dsk, length=None, layer=None):
        self._key = key
        self._dask = dsk
        self._length = length

        # NOTE: Layer is used by `to_delayed` in other collections, but not in normal Delayed use
        self._layer = layer or key
        if isinstance(dsk, HighLevelGraph) and self._layer not in dsk.layers:
            raise ValueError(
                f"Layer {self._layer} not in the HighLevelGraph's layers: {list(dsk.layers)}"
            )

    @property
    def key(self):
        return self._key

    @property
    def dask(self):
        return self._dask

    def __dask_graph__(self) -> Graph:
        return self.dask

    def __dask_keys__(self) -> NestedKeys:
        return [self.key]

    def __dask_layers__(self) -> Sequence[str]:
        return (self._layer,)

    def __dask_tokenize__(self):
        return self.key

    __dask_scheduler__ = staticmethod(DEFAULT_GET)
    __dask_optimize__ = globalmethod(optimize, key="delayed_optimize")

    def __dask_postcompute__(self):
        return single_key, ()

    def __dask_postpersist__(self):
        return self._rebuild, ()

    def _rebuild(self, dsk, *, rename=None):
        key = replace_name_in_key(self.key, rename) if rename else self.key
        if isinstance(dsk, HighLevelGraph) and len(dsk.layers) == 1:
            # FIXME Delayed is currently the only collection type that supports both high- and low-level graphs.
            # The HLG output of `optimize` will have a layer name that doesn't match `key`.
            # Remove this when Delayed is HLG-only (because `optimize` will only be passed HLGs, so it won't have
            # to generate random layer names).
            layer = next(iter(dsk.layers))
        else:
            layer = None
        return Delayed(key, dsk, self._length, layer=layer)

    def __repr__(self):
        return f"Delayed({self.key!r})"

    def __hash__(self):
        return hash(self.key)

    def __dir__(self):
        return dir(type(self))

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(f"Attribute {attr} not found")

        if attr == "visualise":
            # added to warn users in case of spelling error
            # for more details: https://github.com/dask/dask/issues/5721
            warnings.warn(
                "dask.delayed objects have no `visualise` method. "
                "Perhaps you meant `visualize`?"
            )

        return DelayedAttr(self, attr)

    def __setattr__(self, attr, val):
        try:
            object.__setattr__(self, attr, val)
        except AttributeError:
            # attr is neither in type(self).__slots__ nor in the __slots__ of any of its
            # parent classes, and all the parent classes define __slots__ too.
            # This last bit needs to be unit tested: if any of the parent classes omit
            # the __slots__ declaration, self will gain a __dict__ and this branch will
            # become unreachable.
            raise TypeError("Delayed objects are immutable")

    def __setitem__(self, index, val):
        raise TypeError("Delayed objects are immutable")

    def __iter__(self):
        if self._length is None:
            raise TypeError("Delayed objects of unspecified length are not iterable")
        for i in range(self._length):
            yield self[i]

    def __len__(self):
        if self._length is None:
            raise TypeError("Delayed objects of unspecified length have no len()")
        return self._length

    def __call__(self, *args, pure=None, dask_key_name=None, **kwargs):
        func = delayed(apply, pure=pure)
        if dask_key_name is not None:
            return func(self, args, kwargs, dask_key_name=dask_key_name)
        return func(self, args, kwargs)

    def __bool__(self):
        raise TypeError("Truth of Delayed objects is not supported")

    __nonzero__ = __bool__

    def __get__(self, instance, cls):
        if instance is None:
            return self
        return types.MethodType(self, instance)

    @classmethod
    def _get_binary_operator(cls, op, inv=False):
        method = delayed(right(op) if inv else op, pure=True)
        return lambda *args, **kwargs: method(*args, **kwargs)

    _get_unary_operator = _get_binary_operator


def call_function(func, func_token, args, kwargs, pure=None, nout=None):
    dask_key_name = kwargs.pop("dask_key_name", None)
    pure = kwargs.pop("pure", pure)

    if dask_key_name is None:
        name = f"{funcname(func)}-{tokenize(func_token, *args, pure=pure, **kwargs)}"
    else:
        name = dask_key_name

    args2 = []
    collections = []
    for arg in args:
        arg_task, subcollections = unpack_collections(arg)
        args2.append(arg_task)
        collections.extend(subcollections)

    if kwargs:
        dask_kwargs, kwargs_collections = unpack_collections(kwargs)
        collections.extend(kwargs_collections)
    else:
        # An empty mapping needs no traversal: it can only unpack to itself, and
        # ``Task`` receives the same empty keyword mapping either way.
        dask_kwargs = kwargs
    task = Task(name, func, *args2, **dask_kwargs)

    graph = _graph_from_collections(name, {name: task}, collections)
    return Delayed(name, graph, length=nout)


class DelayedLeaf(Delayed):
    __slots__ = ("_obj", "_pure", "_nout")

    def __init__(self, obj, key, pure=None, nout=None):
        super().__init__(key, None, length=nout)
        self._obj = obj
        self._pure = pure
        self._nout = nout

    @property
    def dask(self):
        return HighLevelGraph({self._key: _leaf_layer_dict(self)}, {self._key: set()})

    def __call__(self, *args, **kwargs):
        return call_function(
            self._obj, self._key, args, kwargs, pure=self._pure, nout=self._nout
        )

    @property
    def __name__(self):
        return self._obj.__name__

    @property
    def __doc__(self):
        return self._obj.__doc__

    @property
    def __wrapped__(self):
        return self._obj


class DelayedAttr(Delayed):
    __slots__ = ("_obj", "_attr")

    def __init__(self, obj, attr):
        key = f"getattr-{tokenize(obj, attr, pure=True)}"
        super().__init__(key, None)
        self._obj = obj
        self._attr = attr

    def __getattr__(self, attr):
        # Calling np.dtype(dask.delayed(...)) used to result in a segfault, as
        # numpy recursively tries to get `dtype` from the object. This is
        # likely a bug in numpy. For now, we can do a dumb for if
        # `x.dtype().dtype()` is called (which shouldn't ever show up in real
        # code). See https://github.com/dask/dask/pull/4374#issuecomment-454381465
        if attr == "dtype" and self._attr == "dtype":
            raise AttributeError("Attribute dtype not found")
        return super().__getattr__(attr)

    @property
    def dask(self):
        layer = {self._key: (getattr, self._obj._key, self._attr)}
        return _graph_from_collections(self._key, layer, (self._obj,))

    def __call__(self, *args, **kwargs):
        return call_function(
            methodcaller(self._attr), self._attr, (self._obj,) + args, kwargs
        )


for op in [
    operator.abs,
    operator.neg,
    operator.pos,
    operator.invert,
    operator.add,
    operator.sub,
    operator.mul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    operator.pow,
    operator.and_,
    operator.or_,
    operator.xor,
    operator.lshift,
    operator.rshift,
    operator.eq,
    operator.ge,
    operator.gt,
    operator.ne,
    operator.le,
    operator.lt,
    operator.getitem,
]:
    Delayed._bind_operator(op)


try:
    Delayed._bind_operator(operator.matmul)
except AttributeError:
    pass


def single_key(seq):
    """Pick out the only element of this list, a list of keys"""
    return seq[0]
