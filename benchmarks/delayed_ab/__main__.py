from __future__ import annotations

# ``from .main import main`` binds exactly ``benchmarks.delayed_ab.main``: this
# module is executed as ``benchmarks.delayed_ab.__main__``. The absolute spelling
# cannot be used while ``benchmarks/__init__.py`` is absent -- mypy then maps every
# module of this tree under two names and aborts the whole repository's type check.
# ``main.py``'s import block records that error and the two out-of-scope remedies.
from .main import main

raise SystemExit(main())
