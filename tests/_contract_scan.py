"""Derive the contract's inventories from source, instead of listing them by hand.

Not a test module. ``tests/test_contract.py`` imports these functions to
parametrize itself, and ``tests/test_contract_detects.py`` points them at
mutated copies to prove they see what a literal list could not.

Three lists used to be written out by hand: the exception taxonomy, the ports,
and the ``(adapter, port)`` pairs. A hand-written list is fail-open. It cannot
see anything added after it was written, which is not hypothetical here:
``IncompatibleStore`` reached ``errors.__all__`` while missing from
``denselinkage.core`` entirely, and no test noticed (issue #31, PR #29).

Separate from ``tests/_api_snapshot.py`` on purpose. That module answers "did
the frozen surface move since the last release", and its repair is a deliberate
``--regenerate --authority``. This one answers "is the code self-consistent
right now", and its repair is always to fix the source. Sharing a module would
put one recovery in front of the other.

Everything here reads the AST and never imports the package, so every function
takes a ``package_root`` and can be pointed at a copy on disk. That is what lets
the negative controls plant a violation in a scratch tree, which runtime
discovery could not do.
"""

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests._api_snapshot import FROZEN_MODULES, PACKAGE_ROOT, parse_module

PORTS_MODULE = "core/ports.py"
ERRORS_MODULE = "core/errors.py"
CORE_INIT = "core/__init__.py"
ROOT_ERROR = "DenseLinkageError"

#: The adapter rule applies to every public module the freeze does not cover.
#: ``core/__init__.py`` joins the frozen list because it is a re-export facade
#: with no classes of its own.
_OUT_OF_ADAPTER_SCOPE = frozenset(FROZEN_MODULES) | {CORE_INIT}

#: Members that make a class a candidate adapter. A class with no public
#: callable implements no port, which is what excludes ``RetryPolicy`` by
#: predicate rather than by name.
_CALLABLE_KINDS = frozenset({"method", "classmethod", "staticmethod", "property"})

#: name, kind, has_default. The default *value* is deliberately excluded: an
#: adapter may narrow a default without breaking the port's calling convention.
Parameter = tuple[str, str, bool]
Signature = tuple[Parameter, ...]


@dataclass(frozen=True)
class ClassDecl:
    """One class declaration, located, with its bases resolved to bare names."""

    module: str
    name: str
    bases: tuple[str, ...]
    lineno: int

    @property
    def where(self) -> str:
        return f"src/denselinkage/{self.module}:{self.lineno}"


def is_private(relative: str) -> bool:
    """A module is private when any path part starts with an underscore.

    ``__init__.py`` is the exception: it is the package's public face.
    ADR-0006 puts private names outside the contract, so private modules are
    outside the adapter rule too.
    """
    parts = relative.split("/")
    return any(part.startswith("_") and part != "__init__.py" for part in parts)


def iter_modules(package_root: Path = PACKAGE_ROOT) -> list[tuple[Path, str]]:
    """Every module in the package, as (path, posix-relative), sorted."""
    return [
        (path, path.relative_to(package_root).as_posix())
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def _alias_map(tree: ast.Module) -> dict[str, str]:
    """Import aliases, so ``DenseLinkageError as _DLE`` still resolves.

    Only aliased names are recorded; an unaliased import already arrives under
    the name the class declaration uses.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name.rsplit(".", 1)[-1]
    return aliases


def _base_name(node: ast.expr, aliases: dict[str, str]) -> str:
    """The bare class name of a base expression.

    Drops a subscript (``Protocol[ComponentT]``) and any dotted prefix
    (``errors.DenseLinkageError``), then resolves an import alias.
    """
    rendered = ast.unparse(node).split("[", 1)[0].rsplit(".", 1)[-1]
    return aliases.get(rendered, rendered)


def class_declarations(package_root: Path = PACKAGE_ROOT) -> list[ClassDecl]:
    """Every class in the package, private modules included.

    ``ast.walk`` rather than a module-body scan, so a class nested inside a
    function is still seen. Private modules are included because an exception
    smuggled into ``_store/`` is exactly what the outside-declaration rule
    exists to catch.
    """
    found: list[ClassDecl] = []
    for path, relative in iter_modules(package_root):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = _alias_map(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found.append(
                    ClassDecl(
                        module=relative,
                        name=node.name,
                        bases=tuple(_base_name(b, aliases) for b in node.bases),
                        lineno=node.lineno,
                    )
                )
    return found


def ports(package_root: Path = PACKAGE_ROOT) -> dict[str, dict[str, Any]]:
    """Every ``Protocol`` in ``core/ports.py``, keyed by name."""
    module = parse_module(package_root / PORTS_MODULE, PORTS_MODULE)
    return {c["name"]: c for c in module["classes"] if c["is_protocol"]}


def callables(record: dict[str, Any]) -> dict[str, Signature]:
    """Public callable members of a parsed class, keyed by name.

    ``__init__`` is excluded: it is a construction detail an adapter is free to
    choose, not part of the port it implements.
    """
    out: dict[str, Signature] = {}
    for member in record["members"]:
        if member["kind"] in _CALLABLE_KINDS and member["name"] != "__init__":
            out[member["name"]] = tuple(
                (p["name"], p["kind"], p["has_default"])
                for p in member["parameters"]
                if p["name"] != "self"
            )
    return out


def port_members(package_root: Path = PACKAGE_ROOT) -> dict[str, dict[str, Signature]]:
    """Every port's required members and their parameter shapes."""
    return {name: callables(record) for name, record in ports(package_root).items()}


def _iter_adapter_records(
    package_root: Path,
) -> Iterator[tuple[str, dict[str, Any]]]:
    for path, relative in iter_modules(package_root):
        if relative in _OUT_OF_ADAPTER_SCOPE or is_private(relative):
            continue
        for record in parse_module(path, relative)["classes"]:
            if callables(record):
                yield relative, record


def adapters(package_root: Path = PACKAGE_ROOT) -> list[dict[str, Any]]:
    """Every public class outside the frozen surface that could implement a port.

    Closed-world by construction: a new adapter is picked up the moment it is
    written, which is the whole point. A class with no public callable is not a
    candidate, which excludes value objects such as ``RetryPolicy`` without
    naming them.
    """
    return [
        {"module": relative, "record": record}
        for relative, record in _iter_adapter_records(package_root)
    ]


def declared(record: dict[str, Any], port_names: set[str]) -> tuple[str, ...]:
    """The ports a class explicitly subclasses."""
    return tuple(base for base in record["bases"] if base in port_names)


def missing_members(
    record: dict[str, Any], required: dict[str, Signature]
) -> list[str]:
    """Port members the class does not provide at all."""
    return sorted(set(required) - set(callables(record)))


def incompatible_members(
    record: dict[str, Any], required: dict[str, Signature]
) -> list[str]:
    """Members whose parameters do not satisfy the port, by the override rule.

    The port's parameters must be a prefix of the adapter's, and anything extra
    must be keyword-only with a default. That is mypy's compatible-override
    rule: a caller holding the port must be able to call the adapter.
    """
    provided = callables(record)
    bad: list[str] = []
    for name, want in required.items():
        got = provided.get(name)
        if got is None:
            continue
        if got[: len(want)] != want:
            bad.append(name)
            continue
        extra = got[len(want) :]
        if any(
            kind != "keyword_only" or not has_default for _, kind, has_default in extra
        ):
            bad.append(name)
    return sorted(bad)


def taxonomy(package_root: Path = PACKAGE_ROOT) -> list[dict[str, Any]]:
    """Every class declared in ``core/errors.py``."""
    module = parse_module(package_root / ERRORS_MODULE, ERRORS_MODULE)
    classes: list[dict[str, Any]] = module["classes"]
    return classes


def module_all(relative: str, package_root: Path = PACKAGE_ROOT) -> list[str] | None:
    """A module's ``__all__``, or None when it declares none."""
    names: list[str] | None = parse_module(package_root / relative, relative)["__all__"]
    return names


def error_family_outside(package_root: Path = PACKAGE_ROOT) -> list[ClassDecl]:
    """Every ``DenseLinkageError`` descendant declared outside ``core/errors.py``.

    A fixed point, so a subclass of a subclass is caught too. The taxonomy is a
    published contract: a caller writes ``except DenseLinkageError`` and expects
    the set of things it catches to be the documented one.
    """
    declarations = class_declarations(package_root)
    family = {ROOT_ERROR}
    while True:
        grown = {
            decl.name
            for decl in declarations
            if any(base in family for base in decl.bases)
        }
        if grown <= family:
            break
        family |= grown
    return sorted(
        (
            decl
            for decl in declarations
            if decl.name in family
            and decl.name != ROOT_ERROR
            and decl.module != ERRORS_MODULE
        ),
        key=lambda d: (d.module, d.lineno),
    )
