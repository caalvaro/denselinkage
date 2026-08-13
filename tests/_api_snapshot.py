"""Derive the frozen public API from source, and diff it against the snapshot.

Not a test module. ``tests/test_frozen_surface.py`` imports the derivation and
the diff; ``python -m tests._api_snapshot`` regenerates the committed snapshot.

The derivation reads the **AST**, never the imported objects, so it records
annotations as they are written (``'SearchableIndex'``, ``Vectors``) rather than
an interpreter-dependent ``repr``. ``ast.unparse`` normalises formatting, so
neither a ``ruff format`` pass nor a docstring edit can move the snapshot; a
signature, a field type, a default, a field's position or a decorator argument
can. Output is byte-identical on CPython 3.10 through 3.13, which is what lets a
single committed file be compared on every leg of the CI matrix.

Scope is the parsed public API of the modules in :data:`FROZEN_MODULES`
(ADR-0007, widening ADR-0006 from ``core/`` to the surface
``docs/development/freeze-gate.md`` enumerates). Derivation is closed-world:
every public class, member, field, decorator argument and module ``__all__`` in
those modules is recorded, so a new symbol is gated the moment it appears rather
than when someone remembers to add it to a list.
"""

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "src" / "denselinkage"
SNAPSHOT_PATH = Path(__file__).resolve().parent / "api_snapshot.json"

SCHEMA_VERSION = 1

# The frozen surface, per docs/development/freeze-gate.md:91-108 as widened by
# ADR-0007. Ordered as that document groups them, so a reader can map the
# snapshot back to the claim it enforces.
FROZEN_MODULES = (
    # Ports
    "core/ports.py",
    # Models
    "core/models.py",
    # Results
    "core/results.py",
    # Errors
    "core/errors.py",
    # Orchestration
    "linkage/dense_linker.py",
    "linkage/linkage_index.py",
    # Metrics
    "metrics/linkage.py",
    "metrics/blocking.py",
    "metrics/clustering.py",
    "metrics/tuning.py",
    "metrics/adjusted.py",
)

# ``__init__`` is a calling convention third parties depend on; every other
# dunder is machinery, and a single leading underscore is private by ADR-0006.
_KEPT_DUNDERS = frozenset({"__init__"})

_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _is_public(name: str) -> bool:
    if name in _KEPT_DUNDERS:
        return True
    return not name.startswith("_")


def _render(node: ast.AST | None) -> str | None:
    return None if node is None else ast.unparse(node)


def _decorator(node: ast.expr) -> dict[str, Any]:
    """Structured decorator, so ``frozen=True`` is a comparable field.

    ``@dataclass(frozen=True, slots=True)`` and ``@dataclass`` differ in ways
    the contract depends on, and a rendered string would only let a message say
    "the decorator changed".
    """
    if isinstance(node, ast.Call):
        return {
            "name": ast.unparse(node.func),
            "args": [ast.unparse(a) for a in node.args],
            "kwargs": {
                kw.arg: ast.unparse(kw.value)
                for kw in node.keywords
                if kw.arg is not None
            },
        }
    return {"name": ast.unparse(node), "args": [], "kwargs": {}}


def _parameters(args: ast.arguments) -> list[dict[str, Any]]:
    """Every parameter with its kind, annotation and default.

    Kind is recorded because moving a parameter between positional and
    keyword-only is a breaking change that leaves the name and annotation
    untouched.
    """
    out: list[dict[str, Any]] = []

    positional = list(args.posonlyargs) + list(args.args)
    # Defaults right-align against the positional parameters.
    pad = len(positional) - len(args.defaults)
    for i, arg in enumerate(positional):
        default = args.defaults[i - pad] if i >= pad else None
        out.append(
            {
                "name": arg.arg,
                "kind": (
                    "positional_only" if arg in args.posonlyargs else "positional"
                ),
                "annotation": _render(arg.annotation),
                "has_default": default is not None,
                "default": _render(default),
            }
        )
    if args.vararg is not None:
        out.append(
            {
                "name": args.vararg.arg,
                "kind": "var_positional",
                "annotation": _render(args.vararg.annotation),
                "has_default": False,
                "default": None,
            }
        )
    # `kw_defaults` is padded with None to match `kwonlyargs`, so the two are
    # always the same length; `strict=True` asserts that rather than assuming it.
    for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        out.append(
            {
                "name": arg.arg,
                "kind": "keyword_only",
                "annotation": _render(arg.annotation),
                "has_default": kw_default is not None,
                "default": _render(kw_default),
            }
        )
    if args.kwarg is not None:
        out.append(
            {
                "name": args.kwarg.arg,
                "kind": "var_keyword",
                "annotation": _render(args.kwarg.annotation),
                "has_default": False,
                "default": None,
            }
        )
    return out


def _member_kind(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    names = {d["name"] for d in (_decorator(d) for d in node.decorator_list)}
    if "property" in names:
        return "property"
    if "classmethod" in names:
        return "classmethod"
    if "staticmethod" in names:
        return "staticmethod"
    return "method"


def _function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    return {
        "kind": _member_kind(node),
        "name": node.name,
        "decorators": [_decorator(d) for d in node.decorator_list],
        "parameters": _parameters(node.args),
        "returns": _render(node.returns),
        "signature": f"{node.name}{ast.unparse(node.args).join(('(', ')'))}",
        "is_async": isinstance(node, ast.AsyncFunctionDef),
    }


def _column_schema(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """The ``columns = [...]`` literal inside a ``to_frame`` body.

    ``freeze-gate.md:98-99`` freezes the ``to_frame`` column schema, which is
    written as a function-local list rather than a module constant, so it is
    unreachable by any class- or signature-level rule.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "columns" for t in sub.targets
        ):
            return ast.unparse(sub.value)
    return None


def _class(node: ast.ClassDef) -> dict[str, Any]:
    bases = [ast.unparse(b) for b in node.bases]
    members: list[dict[str, Any]] = []

    # Source order is preserved: dataclass field ordering is part of the
    # contract, because it is the positional constructor signature.
    for sub in node.body:
        if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            if not _is_public(sub.target.id):
                continue
            members.append(
                {
                    "kind": "field",
                    "name": sub.target.id,
                    "annotation": _render(sub.annotation),
                    "has_default": sub.value is not None,
                    "default": _render(sub.value),
                }
            )
        elif isinstance(sub, _FUNCTION_NODES) and _is_public(sub.name):
            entry = _function(sub)
            schema = _column_schema(sub)
            if schema is not None:
                entry["column_schema"] = schema
            members.append(entry)

    return {
        "name": node.name,
        "bases": bases,
        "decorators": [_decorator(d) for d in node.decorator_list],
        "is_protocol": any(b == "Protocol" or b.startswith("Protocol[") for b in bases),
        "members": members,
    }


def _iter_statements(tree: ast.Module) -> list[ast.stmt]:
    """Module body, descending one level into ``if`` blocks.

    ``Vectors`` is defined in both branches of ``if TYPE_CHECKING:``
    (``core/ports.py:24,26``) and is the annotation text on four port members, so
    a plain ``tree.body`` walk would never see it and a dtype change would leave
    every recorded signature byte-identical.
    """
    statements: list[ast.stmt] = []
    for node in tree.body:
        statements.append(node)
        if isinstance(node, ast.If):
            statements.extend(node.body)
            statements.extend(node.orelse)
    return statements


def _module(path: Path, relative: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    dunder_all: list[str] | None = None

    for node in _iter_statements(tree):
        if isinstance(node, ast.ClassDef):
            if _is_public(node.name):
                classes.append(_class(node))
        elif isinstance(node, _FUNCTION_NODES):
            if _is_public(node.name):
                functions.append(_function(node))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "__all__":
                    dunder_all = [
                        elt.value
                        for elt in ast.walk(node.value)
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
                elif _is_public(target.id):
                    aliases.append(
                        {"name": target.id, "value": ast.unparse(node.value)}
                    )

    return {
        "path": relative,
        "__all__": dunder_all,
        "aliases": aliases,
        "classes": classes,
        "functions": functions,
    }


def extract_surface(package_root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """The frozen public API, derived from source."""
    modules = [
        _module(package_root / relative, relative) for relative in FROZEN_MODULES
    ]
    core_init = _module(package_root / "core" / "__init__.py", "core/__init__.py")
    n_protocols = sum(1 for m in modules for c in m["classes"] if c["is_protocol"])
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_count": n_protocols,
        "core_all": core_init["__all__"],
        "modules": modules,
    }


def serialize(surface: dict[str, Any]) -> str:
    """Canonical on-disk form. Trailing newline so the file is POSIX-clean."""
    return json.dumps(surface, indent=2, ensure_ascii=False) + "\n"


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):  # pragma: no cover - malformed file
        raise TypeError(f"{path} does not contain a JSON object")
    return data


# --------------------------------------------------------------------------
# Diffing, and the severity classification the failure message reports.
# --------------------------------------------------------------------------

#: A whole new ``Protocol`` ships in a minor release; a member added to an
#: existing one does not. ADR-0003's add/remove asymmetry table is the authority
#: and the distinction is the entire reason this gate reports severity rather
#: than a bare "the snapshot changed".
ADDITIVE = "ADDITIVE"
BREAKING = "BREAKING"
RELAXING = "RELAXING"


@dataclass(frozen=True)
class Change:
    """One difference between the committed snapshot and the current source."""

    severity: str
    module: str
    symbol: str
    verb: str
    was: str | None = None
    now: str | None = None
    note: str | None = None


def _by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index by ``(name, nth occurrence)``, never by name alone.

    A name can legitimately appear more than once: ``Vectors`` is assigned in
    both branches of ``if TYPE_CHECKING:`` (``core/ports.py:24,26``), and an
    ``@overload`` chain produces several definitions of one member. Keying by
    name alone silently keeps the last and makes every earlier one unwatched,
    which hid a ``Vectors`` dtype change entirely.
    """
    counts: dict[str, int] = {}
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        name = item["name"]
        nth = counts.get(name, 0)
        counts[name] = nth + 1
        indexed[name if nth == 0 else f"{name}#{nth + 1}"] = item
    return indexed


def _member_display(member: dict[str, Any]) -> str:
    if member["kind"] == "field":
        default = f" = {member['default']}" if member["has_default"] else ""
        return f"{member['name']}: {member['annotation']}{default}"
    params = ", ".join(_parameter_display(p) for p in member["parameters"])
    returns = f" -> {member['returns']}" if member["returns"] else ""
    return f"{member['name']}({params}){returns}"


def _parameter_display(param: dict[str, Any]) -> str:
    text = str(param["name"])
    if param["kind"] == "var_positional":
        text = f"*{text}"
    elif param["kind"] == "var_keyword":
        text = f"**{text}"
    if param["annotation"]:
        text = f"{text}: {param['annotation']}"
    if param["has_default"]:
        text = f"{text} = {param['default']}"
    return text


def _diff_members(
    module: str, owner: str, expected: dict[str, Any], actual: dict[str, Any]
) -> list[Change]:
    changes: list[Change] = []
    is_protocol = expected.get("is_protocol", False)
    old = _by_name(expected["members"])
    new = _by_name(actual["members"])

    for name in old.keys() - new.keys():
        changes.append(
            Change(
                # Removing a port member is "relaxing" under ADR-0003, but it
                # still deletes something a third party may call.
                severity=RELAXING if is_protocol else BREAKING,
                module=module,
                symbol=f"{owner}.{name}",
                verb="removed",
                was=_member_display(old[name]),
            )
        )
    for name in new.keys() - old.keys():
        member = new[name]
        additive = (
            not is_protocol and member["kind"] == "field" and member["has_default"]
        )
        changes.append(
            Change(
                severity=ADDITIVE if additive else BREAKING,
                module=module,
                symbol=f"{owner}.{name}",
                verb="added",
                now=_member_display(member),
                note=(
                    "every third-party implementer must now provide this member"
                    if is_protocol
                    else None
                ),
            )
        )
    for name in old.keys() & new.keys():
        if old[name] != new[name]:
            changes.append(
                Change(
                    severity=BREAKING,
                    module=module,
                    symbol=f"{owner}.{name}",
                    verb="changed",
                    was=_member_display(old[name]),
                    now=_member_display(new[name]),
                )
            )

    old_order = [m["name"] for m in expected["members"] if m["kind"] == "field"]
    new_order = [m["name"] for m in actual["members"] if m["kind"] == "field"]
    if old_order != new_order and set(old_order) == set(new_order):
        changes.append(
            Change(
                severity=BREAKING,
                module=module,
                symbol=owner,
                verb="reordered fields",
                was=", ".join(old_order),
                now=", ".join(new_order),
                note="field order is the positional constructor signature",
            )
        )
    return changes


def _diff_module(expected: dict[str, Any], actual: dict[str, Any]) -> list[Change]:
    module = expected["path"]
    changes: list[Change] = []

    old_classes = _by_name(expected["classes"])
    new_classes = _by_name(actual["classes"])
    for name in old_classes.keys() - new_classes.keys():
        changes.append(Change(BREAKING, module, name, "removed", was=f"class {name}"))
    for name in new_classes.keys() - old_classes.keys():
        cls = new_classes[name]
        changes.append(
            Change(
                severity=ADDITIVE,
                module=module,
                symbol=name,
                verb="added",
                now=f"class {name}({', '.join(cls['bases'])})",
                note=(
                    "a NEW Protocol is additive and ships in a MINOR release"
                    if cls["is_protocol"]
                    else "a new type is additive and ships in a MINOR release"
                ),
            )
        )
    for name in old_classes.keys() & new_classes.keys():
        old_cls, new_cls = old_classes[name], new_classes[name]
        if old_cls["bases"] != new_cls["bases"]:
            changes.append(
                Change(
                    BREAKING,
                    module,
                    name,
                    "changed bases",
                    was=", ".join(old_cls["bases"]),
                    now=", ".join(new_cls["bases"]),
                )
            )
        if old_cls["decorators"] != new_cls["decorators"]:
            changes.append(
                Change(
                    severity=BREAKING,
                    module=module,
                    symbol=name,
                    verb="changed decorators",
                    was=_decorator_display(old_cls["decorators"]),
                    now=_decorator_display(new_cls["decorators"]),
                    note=(
                        "dataclass arguments are contract: frozen= is the purity "
                        "invariant and kw_only= is the calling convention"
                    ),
                )
            )
        changes.extend(_diff_members(module, name, old_cls, new_cls))

    old_funcs = _by_name(expected["functions"])
    new_funcs = _by_name(actual["functions"])
    for name in old_funcs.keys() - new_funcs.keys():
        changes.append(
            Change(
                BREAKING, module, name, "removed", was=_member_display(old_funcs[name])
            )
        )
    for name in new_funcs.keys() - old_funcs.keys():
        changes.append(
            Change(
                ADDITIVE,
                module,
                name,
                "added",
                now=_member_display(new_funcs[name]),
                note="a new public function is additive and ships in a MINOR release",
            )
        )
    for name in old_funcs.keys() & new_funcs.keys():
        if old_funcs[name] != new_funcs[name]:
            changes.append(
                Change(
                    BREAKING,
                    module,
                    name,
                    "changed",
                    was=_member_display(old_funcs[name]),
                    now=_member_display(new_funcs[name]),
                )
            )

    old_aliases = _by_name(expected["aliases"])
    new_aliases = _by_name(actual["aliases"])
    for name in old_aliases.keys() & new_aliases.keys():
        if old_aliases[name]["value"] != new_aliases[name]["value"]:
            changes.append(
                Change(
                    BREAKING,
                    module,
                    name,
                    "changed",
                    was=f"{name} = {old_aliases[name]['value']}",
                    now=f"{name} = {new_aliases[name]['value']}",
                )
            )
    for name in old_aliases.keys() - new_aliases.keys():
        changes.append(Change(BREAKING, module, name, "removed"))
    for name in new_aliases.keys() - old_aliases.keys():
        changes.append(Change(ADDITIVE, module, name, "added"))

    if expected["__all__"] != actual["__all__"]:
        changes.append(
            Change(
                BREAKING,
                module,
                "__all__",
                "changed",
                was=_all_display(expected["__all__"]),
                now=_all_display(actual["__all__"]),
            )
        )
    return changes


def _decorator_display(decorators: list[dict[str, Any]]) -> str:
    rendered = []
    for dec in decorators:
        parts = list(dec["args"]) + [f"{k}={v}" for k, v in dec["kwargs"].items()]
        rendered.append(
            f"@{dec['name']}({', '.join(parts)})" if parts else f"@{dec['name']}"
        )
    return " ".join(rendered) or "(none)"


def _all_display(names: list[str] | None) -> str:
    if names is None:
        return "(absent)"
    return f"{len(names)} names: {', '.join(names)}"


def diff_surface(expected: dict[str, Any], actual: dict[str, Any]) -> list[Change]:
    """Every difference, most severe first."""
    changes: list[Change] = []

    old_modules = {m["path"]: m for m in expected["modules"]}
    new_modules = {m["path"]: m for m in actual["modules"]}
    for path in sorted(old_modules.keys() - new_modules.keys()):
        changes.append(Change(BREAKING, path, path, "module removed from the surface"))
    for path in sorted(new_modules.keys() - old_modules.keys()):
        changes.append(Change(ADDITIVE, path, path, "module added to the surface"))
    for path in sorted(old_modules.keys() & new_modules.keys()):
        changes.extend(_diff_module(old_modules[path], new_modules[path]))

    if expected["core_all"] != actual["core_all"]:
        changes.append(
            Change(
                BREAKING,
                "core/__init__.py",
                "denselinkage.core.__all__",
                "changed",
                was=_all_display(expected["core_all"]),
                now=_all_display(actual["core_all"]),
            )
        )
    if expected["protocol_count"] != actual["protocol_count"]:
        changes.append(
            Change(
                severity=ADDITIVE
                if actual["protocol_count"] > expected["protocol_count"]
                else BREAKING,
                module="core/ports.py",
                symbol="Protocol count",
                verb="changed",
                was=str(expected["protocol_count"]),
                now=str(actual["protocol_count"]),
                note="the paper reports this number; it must move deliberately",
            )
        )

    order = {BREAKING: 0, RELAXING: 1, ADDITIVE: 2}
    return sorted(changes, key=lambda c: (order[c.severity], c.module, c.symbol))


# --------------------------------------------------------------------------
# The failure message. A reader who does not already know the architecture must
# learn the rule, its authority, and what to do next — otherwise the cheapest
# apparent repair is to weaken the gate, which is the wrong repair.
# --------------------------------------------------------------------------

_HEADER = "FROZEN CONTRACT VIOLATION"

_RULE = """\
The rule is EXTEND, NEVER MODIFY. See the add/remove asymmetry table in
docs/ADRs/0003-pre-freeze-contract-ratification.md, and the freeze scope in
docs/ADRs/0007-freeze-scope-enumerated-surface.md.
  - a NEW Protocol, type or function is additive and ships in a MINOR release
  - ANY change to an existing Protocol is BREAKING FOR IMPLEMENTERS: mypy cannot
    see third-party implementers, so this repository stays green while their
    code breaks
  - a new field WITH a default on a frozen dataclass is additive; without one,
    or reordered, it breaks every positional construction"""

_ESCALATION = """\
What to do next:
  - if this change is deliberate and additive, regenerate the snapshot (below)
    and say so in the PR; the regeneration commit is the record
  - if it is breaking, it needs a MAJOR version. See
    docs/development/releasing.md for the version table and the pre-release step
  - if it was accidental, revert the signature rather than the snapshot

DO NOT edit tests/api_snapshot.json by hand to make this pass. The snapshot is
the frozen contract, not a cache of the current source. Regenerate it with:

    python -m tests._api_snapshot --regenerate --authority <ADR-#### or #issue>"""


def format_violation(changes: list[Change]) -> str:
    """Render a diff as the gate's failure message."""
    lines = [
        f"{_HEADER} - {len(changes)} change(s) against tests/api_snapshot.json",
        "",
    ]
    for severity in (BREAKING, RELAXING, ADDITIVE):
        group = [c for c in changes if c.severity == severity]
        if not group:
            continue
        lines.append(f"{severity} ({len(group)}):")
        for change in group:
            lines.append(f"  * {change.module} :: {change.symbol}: {change.verb}")
            if change.was is not None:
                lines.append(f"      was: {change.was}")
            if change.now is not None:
                lines.append(f"      now: {change.now}")
            if change.note is not None:
                lines.append(f"    {change.note}")
        lines.append("")
    lines.append(_RULE)
    lines.append("")
    lines.append(_ESCALATION)
    return "\n".join(lines)


def _regenerate(authority: str) -> int:
    surface = extract_surface()
    surface["authority"] = authority
    previous = load_snapshot() if SNAPSHOT_PATH.exists() else None
    if previous is not None:
        stored = dict(previous)
        stored.pop("authority", None)
        current = dict(surface)
        current.pop("authority", None)
        changes = diff_surface(stored, current)
        if not changes:
            print("No change to the frozen surface; snapshot left alone.")
            return 0
        print(format_violation(changes))
        print()
    SNAPSHOT_PATH.write_text(serialize(surface), encoding="utf-8")
    print(f"Snapshot regenerated under authority {authority}: {SNAPSHOT_PATH}")
    print("Commit it with the change, and cite that authority in the PR.")
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - entry point
    parser = argparse.ArgumentParser(
        prog="python -m tests._api_snapshot",
        description=(
            "Regenerate the frozen-API snapshot. Regenerating is the signal that "
            "the public contract moved, so it requires an explicit flag and a "
            "cited authority; it is not a routine `--update`."
        ),
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rewrite tests/api_snapshot.json from the current source",
    )
    parser.add_argument(
        "--authority",
        metavar="REF",
        help=(
            "the ADR or issue authorising the contract change, e.g. ADR-0007 or "
            "#42. Recorded in the snapshot so the change is traceable."
        ),
    )
    args = parser.parse_args(argv)

    if not args.regenerate:
        parser.error(
            "refusing to act without --regenerate. This tool rewrites the frozen "
            "contract; run the test suite to check the surface instead."
        )
    if not args.authority:
        parser.error(
            "refusing to regenerate without --authority. A contract change needs "
            "a decision behind it: pass the ADR (ADR-0007) or issue (#42) that "
            "authorises it. If neither exists, the change is not ready."
        )
    return _regenerate(args.authority)


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    sys.exit(main())
