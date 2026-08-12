"""Shared payload parsing for the hooks in this directory.

Both hooks answer the same question first — which file is the tool about to
write — and both must answer it the same way, or the guard and the formatter
disagree about what counts as inside the repository.

The repository root is derived from ``__file__``, never from the process cwd. A
hook command is a relative path in ``.claude/settings.json`` and the cwd is
whatever directory the session was started in, so cwd is not a reliable anchor;
``__file__`` is.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def target_path(payload: object) -> Path | None:
    """The absolute path the tool is about to write, or ``None``.

    ``None`` for any payload this cannot read with certainty: a non-object, a
    missing ``tool_input``, a tool that names no file. Every caller treats
    ``None`` as "not my business" and defers to the normal permission flow.

    ``notebook_path`` is read as well as ``file_path`` because the tool-name
    matcher is a regex: ``NotebookEdit`` matches it, and its payload names the
    file under the other key.
    """
    if not isinstance(payload, dict):
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    raw = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    # A relative path is resolved against the repository root, not the cwd: the
    # harness passes absolute paths, and the root is the only anchor a hook can
    # trust for the rest.
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def read_stdin_target() -> Path | None:
    """``target_path`` of the JSON payload on stdin; ``None`` on any error."""
    try:
        return target_path(json.load(sys.stdin))
    except Exception:
        return None
