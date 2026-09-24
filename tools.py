# tools.py
"""Agent tools. Every tool works inside the SWE-bench container (repo at /testbed)."""
import ast
import difflib
import io
import posixpath
import subprocess
import tarfile
import warnings

from langchain_core.tools import tool

import container as ct

# agent_graph.py uses this prefix to detect successful edits.
EDIT_OK_PREFIX = "Successfully updated"

_MAP_IGNORE = {".git", "__pycache__", "tests", "test", "docs", "build", "dist"}
_TREE_IGNORE = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".tox"}
_MAX_MAP_FILES = 10     # skeletons shown by repo_map
_MAX_MAP_SCAN = 200     # files fetched from the container per repo_map call


def _truncate_output(text: str, max_lines: int = 100) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    half = max_lines // 2
    return "\n".join(lines[:half] + [f"\n... [TRUNCATED {len(lines) - max_lines} LINES] ...\n"] + lines[-half:])


def _split_lines(text: str) -> list[str]:
    """Split on \\n only, so line numbers agree between view_file and edit_file.

    (str.splitlines() also splits on form feeds and unicode separators, which
    would shift the numbers in files that contain them.)
    """
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _relative_to(rel: str, path: str) -> str:
    """`path` is ROOT-relative; return it relative to directory `rel` (which may be '.')."""
    return path if rel == "." else path[len(rel) + 1:]


class SkeletonTransformer(ast.NodeTransformer):
    """Walks the AST and strips out all function/method implementations."""

    def visit_FunctionDef(self, node):
        node.body = [ast.Pass()]
        return node

    def visit_AsyncFunctionDef(self, node):
        node.body = [ast.Pass()]
        return node

    def visit_ClassDef(self, node):
        new_body = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                new_body.append(self.visit(child))
            elif isinstance(child, ast.Assign):
                new_body.append(child)
        if not new_body:
            new_body = [ast.Pass()]
        node.body = new_body
        return node

    def visit_Module(self, node):
        new_body = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                new_body.append(self.visit(child))
        node.body = new_body
        return node


def generate_file_skeleton(source: str) -> str:
    """Parses Python source and returns its skeleton."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
        transformed_tree = SkeletonTransformer().visit(tree)
        return ast.unparse(transformed_tree)
    except SyntaxError:
        return "# [Syntax Error: Could not parse file]"
    except Exception as e:
        return f"# [Error parsing file: {e}]"


def _fetch_sources(paths: list[str]) -> dict[str, str]:
    """Read many files from the container in ONE docker exec (tar stream)."""
    res = ct.exec_raw(
        ["tar", "--null", "-T", "-", "-cf", "-"],
        stdin=("\0".join(paths) + "\0").encode(),
        timeout=60,
    )
    out: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(res.stdout)) as tf:
            for member in tf.getmembers():
                f = tf.extractfile(member)
                if f:
                    out[member.name] = ct.text(f.read())
    except tarfile.TarError:
        pass
    return out


@tool
def repo_map(target_directory: str = ".") -> str:
    """
    Generates a structural map of Python files in a directory.
    Returns the skeleton of all classes, methods, and functions without their implementations.
    Pass a specific subpackage, never the repository root. Use search_code first to find the right file.

    Args:
        target_directory: Directory to map, relative to the repository root.

    Return:
        Skeletons of up to 10 Python files.
    """
    target = ct.resolve(target_directory)
    if target is None or not ct.exists(target, "d"):
        return f"Error: Directory '{target_directory}' does not exist."
    rel = posixpath.relpath(target, ct.ROOT)

    candidates = []
    for p in sorted(ct.tracked_files(rel)):
        if not p.endswith(".py"):
            continue
        dir_parts = _relative_to(rel, p).split("/")[:-1]
        if any(d in _MAP_IGNORE or d.startswith(".") for d in dir_parts):
            continue
        candidates.append(p)

    if not candidates:
        return f"No Python files found in '{target_directory}' (excluding tests/ hidden dirs)."

    scanned = candidates[:_MAX_MAP_SCAN]
    sources = _fetch_sources(scanned)
    mapped_files = []
    for p in scanned:
        skeleton = generate_file_skeleton(sources.get(p, ""))
        if skeleton.strip():
            mapped_files.append(f"### File: {p}\n```python\n{skeleton}\n```")

    if len(mapped_files) > _MAX_MAP_FILES:
        output = "\n\n".join(mapped_files[:_MAX_MAP_FILES])
        output += f"\n\n... [TRUNCATED: {len(mapped_files) - _MAX_MAP_FILES} more files omitted. Narrow your search.]"
        return output
    result = "\n\n".join(mapped_files)
    if len(candidates) > _MAX_MAP_SCAN:
        result += f"\n\n... [Only the first {_MAX_MAP_SCAN} of {len(candidates)} files were scanned. Narrow your search.]"
    return result


@tool
def search_code(query: str, directory: str = ".") -> str:
    """Search for a symbol, function, or string across files using git grep."""
    target = ct.resolve(directory)
    if target is None:
        return f"Error: Directory '{directory}' is outside the repository."
    rel = posixpath.relpath(target, ct.ROOT)
    try:
        res = ct.exec_raw(["git", "grep", "-n", "-I", "-e", query, "--", rel], timeout=30)
    except Exception as e:
        return f"Error executing search: {str(e)}"
    out = ct.text(res.stdout).strip()
    if not out:
        if res.returncode > 1:  # 1 just means "no matches"
            return f"Error executing search: {ct.text(res.stderr).strip()}"
        return f"No matches found for '{query}'."
    return _truncate_output(out, max_lines=60)


@tool
def explore_directory(path: str = ".") -> str:
    """Lists files and folders up to 2 levels deep to inspect repository layout."""
    target = ct.resolve(path)
    if target is None or not ct.exists(target, "d"):
        return f"Error: Directory '{path}' does not exist."
    rel = posixpath.relpath(target, ct.ROOT)

    entries = []  # each entry is the path split into parts, relative to `target`
    for p in ct.tracked_files(rel):
        parts = _relative_to(rel, p).split("/")
        if any(d in _TREE_IGNORE or d.startswith(".") for d in parts[:-1]) or parts[-1].startswith("."):
            continue
        entries.append(parts)

    lines = [f"{'.' if rel == '.' else posixpath.basename(target)}/"]

    def emit(names: list[str], indent: str) -> None:
        for name in sorted(names)[:15]:
            lines.append(f"{indent}  {name}")
        if len(names) > 15:
            lines.append(f"{indent}  ... [{len(names) - 15} more files]")

    emit([e[0] for e in entries if len(e) == 1], "")
    for d in sorted({e[0] for e in entries if len(e) > 1}):
        lines.append(f"  {d}/")
        emit([e[1] for e in entries if len(e) == 2 and e[0] == d], "  ")

    return _truncate_output("\n".join(lines), max_lines=80)


@tool
def view_file(file_path: str, start_line: int = 1, end_line: int = 150) -> str:
    """View lines from a file with line numbers. View at least 60 lines at a time."""
    target = ct.resolve(file_path)
    raw = ct.read_bytes(target) if target else None
    if raw is None:
        return f"Error: File '{file_path}' does not exist inside the repository."

    lines = _split_lines(ct.text(raw))
    start = max(1, start_line)
    end = min(len(lines), end_line)
    if start > len(lines):
        return f"File has only {len(lines)} lines."
    if end < start:
        return f"Error: end_line ({end_line}) must be >= start_line ({start_line}). File has {len(lines)} lines."
    return "".join(f"{i + start:4d} | {line}\n" for i, line in enumerate(lines[start - 1:end]))


def _numbered(lines: list[str], lo: int, hi: int) -> str:
    """Lines lo..hi-1 (0-based) with 1-based line numbers."""
    return "\n".join(f"{n + 1:4d} | {lines[n]}" for n in range(lo, hi))


def _not_found_message(content: str, old_str: str) -> str:
    msg = (
        "Error: old_str was not found. It must match the file exactly, including "
        "indentation and blank lines. Copy it from a view_file output without the line-number prefix."
    )
    lines = _split_lines(content)
    first = next((ln.strip() for ln in old_str.splitlines() if ln.strip()), "")
    stripped = [ln.strip() for ln in lines]
    match = difflib.get_close_matches(first, stripped, n=1, cutoff=0.6)
    if match:
        i = stripped.index(match[0])
        lo, hi = max(0, i - 1), min(len(lines), i + 11)
        msg += f"\nClosest region in the file right now:\n{_numbered(lines, lo, hi)}"
    return msg


def _syntax_error(source: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ast.parse(source)
            return None
        except SyntaxError as e:
            return e


@tool
def edit_file(file_path: str, old_str: str, new_str: str) -> str:
    """Replace exactly one occurrence of old_str with new_str in a file.

    old_str must match the file text exactly (indentation included) and appear exactly once,
    so include a few surrounding lines if needed. There are no line numbers to go stale.
    Python syntax is validated and invalid edits are not applied. Returns the updated region.
    """
    if not old_str:
        return "Error: old_str is empty."
    if old_str == new_str:
        return "Error: old_str and new_str are identical, so nothing would change. Make a real change."

    target = ct.resolve(file_path)
    raw = ct.read_bytes(target) if target else None
    if raw is None:
        return f"Error: File '{file_path}' does not exist."
    try:
        content = raw.decode("utf-8")  # bytes -> str keeps line endings untouched
    except UnicodeDecodeError:
        return f"Error: '{file_path}' is not valid UTF-8, refusing to edit it."

    # Keep the file's own line endings.
    if "\r\n" in content:
        old_str = old_str.replace("\r\n", "\n").replace("\n", "\r\n")
        new_str = new_str.replace("\r\n", "\n").replace("\n", "\r\n")

    count = content.count(old_str)
    if count == 0:
        return _not_found_message(content, old_str)
    if count > 1:
        starts, pos = [], 0
        while len(starts) < 5:
            pos = content.find(old_str, pos)
            if pos == -1:
                break
            starts.append(content[:pos].count("\n") + 1)
            pos += 1
        return (
            f"Error: old_str matches {count} places (starting at lines {starts}). "
            "Add more surrounding lines so it is unique."
        )

    new_content = content.replace(old_str, new_str, 1)

    # Only enforce syntax if the file parsed before the edit (the host's Python
    # may not accept every construct in an old repository).
    if target.endswith(".py") and _syntax_error(content) is None:
        err = _syntax_error(new_content)
        if err is not None:
            return (
                f"Edit NOT applied, SyntaxError: {err.msg} at line {err.lineno} of the resulting file. "
                "Check indentation of new_str and retry."
            )

    ct.write_bytes(target, new_content.encode("utf-8"))

    start_line = content[: content.find(old_str)].count("\n") + 1
    new_lines = _split_lines(new_content)
    span = new_str.count("\n") + 1
    lo, hi = max(0, start_line - 4), min(len(new_lines), start_line - 1 + span + 3)
    return f"{EDIT_OK_PREFIX} {file_path}. Updated region:\n{_numbered(new_lines, lo, hi)}"


def _format_output(rc: int, out: str, err: str, timeout: int) -> str:
    output = ""
    if out:
        output += "--- STDOUT ---\n" + _truncate_output(out, max_lines=80) + "\n"
    if err:
        output += "--- STDERR ---\n" + _truncate_output(err, max_lines=80) + "\n"
    if rc == 124:
        output += f"[Timed out after {timeout}s and was killed]\n"
    elif rc != 0:
        output += f"[exit code {rc}]\n"
    return output if output else f"Command completed with exit code {rc} (No output)."


@tool
def run_bash(command: str) -> str:
    """Execute bash commands (pytest, git, python) inside the container, in the repo root with the project's conda env active."""
    rc, out, err = ct.run_script(command, timeout=60)
    return _format_output(rc, out, err, 60)


@tool
def run_python_repro(code: str) -> str:
    """Execute a standalone Python reproduction snippet inside the container. No reasoning in comments."""
    path = "/tmp/__repro_tmp.py"  # outside the repo, so it can never appear in the patch
    script = (
        f"cat > {path}\n"
        f"PYTHONPATH={ct.ROOT} python -W ignore {path}\n"
        f"rc=$?\n"
        f"rm -f {path}\n"
        f"exit $rc"
    )
    rc, out, err = ct.run_script(script, stdin=code.encode("utf-8"), timeout=45)
    output = (out + "\n" + err).strip()
    if rc == 124:
        output += "\n[Timed out after 45s and was killed]"
    return output if output else f"[Process exited with code {rc} and no output]"


ALL_TOOLS = [search_code, explore_directory, repo_map, view_file, edit_file, run_bash, run_python_repro]