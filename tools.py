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
def view_file(file_path: str, start_line: int = 1, end_line: int = 0) -> str:
    """View lines from a file with line numbers. View at least 60 lines at a time.
    
    If end_line is omitted or 0, defaults to start_line + 149 (150 lines window).
    """
    target = ct.resolve(file_path)
    raw = ct.read_bytes(target) if target else None
    if raw is None:
        return f"Error: File '{file_path}' does not exist inside the repository."

    lines = _split_lines(ct.text(raw))
    start = max(1, start_line)
    # If end_line not given (0) or less than start, compute a 150-line window.
    if end_line < start:
        end_line = start + 149
    end = min(len(lines), end_line)
    if start > len(lines):
        return f"File has only {len(lines)} lines. Requested start_line ({start}) is beyond the end of the file."
    output = "".join(f"{i + start:4d} | {line}\n" for i, line in enumerate(lines[start - 1:end]))
    if end >= len(lines):
        output += f"[END OF FILE: {len(lines)} lines total]\n"
    return output


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
def replace_lines(file_path: str, start_line: int, end_line: int, new_str: str) -> str:
    """Replace lines start_line through end_line (inclusive) with new_str.

    start_line and end_line are 1-indexed. To insert lines without replacing any,
    set end_line = start_line - 1. Returns the updated region with surrounding context.
    IMPORTANT: new_str replaces ONLY the specified lines. Do not include lines outside
    that range (e.g. the surrounding if/else/def lines) in new_str.
    """
    if start_line < 1:
        return "Error: start_line must be >= 1."
    if end_line < start_line - 1:
        return f"Error: end_line ({end_line}) cannot be less than start_line - 1 ({start_line - 1})."

    target = ct.resolve(file_path)
    raw = ct.read_bytes(target) if target else None
    if raw is None:
        return f"Error: File '{file_path}' does not exist."
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"Error: '{file_path}' is not valid UTF-8, refusing to edit it."

    lines = _split_lines(content)
    if start_line > len(lines) + 1:
        return f"Error: start_line ({start_line}) is beyond the end of the file (which has {len(lines)} lines)."

    start_idx = start_line - 1
    end_idx = min(end_line, len(lines))

    def _context_block(label: str) -> str:
        lo = max(0, start_idx - 3)
        hi = min(len(lines), end_idx + 3)
        return f"{label}\n{_numbered(lines, lo, hi)}"

    new_lines = _split_lines(new_str) if new_str else []

    if lines[start_idx:end_idx] == new_lines:
        return (
            "Error: The new code is exactly identical to the lines being replaced. Nothing would change.\n"
            + _context_block("Current file content around target lines:")
        )

    resulting_lines = lines[:start_idx] + new_lines + lines[end_idx:]
    new_content = "\n".join(resulting_lines)
    if content.endswith("\n") and not new_content.endswith("\n"):
        new_content += "\n"

    if target.endswith(".py") and _syntax_error(content) is None:
        err = _syntax_error(new_content)
        if err is not None:
            return (
                f"Edit NOT applied, SyntaxError: {err.msg} at line {err.lineno} of the resulting file.\n"
                "Remember: new_str replaces ONLY lines start_line..end_line. Do NOT include surrounding "
                "lines (e.g. the 'else:' or 'def' that comes before start_line) in new_str.\n"
                + _context_block("Current file content around target lines (for reference):")
            )

    ct.write_bytes(target, new_content.encode("utf-8"))

    span = len(new_lines)
    lo, hi = max(0, start_idx - 4), min(len(resulting_lines), start_idx + span + 3)
    return f"{EDIT_OK_PREFIX} {file_path}. Updated region:\n{_numbered(resulting_lines, lo, hi)}"


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


@tool
def run_pytest(test_target: str = "") -> str:
    """Run pytest inside the container using the project's active conda environment.
    
    Args:
        test_target: Path to a test file or specific test function, e.g.:
                     'astropy/io/ascii/tests/test_rst.py'
                     or 'astropy/io/ascii/tests/test_rst.py::test_write_normal'.
                     If omitted or empty, runs pytest on default discovery.
    """
    target = test_target.strip()
    cmd = f"pytest -v {target}" if target else "pytest"
    rc, out, err = ct.run_script(cmd, timeout=90)
    return _format_output(rc, out, err, 90)


ALL_TOOLS = [search_code, explore_directory, repo_map, view_file, replace_lines, run_bash, run_python_repro, run_pytest]