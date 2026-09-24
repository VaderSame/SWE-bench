import os
import ast
import difflib
import subprocess
from pathlib import Path
from langchain_core.tools import tool

WORKSPACE: Path = Path(".")

# agent_graph.py uses this prefix to detect successful edits.
EDIT_OK_PREFIX = "Successfully updated"


def set_workspace(path: Path):
    global WORKSPACE
    WORKSPACE = path.resolve()


def _truncate_output(text: str, max_lines: int = 100) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    half = max_lines // 2
    return "\n".join(lines[:half] + [f"\n... [TRUNCATED {len(lines) - max_lines} LINES] ...\n"] + lines[-half:])


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


def generate_file_skeleton(filepath: Path) -> str:
    """Parses a Python file and returns its skeleton."""
    try:
        content = filepath.read_text(encoding="utf-8")
        tree = ast.parse(content)
        transformer = SkeletonTransformer()
        transformed_tree = transformer.visit(tree)
        return ast.unparse(transformed_tree)
    except SyntaxError:
        return "# [Syntax Error: Could not parse file]"
    except Exception as e:
        return f"# [Error parsing file: {e}]"



@tool
def repo_map(target_directory: str = ".") -> str:
    """
    Generates a structural map of Python files in a directory.
    Returns the skeleton of all classes, methods, and functions without their implementations.
    Pass a specific subpackage, never the repository root. Use search_code first to find the right file.
    
    Args:
    
    Return:
    """
    target = (WORKSPACE / target_directory).resolve()
    if not target.is_relative_to(WORKSPACE) or not target.exists():
        return f"Error: Directory '{target_directory}' does not exist."

    ignore_dirs = {".git", "__pycache__", "tests", "test", "docs", "build", "dist"}
    mapped_files = []

    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
        for file in files:
            if file.endswith(".py"):
                filepath = Path(root) / file
                relative_path = filepath.relative_to(WORKSPACE)
                skeleton = generate_file_skeleton(filepath)
                if skeleton.strip():
                    mapped_files.append(f"### File: {relative_path}\n```python\n{skeleton}\n```")

    if not mapped_files:
        return f"No Python files found in '{target_directory}' (excluding tests/ hidden dirs)."

    MAX_FILES = 10
    if len(mapped_files) > MAX_FILES:
        output = "\n\n".join(mapped_files[:MAX_FILES])
        output += f"\n\n... [TRUNCATED: {len(mapped_files) - MAX_FILES} more files omitted. Narrow your search.]"
        return output

    return "\n\n".join(mapped_files)


@tool
def search_code(query: str, directory: str = ".") -> str:
    """Search for a symbol, function, or string across files using git grep."""
    try:
        res = subprocess.run(
            ["git", "grep", "-n", "-I", query, "--", directory],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if not res.stdout.strip():
            return f"No matches found for '{query}'."
        return _truncate_output(res.stdout.strip(), max_lines=60)
    except Exception as e:
        return f"Error executing search: {str(e)}"


@tool
def explore_directory(path: str = ".") -> str:
    """Lists files and folders up to 2 levels deep to inspect repository layout."""
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE) or not target.exists():
        return f"Error: Directory '{path}' does not exist."

    ignore_dirs = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".tox"}
    tree_lines = []
    base_depth = len(target.parts)

    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
        depth = len(Path(root).parts) - base_depth
        if depth >= 2:
            dirs.clear()
            continue

        indent = "  " * depth
        folder_name = Path(root).name or "."
        tree_lines.append(f"{indent}{folder_name}/")

        for f in sorted(files)[:15]:
            if not f.startswith("."):
                tree_lines.append(f"{indent}  {f}")
        if len(files) > 15:
            tree_lines.append(f"{indent}  ... [{len(files) - 15} more files]")

    return _truncate_output("\n".join(tree_lines), max_lines=80)


@tool
def view_file(file_path: str, start_line: int = 1, end_line: int = 150) -> str:
    """View lines from a file with line numbers. View at least 60 lines at a time."""
    target = (WORKSPACE / file_path).resolve()
    if not target.is_relative_to(WORKSPACE) or not target.exists():
        return f"Error: File '{file_path}' does not exist inside workspace."

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        start = max(1, start_line)
        end = min(len(lines), end_line)
        if end < start:
            return f"Error: end_line ({end_line}) must be >= start_line ({start_line}). File has {len(lines)} lines."
        if start > len(lines):
            return f"File has only {len(lines)} lines."
        selected = lines[start - 1 : end]
        output = [f"{i + start:4d} | {line}" for i, line in enumerate(selected)]
        return "".join(output)
    except Exception as e:
        return f"Error reading file: {str(e)}"


def _numbered(lines: list[str], lo: int, hi: int) -> str:
    """Lines lo..hi-1 (0-based) with 1-based line numbers."""
    return "\n".join(f"{n + 1:4d} | {lines[n]}" for n in range(lo, hi))


def _not_found_message(content: str, old_str: str) -> str:
    msg = (
        "Error: old_str was not found. It must match the file exactly, including "
        "indentation and blank lines. Copy it from a view_file output without the line-number prefix."
    )
    lines = content.splitlines()
    first = next((ln.strip() for ln in old_str.splitlines() if ln.strip()), "")
    stripped = [ln.strip() for ln in lines]
    match = difflib.get_close_matches(first, stripped, n=1, cutoff=0.6)
    if match:
        i = stripped.index(match[0])
        lo, hi = max(0, i - 1), min(len(lines), i + 11)
        msg += f"\nClosest region in the file right now:\n{_numbered(lines, lo, hi)}"
    return msg


@tool
def edit_file(file_path: str, old_str: str, new_str: str) -> str:
    """Replace exactly one occurrence of old_str with new_str in a file.

    old_str must match the file text exactly (indentation included) and appear exactly once,
    so include a few surrounding lines if needed. There are no line numbers to go stale.
    Python syntax is validated and invalid edits are not applied. Returns the updated region.
    """
    target = (WORKSPACE / file_path).resolve()
    if not target.is_relative_to(WORKSPACE) or not target.exists():
        return f"Error: File '{file_path}' does not exist."
    if not old_str:
        return "Error: old_str is empty."
    if old_str == new_str:
        return "Error: old_str and new_str are identical, so nothing would change. Make a real change."

    with open(target, "r", encoding="utf-8", newline="") as f:
        content = f.read()

    # Keep the file's own line endings (matters on Windows checkouts).
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

    if target.suffix == ".py":
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            return (
                f"Edit NOT applied, SyntaxError: {e.msg} at line {e.lineno} of the resulting file. "
                "Check indentation of new_str and retry."
            )

    with open(target, "w", encoding="utf-8", newline="") as f:
        f.write(new_content)

    start_line = content[: content.find(old_str)].count("\n") + 1
    new_lines = new_content.splitlines()
    span = new_str.count("\n") + 1
    lo, hi = max(0, start_line - 4), min(len(new_lines), start_line - 1 + span + 3)
    return f"{EDIT_OK_PREFIX} {file_path}. Updated region:\n{_numbered(new_lines, lo, hi)}"


@tool
def run_bash(command: str) -> str:
    """Execute bash commands (pytest, git, python) inside the container."""
    container = os.getenv("DOCKER_CONTAINER")
    if container:
        cmd = ["docker", "exec", "-e", "PYTHONPATH=/workspace", "-w", "/workspace", container, "bash", "-c", command]
        shell = False
    else:
        cmd = command
        shell = True

    try:
        res = subprocess.run(cmd, cwd=WORKSPACE if not container else None, shell=shell, capture_output=True, text=True, timeout=60)
        output = ""
        if res.stdout:
            output += "--- STDOUT ---\n" + _truncate_output(res.stdout, max_lines=80) + "\n"
        if res.stderr:
            output += "--- STDERR ---\n" + _truncate_output(res.stderr, max_lines=80) + "\n"
        return output if output else f"Command completed with exit code {res.returncode} (No output)."
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool
def run_python_repro(code: str) -> str:
    """Execute a standalone Python reproduction snippet inside the container. No reasoning in comments."""
    repro_file = WORKSPACE / "__repro_tmp.py"
    container = os.getenv("DOCKER_CONTAINER")
    try:
        repro_file.write_text(code, encoding="utf-8")
        if container:
            cmd = ["docker", "exec", "-e", "PYTHONPATH=/workspace", "-w", "/workspace", container, "python", "-W", "ignore", "__repro_tmp.py"]
            shell = False
        else:
            cmd = ["python", "-W", "ignore", "__repro_tmp.py"]
            shell = True

        res = subprocess.run(cmd, cwd=WORKSPACE if not container else None, shell=shell, capture_output=True, text=True, timeout=45)
        output = (res.stdout + "\n" + res.stderr).strip()
        return output if output else "[Process exited with code 0 and no output]"
    except Exception as e:
        return f"Error executing repro: {str(e)}"
    finally:
        if repro_file.exists():
            repro_file.unlink()


ALL_TOOLS = [search_code, explore_directory, repo_map, view_file, edit_file, run_bash, run_python_repro]