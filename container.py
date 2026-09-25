# container.py
"""Everything that touches the SWE-bench container lives here.

The official SWE-bench image ships the repository at /testbed (already at the
right commit and built) plus a conda env called `testbed`. Nothing is
bind-mounted: every read, edit, search and command goes through `docker exec`,
so the container is the single source of truth.
"""
import os
import posixpath
import subprocess

import structlog

log = structlog.get_logger("container")

ROOT = "/testbed"
BASELINE_TAG = "swe-baseline"

# Same activation the official eval scripts use. Fails loudly instead of
# silently falling back to the base python.
CONDA_ACTIVATE = (
    "source /opt/miniconda3/bin/activate && conda activate testbed "
    "|| { echo 'conda activation failed' >&2; exit 97; }"
)


def text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def container_name() -> str:
    name = os.getenv("DOCKER_CONTAINER")
    if not name:
        raise RuntimeError("DOCKER_CONTAINER is not set. Start the container first (run.ps1 does this).")
    return name


def exec_raw(args: list[str], stdin: bytes | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    """`docker exec` with an argv list (no shell, no quoting problems). Output stays bytes.

    stdin is always a pipe (empty by default) so docker can never wait on the
    parent's terminal.
    """
    cmd = ["docker", "exec", "-i", "-w", ROOT, container_name(), *args]
    return subprocess.run(
        cmd,
        input=stdin if stdin is not None else b"",
        capture_output=True,
        timeout=timeout,
    )


def run_script(script: str, stdin: bytes | None = None, timeout: int = 60) -> tuple[int, str, str]:
    """Run a bash script in /testbed with the `testbed` conda env active.

    The in-container `timeout` kills the whole process group, so a hung pytest
    does not keep running after we give up. Returns (returncode, stdout, stderr);
    returncode 124 means the timeout fired.
    """
    full = f"{CONDA_ACTIVATE}\ncd {ROOT}\n{script}"
    try:
        res = exec_raw(["timeout", "-k", "5", str(timeout), "bash", "-c", full], stdin=stdin, timeout=timeout + 20)
    except subprocess.TimeoutExpired:
        return 124, "", f"docker exec did not return within {timeout + 20}s"
    return res.returncode, text(res.stdout), text(res.stderr)


# ---------------------------------------------------------------------------
# Paths and files
# ---------------------------------------------------------------------------
def resolve(path: str) -> str | None:
    """Map an agent-supplied path to an absolute path inside ROOT (None if it escapes ROOT)."""
    p = (path or ".").replace("\\", "/")
    full = posixpath.normpath(posixpath.join(ROOT, p))
    return full if full == ROOT or full.startswith(ROOT + "/") else None


def exists(path: str, kind: str = "e") -> bool:
    """kind: 'e' anything, 'f' regular file, 'd' directory."""
    return exec_raw(["test", f"-{kind}", path]).returncode == 0


def read_bytes(path: str) -> bytes | None:
    res = exec_raw(["cat", "--", path])
    return res.stdout if res.returncode == 0 else None


def write_bytes(path: str, data: bytes) -> None:
    # Plain `cat >` (no temp file + mv) so the file keeps its mode and owner.
    res = exec_raw(["bash", "-c", 'cat > "$1"', "_", path], stdin=data)
    if res.returncode != 0:
        raise RuntimeError(f"write to {path} failed: {text(res.stderr).strip()}")


def tracked_files(rel: str = ".") -> list[str]:
    """Git-tracked files under `rel`, as paths relative to ROOT."""
    res = exec_raw(["git", "ls-files", "-z", "--", rel], timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {text(res.stderr).strip()}")
    return [p for p in text(res.stdout).split("\0") if p]


# ---------------------------------------------------------------------------
# Git baseline and patch extraction
# ---------------------------------------------------------------------------
def git_diff() -> str:
    """The agent's patch: tracked-file changes relative to the baseline tag."""
    res = exec_raw(["git", "-c", "core.fileMode=false", "diff", BASELINE_TAG], timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"git diff failed: {text(res.stderr).strip()}")
    out = text(res.stdout)
    return out if out.strip() else ""


def prepare(base_commit: str) -> None:
    """Get the container into a known state before the agent starts.

    Deliberately no `git reset --hard <base_commit>` and no `git clean`: the image
    may carry install-time tweaks and holds built (ignored) extension files that
    must survive. Instead we pin a baseline tag, and the patch is the diff against
    it, so only the agent's own edits ever appear in the patch.
    """

    def git(*args: str, check: bool = True) -> str:
        res = exec_raw(["git", *args], timeout=120)
        if check and res.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {text(res.stderr).strip()}")
        return text(res.stdout).strip()

    git("config", "--global", "--replace-all", "safe.directory", ROOT)

    has_baseline = exec_raw(["git", "rev-parse", "-q", "--verify", f"refs/tags/{BASELINE_TAG}"]).returncode == 0
    if has_baseline:
        git("reset", "--hard", BASELINE_TAG)
        log.info("baseline_reset", tag=BASELINE_TAG)
    else:
        if git("status", "--porcelain", "--untracked-files=no"):
            log.warning("uncommitted_changes", action="committing_as_baseline")
            git("-c", "user.name=swe-agent", "-c", "user.email=swe-agent@localhost",
                "commit", "-q", "-a", "-m", "baseline")
        git("tag", BASELINE_TAG)

    head = git("rev-parse", "HEAD")
    if head == base_commit:
        log.info("baseline_ok", root=ROOT, commit=base_commit[:10])
    else:
        res = exec_raw(["git", "diff", "--shortstat", base_commit, "HEAD"], timeout=60)
        if res.returncode != 0:
            detail = "base_commit not found in the image's repository"
        else:
            detail = text(res.stdout).strip() or "no file differences"
        log.warning("baseline_mismatch", head=head[:10], base_commit=base_commit[:10], diff=detail)

    rc, out, err = run_script('python -c "import sys; print(sys.version.split()[0], sys.executable)"')
    if rc != 0:
        raise RuntimeError(f"conda env 'testbed' is not usable: {(err or out).strip()}")
    log.info("python_env_ok", python=out.strip())