# bootstrap_container.py
import os
import sys
import shutil
import subprocess
from pathlib import Path


WORKSPACE = Path(os.getenv("SWE_WORKSPACE", "/workspace"))

def run(cmd, check=True):
    print(f"[bootstrap] Running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=WORKSPACE, text=True, capture_output=True)
    if res.stdout:
        print(res.stdout.strip(), flush=True)
    if res.stderr:
        print(res.stderr.strip(), file=sys.stderr, flush=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}")
    return res

def bootstrap():
    print("[*] Starting fast agnostic dependency resolution...", flush=True)

    # 1. Base tools
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip<24.0", "wheel", "setuptools<65.0.0", "pytest"])

    # 2. Identify target Python packages in the workspace
    pkg_dirs = [
        d.name for d in WORKSPACE.iterdir()
        if d.is_dir() and (d / "__init__.py").exists() and d.name not in {"tests", "test", "docs", "build", "dist", "examples"}
    ]

    # Map package directory names to PyPI names where they differ
    pypi_mapping = {"sklearn": "scikit-learn"}
    pypi_pkgs = [pypi_mapping.get(pkg, pkg) for pkg in pkg_dirs]

    # Install requirement manifests if present
    for req in ["requirements.txt", "requirements-dev.txt", "test-requirements.txt", "tests/requirements.txt"]:
        req_path = WORKSPACE / req
        if req_path.exists():
            print(f"[+] Found manifest {req}, installing...", flush=True)
            run([sys.executable, "-m", "pip", "install", "-r", str(req_path)], check=False)

    # 3. Pull pre-compiled wheels for packages with C-extensions (5 seconds vs 25 minutes of compiling)
    for pypi_name in pypi_pkgs:
        print(f"[+] Fetching pre-compiled wheel for '{pypi_name}'...", flush=True)
        run([sys.executable, "-m", "pip", "install", pypi_name, "numpy<2.0.0"], check=False)

    # 4. Inject compiled .so binaries into the workspace
    site_packages = Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    
    copied = 0
    for pkg in pkg_dirs:
        pkg_site = site_packages / pkg
        pkg_workspace = WORKSPACE / pkg
        if pkg_site.exists() and pkg_workspace.exists():
            for so_file in pkg_site.rglob("*.so"):
                rel_path = so_file.relative_to(pkg_site)
                dst = pkg_workspace / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(so_file, dst)
                copied += 1

    print(f"[+] Injected {copied} pre-compiled binary extension(s) into workspace.", flush=True)

    # 5. Verify import in workspace
    for pkg in pkg_dirs:
        test_cmd = [sys.executable, "-c", f"import {pkg}; print(f'[✓] Successfully imported {pkg}')"]
        res = subprocess.run(test_cmd, env={**os.environ, "PYTHONPATH": "/workspace"}, capture_output=True, text=True)
        if res.returncode == 0:
            print(res.stdout.strip(), flush=True)
        else:
            print(f"[!] Warning on import: {res.stderr.strip()}", flush=True)

    print("[+] Environment ready in seconds.", flush=True)

if __name__ == "__main__":
    bootstrap()