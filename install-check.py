#!/usr/bin/env python3
"""
install_check.py
================
Pre-flight check: verifies all dependencies are installed and the
project structure is intact.  Run this ONCE after cloning/unzipping.

Usage
-----
    python install_check.py

Author : Traffic Surveillance System
Version: 1.0.0
"""

from __future__ import annotations

import importlib
import os
import sys


REQUIRED_PACKAGES = [
    ("ultralytics",  "ultralytics"),
    ("cv2",          "opencv-python"),
    ("numpy",        "numpy"),
    ("scipy",        "scipy"),
    ("pandas",       "pandas"),
    ("yaml",         "PyYAML"),
    ("matplotlib",   "matplotlib"),
    ("colorlog",     "colorlog"),
    ("tqdm",         "tqdm"),
    ("PIL",          "Pillow"),
]

REQUIRED_DIRS = [
    "config", "core", "utils", "visualization",
    "input",  "output", "logs",
]

REQUIRED_FILES = [
    "main.py",
    "pipeline.py",
    "batch_process.py",
    "setup_roi.py",
    "config/config.yaml",
    "config/config_loader.py",
    "core/vehicle.py",
    "core/detector.py",
    "core/tracker.py",
    "core/roi_counter.py",
    "utils/geometry.py",
    "utils/logger.py",
    "visualization/overlay.py",
    "visualization/reporter.py",
]


def check_python_version() -> bool:
    """Ensure Python 3.9+."""
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 9)
    status = "✅" if ok else "❌"
    print(f"  {status}  Python {major}.{minor}  (need 3.9+)")
    return ok


def check_packages() -> bool:
    """Check all required Python packages are importable."""
    all_ok = True
    for module, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module)
            print(f"  ✅  {pip_name}")
        except ImportError:
            print(f"  ❌  {pip_name}  ← run: pip install {pip_name}")
            all_ok = False
    return all_ok


def check_structure() -> bool:
    """Verify project directories and files exist."""
    all_ok = True
    for d in REQUIRED_DIRS:
        exists = os.path.isdir(d)
        status = "✅" if exists else "⚠️ "
        print(f"  {status}  {d}/")
        if not exists:
            os.makedirs(d, exist_ok=True)
            print(f"       → Created '{d}/'")

    for f in REQUIRED_FILES:
        exists = os.path.isfile(f)
        status = "✅" if exists else "❌"
        print(f"  {status}  {f}")
        if not exists:
            all_ok = False
    return all_ok


def check_yolo_weights() -> None:
    """Check if YOLO weights are cached (informational only)."""
    from pathlib import Path
    home = Path.home()
    cache_dirs = [
        home / ".cache" / "ultralytics",
        Path("models"),
    ]
    found = False
    for cd in cache_dirs:
        pts = list(cd.glob("*.pt")) if cd.is_dir() else []
        for pt in pts:
            print(f"  ✅  Found cached weights: {pt}")
            found = True
    if not found:
        print("  ℹ️   No cached YOLO weights found – they will be auto-downloaded on first run.")


def main() -> None:
    """Run all checks and report."""
    print("\n" + "═" * 55)
    print("  TRAFFIC SURVEILLANCE SYSTEM – Installation Check")
    print("═" * 55)

    print("\n[1] Python Version")
    py_ok = check_python_version()

    print("\n[2] Required Packages")
    pkg_ok = check_packages()

    print("\n[3] Project Structure")
    struct_ok = check_structure()

    print("\n[4] YOLO Model Weights")
    check_yolo_weights()

    print("\n" + "═" * 55)
    if py_ok and pkg_ok and struct_ok:
        print("  ✅  All checks passed!  Ready to run.")
        print("\n  Next steps:")
        print("  1. Drop a video into the  input/  folder")
        print("  2. (Optional) python setup_roi.py --video input/<your_video.mp4>")
        print("  3. python main.py")
    else:
        print("  ⚠️   Some checks failed – fix the items marked ❌ above.")
        print("  Run:  pip install -r requirements.txt")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
