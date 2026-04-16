"""
batch_process.py
================
Convenience wrapper to process all videos inside ``input/`` using
the Traffic Surveillance Pipeline.

Identical to running ``python main.py`` without arguments, but adds
a simple progress summary table printed at the end.

Usage
-----
    python batch_process.py
    python batch_process.py --folder /path/to/videos
    python batch_process.py --config config/custom.yaml

Author : Traffic Surveillance System
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline import SurveillancePipeline
from utils.logger import get_logger

log = get_logger("batch_process")

SUPPORTED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def main() -> None:
    """Batch-process all videos in the input folder."""
    parser = argparse.ArgumentParser(
        description="Batch-process all videos in a folder."
    )
    parser.add_argument("--folder", "-f", default="input",
                        help="Input folder (default: input/)")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to YAML config file.")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        folder.mkdir(parents=True, exist_ok=True)
        log.warning("Created empty input folder '%s'. Add videos and re-run.", folder)
        return

    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
    if not videos:
        log.warning("No videos found in '%s'.", folder)
        return

    pipeline = SurveillancePipeline(config_path=args.config)

    results  = []
    for vpath in videos:
        log.info("Processing: %s", vpath.name)
        t0 = time.perf_counter()
        try:
            summary = pipeline.run(str(vpath))
            elapsed = time.perf_counter() - t0
            results.append({
                "name":    vpath.name,
                "status":  "OK",
                "total_in":  summary.get("total_in",  0),
                "total_out": summary.get("total_out", 0),
                "elapsed":   elapsed,
            })
        except Exception as exc:
            results.append({
                "name":    vpath.name,
                "status":  f"ERROR: {exc}",
                "total_in":  "-",
                "total_out": "-",
                "elapsed":   time.perf_counter() - t0,
            })

    # ── Summary table ─────────────────────────────────────────
    sep = "─" * 70
    print(f"\n{sep}")
    print("  BATCH PROCESSING SUMMARY")
    print(sep)
    print(f"  {'File':<30} {'Status':<10} {'IN':>6} {'OUT':>6} {'Time(s)':>8}")
    print(sep)
    for r in results:
        tin  = str(r["total_in"])
        tout = str(r["total_out"])
        print(
            f"  {r['name']:<30} {r['status']:<10} "
            f"{tin:>6} {tout:>6} {r['elapsed']:>8.1f}"
        )
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
