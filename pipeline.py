"""
pipeline.py
===========
Central orchestration class for the Traffic Surveillance System.

The ``SurveillancePipeline`` class ties together all sub-systems:
  • YOLODetector       – YOLOv11 inference
  • VehicleTracker     – Centroid-based multi-object tracking
  • ROICounter         – Region-of-interest line counting
  • OverlayRenderer    – OpenCV annotation overlay
  • Reporter           – CSV / JSON report generation

Usage
-----
    from pipeline import SurveillancePipeline

    pipe = SurveillancePipeline(config_path="config/config.yaml")
    pipe.run(video_path="input/traffic.mp4")

Author : Traffic Surveillance System
Version: 1.0.0
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import cv2

from config.config_loader import ConfigLoader
from core.detector import YOLODetector
from core.roi_counter import ROICounter
from core.tracker import VehicleTracker
from utils.logger import get_logger
from visualization.overlay import OverlayRenderer
from visualization.reporter import Reporter

log = get_logger(__name__)


class SurveillancePipeline:
    """
    End-to-end traffic surveillance processing pipeline.

    Instantiate once and call ``run(video_path)`` for each video.
    The pipeline is fully restartable — calling ``run`` multiple
    times re-initialises internal state automatically.

    Parameters
    ----------
    config_path : Path to the YAML config file.
                  Defaults to ``config/config.yaml`` (auto-resolved).
    """

    # Supported input file extensions
    _VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._cfg = ConfigLoader(config_path)
        lcfg      = self._cfg.logging_cfg

        # Re-create logger with config-level
        global log
        log = get_logger(
            __name__,
            level      = lcfg.get("level",       "INFO"),
            log_to_file= lcfg.get("log_to_file",  True),
            log_dir    = lcfg.get("log_dir",      "logs"),
        )

        # ── Build sub-systems ────────────────────────────────
        log.info("Initialising Traffic Surveillance Pipeline …")
        self._detector  = YOLODetector(self._cfg)
        self._tracker   = VehicleTracker(self._cfg)
        self._roi       = ROICounter(self._cfg)

        log.info("Pipeline ready. Drop videos into the 'input/' folder and call run().")

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def run(self, video_path: str) -> dict:
        """
        Process one video file end-to-end.

        Parameters
        ----------
        video_path : Path to the input video.

        Returns
        -------
        dict – Final count summary from the ROICounter.
        """
        video_path = os.path.abspath(video_path)
        self._validate_video(video_path)

        video_name  = Path(video_path).stem
        output_dir  = self._cfg.get("output") or {}
        out_root    = "output"
        os.makedirs(out_root, exist_ok=True)

        log.info("═" * 60)
        log.info("Processing video: %s", video_path)
        log.info("═" * 60)

        # ── Open capture ─────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Unable to open video: {video_path}")

        frame_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        log.info(
            "Video: %dx%d @ %.1f fps | frames=%d (~%.1f s)",
            frame_w, frame_h, src_fps, total_frs, total_frs / src_fps,
        )

        # ── Reset state for new video ─────────────────────────
        self._tracker.reset()
        self._roi.reset_counts()
        self._roi.initialise(frame_w, frame_h)

        # ── Build renderer & reporter ─────────────────────────
        renderer = OverlayRenderer(self._cfg, self._roi)
        reporter = Reporter(out_root, video_name)

        # ── Setup output writer ───────────────────────────────
        writer      = None
        out_vid_path = ""
        ocfg        = self._cfg.output

        if ocfg.get("save_video", True):
            out_fps    = ocfg.get("fps_override") or src_fps
            codec      = ocfg.get("codec", "mp4v")
            fourcc     = cv2.VideoWriter_fourcc(*codec)
            out_vid_path = os.path.join(out_root, f"{video_name}_annotated.mp4")
            writer = cv2.VideoWriter(
                out_vid_path, fourcc, out_fps, (frame_w, frame_h)
            )
            log.info("Output video: %s", out_vid_path)

        # ── Frame loop ────────────────────────────────────────
        frame_idx   = 0
        t_start     = time.perf_counter()
        prev_tracks = {}

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break   # End of video

                # 1. Detection
                detections = self._detector.detect(frame)

                # 2. Tracking
                active_tracks = self._tracker.update(detections, frame_idx)

                # 3. Count line crossings
                self._roi.process_frame(active_tracks)

                # 4. Record newly deregistered tracks for the report
                deregistered = set(prev_tracks.keys()) - set(active_tracks.keys())
                for tid in deregistered:
                    reporter.record_track(prev_tracks[tid])
                prev_tracks = dict(active_tracks)

                # 5. Overlay annotation
                annotated = renderer.draw(frame, active_tracks, frame_idx)

                # 6. Write output frame
                if writer:
                    writer.write(annotated)

                # 7. Progress logging every 100 frames
                frame_idx += 1
                if frame_idx % 100 == 0:
                    elapsed = time.perf_counter() - t_start
                    fps_proc = frame_idx / elapsed
                    pct      = (frame_idx / max(total_frs, 1)) * 100
                    log.info(
                        "Progress: %5d / %5d frames  (%.1f%%)  "
                        "proc_fps=%.1f  tracks=%d",
                        frame_idx, total_frs, pct,
                        fps_proc, len(active_tracks),
                    )

        finally:
            # Record any remaining active tracks in the report
            for vehicle in prev_tracks.values():
                reporter.record_track(vehicle)

            cap.release()
            if writer:
                writer.release()

        # ── Write reports ─────────────────────────────────────
        elapsed_total = time.perf_counter() - t_start
        log.info(
            "Processing complete. %d frames in %.1f s (%.1f fps avg)",
            frame_idx, elapsed_total, frame_idx / max(elapsed_total, 0.001),
        )

        if ocfg.get("save_csv_report", True):
            reporter.write_csv()

        if ocfg.get("save_summary_json", True):
            reporter.write_json_summary(
                self._roi, frame_idx, src_fps, out_vid_path
            )

        reporter.print_summary(self._roi)

        return self._roi.summary()

    def run_folder(self, folder_path: str = "input") -> None:
        """
        Process all supported video files found in ``folder_path``.

        This is the recommended entry-point when the user has dropped
        multiple videos into the ``input/`` folder.

        Parameters
        ----------
        folder_path : Directory to scan for video files.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Input folder not found: {folder}")

        videos = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in self._VIDEO_EXTENSIONS
        )

        if not videos:
            log.warning(
                "No supported video files found in '%s'. "
                "Supported extensions: %s",
                folder_path, ", ".join(sorted(self._VIDEO_EXTENSIONS)),
            )
            return

        log.info("Found %d video(s) in '%s'.", len(videos), folder_path)
        for idx, vpath in enumerate(videos, start=1):
            log.info("── Video %d / %d ──", idx, len(videos))
            try:
                self.run(str(vpath))
            except Exception as exc:
                log.error("Failed to process '%s': %s", vpath.name, exc, exc_info=True)

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _validate_video(self, path: str) -> None:
        """Raise informative errors for invalid video paths."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        ext = Path(path).suffix.lower()
        if ext not in self._VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {', '.join(sorted(self._VIDEO_EXTENSIONS))}"
            )
