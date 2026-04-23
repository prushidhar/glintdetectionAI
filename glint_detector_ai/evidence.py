from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from glint_detector_ai.config import ActionConfig
from glint_detector_ai.models import BaristaDecision, FrameAnalysis, RiskAssessment


class EvidenceCaptureManager:
    def __init__(self, log_dir: Path, config: ActionConfig) -> None:
        self.config = config
        configured_root = config.evidence_dir or Path("evidence")
        if configured_root.is_absolute():
            self.root = configured_root
        else:
            self.root = log_dir / configured_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "evidence-index.jsonl"
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=max(1, config.clip_context_frames))
        self.source_fps = config.clip_fps
        self.source_label = "unknown"

    def set_source_context(self, source_label: str, source_fps: float | None) -> None:
        self.source_label = source_label
        if source_fps and source_fps > 0:
            self.source_fps = source_fps

    def observe_frame(self, frame: np.ndarray) -> None:
        self.frame_buffer.append(frame.copy())

    def capture(
        self,
        event_type: str,
        frame: np.ndarray,
        mask: np.ndarray | None,
        analysis: FrameAnalysis,
        assessment: RiskAssessment,
        decision: BaristaDecision,
    ) -> dict[str, object]:
        event_id = (
            f"{analysis.timestamp:%Y%m%d-%H%M%S-%f}-"
            f"{event_type.replace('_', '-')}-f{analysis.frame_index}"
        )
        event_dir = self.root / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        frame_relative_path = self._write_image(event_dir / "frame.jpg", frame)
        annotated_frame_path = None
        if self.config.save_annotated_frame:
            annotated_frame = self._annotate_frame(frame, analysis, assessment, decision)
            annotated_frame_path = self._write_image(event_dir / "annotated.jpg", annotated_frame)

        mask_relative_path = None
        if mask is not None:
            mask_relative_path = self._write_image(event_dir / "mask.png", mask)

        track_crops = self._save_track_crops(event_dir, frame, analysis, assessment)
        clip_relative_path = None
        if self.config.save_context_clip and self.frame_buffer:
            clip_relative_path = self._write_clip(event_dir / "context.mp4")

        manifest = {
            "id": event_id,
            "event_type": event_type,
            "timestamp": analysis.timestamp.isoformat(),
            "frame_index": analysis.frame_index,
            "risk_level": assessment.level.value,
            "score": round(assessment.score, 3),
            "smoothed_score": round(decision.smoothed_score, 3),
            "confidence": round(assessment.confidence, 3),
            "backend": analysis.backend,
            "source_label": self.source_label,
            "reason": assessment.reason,
            "operator_message": decision.operator_message,
            "incident_id": decision.incident_id,
            "incident_state": decision.incident_state,
            "incident_age_frames": decision.incident_age_frames,
            "zone_label": decision.zone_label,
            "service_mode": decision.service_mode,
            "priority": decision.priority.value,
            "active_tracks": assessment.active_tracks,
            "dominant_track_id": assessment.dominant_track_id,
            "frame_path": frame_relative_path,
            "annotated_frame_path": annotated_frame_path,
            "mask_path": mask_relative_path,
            "clip_path": clip_relative_path,
            "track_crops": track_crops,
            "recommended_actions": [
                {
                    "kind": action.kind,
                    "label": action.label,
                    "detail": action.detail,
                }
                for action in decision.recommended_actions
            ],
        }

        manifest_path = event_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest_path"] = self._relative_path(manifest_path)

        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest) + "\n")

        return manifest

    def _save_track_crops(
        self,
        event_dir: Path,
        frame: np.ndarray,
        analysis: FrameAnalysis,
        assessment: RiskAssessment,
    ) -> list[dict[str, object]]:
        track_crops: list[dict[str, object]] = []
        ordered_tracks = sorted(
            analysis.tracked_glints,
            key=lambda track: (
                track.track_id != assessment.dominant_track_id,
                not track.confirmed,
                -track.confidence,
                -track.area,
            ),
        )

        for track in ordered_tracks[: max(0, self.config.max_track_crops)]:
            crop = self._extract_crop(frame, track.x, track.y, track.area)
            if crop.size == 0:
                continue
            crop_path = event_dir / f"track-{track.track_id}.jpg"
            relative_path = self._write_image(crop_path, crop)
            track_crops.append(
                {
                    "track_id": track.track_id,
                    "label": track.label,
                    "confidence": round(track.confidence, 3),
                    "confirmed": track.confirmed,
                    "path": relative_path,
                }
            )
        return track_crops

    def _extract_crop(self, frame: np.ndarray, x: int, y: int, area: float) -> np.ndarray:
        radius = max(24, int((max(area, 16.0) ** 0.5) * 3))
        left = max(0, x - radius)
        right = min(frame.shape[1], x + radius)
        top = max(0, y - radius)
        bottom = min(frame.shape[0], y + radius)
        if left >= right or top >= bottom:
            return np.empty((0, 0, 3), dtype=frame.dtype)
        return frame[top:bottom, left:right]

    def _annotate_frame(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysis,
        assessment: RiskAssessment,
        decision: BaristaDecision,
    ) -> np.ndarray:
        annotated = frame.copy()
        color = {
            "watch": (80, 200, 120),
            "investigate": (0, 210, 255),
            "intervene": (0, 0, 255),
        }.get(decision.service_mode, (255, 255, 255))

        for track in analysis.tracked_glints:
            radius = 12 if track.confirmed else 8
            track_color = color if track.track_id == decision.focus_track_id else (180, 180, 180)
            cv2.circle(annotated, (track.x, track.y), radius, track_color, 2)
            cv2.putText(
                annotated,
                f"T{track.track_id}",
                (track.x + 10, track.y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                track_color,
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 92), (20, 20, 20), thickness=-1)
        lines = [
            f"{decision.service_mode.upper()} | {decision.priority.value.upper()} | Incident {decision.incident_id or '-'}",
            f"Zone {decision.zone_label or '-'} | Score {assessment.score:.2f} | Smoothed {decision.smoothed_score:.2f}",
            decision.operator_message,
        ]
        for line_index, text in enumerate(lines):
            cv2.putText(
                annotated,
                text,
                (12, 22 + (line_index * 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color if line_index == 0 else (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        return annotated

    def _write_clip(self, path: Path) -> str | None:
        frames = list(self.frame_buffer)
        if not frames:
            return None
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, float(self.source_fps)),
            (width, height),
        )
        try:
            for frame in frames:
                if frame.shape[0] != height or frame.shape[1] != width:
                    continue
                writer.write(frame)
        finally:
            writer.release()

        if not path.exists():
            return None
        return self._relative_path(path)

    def _write_image(self, path: Path, image: np.ndarray) -> str:
        success = cv2.imwrite(str(path), image)
        if not success:
            raise RuntimeError(f"Failed to write evidence image to {path}")
        return self._relative_path(path)

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()
