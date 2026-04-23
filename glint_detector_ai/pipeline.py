from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from glint_detector_ai.actions import ActionCoordinator
from glint_detector_ai.barista import AIBarista
from glint_detector_ai.config import BaristaConfig, DetectionConfig, PipelineConfig, TrackerConfig
from glint_detector_ai.models import BaristaDecision, FrameAnalysis, RiskAssessment, RiskLevel
from glint_detector_ai.perception import build_detector
from glint_detector_ai.reasoning import TemporalRiskAgent
from glint_detector_ai.tracking import GlintTracker


class GlintMonitoringPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.detector = build_detector(config.detection)
        self.tracker = GlintTracker(config.tracker)
        self.agent = TemporalRiskAgent(config.reasoning)
        self.barista = AIBarista(config.barista)
        self.actions = ActionCoordinator(config.actions)

    def run(self) -> int:
        source = self.config.video_path if self.config.video_path else self.config.camera_index
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            print(
                f"Unable to open source {source!r}. "
                "Check the device connection, file path, or source configuration."
            )
            return 1
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        self.actions.set_source_context(self._source_label(), source_fps if source_fps > 0 else None)

        frame_index = 0
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    if self.config.video_path:
                        break
                    print("Camera frame read failed; stopping pipeline.")
                    return 1

                if self.config.frame_stride > 1 and frame_index % self.config.frame_stride != 0:
                    frame_index += 1
                    continue

                analysis, mask = self.detector.analyze_frame(frame, frame_index)
                tracked_glints = self.tracker.update(analysis)
                analysis = replace(analysis, tracked_glints=tracked_glints)
                assessment = self.agent.assess(analysis)
                decision = self.barista.serve(analysis, assessment)
                self.actions.process(frame, mask, analysis, assessment, decision)

                if not self.config.headless:
                    overlay = self._build_overlay(frame, analysis, assessment, decision)
                    cv2.imshow("GlintDetectorAI", overlay)
                    if self.config.show_mask:
                        cv2.imshow("GlintDetectorAI Mask", mask)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_index += 1
        except KeyboardInterrupt:
            print("Pipeline interrupted by user.")
        finally:
            capture.release()
            if not self.config.headless:
                cv2.destroyAllWindows()

        return 0

    def _build_overlay(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysis,
        assessment: RiskAssessment,
        decision: BaristaDecision,
    ) -> np.ndarray:
        overlay = frame.copy()

        risk_color = {
            RiskLevel.SAFE: (80, 200, 120),
            RiskLevel.LOW: (0, 210, 255),
            RiskLevel.HIGH: (0, 0, 255),
        }[assessment.level]

        if analysis.tracked_glints:
            for track in analysis.tracked_glints:
                track_color = risk_color if track.confirmed else (160, 160, 160)
                radius = 12 if track.confirmed else 8
                cv2.circle(overlay, (track.x, track.y), radius, track_color, 2)
                cv2.putText(
                    overlay,
                    f"T{track.track_id} {track.label} {track.confidence:.2f}",
                    (track.x + 12, track.y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    track_color,
                    1,
                    cv2.LINE_AA,
                )
        else:
            for detection in analysis.detections:
                cv2.circle(overlay, (detection.x, detection.y), 10, risk_color, 2)
                cv2.putText(
                    overlay,
                    f"{detection.label} {detection.confidence:.2f}",
                    (detection.x + 12, detection.y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    risk_color,
                    1,
                    cv2.LINE_AA,
                )

        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 180), (18, 18, 18), thickness=-1)
        info_lines = [
            f"Risk: {assessment.level.value.upper()}",
            f"Score: {assessment.score:.2f} | Confidence: {assessment.confidence:.2f}",
            f"Backend: {analysis.backend} | Detections: {analysis.detection_count} | Tracks: {assessment.active_tracks}",
            f"Active rate: {assessment.detection_rate:.2f} | Persistent tracks: {assessment.persistent_tracks}",
            f"Source: {self._source_label()} | Frame stride: {self.config.frame_stride}",
            f"Barista: {decision.service_mode.upper()} | Priority: {decision.priority.value.upper()} | Focus: {decision.focus_track_id}",
            f"Incident: {decision.incident_id or '-'} | State: {decision.incident_state.upper()} | Zone: {decision.zone_label or '-'}",
            f"Smoothed score: {decision.smoothed_score:.2f} | Dominant track: {assessment.dominant_track_id} | Stability: {assessment.stability_score:.2f}",
            f"Barista says: {decision.operator_message}",
            f"Reason: {assessment.reason}",
        ]

        for line_index, text in enumerate(info_lines):
            cv2.putText(
                overlay,
                text,
                (12, 22 + (line_index * 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                risk_color if line_index == 0 else (245, 245, 245),
                1,
                cv2.LINE_AA,
            )

        return overlay

    def _source_label(self) -> str:
        if self.config.video_path:
            return Path(self.config.video_path).name
        return f"camera:{self.config.camera_index}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GlintDetectorAI MVP: webcam-based lens glint detection and risk assessment."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--video-path",
        default=None,
        help="Optional video file path to analyze instead of a live camera.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame for higher throughput.",
    )
    parser.add_argument(
        "--detector-backend",
        default="heuristic",
        choices=("heuristic", "yolo"),
        help="Detection backend to use.",
    )
    parser.add_argument(
        "--brightness-threshold",
        type=int,
        default=240,
        help="Grayscale threshold used to isolate bright spots.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=4.0,
        help="Minimum contour area to consider as a glint.",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=180.0,
        help="Maximum contour area to consider as a glint.",
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Display the binary threshold mask in a second window.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run capture, reasoning, and logging without UI windows.",
    )
    parser.add_argument(
        "--disable-audio-alert",
        action="store_true",
        help="Disable system beeps for high-risk events.",
    )
    parser.add_argument(
        "--yolo-model-path",
        default=None,
        help="Optional Ultralytics YOLO model path for the yolo backend.",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO detections.",
    )
    parser.add_argument(
        "--processing-scale",
        type=float,
        default=1.0,
        help="Downscale factor for heuristic detection, between 0.2 and 1.0.",
    )
    parser.add_argument(
        "--disable-clahe",
        action="store_true",
        help="Disable CLAHE contrast enhancement in the heuristic detector.",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=2.4,
        help="Reject contours more elongated than this aspect ratio.",
    )
    parser.add_argument(
        "--max-track-distance",
        type=float,
        default=40.0,
        help="Maximum pixel distance used to match glints to an existing track.",
    )
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=2,
        help="How many frames a track can disappear before it is retired.",
    )
    parser.add_argument(
        "--tracker-confirmation-hits",
        type=int,
        default=2,
        help="Number of detections required before a track is marked confirmed.",
    )
    parser.add_argument(
        "--barista-investigate-threshold",
        type=float,
        default=0.35,
        help="Smoothed score threshold where Barista moves into investigate mode.",
    )
    parser.add_argument(
        "--barista-escalate-threshold",
        type=float,
        default=0.72,
        help="Smoothed score threshold where Barista escalates to intervene mode.",
    )
    parser.add_argument(
        "--barista-smoothing-alpha",
        type=float,
        default=0.35,
        help="Smoothing factor for Barista incident score memory.",
    )
    parser.add_argument(
        "--incident-cooldown-frames",
        type=int,
        default=18,
        help="How many processed frames Barista keeps an incident open while activity cools.",
    )
    parser.add_argument(
        "--zone-rows",
        type=int,
        default=3,
        help="Number of zone rows used for auditorium hotspot mapping.",
    )
    parser.add_argument(
        "--zone-cols",
        type=int,
        default=5,
        help="Number of zone columns used for auditorium hotspot mapping.",
    )
    parser.add_argument(
        "--disable-evidence",
        action="store_true",
        help="Disable snapshot and crop capture for suspicious events.",
    )
    parser.add_argument(
        "--evidence-dir",
        default=None,
        help="Directory used for saved evidence artifacts relative to runtime_logs/ by default.",
    )
    parser.add_argument(
        "--disable-mask-evidence",
        action="store_true",
        help="Do not store threshold masks alongside evidence captures.",
    )
    parser.add_argument(
        "--disable-low-risk-evidence",
        action="store_true",
        help="Only capture evidence for high-risk events.",
    )
    parser.add_argument(
        "--max-track-crops",
        type=int,
        default=3,
        help="Maximum number of tracked-glint crops to store per evidence capture.",
    )
    parser.add_argument(
        "--disable-annotated-evidence",
        action="store_true",
        help="Do not save Barista-annotated frames inside evidence bundles.",
    )
    parser.add_argument(
        "--disable-context-clip",
        action="store_true",
        help="Do not save rolling MP4 clips with evidence bundles.",
    )
    parser.add_argument(
        "--clip-context-frames",
        type=int,
        default=24,
        help="How many processed frames to retain for rolling evidence clips.",
    )
    parser.add_argument(
        "--clip-fps",
        type=float,
        default=8.0,
        help="Fallback FPS used when saving context clips.",
    )
    parser.add_argument(
        "--webhook-url",
        default=None,
        help="Optional webhook endpoint that receives alert payloads.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    detection = DetectionConfig(
        backend=args.detector_backend,
        brightness_threshold=args.brightness_threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        yolo_model_path=args.yolo_model_path,
        yolo_confidence_threshold=args.yolo_confidence,
        max_aspect_ratio=args.max_aspect_ratio,
        processing_scale=args.processing_scale,
        enable_clahe=not args.disable_clahe,
    )
    tracker = TrackerConfig(
        max_match_distance=args.max_track_distance,
        max_missed_frames=args.max_missed_frames,
        min_confirmed_hits=args.tracker_confirmation_hits,
    )
    barista = BaristaConfig(
        investigate_score_threshold=args.barista_investigate_threshold,
        escalate_score_threshold=args.barista_escalate_threshold,
        score_smoothing_alpha=args.barista_smoothing_alpha,
        incident_cooldown_frames=args.incident_cooldown_frames,
        zone_rows=max(1, args.zone_rows),
        zone_cols=max(1, args.zone_cols),
    )
    return PipelineConfig(
        camera_index=args.camera_index,
        video_path=args.video_path,
        frame_stride=max(1, args.frame_stride),
        show_mask=args.show_mask,
        headless=args.headless,
        detection=detection,
        tracker=tracker,
        barista=barista,
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)
    config.actions.enable_audio_alert = not args.disable_audio_alert
    config.actions.save_evidence = not args.disable_evidence
    config.actions.evidence_dir = Path(args.evidence_dir) if args.evidence_dir else None
    config.actions.save_mask_image = not args.disable_mask_evidence
    config.actions.capture_low_risk_evidence = not args.disable_low_risk_evidence
    config.actions.max_track_crops = max(0, args.max_track_crops)
    config.actions.save_annotated_frame = not args.disable_annotated_evidence
    config.actions.save_context_clip = not args.disable_context_clip
    config.actions.clip_context_frames = max(1, args.clip_context_frames)
    config.actions.clip_fps = max(1.0, args.clip_fps)
    config.actions.webhook_url = args.webhook_url
    pipeline = GlintMonitoringPipeline(config)
    return pipeline.run()
