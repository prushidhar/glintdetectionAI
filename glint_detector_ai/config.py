from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DetectionConfig:
    backend: str = "heuristic"
    brightness_threshold: int = 240
    min_area: float = 4.0
    max_area: float = 180.0
    blur_kernel_size: int = 5
    morphology_kernel_size: int = 3
    circularity_threshold: float = 0.2
    max_aspect_ratio: float = 2.4
    processing_scale: float = 1.0
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    yolo_model_path: str | None = None
    yolo_confidence_threshold: float = 0.25
    yolo_target_labels: tuple[str, ...] = ("lens_glint", "cell phone", "camera")


@dataclass(slots=True)
class TrackerConfig:
    max_match_distance: float = 40.0
    max_missed_frames: int = 2
    min_confirmed_hits: int = 2


@dataclass(slots=True)
class ReasoningConfig:
    history_size: int = 12
    max_tracking_distance: float = 35.0
    min_persistent_frames: int = 3
    low_risk_threshold: float = 0.32
    high_risk_threshold: float = 0.68


@dataclass(slots=True)
class BaristaConfig:
    history_size: int = 10
    investigate_score_threshold: float = 0.35
    escalate_score_threshold: float = 0.72
    low_risk_attention_confidence: float = 0.55
    score_smoothing_alpha: float = 0.35
    incident_cooldown_frames: int = 18
    zone_rows: int = 3
    zone_cols: int = 5


@dataclass(slots=True)
class ActionConfig:
    enable_audio_alert: bool = True
    repeat_high_risk_alert_seconds: float = 3.0
    log_dir: Path = field(default_factory=lambda: Path("runtime_logs"))
    save_evidence: bool = True
    evidence_dir: Path | None = None
    save_mask_image: bool = True
    max_track_crops: int = 3
    evidence_cooldown_seconds: float = 5.0
    capture_low_risk_evidence: bool = True
    webhook_url: str | None = None
    webhook_timeout_seconds: float = 4.0
    webhook_on_levels: tuple[str, ...] = ("high risk",)
    save_annotated_frame: bool = True
    save_context_clip: bool = True
    clip_context_frames: int = 24
    clip_fps: float = 8.0


@dataclass(slots=True)
class PipelineConfig:
    camera_index: int = 0
    video_path: str | None = None
    frame_stride: int = 1
    show_mask: bool = False
    headless: bool = False
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    barista: BaristaConfig = field(default_factory=BaristaConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
