from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low risk"
    HIGH = "high risk"


class ServicePriority(str, Enum):
    ROUTINE = "routine"
    ATTENTION = "attention"
    URGENT = "urgent"


@dataclass(slots=True, frozen=True)
class GlintDetection:
    x: int
    y: int
    area: float
    intensity: float
    circularity: float
    label: str = "lens_glint"
    confidence: float = 1.0


@dataclass(slots=True, frozen=True)
class TrackedGlint:
    track_id: int
    x: int
    y: int
    area: float
    intensity: float
    circularity: float
    label: str
    confidence: float
    age: int
    hits: int
    consecutive_hits: int
    missed_frames: int
    average_motion: float
    confirmed: bool


@dataclass(slots=True, frozen=True)
class FrameAnalysis:
    frame_index: int
    timestamp: datetime
    detections: list[GlintDetection]
    max_intensity: float
    backend: str = "heuristic"
    tracked_glints: list[TrackedGlint] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0

    @property
    def detection_count(self) -> int:
        return len(self.detections)


@dataclass(slots=True, frozen=True)
class RiskAssessment:
    level: RiskLevel
    score: float
    confidence: float
    reason: str
    detection_rate: float
    persistent_tracks: int
    stability_score: float
    active_tracks: int
    dominant_track_id: int | None
    dominant_track_persistence: float


@dataclass(slots=True, frozen=True)
class BaristaAction:
    kind: str
    label: str
    detail: str


@dataclass(slots=True, frozen=True)
class BaristaDecision:
    service_mode: str
    priority: ServicePriority
    focus_track_id: int | None
    zone_label: str | None
    incident_id: str | None
    incident_state: str
    incident_age_frames: int
    smoothed_score: float
    operator_message: str
    rationale: str
    recommended_actions: list[BaristaAction]
    capture_evidence: bool
    audible_alert: bool
    send_webhook: bool


@dataclass(slots=True, frozen=True)
class LoggedEvent:
    event_type: str
    timestamp: datetime
    frame_index: int
    risk_level: str
    score: float
    confidence: float
    reason: str
    backend: str
    active_tracks: int
    dominant_track_id: int | None
    tracked_glints: list[TrackedGlint]
    detections: list[GlintDetection]
    barista_service_mode: str | None = None
    barista_priority: str | None = None
    barista_focus_track_id: int | None = None
    barista_zone_label: str | None = None
    barista_incident_id: str | None = None
    barista_incident_state: str | None = None
    barista_incident_age_frames: int | None = None
    barista_smoothed_score: float | None = None
    barista_operator_message: str | None = None
    barista_rationale: str | None = None
    barista_actions: list[BaristaAction] = field(default_factory=list)
    evidence_id: str | None = None
    evidence_manifest_path: str | None = None
    evidence_frame_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "frame_index": self.frame_index,
            "risk_level": self.risk_level,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "backend": self.backend,
            "active_tracks": self.active_tracks,
            "dominant_track_id": self.dominant_track_id,
            "tracked_glints": [asdict(track) for track in self.tracked_glints],
            "detections": [asdict(detection) for detection in self.detections],
            "barista_service_mode": self.barista_service_mode,
            "barista_priority": self.barista_priority,
            "barista_focus_track_id": self.barista_focus_track_id,
            "barista_zone_label": self.barista_zone_label,
            "barista_incident_id": self.barista_incident_id,
            "barista_incident_state": self.barista_incident_state,
            "barista_incident_age_frames": self.barista_incident_age_frames,
            "barista_smoothed_score": self.barista_smoothed_score,
            "barista_operator_message": self.barista_operator_message,
            "barista_rationale": self.barista_rationale,
            "barista_actions": [asdict(action) for action in self.barista_actions],
            "evidence_id": self.evidence_id,
            "evidence_manifest_path": self.evidence_manifest_path,
            "evidence_frame_path": self.evidence_frame_path,
        }
