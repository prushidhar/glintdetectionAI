from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from glint_detector_ai.config import ReasoningConfig
from glint_detector_ai.models import FrameAnalysis, RiskAssessment, RiskLevel, TrackedGlint


@dataclass(slots=True)
class TrackWindowStats:
    track_id: int
    frames_seen: set[int] = field(default_factory=set)
    motion_samples: list[float] = field(default_factory=list)
    confirmed_frames: int = 0

    def add(self, frame_index: int, tracked_glint: TrackedGlint) -> None:
        self.frames_seen.add(frame_index)
        self.motion_samples.append(tracked_glint.average_motion)
        if tracked_glint.confirmed:
            self.confirmed_frames += 1

    @property
    def persistence(self) -> int:
        return len(self.frames_seen)

    @property
    def confirmation_ratio(self) -> float:
        if not self.frames_seen:
            return 0.0
        return self.confirmed_frames / len(self.frames_seen)

    @property
    def average_motion(self) -> float:
        if not self.motion_samples:
            return 0.0
        return sum(self.motion_samples) / len(self.motion_samples)


class TemporalRiskAgent:
    """Agent-inspired temporal reasoning over recent glint observations."""

    def __init__(self, config: ReasoningConfig) -> None:
        self.config = config
        self.history: deque[FrameAnalysis] = deque(maxlen=config.history_size)

    def assess(self, analysis: FrameAnalysis) -> RiskAssessment:
        self.history.append(analysis)
        history_frames = list(self.history)
        if not history_frames:
            return self._safe_assessment("No history available yet.")

        detections_per_frame = [frame.detection_count for frame in history_frames]
        active_frames = sum(1 for count in detections_per_frame if count > 0)
        detection_rate = active_frames / len(history_frames)
        track_stats = self._build_track_stats(history_frames)
        persistent_tracks = [
            track
            for track in track_stats.values()
            if track.persistence >= self.config.min_persistent_frames
        ]

        persistence_score = self._mean(
            track.persistence / len(history_frames) for track in persistent_tracks
        )
        stability_score = self._mean(
            self._stability_from_motion(track.average_motion) for track in persistent_tracks
        )
        confirmation_score = self._mean(track.confirmation_ratio for track in persistent_tracks)
        dominant_track = max(
            track_stats.values(),
            key=lambda track: (track.persistence, track.confirmation_ratio),
            default=None,
        )
        dominant_track_persistence = (
            dominant_track.persistence / len(history_frames) if dominant_track else 0.0
        )

        score = min(
            1.0,
            (detection_rate * 0.25)
            + (persistence_score * 0.3)
            + (stability_score * 0.2)
            + (confirmation_score * 0.15)
            + (dominant_track_persistence * 0.1),
        )

        level = self._classify_level(
            active_frames=active_frames,
            score=score,
            persistent_tracks=persistent_tracks,
            detection_rate=detection_rate,
            stability_score=stability_score,
            dominant_track_persistence=dominant_track_persistence,
            current_tracks=len(analysis.tracked_glints),
        )
        confidence = min(
            1.0,
            0.3
            + (len(history_frames) / self.config.history_size) * 0.45
            + (confirmation_score * 0.25),
        )

        reason = self._build_reason(
            active_frames=active_frames,
            history_size=len(history_frames),
            persistent_tracks=len(persistent_tracks),
            detection_rate=detection_rate,
            stability_score=stability_score,
            dominant_track=dominant_track,
            current_tracks=len(analysis.tracked_glints),
        )

        return RiskAssessment(
            level=level,
            score=score,
            confidence=confidence,
            reason=reason,
            detection_rate=detection_rate,
            persistent_tracks=len(persistent_tracks),
            stability_score=stability_score,
            active_tracks=len(analysis.tracked_glints),
            dominant_track_id=dominant_track.track_id if dominant_track else None,
            dominant_track_persistence=dominant_track_persistence,
        )

    def _build_track_stats(self, history_frames: list[FrameAnalysis]) -> dict[int, TrackWindowStats]:
        track_stats: dict[int, TrackWindowStats] = {}
        for frame in history_frames:
            for tracked_glint in frame.tracked_glints:
                stats = track_stats.setdefault(
                    tracked_glint.track_id,
                    TrackWindowStats(track_id=tracked_glint.track_id),
                )
                stats.add(frame.frame_index, tracked_glint)
        return track_stats

    def _classify_level(
        self,
        active_frames: int,
        score: float,
        persistent_tracks: list[TrackWindowStats],
        detection_rate: float,
        stability_score: float,
        dominant_track_persistence: float,
        current_tracks: int,
    ) -> RiskLevel:
        if active_frames == 0:
            return RiskLevel.SAFE

        if score >= self.config.high_risk_threshold:
            return RiskLevel.HIGH

        if persistent_tracks and dominant_track_persistence >= 0.5 and stability_score >= 0.55:
            return RiskLevel.HIGH

        if score >= self.config.low_risk_threshold or active_frames >= 2 or current_tracks > 0:
            return RiskLevel.LOW

        return RiskLevel.SAFE

    def _build_reason(
        self,
        active_frames: int,
        history_size: int,
        persistent_tracks: int,
        detection_rate: float,
        stability_score: float,
        dominant_track: TrackWindowStats | None,
        current_tracks: int,
    ) -> str:
        if active_frames == 0:
            return "No bright spot candidates observed in the current history window."

        dominant_track_text = (
            f"dominant track #{dominant_track.track_id} persistence "
            f"{dominant_track.persistence / history_size:.2f}"
            if dominant_track
            else "no dominant track yet"
        )
        return (
            f"Detections seen in {active_frames}/{history_size} recent frames, "
            f"{persistent_tracks} persistent track(s), "
            f"{current_tracks} active track(s) now, "
            f"detection rate {detection_rate:.2f}, "
            f"stability {stability_score:.2f}, "
            f"{dominant_track_text}."
        )

    def _safe_assessment(self, reason: str) -> RiskAssessment:
        return RiskAssessment(
            level=RiskLevel.SAFE,
            score=0.0,
            confidence=0.0,
            reason=reason,
            detection_rate=0.0,
            persistent_tracks=0,
            stability_score=0.0,
            active_tracks=0,
            dominant_track_id=None,
            dominant_track_persistence=0.0,
        )

    @staticmethod
    def _mean(values: object) -> float:
        values_list = list(values)
        if not values_list:
            return 0.0
        return sum(values_list) / len(values_list)

    def _stability_from_motion(self, average_motion: float) -> float:
        return max(0.0, 1.0 - (average_motion / self.config.max_tracking_distance))
