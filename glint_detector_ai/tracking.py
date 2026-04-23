from __future__ import annotations

from dataclasses import dataclass
from math import dist

from glint_detector_ai.config import TrackerConfig
from glint_detector_ai.models import FrameAnalysis, GlintDetection, TrackedGlint


@dataclass(slots=True)
class TrackState:
    track_id: int
    x: float
    y: float
    area: float
    intensity: float
    circularity: float
    label: str
    confidence: float
    age: int = 1
    hits: int = 1
    consecutive_hits: int = 1
    missed_frames: int = 0
    total_motion: float = 0.0
    last_seen_frame: int = 0

    @classmethod
    def from_detection(
        cls,
        track_id: int,
        detection: GlintDetection,
        frame_index: int,
    ) -> "TrackState":
        return cls(
            track_id=track_id,
            x=float(detection.x),
            y=float(detection.y),
            area=detection.area,
            intensity=detection.intensity,
            circularity=detection.circularity,
            label=detection.label,
            confidence=detection.confidence,
            last_seen_frame=frame_index,
        )

    def update(self, detection: GlintDetection, frame_index: int) -> None:
        self.total_motion += dist((self.x, self.y), (detection.x, detection.y))
        self.x = float(detection.x)
        self.y = float(detection.y)
        self.area = detection.area
        self.intensity = detection.intensity
        self.circularity = detection.circularity
        self.label = detection.label
        self.confidence = detection.confidence
        self.age += 1
        self.hits += 1
        self.consecutive_hits += 1
        self.missed_frames = 0
        self.last_seen_frame = frame_index

    def mark_missed(self) -> None:
        self.age += 1
        self.missed_frames += 1
        self.consecutive_hits = 0

    @property
    def average_motion(self) -> float:
        if self.hits <= 1:
            return 0.0
        return self.total_motion / (self.hits - 1)

    def snapshot(self, config: TrackerConfig) -> TrackedGlint:
        return TrackedGlint(
            track_id=self.track_id,
            x=int(round(self.x)),
            y=int(round(self.y)),
            area=self.area,
            intensity=self.intensity,
            circularity=self.circularity,
            label=self.label,
            confidence=self.confidence,
            age=self.age,
            hits=self.hits,
            consecutive_hits=self.consecutive_hits,
            missed_frames=self.missed_frames,
            average_motion=self.average_motion,
            confirmed=self.hits >= config.min_confirmed_hits,
        )


class GlintTracker:
    """Simple nearest-neighbor multi-object tracker for glint candidates."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracks: dict[int, TrackState] = {}
        self._next_track_id = 1

    def update(self, analysis: FrameAnalysis) -> list[TrackedGlint]:
        detections = analysis.detections
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        candidate_matches: list[tuple[float, float, int, int]] = []

        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                if detection.label != track.label:
                    continue
                distance = dist((track.x, track.y), (detection.x, detection.y))
                if distance <= self.config.max_match_distance:
                    candidate_matches.append(
                        (distance, -detection.confidence, track_id, detection_index)
                    )

        candidate_matches.sort(key=lambda item: (item[0], item[1]))

        for _, _, track_id, detection_index in candidate_matches:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            self._tracks[track_id].update(detections[detection_index], analysis.frame_index)
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks:
                continue
            track.mark_missed()
            if track.missed_frames > self.config.max_missed_frames:
                del self._tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = TrackState.from_detection(
                track_id=track_id,
                detection=detection,
                frame_index=analysis.frame_index,
            )

        visible_tracks = [
            track.snapshot(self.config)
            for track in self._tracks.values()
            if track.last_seen_frame == analysis.frame_index
        ]
        visible_tracks.sort(key=lambda track: track.track_id)
        return visible_tracks
