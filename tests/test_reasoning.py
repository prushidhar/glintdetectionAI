from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from glint_detector_ai.config import ReasoningConfig
from glint_detector_ai.models import FrameAnalysis, GlintDetection, RiskLevel, TrackedGlint
from glint_detector_ai.reasoning import TemporalRiskAgent


def make_frame(
    frame_index: int,
    detections: list[GlintDetection],
    timestamp: datetime,
    tracked_glints: list[TrackedGlint] | None = None,
) -> FrameAnalysis:
    return FrameAnalysis(
        frame_index=frame_index,
        timestamp=timestamp,
        detections=detections,
        max_intensity=max((detection.intensity for detection in detections), default=0.0),
        tracked_glints=tracked_glints or [],
    )


class TemporalRiskAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = TemporalRiskAgent(ReasoningConfig(history_size=6, min_persistent_frames=3))
        self.start = datetime(2026, 1, 1, 12, 0, 0)

    def test_empty_history_remains_safe(self) -> None:
        assessment = self.agent.assess(make_frame(0, [], self.start))
        self.assertEqual(assessment.level, RiskLevel.SAFE)
        self.assertEqual(assessment.score, 0.0)

    def test_sporadic_detection_becomes_low_risk(self) -> None:
        analysis_frames = [
            make_frame(0, [], self.start),
            make_frame(
                1,
                [GlintDetection(x=100, y=80, area=8.0, intensity=252.0, circularity=0.9)],
                self.start + timedelta(seconds=1),
                tracked_glints=[
                    TrackedGlint(
                        track_id=1,
                        x=100,
                        y=80,
                        area=8.0,
                        intensity=252.0,
                        circularity=0.9,
                        label="lens_glint",
                        confidence=0.8,
                        age=1,
                        hits=1,
                        consecutive_hits=1,
                        missed_frames=0,
                        average_motion=0.0,
                        confirmed=False,
                    )
                ],
            ),
            make_frame(2, [], self.start + timedelta(seconds=2)),
            make_frame(
                3,
                [GlintDetection(x=160, y=120, area=9.0, intensity=251.0, circularity=0.8)],
                self.start + timedelta(seconds=3),
                tracked_glints=[
                    TrackedGlint(
                        track_id=2,
                        x=160,
                        y=120,
                        area=9.0,
                        intensity=251.0,
                        circularity=0.8,
                        label="lens_glint",
                        confidence=0.82,
                        age=1,
                        hits=1,
                        consecutive_hits=1,
                        missed_frames=0,
                        average_motion=0.0,
                        confirmed=False,
                    )
                ],
            ),
        ]

        assessment = None
        for frame in analysis_frames:
            assessment = self.agent.assess(frame)

        self.assertIsNotNone(assessment)
        self.assertEqual(assessment.level, RiskLevel.LOW)

    def test_consistent_cluster_reaches_high_risk(self) -> None:
        assessment = None
        for frame_index in range(6):
            detection = GlintDetection(
                x=200 + (frame_index % 2),
                y=110 + (frame_index % 2),
                area=12.0,
                intensity=254.0,
                circularity=0.92,
            )
            frame = make_frame(
                frame_index,
                [detection],
                self.start + timedelta(seconds=frame_index),
                tracked_glints=[
                    TrackedGlint(
                        track_id=1,
                        x=200 + (frame_index % 2),
                        y=110 + (frame_index % 2),
                        area=12.0,
                        intensity=254.0,
                        circularity=0.92,
                        label="lens_glint",
                        confidence=0.95,
                        age=frame_index + 1,
                        hits=frame_index + 1,
                        consecutive_hits=frame_index + 1,
                        missed_frames=0,
                        average_motion=1.0,
                        confirmed=frame_index >= 1,
                    )
                ],
            )
            assessment = self.agent.assess(frame)

        self.assertIsNotNone(assessment)
        self.assertEqual(assessment.level, RiskLevel.HIGH)
        self.assertGreaterEqual(assessment.persistent_tracks, 1)


if __name__ == "__main__":
    unittest.main()
