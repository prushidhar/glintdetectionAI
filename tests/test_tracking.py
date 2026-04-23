from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from glint_detector_ai.config import TrackerConfig
from glint_detector_ai.models import FrameAnalysis, GlintDetection
from glint_detector_ai.tracking import GlintTracker


def make_analysis(
    frame_index: int,
    detections: list[GlintDetection],
    timestamp: datetime,
) -> FrameAnalysis:
    return FrameAnalysis(
        frame_index=frame_index,
        timestamp=timestamp,
        detections=detections,
        max_intensity=max((d.intensity for d in detections), default=0.0),
    )


class GlintTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = GlintTracker(
            TrackerConfig(max_match_distance=20.0, max_missed_frames=2, min_confirmed_hits=2)
        )
        self.start = datetime(2026, 1, 1, 12, 0, 0)

    def test_consistent_glint_keeps_same_track_id(self) -> None:
        first = self.tracker.update(
            make_analysis(
                0,
                [GlintDetection(x=100, y=80, area=8.0, intensity=252.0, circularity=0.9)],
                self.start,
            )
        )
        second = self.tracker.update(
            make_analysis(
                1,
                [GlintDetection(x=108, y=82, area=8.5, intensity=251.0, circularity=0.88)],
                self.start + timedelta(seconds=1),
            )
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertTrue(second[0].confirmed)

    def test_distant_glint_creates_new_track(self) -> None:
        self.tracker.update(
            make_analysis(
                0,
                [GlintDetection(x=100, y=80, area=8.0, intensity=252.0, circularity=0.9)],
                self.start,
            )
        )
        second = self.tracker.update(
            make_analysis(
                1,
                [GlintDetection(x=220, y=160, area=8.5, intensity=250.0, circularity=0.87)],
                self.start + timedelta(seconds=1),
            )
        )

        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].track_id, 2)

    def test_track_survives_short_dropout(self) -> None:
        self.tracker.update(
            make_analysis(
                0,
                [GlintDetection(x=140, y=100, area=10.0, intensity=253.0, circularity=0.91)],
                self.start,
            )
        )
        self.tracker.update(make_analysis(1, [], self.start + timedelta(seconds=1)))
        reappeared = self.tracker.update(
            make_analysis(
                2,
                [GlintDetection(x=147, y=104, area=10.2, intensity=252.0, circularity=0.89)],
                self.start + timedelta(seconds=2),
            )
        )

        self.assertEqual(len(reappeared), 1)
        self.assertEqual(reappeared[0].track_id, 1)


if __name__ == "__main__":
    unittest.main()
