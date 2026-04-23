from __future__ import annotations

import unittest
from datetime import datetime

from glint_detector_ai.barista import AIBarista
from glint_detector_ai.config import BaristaConfig
from glint_detector_ai.models import (
    FrameAnalysis,
    GlintDetection,
    RiskAssessment,
    RiskLevel,
    ServicePriority,
    TrackedGlint,
)


def make_analysis(track_id: int | None = None) -> FrameAnalysis:
    tracked_glints = []
    if track_id is not None:
        tracked_glints.append(
            TrackedGlint(
                track_id=track_id,
                x=200,
                y=120,
                area=10.0,
                intensity=253.0,
                circularity=0.9,
                label="lens_glint",
                confidence=0.92,
                age=4,
                hits=4,
                consecutive_hits=4,
                missed_frames=0,
                average_motion=1.2,
                confirmed=True,
            )
        )
    return FrameAnalysis(
        frame_index=1,
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        detections=[GlintDetection(x=200, y=120, area=10.0, intensity=253.0, circularity=0.9)],
        max_intensity=253.0,
        backend="heuristic",
        tracked_glints=tracked_glints,
        frame_width=500,
        frame_height=300,
    )


def make_assessment(
    level: RiskLevel,
    score: float,
    confidence: float,
    dominant_track_id: int | None = None,
    dominant_track_persistence: float = 0.0,
) -> RiskAssessment:
    return RiskAssessment(
        level=level,
        score=score,
        confidence=confidence,
        reason="Synthetic assessment for testing.",
        detection_rate=0.5,
        persistent_tracks=1 if dominant_track_id is not None else 0,
        stability_score=0.8,
        active_tracks=1 if dominant_track_id is not None else 0,
        dominant_track_id=dominant_track_id,
        dominant_track_persistence=dominant_track_persistence,
    )


class AIBaristaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.barista = AIBarista(BaristaConfig())

    def test_safe_assessment_stays_in_watch_mode(self) -> None:
        decision = self.barista.serve(
            make_analysis(),
            make_assessment(RiskLevel.SAFE, score=0.1, confidence=0.4),
        )

        self.assertEqual(decision.service_mode, "watch")
        self.assertEqual(decision.priority, ServicePriority.ROUTINE)
        self.assertFalse(decision.capture_evidence)
        self.assertIn("calm", decision.operator_message.lower())
        self.assertEqual(decision.incident_state, "none")

    def test_low_risk_assessment_triggers_investigate_mode(self) -> None:
        decision = self.barista.serve(
            make_analysis(track_id=3),
            make_assessment(
                RiskLevel.LOW,
                score=0.48,
                confidence=0.72,
                dominant_track_id=3,
                dominant_track_persistence=0.6,
            ),
        )

        self.assertEqual(decision.service_mode, "investigate")
        self.assertEqual(decision.priority, ServicePriority.ATTENTION)
        self.assertTrue(decision.capture_evidence)
        self.assertFalse(decision.audible_alert)
        self.assertEqual(decision.focus_track_id, 3)
        self.assertEqual(decision.zone_label, "B3")
        self.assertEqual(decision.incident_id, "INC-0001")
        self.assertEqual(decision.incident_state, "active")

    def test_high_risk_assessment_triggers_intervention(self) -> None:
        self.barista.serve(
            make_analysis(track_id=2),
            make_assessment(
                RiskLevel.LOW,
                score=0.45,
                confidence=0.67,
                dominant_track_id=2,
                dominant_track_persistence=0.4,
            ),
        )
        decision = self.barista.serve(
            make_analysis(track_id=2),
            make_assessment(
                RiskLevel.HIGH,
                score=0.86,
                confidence=0.9,
                dominant_track_id=2,
                dominant_track_persistence=0.8,
            ),
        )

        self.assertEqual(decision.service_mode, "intervene")
        self.assertEqual(decision.priority, ServicePriority.URGENT)
        self.assertTrue(decision.capture_evidence)
        self.assertTrue(decision.audible_alert)
        self.assertTrue(decision.send_webhook)
        self.assertTrue(any(action.kind == "alert" for action in decision.recommended_actions))
        self.assertEqual(decision.incident_id, "INC-0001")
        self.assertEqual(decision.zone_label, "B3")


if __name__ == "__main__":
    unittest.main()
