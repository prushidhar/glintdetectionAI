from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from glint_detector_ai.dashboard import load_events, load_evidence_index, summarize_events


@dataclass(order=True)
class FakeLogFile:
    name: str
    content: str

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.content


class FakeLogDir:
    def __init__(self, files: list[FakeLogFile]) -> None:
        self._files = files

    def exists(self) -> bool:
        return True

    def glob(self, pattern: str) -> list[FakeLogFile]:
        if pattern == "events-*.jsonl":
            return [file for file in self._files if file.name.startswith("events-")]
        if pattern == "evidence-index.jsonl":
            return [file for file in self._files if file.name == "evidence-index.jsonl"]
        return self._files


class DashboardSummaryTests(unittest.TestCase):
    def test_load_and_summarize_events(self) -> None:
        payloads = [
            {
                "event_type": "risk_state_changed",
                "timestamp": "2026-01-01T12:00:00",
                "frame_index": 10,
                "risk_level": "low risk",
                "score": 0.44,
                "confidence": 0.76,
                "reason": "Emerging pattern.",
                "backend": "heuristic",
                "active_tracks": 1,
                "dominant_track_id": 1,
                "barista_service_mode": "investigate",
                "barista_zone_label": "B2",
                "barista_incident_id": "INC-0001",
                "tracked_glints": [{"track_id": 1}],
                "detections": [{"x": 100, "y": 80}],
            },
            {
                "event_type": "high_risk_repeat",
                "timestamp": "2026-01-01T12:00:05",
                "frame_index": 15,
                "risk_level": "high risk",
                "score": 0.88,
                "confidence": 0.91,
                "reason": "Persistent track.",
                "backend": "yolo:glint",
                "active_tracks": 1,
                "dominant_track_id": 1,
                "barista_service_mode": "intervene",
                "barista_zone_label": "B2",
                "barista_incident_id": "INC-0001",
                "tracked_glints": [{"track_id": 1}],
                "detections": [{"x": 102, "y": 82}],
            },
        ]
        fake_dir = FakeLogDir(
            [
                FakeLogFile(
                    name="events-20260101-120000.jsonl",
                    content="\n".join(json.dumps(payload) for payload in payloads) + "\n",
                ),
                FakeLogFile(
                    name="evidence-index.jsonl",
                    content=json.dumps(
                        {
                            "id": "capture-1",
                            "timestamp": "2026-01-01T12:00:05",
                            "risk_level": "high risk",
                            "backend": "yolo:glint",
                            "frame_path": "capture-1/frame.jpg",
                            "manifest_path": "capture-1/manifest.json",
                            "track_crops": [{"track_id": 1, "path": "capture-1/track-1.jpg"}],
                        }
                    )
                    + "\n",
                ),
            ]
        )
        events = load_events(fake_dir, limit=10)  # type: ignore[arg-type]
        evidence = load_evidence_index(fake_dir, limit=10)  # type: ignore[arg-type]
        summary = summarize_events(events, evidence)

        self.assertEqual(len(events), 2)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(summary["total_events"], 2)
        self.assertEqual(summary["risk_counts"]["high risk"], 1)
        self.assertEqual(summary["backend_counts"]["heuristic"], 1)
        self.assertEqual(summary["latest_event"]["risk_level"], "high risk")
        self.assertEqual(summary["top_tracks"][0]["track_id"], "1")
        self.assertEqual(summary["total_evidence"], 1)
        self.assertEqual(summary["high_risk_evidence"], 1)
        self.assertEqual(summary["service_mode_counts"]["intervene"], 1)
        self.assertEqual(summary["zone_counts"]["B2"], 2)
        self.assertEqual(summary["recent_incidents"][0]["incident_id"], "INC-0001")


if __name__ == "__main__":
    unittest.main()
