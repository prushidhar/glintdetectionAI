from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from glint_detector_ai.config import ActionConfig
from glint_detector_ai.evidence import EvidenceCaptureManager
from glint_detector_ai.models import BaristaDecision, FrameAnalysis, LoggedEvent, RiskAssessment, RiskLevel

try:
    import winsound
except ImportError:  # pragma: no cover - only available on Windows
    winsound = None


class EventLogger:
    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime("events-%Y%m%d-%H%M%S.jsonl")
        self.path = log_dir / filename

    def log_event(
        self,
        event_type: str,
        analysis: FrameAnalysis,
        assessment: RiskAssessment,
        decision: BaristaDecision,
        evidence: dict[str, object] | None = None,
    ) -> None:
        payload = LoggedEvent(
            event_type=event_type,
            timestamp=analysis.timestamp,
            frame_index=analysis.frame_index,
            risk_level=assessment.level.value,
            score=assessment.score,
            confidence=assessment.confidence,
            reason=assessment.reason,
            backend=analysis.backend,
            active_tracks=assessment.active_tracks,
            dominant_track_id=assessment.dominant_track_id,
            tracked_glints=analysis.tracked_glints,
            detections=analysis.detections,
            barista_service_mode=decision.service_mode,
            barista_priority=decision.priority.value,
            barista_focus_track_id=decision.focus_track_id,
            barista_zone_label=decision.zone_label,
            barista_incident_id=decision.incident_id,
            barista_incident_state=decision.incident_state,
            barista_incident_age_frames=decision.incident_age_frames,
            barista_smoothed_score=round(decision.smoothed_score, 3),
            barista_operator_message=decision.operator_message,
            barista_rationale=decision.rationale,
            barista_actions=decision.recommended_actions,
            evidence_id=(str(evidence["id"]) if evidence and evidence.get("id") else None),
            evidence_manifest_path=(
                str(evidence["manifest_path"])
                if evidence and evidence.get("manifest_path")
                else None
            ),
            evidence_frame_path=(
                str(evidence["frame_path"])
                if evidence and evidence.get("frame_path")
                else None
            ),
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload.to_dict()) + "\n")


class WebhookNotifier:
    def __init__(self, url: str, timeout_seconds: float) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    def send(
        self,
        event_type: str,
        analysis: FrameAnalysis,
        assessment: RiskAssessment,
        decision: BaristaDecision,
        evidence: dict[str, object] | None,
    ) -> None:
        payload = {
            "event_type": event_type,
            "timestamp": analysis.timestamp.isoformat(),
            "frame_index": analysis.frame_index,
            "risk_level": assessment.level.value,
            "score": round(assessment.score, 3),
            "confidence": round(assessment.confidence, 3),
            "reason": assessment.reason,
            "backend": analysis.backend,
            "active_tracks": assessment.active_tracks,
            "dominant_track_id": assessment.dominant_track_id,
            "barista_service_mode": decision.service_mode,
            "barista_priority": decision.priority.value,
            "barista_zone_label": decision.zone_label,
            "barista_incident_id": decision.incident_id,
            "barista_incident_state": decision.incident_state,
            "barista_smoothed_score": round(decision.smoothed_score, 3),
            "barista_message": decision.operator_message,
            "barista_actions": [action.kind for action in decision.recommended_actions],
            "evidence": evidence,
        }
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds):
                return
        except URLError as exc:
            print(f"Webhook delivery failed: {exc}")


class ActionCoordinator:
    def __init__(self, config: ActionConfig) -> None:
        self.config = config
        self.logger = EventLogger(config.log_dir)
        self.evidence = EvidenceCaptureManager(config.log_dir, config)
        self.webhook = (
            WebhookNotifier(config.webhook_url, config.webhook_timeout_seconds)
            if config.webhook_url
            else None
        )
        self.last_level = RiskLevel.SAFE
        self.last_high_alert_at: datetime | None = None
        self.last_evidence_at: datetime | None = None
        self.source_label = "unknown"

    def set_source_context(self, source_label: str, source_fps: float | None) -> None:
        self.source_label = source_label
        self.evidence.set_source_context(source_label, source_fps)

    def process(
        self,
        frame: np.ndarray,
        mask: np.ndarray | None,
        analysis: FrameAnalysis,
        assessment: RiskAssessment,
        decision: BaristaDecision,
    ) -> None:
        self.evidence.observe_frame(frame)
        level_changed = assessment.level != self.last_level
        should_repeat_high_alert = (not level_changed) and self._should_repeat_high_risk_alert(
            analysis.timestamp,
            assessment.level,
        )

        if level_changed:
            evidence = self._capture_evidence_if_needed(
                event_type="risk_state_changed",
                frame=frame,
                mask=mask,
                analysis=analysis,
                assessment=assessment,
                decision=decision,
            )
            self.logger.log_event(
                "risk_state_changed",
                analysis,
                assessment,
                decision,
                evidence=evidence,
            )
            self._send_webhook_if_needed(
                event_type="risk_state_changed",
                analysis=analysis,
                assessment=assessment,
                decision=decision,
                evidence=evidence,
            )
            print(
                f"[{analysis.timestamp:%H:%M:%S}] Risk changed to "
                f"{assessment.level.value.upper()} | score={assessment.score:.2f} | "
                f"backend={analysis.backend} | zone={decision.zone_label or '-'} | "
                f"incident={decision.incident_id or '-'} | tracks={assessment.active_tracks} | "
                f"mode={decision.service_mode.upper()} | {decision.operator_message}"
                f"{self._evidence_suffix(evidence)}"
            )

        if should_repeat_high_alert:
            evidence = self._capture_evidence_if_needed(
                event_type="high_risk_repeat",
                frame=frame,
                mask=mask,
                analysis=analysis,
                assessment=assessment,
                decision=decision,
            )
            self.logger.log_event(
                "high_risk_repeat",
                analysis,
                assessment,
                decision,
                evidence=evidence,
            )
            self._send_webhook_if_needed(
                event_type="high_risk_repeat",
                analysis=analysis,
                assessment=assessment,
                decision=decision,
                evidence=evidence,
            )
            print(
                f"[{analysis.timestamp:%H:%M:%S}] HIGH RISK sustained | "
                f"score={assessment.score:.2f} | dominant_track={assessment.dominant_track_id} | "
                f"incident={decision.incident_id or '-'} | zone={decision.zone_label or '-'} | "
                f"{decision.operator_message}"
                f"{self._evidence_suffix(evidence)}"
            )
            if decision.audible_alert:
                self._play_audio_alert()
            self.last_high_alert_at = analysis.timestamp

        if level_changed and decision.audible_alert:
            self._play_audio_alert()
            self.last_high_alert_at = analysis.timestamp

        self.last_level = assessment.level

    def _should_repeat_high_risk_alert(
        self,
        now: datetime,
        level: RiskLevel,
    ) -> bool:
        if level != RiskLevel.HIGH:
            return False
        if self.last_high_alert_at is None:
            return False
        elapsed = (now - self.last_high_alert_at).total_seconds()
        return elapsed >= self.config.repeat_high_risk_alert_seconds

    def _capture_evidence_if_needed(
        self,
        event_type: str,
        frame: np.ndarray,
        mask: np.ndarray | None,
        analysis: FrameAnalysis,
        assessment: RiskAssessment,
        decision: BaristaDecision,
    ) -> dict[str, object] | None:
        if not self.config.save_evidence:
            return None
        if assessment.level == RiskLevel.SAFE:
            return None
        if not decision.capture_evidence:
            return None
        if assessment.level == RiskLevel.LOW and not self.config.capture_low_risk_evidence:
            return None
        if self.last_evidence_at is not None:
            elapsed = (analysis.timestamp - self.last_evidence_at).total_seconds()
            if elapsed < self.config.evidence_cooldown_seconds:
                return None

        evidence_mask = mask if self.config.save_mask_image else None
        evidence = self.evidence.capture(
            event_type=event_type,
            frame=frame,
            mask=evidence_mask,
            analysis=analysis,
            assessment=assessment,
            decision=decision,
        )
        self.last_evidence_at = analysis.timestamp
        return evidence

    def _send_webhook_if_needed(
        self,
        event_type: str,
        analysis: FrameAnalysis,
        assessment: RiskAssessment,
        decision: BaristaDecision,
        evidence: dict[str, object] | None,
    ) -> None:
        if self.webhook is None:
            return
        if not decision.send_webhook:
            return
        enabled_levels = {level.casefold() for level in self.config.webhook_on_levels}
        if enabled_levels and assessment.level.value.casefold() not in enabled_levels:
            return
        self.webhook.send(event_type, analysis, assessment, decision, evidence)

    @staticmethod
    def _evidence_suffix(evidence: dict[str, object] | None) -> str:
        if not evidence or not evidence.get("id"):
            return ""
        return f" | evidence={evidence['id']}"

    def _play_audio_alert(self) -> None:
        if not self.config.enable_audio_alert or winsound is None:
            return
        winsound.Beep(1800, 250)
