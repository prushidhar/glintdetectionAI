from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from glint_detector_ai.config import BaristaConfig
from glint_detector_ai.models import (
    BaristaAction,
    BaristaDecision,
    FrameAnalysis,
    RiskAssessment,
    RiskLevel,
    ServicePriority,
)


@dataclass(slots=True)
class IncidentCase:
    incident_id: str
    started_frame: int
    last_seen_frame: int
    zone_label: str | None
    focus_track_id: int | None
    highest_score: float


class AIBarista:
    """Service-oriented decision layer inspired by the AI Barista pattern."""

    def __init__(self, config: BaristaConfig) -> None:
        self.config = config
        self.history: deque[RiskAssessment] = deque(maxlen=config.history_size)
        self._smoothed_score = 0.0
        self._incident_counter = 0
        self._active_incident: IncidentCase | None = None

    def serve(self, analysis: FrameAnalysis, assessment: RiskAssessment) -> BaristaDecision:
        self.history.append(assessment)
        self._smoothed_score = self._smooth_score(assessment.score)
        focus_track_id = self._resolve_focus_track(analysis, assessment)
        zone_label = self._zone_label(analysis, focus_track_id)
        trend = self._trend()
        service_mode = self._service_mode(assessment)
        incident = self._update_incident(
            frame_index=analysis.frame_index,
            focus_track_id=focus_track_id,
            zone_label=zone_label,
            service_mode=service_mode,
            score=max(assessment.score, self._smoothed_score),
        )
        priority = self._priority(service_mode)
        recommended_actions = self._recommended_actions(
            analysis=analysis,
            assessment=assessment,
            service_mode=service_mode,
            trend=trend,
            zone_label=zone_label,
            incident=incident,
        )
        operator_message = self._operator_message(
            service_mode=service_mode,
            focus_track_id=focus_track_id,
            zone_label=zone_label,
            incident=incident,
        )
        rationale = self._rationale(
            assessment=assessment,
            trend=trend,
            zone_label=zone_label,
            incident=incident,
        )

        return BaristaDecision(
            service_mode=service_mode,
            priority=priority,
            focus_track_id=focus_track_id,
            zone_label=zone_label,
            incident_id=incident.incident_id if incident else None,
            incident_state=self._incident_state(service_mode, incident),
            incident_age_frames=(
                (analysis.frame_index - incident.started_frame + 1) if incident else 0
            ),
            smoothed_score=self._smoothed_score,
            operator_message=operator_message,
            rationale=rationale,
            recommended_actions=recommended_actions,
            capture_evidence=service_mode in {"investigate", "intervene"} or trend == "rising",
            audible_alert=service_mode == "intervene",
            send_webhook=service_mode == "intervene",
        )

    def _smooth_score(self, current_score: float) -> float:
        alpha = min(1.0, max(0.0, self.config.score_smoothing_alpha))
        if not self.history or len(self.history) == 1:
            return current_score
        return (alpha * current_score) + ((1.0 - alpha) * self._smoothed_score)

    def _service_mode(self, assessment: RiskAssessment) -> str:
        if assessment.level == RiskLevel.HIGH or self._smoothed_score >= self.config.escalate_score_threshold:
            return "intervene"
        if (
            assessment.level == RiskLevel.LOW
            or self._smoothed_score >= self.config.investigate_score_threshold
            or (
                assessment.active_tracks > 0
                and assessment.confidence >= self.config.low_risk_attention_confidence
            )
        ):
            return "investigate"
        return "watch"

    def _priority(self, service_mode: str) -> ServicePriority:
        if service_mode == "intervene":
            return ServicePriority.URGENT
        if service_mode == "investigate":
            return ServicePriority.ATTENTION
        return ServicePriority.ROUTINE

    def _trend(self) -> str:
        if len(self.history) < 2:
            return "new"
        latest = self.history[-1]
        previous = self.history[-2]
        if latest.score > previous.score + 0.08:
            return "rising"
        if latest.score < previous.score - 0.08:
            return "cooling"
        return "steady"

    def _resolve_focus_track(self, analysis: FrameAnalysis, assessment: RiskAssessment) -> int | None:
        if assessment.dominant_track_id is not None:
            return assessment.dominant_track_id
        if not analysis.tracked_glints:
            return None
        return max(
            analysis.tracked_glints,
            key=lambda track: (track.confirmed, track.confidence, track.hits),
        ).track_id

    def _zone_label(self, analysis: FrameAnalysis, focus_track_id: int | None) -> str | None:
        if focus_track_id is None or analysis.frame_width <= 0 or analysis.frame_height <= 0:
            return None
        track = next(
            (tracked for tracked in analysis.tracked_glints if tracked.track_id == focus_track_id),
            None,
        )
        if track is None:
            return None

        zone_rows = max(1, self.config.zone_rows)
        zone_cols = max(1, self.config.zone_cols)
        row_index = min(zone_rows - 1, int((track.y / analysis.frame_height) * zone_rows))
        col_index = min(zone_cols - 1, int((track.x / analysis.frame_width) * zone_cols))
        row_label = chr(ord("A") + row_index)
        return f"{row_label}{col_index + 1}"

    def _update_incident(
        self,
        frame_index: int,
        focus_track_id: int | None,
        zone_label: str | None,
        service_mode: str,
        score: float,
    ) -> IncidentCase | None:
        if self._active_incident and (
            frame_index - self._active_incident.last_seen_frame
            > self.config.incident_cooldown_frames
        ):
            self._active_incident = None

        if service_mode == "watch":
            return self._active_incident

        if (
            self._active_incident is None
            or (
                zone_label is not None
                and self._active_incident.zone_label is not None
                and zone_label != self._active_incident.zone_label
                and frame_index - self._active_incident.last_seen_frame > 2
            )
        ):
            self._incident_counter += 1
            self._active_incident = IncidentCase(
                incident_id=f"INC-{self._incident_counter:04d}",
                started_frame=frame_index,
                last_seen_frame=frame_index,
                zone_label=zone_label,
                focus_track_id=focus_track_id,
                highest_score=score,
            )
        else:
            self._active_incident.last_seen_frame = frame_index
            self._active_incident.focus_track_id = focus_track_id
            self._active_incident.zone_label = zone_label or self._active_incident.zone_label
            self._active_incident.highest_score = max(self._active_incident.highest_score, score)

        return self._active_incident

    def _incident_state(self, service_mode: str, incident: IncidentCase | None) -> str:
        if incident is None:
            return "none"
        if service_mode == "watch":
            return "cooling"
        return "active"

    def _recommended_actions(
        self,
        analysis: FrameAnalysis,
        assessment: RiskAssessment,
        service_mode: str,
        trend: str,
        zone_label: str | None,
        incident: IncidentCase | None,
    ) -> list[BaristaAction]:
        zone_text = zone_label or "unknown zone"
        actions: list[BaristaAction] = [
            BaristaAction(
                kind="observe",
                label="Monitor Feed",
                detail=f"Keep watching backend {analysis.backend} and activity near zone {zone_text}.",
            )
        ]

        if incident is not None:
            actions.append(
                BaristaAction(
                    kind="case",
                    label="Maintain Incident",
                    detail=f"Keep incident {incident.incident_id} attached to follow-up actions.",
                )
            )

        if service_mode in {"investigate", "intervene"}:
            actions.append(
                BaristaAction(
                    kind="capture",
                    label="Preserve Evidence",
                    detail="Save annotated frames, masks, track crops, and rolling clip context.",
                )
            )

        if service_mode == "investigate":
            actions.append(
                BaristaAction(
                    kind="review",
                    label="Request Quiet Review",
                    detail=f"Ask staff to inspect aisle and seat block near zone {zone_text} discreetly.",
                )
            )

        if service_mode == "intervene":
            actions.append(
                BaristaAction(
                    kind="alert",
                    label="Dispatch Staff",
                    detail=f"Escalate immediately to floor staff for zone {zone_text}.",
                )
            )
            actions.append(
                BaristaAction(
                    kind="remote",
                    label="Push Incident Alert",
                    detail="Send webhook payload and evidence bundle to remote operators.",
                )
            )

        if trend == "rising":
            actions.append(
                BaristaAction(
                    kind="trend",
                    label="Accelerate Response",
                    detail="Risk is rising; shorten manual review and prepare intervention.",
                )
            )

        if assessment.active_tracks > 1:
            actions.append(
                BaristaAction(
                    kind="scan",
                    label="Expand Scan Radius",
                    detail="Multiple tracks are active; verify adjacent seats for supporting evidence.",
                )
            )

        return actions

    def _operator_message(
        self,
        service_mode: str,
        focus_track_id: int | None,
        zone_label: str | None,
        incident: IncidentCase | None,
    ) -> str:
        zone_text = f" near zone {zone_label}" if zone_label else ""
        incident_text = f" [{incident.incident_id}]" if incident else ""
        track_text = f" on track #{focus_track_id}" if focus_track_id is not None else ""

        if service_mode == "watch":
            return f"Barista{incident_text}: scene looks calm, continue passive monitoring{zone_text}."
        if service_mode == "investigate":
            return (
                f"Barista{incident_text}: suspicious reflective behavior is building"
                f"{track_text}{zone_text}. Review discreetly while evidence is preserved."
            )
        return (
            f"Barista{incident_text}: likely recording attempt{track_text}{zone_text}. "
            "Escalate to staff now and keep evidence attached."
        )

    def _rationale(
        self,
        assessment: RiskAssessment,
        trend: str,
        zone_label: str | None,
        incident: IncidentCase | None,
    ) -> str:
        zone_text = zone_label or "unknown zone"
        incident_text = incident.incident_id if incident else "no active incident"
        return (
            f"Trend is {trend}; raw score {assessment.score:.2f}, smoothed score {self._smoothed_score:.2f}, "
            f"confidence {assessment.confidence:.2f}, {assessment.persistent_tracks} persistent track(s), "
            f"zone {zone_text}, incident {incident_text}, "
            f"dominant persistence {assessment.dominant_track_persistence:.2f}."
        )
