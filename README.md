# GlintDetectorAI

GlintDetectorAI is an MVP for a theatre anti-piracy system that watches a live camera feed, detects likely lens glints, and uses multi-frame reasoning to classify the situation as `safe`, `low risk`, or `high risk`.

The codebase follows a perception -> reasoning -> action pipeline:

- Perception: pluggable detection backends with a heuristic OpenCV detector today and YOLO hooks for trained models later.
- Reasoning: a temporal agent that scores consistency, stability, and frequency across tracked glints over recent frames.
- Action: an AI Barista-style decision layer that translates risk into service modes, operator guidance, escalation steps, and event logging.

## Features

- Webcam or video-file real-time capture
- Bright reflective spot detection using grayscale thresholding, CLAHE contrast enhancement, and contour filtering
- Multi-frame risk classification instead of single-frame triggers
- Multi-object glint tracking with stable track IDs across frames
- AI Barista orchestration that turns detections into `watch`, `investigate`, or `intervene` decisions
- Incident management with stable case IDs, cooling windows, and smoothed risk memory
- Theatre zone mapping with hotspot labels such as `A1`, `B3`, or `C5`
- On-screen overlays for detections and risk state
- Event logging for state changes and repeated high-risk alerts
- Automatic evidence capture with raw frames, annotated frames, masks, tracked-glint crops, and rolling context clips
- Optional webhook delivery for alert payloads
- Local dashboard and log viewer over the generated JSONL runtime events, incidents, hotspots, and evidence artifacts
- YOLO-ready detection hooks for future trained-model integration
- Rule-based baseline that can later be replaced with YOLO, IR sensors, or cloud reasoning

## Project structure

```text
app.py
dashboard.py
glint_detector_ai/
  actions.py
  barista.py
  config.py
  dashboard.py
  evidence.py
  models.py
  perception.py
  pipeline.py
  reasoning.py
  tracking.py
tests/
  test_barista.py
  test_dashboard.py
  test_reasoning.py
  test_tracking.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional YOLO support:

```bash
pip install -r requirements-optional.txt
```

## Run

```bash
python app.py
```

Optional flags:

```bash
python app.py --camera-index 0 --show-mask
python app.py --brightness-threshold 245 --min-area 3 --max-area 220
python app.py --headless
python app.py --video-path recordings/theatre-feed.mp4 --frame-stride 2
python app.py --processing-scale 0.6 --max-aspect-ratio 2.0
python app.py --detector-backend yolo --yolo-model-path models/lens_glint.pt
python app.py --zone-rows 4 --zone-cols 8 --incident-cooldown-frames 24
python app.py --barista-investigate-threshold 0.38 --barista-escalate-threshold 0.74
python app.py --clip-context-frames 32 --clip-fps 10
python app.py --webhook-url https://example.com/alerts --disable-low-risk-evidence
```

Controls:

- Press `q` to quit the live viewer.

## Dashboard

Run the local dashboard against the generated event logs:

```bash
python dashboard.py --port 8765
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765) in your browser.

The dashboard shows:

- latest risk level and score
- event counts by type
- risk distribution
- service-mode distribution
- backend usage
- top tracked glints
- recent event table and score trend
- hot theatre zones and incident-aware summaries
- saved evidence gallery with manifest and crop links

## Tracking

The tracker assigns a persistent ID to each glint candidate using nearest-neighbor matching between frames. This gives the agent better temporal context than raw contours alone:

- track persistence: whether the same glint keeps reappearing
- confirmation hits: whether a track is consistent enough to trust
- motion stability: whether the glint remains spatially stable
- missed-frame tolerance: short dropouts do not immediately destroy a track

Useful tuning flags:

```bash
python app.py --max-track-distance 45 --max-missed-frames 3 --tracker-confirmation-hits 2
```

## AI Barista Layer

The project now includes an AI Barista-style service layer that sits between raw risk scoring and final actions. It behaves like an operations lead:

- `watch`: keep observing and avoid noisy interventions
- `investigate`: preserve evidence and prompt discreet staff review
- `intervene`: escalate to operators, sound alerts, and notify remote systems

Each decision includes:

- priority level
- focus track
- incident ID and incident age
- zone label inside the auditorium grid
- smoothed risk score
- operator-facing guidance
- recommended actions
- flags for evidence capture, audible alerts, and webhook delivery

## Evidence And Alerts

When the system sees a suspicious state transition or sustained high-risk pattern, it can automatically save:

- the full frame
- an annotated Barista frame with incident metadata
- the threshold mask
- a short rolling context clip
- crops around the strongest tracked glints
- a JSON manifest describing the alert context

Artifacts are stored under `runtime_logs/evidence/` by default.

Useful alerting flags:

```bash
python app.py --disable-evidence
python app.py --evidence-dir captured_evidence --max-track-crops 5
python app.py --disable-annotated-evidence --disable-context-clip
python app.py --webhook-url https://example.com/alerts
```

## How the risk agent works

The reasoning stage keeps a rolling history of recent frames and inspects tracked glints instead of isolated detections. It looks for:

- Frequency: how often detections appear in the recent window
- Persistence: how many frames the same tracked glint stays active
- Stability: whether the glint remains spatially consistent instead of jumping randomly
- Confirmation: whether a track has enough repeated hits to be reliable

These signals are combined into a normalized score:

- `safe`: no meaningful pattern yet
- `low risk`: sporadic or emerging consistency
- `high risk`: stable, recurring glints across multiple frames

This is intentionally a deterministic MVP so the full system behavior is explainable while the team gathers training data.

## Event logs

Runtime logs are written to `runtime_logs/` as JSONL files. Each event includes:

- timestamp
- frame index
- risk level and score
- reason string
- detector backend
- dominant track ID
- Barista service mode, zone, and incident fields
- tracked glint snapshots
- optional evidence IDs and artifact paths
- detected glint coordinates and metadata

## Verification

Core reasoning, tracking, and dashboard summaries can be checked without OpenCV:

```bash
python -m unittest discover -s tests
```

Syntax can be validated with:

```bash
python -m compileall .
```

## Next steps

- Add IR camera support for dark-auditorium deployments
- Send alerts to a cloud backend or mobile app
- Store clips or thumbnails around high-risk events for operator review
- Train a theatre-specific YOLO model on real lens-glint examples
