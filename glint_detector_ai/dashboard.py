from __future__ import annotations

import argparse
import json
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from mimetypes import guess_type
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


def _load_jsonl_records(
    directory: object,
    pattern: str,
    limit: int | None = None,
) -> list[dict[str, object]]:
    if not directory.exists():
        return []

    records: list[dict[str, object]] = []
    for path in sorted(directory.glob(pattern), reverse=True):
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
        if limit is not None and len(records) >= limit:
            break

    records.reverse()
    return records


def load_events(log_dir: Path, limit: int | None = None) -> list[dict[str, object]]:
    return _load_jsonl_records(log_dir, "events-*.jsonl", limit=limit)


def load_evidence_index(evidence_dir: Path, limit: int | None = None) -> list[dict[str, object]]:
    return _load_jsonl_records(evidence_dir, "evidence-index.jsonl", limit=limit)


def summarize_events(
    events: list[dict[str, object]],
    evidence_items: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    evidence_items = evidence_items or []
    risk_counts = Counter(str(event.get("risk_level", "unknown")) for event in events)
    event_counts = Counter(str(event.get("event_type", "unknown")) for event in events)
    backend_counts = Counter(str(event.get("backend", "unknown")) for event in events)
    service_mode_counts = Counter(str(event.get("barista_service_mode", "unknown")) for event in events)
    zone_counts = Counter(
        str(event.get("barista_zone_label"))
        for event in events
        if event.get("barista_zone_label")
    )
    top_tracks = Counter()
    max_score = 0.0
    incident_map: dict[str, dict[str, object]] = {}

    for event in events:
        max_score = max(max_score, float(event.get("score", 0.0)))
        for tracked_glint in event.get("tracked_glints", []):
            track_id = tracked_glint.get("track_id")
            if track_id is not None:
                top_tracks[str(track_id)] += 1
        incident_id = event.get("barista_incident_id")
        if incident_id:
            incident_map[str(incident_id)] = {
                "incident_id": incident_id,
                "timestamp": event.get("timestamp"),
                "zone_label": event.get("barista_zone_label"),
                "service_mode": event.get("barista_service_mode"),
                "score": round(float(event.get("score", 0.0)), 3),
            }

    high_risk_evidence = sum(
        1 for item in evidence_items if str(item.get("risk_level", "")).casefold() == "high risk"
    )

    return {
        "total_events": len(events),
        "risk_counts": dict(risk_counts),
        "event_counts": dict(event_counts),
        "backend_counts": dict(backend_counts),
        "service_mode_counts": dict(service_mode_counts),
        "zone_counts": dict(zone_counts),
        "max_score": round(max_score, 3),
        "latest_event": events[-1] if events else None,
        "top_tracks": [
            {"track_id": track_id, "frames_seen": count}
            for track_id, count in top_tracks.most_common(5)
        ],
        "recent_incidents": list(incident_map.values())[-5:],
        "recent_timeline": [
            {
                "timestamp": event.get("timestamp"),
                "score": round(float(event.get("score", 0.0)), 3),
                "risk_level": event.get("risk_level", "unknown"),
            }
            for event in events[-20:]
        ],
        "total_evidence": len(evidence_items),
        "high_risk_evidence": high_risk_evidence,
        "latest_evidence": evidence_items[-1] if evidence_items else None,
    }


def _dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GlintDetectorAI Dashboard</title>
  <style>
    :root {
      --bg: #081018;
      --panel: rgba(12, 25, 38, 0.9);
      --panel-2: rgba(8, 17, 27, 0.9);
      --text: #ecf5ff;
      --muted: #9db2c7;
      --line: rgba(155, 191, 219, 0.16);
      --accent: #5fd1ff;
      --accent-2: #ffd96a;
      --safe: #64d28d;
      --low: #f0bd4f;
      --high: #ff6a5b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(95, 209, 255, 0.18), transparent 28%),
        radial-gradient(circle at right, rgba(255, 106, 91, 0.14), transparent 24%),
        linear-gradient(160deg, #050a0f, #081018 46%, #0d1824);
      min-height: 100vh;
    }
    main { width: min(1120px, calc(100vw - 32px)); margin: 24px auto; display: grid; gap: 16px; }
    .hero, .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 20px; }
    .hero { padding: 20px 22px; display: flex; justify-content: space-between; gap: 14px; flex-wrap: wrap; }
    h1 { margin: 0 0 8px; font-size: clamp(1.7rem, 4vw, 2.5rem); text-transform: uppercase; letter-spacing: 0.05em; }
    h2 { margin: 0 0 10px; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); }
    p { margin: 0; color: var(--muted); line-height: 1.5; }
    a { color: var(--accent); text-decoration: none; }
    .badge { padding: 10px 14px; border-radius: 999px; background: rgba(255,255,255,0.04); border: 1px solid var(--line); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
    .panel { padding: 18px; background: var(--panel-2); }
    .metric { font-size: 2rem; font-weight: 700; margin: 0 0 8px; }
    .status { display: inline-block; padding: 7px 12px; border-radius: 999px; text-transform: uppercase; font-weight: 700; }
    .safe { color: var(--safe); background: rgba(100, 210, 141, 0.14); }
    .low-risk { color: var(--low); background: rgba(240, 189, 79, 0.14); }
    .high-risk { color: var(--high); background: rgba(255, 106, 91, 0.14); }
    ul { list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }
    li { display: flex; justify-content: space-between; gap: 10px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.06); }
    li:last-child { border-bottom: 0; padding-bottom: 0; }
    .bars, .timeline { display: grid; gap: 10px; }
    .bar-row { display: grid; grid-template-columns: 84px 1fr 36px; gap: 10px; align-items: center; }
    .bar-shell { height: 10px; border-radius: 999px; background: rgba(255,255,255,0.08); overflow: hidden; }
    .bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), #a8f0ff); border-radius: inherit; }
    .timeline { display: flex; gap: 6px; align-items: end; height: 90px; }
    .timeline div { flex: 1; min-height: 5px; border-radius: 8px 8px 2px 2px; background: linear-gradient(180deg, rgba(95,209,255,0.9), rgba(95,209,255,0.2)); }
    .two-col { display: grid; gap: 16px; grid-template-columns: 1.2fr 0.8fr; }
    .evidence-grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
    .evidence-card { border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; overflow: hidden; background: rgba(255,255,255,0.03); }
    .evidence-card img { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: cover; background: rgba(255,255,255,0.05); }
    .evidence-meta { padding: 12px; display: grid; gap: 8px; }
    .thumb-strip { display: flex; gap: 8px; overflow-x: auto; }
    .thumb-strip img { width: 68px; height: 68px; aspect-ratio: 1; border-radius: 10px; object-fit: cover; }
    table { width: 100%; border-collapse: collapse; font-size: 0.93rem; }
    th, td { padding: 12px 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.07); vertical-align: top; }
    th { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .table-wrap { overflow-x: auto; }
    .tiny { font-size: 0.84rem; color: var(--muted); }
    @media (max-width: 880px) {
      .two-col { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) { table { min-width: 860px; } }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div>
        <h1>GlintDetectorAI Ops View</h1>
        <p>Local review surface for risk transitions, tracked glints, detector backend activity, and saved evidence artifacts for escalation or training review.</p>
      </div>
      <div class="badge" id="refresh-note">Waiting for data...</div>
    </section>
    <section class="grid">
      <section class="panel">
        <h2>Latest Risk</h2>
        <div class="metric" id="latest-score">0.00</div>
        <div id="latest-risk" class="status safe">safe</div>
        <p id="latest-reason">No events loaded yet.</p>
      </section>
      <section class="panel">
        <h2>Total Events</h2>
        <div class="metric" id="total-events">0</div>
        <ul id="event-counts"></ul>
      </section>
      <section class="panel">
        <h2>Risk Mix</h2>
        <div class="bars" id="risk-bars"></div>
      </section>
      <section class="panel">
        <h2>Backends</h2>
        <ul id="backend-counts"></ul>
      </section>
      <section class="panel">
        <h2>Top Tracks</h2>
        <ul id="top-tracks"></ul>
      </section>
      <section class="panel">
        <h2>Evidence</h2>
        <div class="metric" id="total-evidence">0</div>
        <ul id="evidence-stats"></ul>
      </section>
      <section class="panel">
        <h2>Service Modes</h2>
        <ul id="service-modes"></ul>
      </section>
      <section class="panel">
        <h2>Hot Zones</h2>
        <ul id="hot-zones"></ul>
      </section>
    </section>
    <section class="panel">
      <h2>Score Trend</h2>
      <div class="timeline" id="timeline"></div>
    </section>
    <section class="two-col">
      <section class="panel">
        <h2>Recent Events</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Risk</th>
                <th>Barista</th>
                <th>Score</th>
                <th>Backend</th>
                <th>Track</th>
                <th>Evidence</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody id="events-body"></tbody>
          </table>
        </div>
      </section>
      <section class="panel">
        <h2>Latest Evidence</h2>
        <div class="evidence-grid" id="evidence-grid"></div>
      </section>
    </section>
  </main>
  <script>
    const listTargets = ["event-counts", "backend-counts", "top-tracks"];

    function renderList(targetId, entries, formatter) {
      const target = document.getElementById(targetId);
      target.innerHTML = "";
      if (!entries.length) {
        target.innerHTML = "<li><span>No data yet</span><span></span></li>";
        return;
      }
      entries.forEach(entry => {
        const li = document.createElement("li");
        const [left, right] = formatter(entry);
        li.innerHTML = `<span>${left}</span><span>${right}</span>`;
        target.appendChild(li);
      });
    }

    function renderBars(values) {
      const target = document.getElementById("risk-bars");
      target.innerHTML = "";
      const entries = Object.entries(values);
      const total = entries.reduce((sum, [, count]) => sum + count, 0) || 1;
      if (!entries.length) {
        target.textContent = "No risk data yet.";
        return;
      }
      entries.forEach(([label, count]) => {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <span>${label}</span>
          <div class="bar-shell"><div class="bar-fill" style="width:${Math.max(6, Math.round((count / total) * 100))}%"></div></div>
          <span>${count}</span>
        `;
        target.appendChild(row);
      });
    }

    function renderTimeline(points) {
      const target = document.getElementById("timeline");
      target.innerHTML = "";
      if (!points.length) {
        target.innerHTML = "<div style='height:6px'></div>";
        return;
      }
      points.forEach(point => {
        const bar = document.createElement("div");
        const height = Math.max(8, Math.round((Number(point.score) || 0) * 84));
        bar.style.height = `${height}px`;
        bar.title = `${point.timestamp} | ${point.risk_level} | ${point.score}`;
        target.appendChild(bar);
      });
    }

    function renderEvents(events) {
      const body = document.getElementById("events-body");
      body.innerHTML = "";
      if (!events.length) {
        body.innerHTML = "<tr><td colspan='8'>No events found.</td></tr>";
        return;
      }
      events.slice().reverse().forEach(event => {
        const risk = String(event.risk_level || "safe");
        const klass = risk.replaceAll(" ", "-").toLowerCase();
        const serviceMode = String(event.barista_service_mode || "watch");
        const evidenceLink = event.evidence_frame_path
          ? `<a href="/artifacts/${event.evidence_frame_path}" target="_blank">frame</a>`
          : "<span class='tiny'>-</span>";
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${event.timestamp || ""}</td>
          <td><span class="status ${klass}">${risk}</span></td>
          <td>${serviceMode}</td>
          <td>${Number(event.score || 0).toFixed(2)}</td>
          <td>${event.backend || "unknown"}</td>
          <td>${event.dominant_track_id ?? "-"}</td>
          <td>${evidenceLink}</td>
          <td>${event.barista_operator_message || event.reason || ""}</td>
        `;
        body.appendChild(row);
      });
    }

    function renderEvidence(evidenceItems) {
      const target = document.getElementById("evidence-grid");
      target.innerHTML = "";
      if (!evidenceItems.length) {
        target.innerHTML = "<p class='tiny'>No saved evidence yet.</p>";
        return;
      }
      evidenceItems.slice().reverse().forEach(item => {
        const risk = String(item.risk_level || "unknown");
        const klass = risk.replaceAll(" ", "-").toLowerCase();
        const cropThumbs = (item.track_crops || []).map(crop =>
          `<a href="/artifacts/${crop.path}" target="_blank"><img src="/artifacts/${crop.path}" alt="Track crop ${crop.track_id}"></a>`
        ).join("");
        const card = document.createElement("article");
        card.className = "evidence-card";
        card.innerHTML = `
          <a href="/artifacts/${item.frame_path}" target="_blank">
            <img src="/artifacts/${item.frame_path}" alt="Evidence frame ${item.id}">
          </a>
          <div class="evidence-meta">
            <div style="display:flex;justify-content:space-between;gap:8px;align-items:center;">
              <span class="status ${klass}">${risk}</span>
              <a class="tiny" href="/artifacts/${item.manifest_path}" target="_blank">manifest</a>
            </div>
            <div class="tiny">${item.timestamp || ""}</div>
            <div>${item.reason || ""}</div>
            <div class="tiny">Track ${item.dominant_track_id ?? "-"} | Backend ${item.backend || "unknown"}</div>
            <div class="thumb-strip">${cropThumbs || "<span class='tiny'>No track crops</span>"}</div>
          </div>
        `;
        target.appendChild(card);
      });
    }

    async function refreshDashboard() {
      const [summaryRes, eventsRes, evidenceRes] = await Promise.all([
        fetch("/api/summary"),
        fetch("/api/events?limit=24"),
        fetch("/api/evidence?limit=8")
      ]);
      const summary = await summaryRes.json();
      const events = await eventsRes.json();
      const evidence = await evidenceRes.json();
      const latest = summary.latest_event || {};
      const latestEvidenceId = (summary.latest_evidence && summary.latest_evidence.id) || "-";
      const latestRisk = String(latest.risk_level || "safe");
      const latestRiskClass = latestRisk.replaceAll(" ", "-").toLowerCase();

      document.getElementById("refresh-note").textContent = `Last refresh ${new Date().toLocaleTimeString()}`;
      document.getElementById("total-events").textContent = String(summary.total_events || 0);
      document.getElementById("total-evidence").textContent = String(summary.total_evidence || 0);
      document.getElementById("latest-score").textContent = Number(latest.score || 0).toFixed(2);
      document.getElementById("latest-reason").textContent = latest.reason || "No events loaded yet.";

      const latestRiskNode = document.getElementById("latest-risk");
      latestRiskNode.textContent = latestRisk;
      latestRiskNode.className = `status ${latestRiskClass}`;

      renderList("event-counts", Object.entries(summary.event_counts || {}), entry => [entry[0], entry[1]]);
      renderList("backend-counts", Object.entries(summary.backend_counts || {}), entry => [entry[0], entry[1]]);
      renderList("top-tracks", summary.top_tracks || [], entry => [`Track #${entry.track_id}`, `${entry.frames_seen} frames`]);
      renderList("evidence-stats", [
        ["High-risk", summary.high_risk_evidence || 0],
        ["Latest", latestEvidenceId]
      ], entry => [entry[0], entry[1]]);
      renderList("service-modes", Object.entries(summary.service_mode_counts || {}), entry => [entry[0], entry[1]]);
      renderList("hot-zones", Object.entries(summary.zone_counts || {}).slice(0, 5), entry => [entry[0], entry[1]]);
      renderBars(summary.risk_counts || {});
      renderTimeline(summary.recent_timeline || []);
      renderEvents(events);
      renderEvidence(evidence);
    }

    refreshDashboard().catch(error => {
      document.getElementById("latest-reason").textContent = `Dashboard error: ${error}`;
    });
    setInterval(() => refreshDashboard().catch(() => {}), 3000);
  </script>
</body>
</html>"""


def _safe_artifact_path(relative_path: str, root: Path) -> Path | None:
    if not relative_path:
        return None
    normalized = Path(unquote(relative_path))
    if normalized.is_absolute():
        return None
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return candidate


def create_handler(log_dir: Path, evidence_dir: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_dashboard_html())
                return
            if parsed.path == "/api/summary":
                events = load_events(log_dir, limit=500)
                evidence_items = load_evidence_index(evidence_dir, limit=200)
                self._send_json(summarize_events(events, evidence_items))
                return
            if parsed.path == "/api/events":
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", ["30"])[0])
                self._send_json(load_events(log_dir, limit=limit))
                return
            if parsed.path == "/api/evidence":
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", ["12"])[0])
                self._send_json(load_evidence_index(evidence_dir, limit=limit))
                return
            if parsed.path.startswith("/artifacts/"):
                artifact_path = _safe_artifact_path(
                    parsed.path.removeprefix("/artifacts/"),
                    evidence_dir,
                )
                if artifact_path is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "Artifact not found")
                    return
                self._send_file(artifact_path)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: object) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_file(self, path: Path) -> None:
            body = path.read_bytes()
            content_type, _ = guess_type(path.name)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the GlintDetectorAI runtime dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    parser.add_argument(
        "--log-dir",
        default="runtime_logs",
        help="Directory containing JSONL runtime event logs.",
    )
    parser.add_argument(
        "--evidence-dir",
        default=None,
        help="Directory containing evidence artifacts. Defaults to runtime_logs/evidence.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir)
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else log_dir / "evidence"
    server = ThreadingHTTPServer((args.host, args.port), create_handler(log_dir, evidence_dir))
    print(f"Dashboard available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Dashboard stopped.")
    finally:
        server.server_close()
    return 0
