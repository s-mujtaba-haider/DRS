"""PDF report for one match review, built only from the stored evidence package.

Evidence frames are re-decoded from the stored videos and annotated with the geometry the
analysis used (crease line and marked points, rope and event positions), so the printed frame is
exactly the evidence the engine saw.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.app.services.reports import _fmt, _kv_table

MAX_FRAMES_PER_ANALYSIS = 3


def _frame(video_path: Path, index: int) -> np.ndarray | None:
    from ml.video.ingest import iter_frames

    rec = next(iter_frames(video_path, start_frame=index, max_frames=1), None)
    return None if rec is None else rec.image.copy()


def _line(img: np.ndarray, p1, p2, color) -> None:
    (x1, y1), (x2, y2) = p1, p2
    d = np.array([x2 - x1, y2 - y1], float)
    n = np.linalg.norm(d) or 1.0
    k = (img.shape[0] + img.shape[1]) / n
    a = (int(x1 - d[0] * k), int(y1 - d[1] * k))
    b = (int(x2 + d[0] * k), int(y2 + d[1] * k))
    cv2.line(img, a, b, color, max(1, img.shape[1] // 640), cv2.LINE_AA)


def _label(img: np.ndarray, text: str) -> None:
    s = max(0.5, img.shape[1] / 1600)
    cv2.putText(img, text, (10, int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)


def _png(img: np.ndarray) -> bytes:
    h, w = img.shape[:2]
    if w > 1280:
        img = cv2.resize(img, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", img)[1].tobytes()


STATE_BGR = {"IN": (128, 222, 74), "NOT_IN_BEYOND": (113, 113, 248), "AIRBORNE_BEHIND": (36, 191, 251), "UNKNOWN": (60, 146, 251)}


def crease_frames(analysis: dict, video_path: Path) -> list[tuple[str, bytes]]:
    """Frames around the wicket break with the crease line and marked bat/foot points."""
    req, res = analysis["request"], analysis["result"]
    line = res.get("crease_line_px")
    obs = {o["frame_index"]: o for o in req.get("observations", [])}
    states = {a["frame_index"]: a["state"] for a in (res.get("crease") or {}).get("assessments", [])}
    wb = req.get("wicket_break") or {}
    wanted = [f for f in (wb.get("last_intact_frame"), wb.get("first_broken_frame")) if f is not None]
    wanted += [f for f in sorted(obs) if f not in wanted]
    out = []
    for f in wanted[:MAX_FRAMES_PER_ANALYSIS]:
        img = _frame(video_path, f)
        if img is None:
            continue
        if line:
            _line(img, line[0], line[1], (248, 189, 56))
        o = obs.get(f)
        if o and o.get("point"):
            x, y = (int(round(v)) for v in o["point"])
            cv2.circle(img, (x, y), max(5, img.shape[1] // 200), STATE_BGR.get(states.get(f, ""), (255, 255, 255)), 2, cv2.LINE_AA)
        tag = " (last intact)" if f == wb.get("last_intact_frame") else " (first broken)" if f == wb.get("first_broken_frame") else ""
        ts = o["timestamp_ms"] if o else (wb.get("last_intact_ms") if f == wb.get("last_intact_frame") else wb.get("first_broken_ms"))
        cap = f"frame {f}{tag}" + (f" t={ts:.1f} ms" if ts is not None else "") + (f" {states[f]}" if f in states else "")
        _label(img, cap)
        out.append((cap, _png(img)))
    return out


def no_ball_frames(analysis: dict, video_path: Path) -> list[tuple[str, bytes]]:
    """Landing frames with the popping/return crease lines and the marked foot points."""
    req, res = analysis["request"], analysis["result"]
    marks = [("front", m) for m in req.get("front_marks", [])] + [("back", m) for m in req.get("back_marks", [])]
    out = []
    for which, m in marks[:MAX_FRAMES_PER_ANALYSIS]:
        img = _frame(video_path, m["frame_index"])
        if img is None:
            continue
        if res.get("crease_line_px"):
            _line(img, *res["crease_line_px"], (248, 189, 56))
        if res.get("return_crease_px"):
            _line(img, *res["return_crease_px"], (238, 211, 34))
        x, y = (int(round(v)) for v in m["point"])
        cv2.circle(img, (x, y), max(5, img.shape[1] // 200), (249, 121, 232) if which == "front" else (36, 191, 251), 2, cv2.LINE_AA)
        cap = f"frame {m['frame_index']} t={m['timestamp_ms']:.1f} ms: {which} foot ({res.get('status')})"
        _label(img, cap)
        out.append((cap, _png(img)))
    return out


def boundary_frames(analysis: dict, video_path: Path) -> list[tuple[str, bytes]]:
    req, res = analysis["request"], analysis["result"]
    rope = (req.get("rope") or {})
    pts = rope.get("points") or []
    decisive = [t for t in res.get("timeline", []) if t.get("decisive")]
    frames = [t.get("frame_index") for t in decisive if t.get("frame_index") is not None]
    frames += [e.get("frame_index") for e in req.get("events", []) if e.get("frame_index") is not None]
    out, seen = [], set()
    for f in frames:
        if f in seen or len(out) >= MAX_FRAMES_PER_ANALYSIS:
            continue
        seen.add(f)
        img = _frame(video_path, f)
        if img is None:
            continue
        if rope.get("frame", "image") == "image" and len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, np.int32)], False, (248, 189, 56), max(1, img.shape[1] // 640), cv2.LINE_AA)
        for e in req.get("events", []):
            if e.get("frame_index") != f:
                continue
            for key in ("position", "last_ground_contact"):
                g = e.get(key)
                if g and g.get("frame", "image") == "image":
                    x, y = (int(round(v)) for v in g["point"])
                    cv2.circle(img, (x, y), max(5, img.shape[1] // 200), (249, 121, 232), 2, cv2.LINE_AA)
        cap = f"frame {f}: " + ", ".join(e["kind"] for e in req.get("events", []) if e.get("frame_index") == f)
        _label(img, cap)
        out.append((cap, _png(img)))
    return out


def _checks_table(checks: list[dict], small) -> Table:
    rows = [["Check", "Law", "Status", "Evidence"]] + [
        [Paragraph(c.get("description", c.get("id", "")), small), c.get("law_reference", ""), c.get("status", ""),
         Paragraph(str(c.get("evidence", "")), small)] for c in checks]
    t = Table(rows, colWidths=[55 * mm, 24 * mm, 24 * mm, 67 * mm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    return t


def build_review_report(pkg: dict, video_paths: dict[str, Path]) -> bytes:
    buf = io.BytesIO()
    m, r, proto = pkg["match"], pkg["review"], pkg["protocol"]
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"Review {r['seq']} - {m['name']}")
    st = getSampleStyleSheet()
    small = st["BodyText"].clone("small", fontSize=8, leading=10)
    story = [Paragraph(f"Decision review #{r['seq']}: {r['decision_type'].replace('_', ' ')}", st["Title"]),
             Paragraph(f"{m['name']} - {m['team_a']} v {m['team_b']} - {m.get('venue') or ''} {m.get('match_date') or ''}", small),
             Paragraph(f"Generated {pkg['generated_at']} - software {pkg['software_version']}", small),
             Paragraph(f"<b>{pkg['disclaimer']}</b>", small), Spacer(1, 6)]

    story += [Paragraph("Review", st["Heading2"]), _kv_table([
        ("Innings / over", f"{r['innings']} / {r.get('over_ball') or 'n/a'}"),
        ("Requested by", "On-field umpire referral" if r["kind"] == "umpire" else r["requested_by"]),
        ("Batter / bowler", f"{r.get('batter') or 'n/a'} / {r.get('bowler') or 'n/a'}"),
        ("On-field decision", r["on_field"].replace("_", " ")),
        ("Status", r["status"]),
        ("Final decision (official)", (r.get("final_decision") or "pending").replace("_", " ")),
        ("Outcome", r.get("outcome") or "pending"),
        ("Decided by / at", f"{r.get('decided_by') or 'n/a'} / {r.get('decided_at') or 'n/a'}"),
        ("Protocol", f"{proto['name']} ({proto['id']} v{proto['version']}, effective {proto.get('effective_from')})"),
        ("Notes", r.get("notes") or ""),
    ])]

    rec = r.get("recommendation")
    if rec:
        story += [Paragraph("System recommendation (advisory)", st["Heading2"]), _kv_table([
            ("Outcome", rec.get("outcome")), ("Suggested decision", str(rec.get("final_decision", "")).replace("_", " ")),
            ("Review retained", _fmt(rec.get("review_retained"))), ("Computed", rec.get("computed_at", "")),
        ])]
        for title, items in (("Reasons", rec.get("reasons", [])), ("Warnings", rec.get("warnings", []))):
            if items:
                story.append(Paragraph(f"<b>{title}</b>", small))
                story += [Paragraph(f"- {x}", small) for x in items]
        comps = rec.get("components") or []
        if comps:
            rows = [["Component", "Status", "Umpire's call", "Evidence"]] + [
                [c.get("name", ""), c.get("status", ""), _fmt(c.get("umpires_call")), Paragraph(str(c.get("evidence", "")), small)] for c in comps]
            t = Table(rows, colWidths=[30 * mm, 35 * mm, 25 * mm, 80 * mm])
            t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [Spacer(1, 3), t]

    story += [Paragraph("Replays (provenance)", st["Heading2"])]
    for v in pkg["videos"]:
        meta, cap = v["meta"], v["meta"].get("capture") or {}
        story.append(_kv_table([
            ("File", v["original_filename"]), ("SHA-256", v["sha256"]),
            ("Resolution / fps", f"{meta.get('width')}x{meta.get('height')} @ {_fmt(meta.get('nominal_fps'), 2)}"),
            ("Frames / duration", f"{_fmt(meta.get('frame_count_decoded') or meta.get('frame_count_declared'))} / {_fmt(meta.get('duration_s'), 2)} s"),
            ("Capture", f"camera {cap.get('camera_id')}, clip group {cap.get('clip_group')}, offset {_fmt(cap.get('time_offset_ms'), 2)} ms, "
                        f"software sync +/-{_fmt(cap.get('sync_uncertainty_ms'), 1)} ms, missing frames {cap.get('missing_frames')}" if cap else "uploaded file (no capture timing)"),
        ]))
        story.append(Spacer(1, 3))

    for a in pkg["analyses"]:
        res, kind = a["result"], a["kind"]
        block = [Paragraph(f"Evidence: {kind.replace('_', ' ')} analysis {a['id']}", st["Heading2"])]
        status = res.get("status") or res.get("verdict")
        rows = [("Status", _fmt(status)), ("Engine", res.get("engine_version", ""))]
        if kind in ("run_out", "stumping"):
            cr = res.get("crease") or {}
            rows += [("Rules summary", res.get("rules_summary", "")),
                     ("Wicket broken within (ms)", _fmt(cr.get("break_window_ms"))),
                     ("Grounding window (ms)", _fmt(cr.get("grounding_window_ms"))),
                     ("Median frame gap (ms)", _fmt(cr.get("median_frame_gap_ms"), 1))]
        if kind == "lbw":
            rows += [("P(hit)", _fmt(res.get("p_hit"))), ("Reconstruction", _fmt(res.get("reconstruction_mode")))]
        block.append(_kv_table(rows))
        if res.get("rule_checks"):
            block += [Spacer(1, 3), _checks_table(res["rule_checks"], small)]
        for title, items in (("Reasons", res.get("reasons", [])), ("Warnings", res.get("warnings", []))):
            if items:
                block.append(Paragraph(f"<b>{title}</b>", small))
                block += [Paragraph(f"- {x}", small) for x in items]
        tl = res.get("timeline") or (res.get("crease") or {}).get("timeline") or []
        if tl:
            block.append(Paragraph("<b>Timeline</b>", small))
            for e in tl:
                t0 = e.get("t_start_ms", e.get("timestamp_ms"))
                block.append(Paragraph(f"- {_fmt(t0, 1)} ms: {e.get('description', '')}", small))
        story.append(KeepTogether(block[:3]))
        story += block[3:]
        vp = video_paths.get(a.get("video_id") or "")
        frames: list[tuple[str, bytes]] = []
        try:
            if vp is not None and kind in ("run_out", "stumping"):
                frames = crease_frames(a, vp)
            elif vp is not None and kind in ("boundary", "catch"):
                frames = boundary_frames(a, vp)
            elif vp is not None and kind == "no_ball":
                frames = no_ball_frames(a, vp)
        except Exception as exc:  # a missing frame never blocks the report
            story.append(Paragraph(f"Evidence frames unavailable: {exc}", small))
        for cap, png in frames:
            story += [Spacer(1, 4), Image(io.BytesIO(png), width=170 * mm, height=96 * mm, kind="proportional"), Paragraph(cap, small)]

    story += [Paragraph("Audit log", st["Heading2"])]
    for e in pkg["events"]:
        story.append(Paragraph(f"{e['created_at']} - {e['action']} by {e['actor']}", small))
    doc.build(story)
    return buf.getvalue()
