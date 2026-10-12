"""PDF analysis report (reportlab). Only measured / stored values are printed."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ml.lbw.engine import DISCLAIMER


def _kv_table(rows: list[tuple[str, str]]) -> Table:
    t = Table([[k, v] for k, v in rows], colWidths=[55 * mm, 115 * mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#444444")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
    ]))
    return t


def _fmt(v, nd=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_job_report(job: dict, video: dict, job_dir: Path, lbw: dict | None, frame_png: bytes | None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"DRS research report {job['id']}")
    st = getSampleStyleSheet()
    small = st["BodyText"].clone("small", fontSize=8, leading=10)
    story = [Paragraph("Cricket DRS research prototype - analysis report", st["Title"]),
             Paragraph(f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} - job {job['id']}", small),
             Paragraph(f"<b>{DISCLAIMER}</b>", small), Spacer(1, 6)]

    meta = video.get("meta", {})
    story += [Paragraph("Video", st["Heading2"]), _kv_table([
        ("File", video.get("original_filename", "")),
        ("Resolution", f"{meta.get('width')}x{meta.get('height')}"),
        ("Declared fps", _fmt(meta.get("nominal_fps"))),
        ("Duration (s)", _fmt(meta.get("duration_s"))),
        ("Codec / container", f"{meta.get('codec')} / {meta.get('container_format')}"),
        ("Variable frame rate", _fmt(meta.get("variable_frame_rate"))),
        ("Metadata warnings", "; ".join(meta.get("warnings", [])) or "none"),
    ])]

    s = job.get("summary") or {}
    det = s.get("detector", {})
    pt = s.get("primary_track") or {}
    tr2 = s.get("trajectory_2d") or {}
    story += [Paragraph("Processing", st["Heading2"]), _kv_table([
        ("Pipeline version", s.get("pipeline_version", "")),
        ("Detector", f"{det.get('model_id')} (cricket-specific: {det.get('cricket_specific')})"),
        ("Detector notes", "; ".join(det.get("notes", []))),
        ("Settings", json.dumps({k: v for k, v in (s.get("config") or {}).items() if k != "tracker"})),
        ("Frames processed", _fmt(s.get("frames_processed"))),
        ("Frames with candidates", _fmt(s.get("frames_with_candidates"))),
        ("Tracks", _fmt(s.get("num_tracks"))),
        ("Primary track observed / spanned", f"{pt.get('num_observed')} / {pt.get('num_frames_spanned')}" if pt else "no track"),
        ("Longest gap (frames)", _fmt(pt.get("longest_gap"))),
        ("2D trajectory model", f"{tr2.get('model')} status={tr2.get('status')} rms={_fmt(tr2.get('rms_px'), 2)}px "
                                f"LOO={_fmt(tr2.get('loo_rms_px'), 2)}px" if tr2 else "n/a"),
        ("Throughput (frames/s)", f"detect {_fmt((s.get('throughput_fps') or {}).get('detect'), 1)}, end-to-end "
                                  f"{_fmt((s.get('throughput_fps') or {}).get('end_to_end'), 1)}"),
        ("Validation", s.get("validation_status", "")),
        ("Warnings", "; ".join(s.get("warnings", [])) or "none"),
    ])]

    if frame_png:
        story += [Spacer(1, 6), Image(io.BytesIO(frame_png), width=170 * mm, height=95 * mm, kind="proportional")]

    if lbw:
        r = lbw["result"]
        wp = r.get("wicket_projection") or {}
        cr = (wp.get("crossing") or {})
        story += [Paragraph("Experimental LBW analysis", st["Heading2"]), _kv_table([
            ("Geometric verdict", r["verdict"]),
            ("P(ball would hit wicket)", _fmt(r.get("p_hit"))),
            ("Rules summary", r["rules_summary"]),
            ("Reconstruction mode", _fmt(r.get("reconstruction_mode"))),
            ("Calibration validated", _fmt(r.get("calibration_validated"))),
            ("Impact time (s) / source", f"{_fmt(r.get('impact_time_s'))} / {r.get('impact_time_source')}"),
            ("Stump-plane crossing x,z (m)", f"{cr.get('mean')} +/- {cr.get('std')}" if cr else "n/a"),
            ("Projection distance (m)", _fmt(wp.get("projection_distance_m"))),
            ("Engine", r.get("engine_version", "")),
        ])]
        rows = [["Check", "Law", "Status", "P", "Evidence"]] + [
            [c["description"], c["law_reference"], c["status"], _fmt(c.get("probability")), Paragraph(c["evidence"], small)]
            for c in r["rule_checks"]]
        t = Table(rows, colWidths=[52 * mm, 22 * mm, 22 * mm, 12 * mm, 62 * mm])
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
        story += [Spacer(1, 4), t, Spacer(1, 4)]
        for title, items in (("Reasons", r.get("reasons", [])), ("Warnings", r.get("warnings", []))):
            if items:
                story.append(Paragraph(f"<b>{title}</b>", small))
                story += [Paragraph(f"- {x}", small) for x in items]

    story += [Paragraph("Limitations", st["Heading2"])]
    for lim in (
        "Detection and tracking have not been validated against labelled ground truth for this video.",
        "Single-camera 3D reconstruction relies on physical assumptions (gravity, drag, bounce priors) and on the stated calibration.",
        "Real-world accuracy of this system is unknown; synthetic benchmarks (docs/EVALUATION.md) do not establish it.",
    ):
        story.append(Paragraph(f"- {lim}", small))
    doc.build(story)
    return buf.getvalue()
