"""Match-day review management: matches, reviews, evidence, protocol recommendation and audit log.

The system never decides a review. It attaches evidence analyses, computes an advisory
recommendation under the selected (versioned) DRS protocol, and records the decision a
human official enters. Every state change is appended to ``review_events``.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.api.deps import db, get_or_404, require_umpire, settings, to_dict
from backend.app.core.config import Settings
from backend.app.database.models import DecisionAnalysis, LbwAnalysis, Match, Review, ReviewEvent, Video

router = APIRouter(prefix="/api", tags=["matches"])

DecisionType = Literal["lbw", "caught", "run_out", "stumping", "boundary", "no_ball", "other"]
EVIDENCE_TYPES = {"lbw": LbwAnalysis, "decision": DecisionAnalysis}
# Which stored analysis kinds may support which decision type.
EVIDENCE_FOR = {"lbw": {"lbw", "contact", "audio_contact", "no_ball"}, "caught": {"contact", "catch", "audio_contact", "no_ball"},
                "run_out": {"run_out", "wicket_break"}, "stumping": {"stumping", "wicket_break", "no_ball"},
                "no_ball": {"no_ball"}, "boundary": {"boundary", "catch"}, "other": None}


@lru_cache
def _software_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).resolve().parents[3]).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _log(s: Session, review_id: str, action: str, payload: dict | None = None, actor: str = "operator") -> None:
    s.add(ReviewEvent(review_id=review_id, action=action, actor=actor, payload=payload or {}))


def _protocol(pid: str):
    from ml.protocol import load_protocol

    try:
        return load_protocol(pid)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(422, f"Unknown DRS protocol '{pid}'") from exc


def _counts(s: Session, m: Match) -> dict:
    from ml.protocol import review_counts

    proto = _protocol(m.protocol_id)
    rows = s.query(Review).filter(Review.match_id == m.id, Review.status == "decided",
                                  Review.innings == m.current_innings).all()
    counts = review_counts(proto, [{"team": r.requested_by, "innings": r.innings, "kind": r.kind, "outcome": r.outcome}
                                   for r in rows])
    empty = {"unsuccessful_used": 0, "remaining": proto.reviews.unsuccessful_reviews_per_innings, "retained_umpires_call": 0}
    return {"current_innings": m.current_innings,
            "teams": {team: counts.get(team, empty) for team in (m.team_a, m.team_b)}}


def _match_dict(s: Session, m: Match) -> dict:
    d = to_dict(m)
    d["reviews"] = _counts(s, m)
    return d


# ------------------------------------------------------------------ protocols
@router.get("/protocols")
def protocols():
    from ml.protocol import list_protocols

    return list_protocols()


@router.get("/protocols/{protocol_id}")
def protocol(protocol_id: str):
    return _protocol(protocol_id).model_dump(mode="json")


# -------------------------------------------------------------------- matches
class MatchCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    team_a: str = Field(min_length=1, max_length=100)
    team_b: str = Field(min_length=1, max_length=100)
    protocol_id: str
    venue: str | None = Field(None, max_length=200)
    match_date: str | None = Field(None, max_length=32)
    batting_team: str | None = None
    officials: dict[str, str] = Field(default_factory=dict)


class MatchUpdate(BaseModel):
    status: Literal["live", "completed", "abandoned"] | None = None
    current_innings: int | None = Field(None, ge=1, le=4)
    batting_team: str | None = None


@router.post("/matches", status_code=201)
def create_match(body: MatchCreate, s: Session = Depends(db)):
    if body.team_a.strip().lower() == body.team_b.strip().lower():
        raise HTTPException(422, "Teams must differ")
    _protocol(body.protocol_id)
    if body.batting_team and body.batting_team not in (body.team_a, body.team_b):
        raise HTTPException(422, "batting_team must be one of the two teams")
    m = Match(**body.model_dump())
    s.add(m)
    s.commit()
    return _match_dict(s, m)


@router.get("/matches")
def list_matches(s: Session = Depends(db)):
    return [_match_dict(s, m) for m in s.query(Match).order_by(Match.created_at.desc()).all()]


@router.get("/matches/{match_id}")
def get_match(match_id: str, s: Session = Depends(db)):
    return _match_dict(s, get_or_404(s, Match, match_id))


@router.patch("/matches/{match_id}")
def update_match(match_id: str, body: MatchUpdate, s: Session = Depends(db)):
    m = get_or_404(s, Match, match_id)
    if body.batting_team is not None and body.batting_team not in (m.team_a, m.team_b):
        raise HTTPException(422, "batting_team must be one of the two teams")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(m, k, v)
    s.commit()
    return _match_dict(s, m)


# -------------------------------------------------------------------- reviews
class ReviewCreate(BaseModel):
    requested_by: str = Field(description="team name for a player review, or 'umpire' for an umpire referral")
    decision_type: DecisionType
    on_field: Literal["OUT", "NOT_OUT", "NONE"]
    over_ball: str | None = Field(None, pattern=r"^\d{1,3}\.[0-9]$")
    batter: str | None = Field(None, max_length=100)
    bowler: str | None = Field(None, max_length=100)
    video_ids: list[str] = Field(default_factory=list)
    notes: str | None = Field(None, max_length=2000)
    actor: str = Field("operator", max_length=100)


@router.post("/matches/{match_id}/reviews", status_code=201)
def create_review(match_id: str, body: ReviewCreate, s: Session = Depends(db)):
    m = get_or_404(s, Match, match_id)
    if m.status != "live":
        raise HTTPException(409, f"Match is {m.status}")
    kind = "umpire" if body.requested_by == "umpire" else "player"
    if kind == "player":
        if body.requested_by not in (m.team_a, m.team_b):
            raise HTTPException(422, "requested_by must be a team in this match or 'umpire'")
        if body.on_field == "NONE":
            raise HTTPException(422, "A player review challenges an on-field decision (OUT or NOT_OUT)")
        if body.decision_type in ("boundary", "no_ball"):
            raise HTTPException(422, f"{body.decision_type.replace('_', ' ')} checks are umpire referrals, not player reviews")
        rules = _protocol(m.protocol_id).reviews
        if not rules.player_reviews_allowed:
            raise HTTPException(409, f"Protocol {m.protocol_id} does not allow player reviews")
        if body.decision_type in rules.not_reviewable_by_players:
            raise HTTPException(422, f"{body.decision_type} decisions are not reviewable by players under this protocol")
        left = _counts(s, m)["teams"][body.requested_by]["remaining"]
        if left is not None and left <= 0:
            raise HTTPException(409, f"{body.requested_by} has no reviews remaining this innings")
        if s.query(Review).filter(Review.match_id == m.id, Review.status == "open",
                                  Review.requested_by == body.requested_by).count():
            raise HTTPException(409, f"{body.requested_by} already has an open review")
    for vid in body.video_ids:
        get_or_404(s, Video, vid)
    seq = s.query(Review).filter(Review.match_id == m.id).count() + 1
    r = Review(match_id=m.id, seq=seq, innings=m.current_innings, kind=kind,
               **body.model_dump(exclude={"actor"}))
    s.add(r)
    s.flush()
    _log(s, r.id, "created", body.model_dump(mode="json"), body.actor)
    s.commit()
    return to_dict(r)


@router.get("/matches/{match_id}/reviews")
def list_reviews(match_id: str, s: Session = Depends(db)):
    get_or_404(s, Match, match_id)
    return [to_dict(r) for r in s.query(Review).filter(Review.match_id == match_id).order_by(Review.seq).all()]


@router.get("/reviews/{review_id}")
def get_review(review_id: str, s: Session = Depends(db)):
    r = get_or_404(s, Review, review_id)
    d = to_dict(r)
    d["events"] = [to_dict(e) for e in s.query(ReviewEvent).filter(ReviewEvent.review_id == r.id)
                   .order_by(ReviewEvent.created_at).all()]
    return d


def _open(s: Session, review_id: str) -> Review:
    r = get_or_404(s, Review, review_id)
    if r.status != "open":
        raise HTTPException(409, f"Review is {r.status}")
    return r


class VideoLink(BaseModel):
    video_id: str
    actor: str = "operator"


@router.post("/reviews/{review_id}/videos")
def add_review_video(review_id: str, body: VideoLink, s: Session = Depends(db)):
    r = _open(s, review_id)
    get_or_404(s, Video, body.video_id)
    if body.video_id not in r.video_ids:
        r.video_ids = [*r.video_ids, body.video_id]
        _log(s, r.id, "video_added", {"video_id": body.video_id}, body.actor)
        s.commit()
    return to_dict(r)


class EvidenceLink(BaseModel):
    type: Literal["lbw", "decision"]
    id: str
    actor: str = "operator"


def _load_evidence(s: Session, type_: str, eid: str):
    obj = s.get(EVIDENCE_TYPES[type_], eid)
    if obj is None:
        raise HTTPException(404, f"{type_} analysis {eid} not found")
    kind = "lbw" if type_ == "lbw" else obj.kind
    return obj, kind


@router.post("/reviews/{review_id}/evidence")
def attach_evidence(review_id: str, body: EvidenceLink, s: Session = Depends(db)):
    r = _open(s, review_id)
    _, kind = _load_evidence(s, body.type, body.id)
    allowed = EVIDENCE_FOR[r.decision_type]
    if allowed is not None and kind not in allowed:
        raise HTTPException(422, f"A {kind} analysis is not evidence for a {r.decision_type} review")
    if not any(e["id"] == body.id for e in r.evidence):
        r.evidence = [*r.evidence, {"type": body.type, "id": body.id, "kind": kind,
                                    "attached_at": datetime.now(UTC).isoformat()}]
        _log(s, r.id, "evidence_attached", {"type": body.type, "id": body.id, "kind": kind}, body.actor)
        s.commit()
    return to_dict(r)


def _evidence_status(decision_type: str, result: dict) -> tuple[str, list[str]]:
    """Translate an evidence engine's status into the protocol's evidence vocabulary."""
    status = str(result.get("status") or "INCONCLUSIVE")
    notes: list[str] = []
    if decision_type in ("run_out", "stumping"):
        summary = result.get("rules_summary")
        if summary == "CONDITIONS_NOT_MET":
            failed = [c["id"] for c in result.get("rule_checks", []) if c.get("status") == "FAIL"]
            return "NOT_OUT", [f"Dismissal conditions not met: {', '.join(failed)}"]
        if status == "OUT_OF_GROUND" and summary != "ALL_CONDITIONS_MET":
            unknown = [c["id"] for c in result.get("rule_checks", []) if c.get("status") == "UNKNOWN"]
            notes.append(f"Batter out of ground, but these conditions are unconfirmed: {', '.join(unknown)}")
        return status, notes
    if decision_type == "boundary":
        return {"BOUNDARY_SIX": "SIX", "BOUNDARY_FOUR": "FOUR"}.get(status, status), notes
    if decision_type == "no_ball":
        # A no-ball check on a dismissal: a fair delivery lets the dismissal stand (OUT), a no ball does not.
        return {"FAIR": "OUT", "NO_BALL": "NOT_OUT"}.get(status, status), notes
    if decision_type == "caught":
        notes.append("Catch analysis covers fielding and boundary conditions only; bat contact is judged separately")
        return {"CATCH_VALID": "CAUGHT", "CATCH_INVALID": "NOT_CAUGHT"}.get(status, status), notes
    return status, notes


PRIMARY_KIND = {"lbw": "lbw", "run_out": "run_out", "stumping": "stumping", "boundary": "boundary", "caught": "catch",
                "no_ball": "no_ball"}
# A no ball means the striker cannot be out LBW, caught, stumped or bowled (Laws 21.18, 32-36, 39).
NO_BALL_NULLIFIES = {"lbw", "caught", "stumping", "no_ball"}


@router.post("/reviews/{review_id}/recommend")
def recommend(review_id: str, s: Session = Depends(db)):
    """Advisory recommendation under the match's DRS protocol from the newest primary evidence."""
    from ml.protocol import lbw_review, simple_review

    r = _open(s, review_id)
    m = s.get(Match, r.match_id)
    proto = _protocol(m.protocol_id)
    want = PRIMARY_KIND.get(r.decision_type)
    prim = [e for e in r.evidence if e["kind"] == want] if want else []
    if not prim:
        rec = {"outcome": "INCONCLUSIVE", "final_decision": "REFER_TO_THIRD_UMPIRE", "review_retained": None,
               "reasons": [f"No {want or r.decision_type} analysis attached; the third umpire decides from the replays"],
               "warnings": [], "protocol_id": proto.id}
    else:
        obj, _ = _load_evidence(s, prim[-1]["type"], prim[-1]["id"])
        if r.decision_type == "lbw":
            if r.on_field == "NONE":
                raise HTTPException(422, "LBW recommendation needs the on-field decision")
            rec = lbw_review(proto, r.on_field, obj.result, review_kind=r.kind).model_dump(mode="json")
        else:
            status, notes = _evidence_status(r.decision_type, obj.result)
            try:
                rec = simple_review(proto, r.on_field, status, review_type=r.decision_type,
                                    review_kind=r.kind).model_dump(mode="json")
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            rec["warnings"] = [*rec.get("warnings", []), *notes]
        rec["evidence_id"] = obj.id
        if r.decision_type == "boundary":
            rec["boundary_status"] = obj.result.get("status")
    nb = [e for e in r.evidence if e["kind"] == "no_ball"]
    if nb and r.decision_type in NO_BALL_NULLIFIES and r.decision_type != "no_ball":
        nb_obj, _ = _load_evidence(s, nb[-1]["type"], nb[-1]["id"])
        nb_status = nb_obj.result.get("status")
        if nb_status == "NO_BALL":
            rec["final_decision"] = "NOT_OUT"
            rec["outcome"] = "OVERTURNED" if r.on_field == "OUT" else "UPHELD"
            rec["reasons"] = [*rec.get("reasons", []), "No ball (Law 21.5): the striker cannot be out "
                              f"{r.decision_type.replace('_', ' ')} off a no ball"]
        elif nb_status != "FAIR":
            rec["warnings"] = [*rec.get("warnings", []), f"No-ball check is {nb_status}: the umpire must decide it"]
    rec.update(advisory=True, computed_at=datetime.now(UTC).isoformat(), software_version=_software_version())
    r.recommendation = rec
    _log(s, r.id, "recommendation", rec, "system")
    s.commit()
    return to_dict(r)


class DecisionIn(BaseModel):
    final_decision: Literal["OUT", "NOT_OUT"] | None = Field(None, description="None for boundary reviews")
    outcome: Literal["UPHELD", "OVERTURNED", "UMPIRES_CALL"]
    boundary_result: Literal["SIX", "FOUR", "NO_BOUNDARY"] | None = None
    decided_by: str = Field(min_length=1, max_length=100)
    notes: str | None = Field(None, max_length=2000)


@router.post("/reviews/{review_id}/decide", dependencies=[Depends(require_umpire)])
def decide(review_id: str, body: DecisionIn, s: Session = Depends(db)):
    r = _open(s, review_id)
    if r.decision_type == "boundary":
        if body.boundary_result is None:
            raise HTTPException(422, "boundary_result is required for a boundary review")
    elif body.final_decision is None:
        raise HTTPException(422, "final_decision is required")
    if r.on_field in ("OUT", "NOT_OUT") and body.final_decision:
        same = body.final_decision == r.on_field
        if body.outcome == "OVERTURNED" and same:
            raise HTTPException(422, "OVERTURNED requires a final decision different from the on-field decision")
        if body.outcome in ("UPHELD", "UMPIRES_CALL") and not same:
            raise HTTPException(422, f"{body.outcome} requires the on-field decision to stand")
    if body.outcome == "UMPIRES_CALL":
        if r.decision_type != "lbw":
            raise HTTPException(422, "Umpire's call applies to LBW ball-tracking reviews only")
        if not _protocol(s.get(Match, r.match_id).protocol_id).lbw.umpires_call_enabled:
            raise HTTPException(422, "This protocol has no umpire's call")
    r.final_decision = body.final_decision or body.boundary_result
    r.outcome, r.decided_by, r.status, r.decided_at = body.outcome, body.decided_by, "decided", datetime.now(UTC)
    if body.notes:
        r.notes = ((r.notes + "\n") if r.notes else "") + body.notes
    _log(s, r.id, "decided", body.model_dump(mode="json"), body.decided_by)
    s.commit()
    return to_dict(r)


class Withdraw(BaseModel):
    reason: str = Field(min_length=1, max_length=500)
    actor: str = "operator"


@router.post("/reviews/{review_id}/withdraw", dependencies=[Depends(require_umpire)])
def withdraw(review_id: str, body: Withdraw, s: Session = Depends(db)):
    r = _open(s, review_id)
    r.status = "withdrawn"
    _log(s, r.id, "withdrawn", body.model_dump(), body.actor)
    s.commit()
    return to_dict(r)


def _package(s: Session, r: Review) -> dict:
    m = s.get(Match, r.match_id)
    analyses = []
    for e in r.evidence:
        obj = s.get(EVIDENCE_TYPES[e["type"]], e["id"])
        if obj is not None:
            vid = getattr(obj, "video_id", None)
            if vid is None and e["type"] == "lbw":
                from backend.app.database.models import Job

                job = s.get(Job, obj.job_id)
                vid = job.video_id if job else None
            analyses.append({**e, "video_id": vid, "request": obj.request, "result": obj.result})
    videos = [{"id": v.id, "original_filename": v.original_filename, "sha256": v.sha256, "meta": v.meta}
              for v in (s.get(Video, vid) for vid in r.video_ids) if v is not None]
    return {"package_version": 1, "generated_at": datetime.now(UTC).isoformat(), "software_version": _software_version(),
            "disclaimer": "Experimental research system. The recommendation is advisory; the recorded final "
                          "decision was entered by a human official.",
            "match": to_dict(m), "protocol": _protocol(m.protocol_id).model_dump(mode="json"),
            "review": to_dict(r), "videos": videos, "analyses": analyses,
            "events": [to_dict(e) for e in s.query(ReviewEvent).filter(ReviewEvent.review_id == r.id)
                       .order_by(ReviewEvent.created_at).all()]}


@router.get("/reviews/{review_id}/package")
def evidence_package(review_id: str, s: Session = Depends(db)):
    """Self-contained evidence package: review, protocol, analyses, video provenance and audit log."""
    return _package(s, get_or_404(s, Review, review_id))


@router.get("/reviews/{review_id}/report.pdf")
def review_report(review_id: str, s: Session = Depends(db), cfg: Settings = Depends(settings)):
    """PDF review report with annotated evidence frames (built from the stored package only)."""
    from backend.app.services.review_report import build_review_report

    r = get_or_404(s, Review, review_id)
    pkg = _package(s, r)
    ids = {a["video_id"] for a in pkg["analyses"] if a.get("video_id")}
    paths = {v.id: cfg.videos_dir / v.stored_name for v in (s.get(Video, i) for i in ids) if v is not None}
    pdf = build_review_report(pkg, paths)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="review_{r.seq}_{r.id}.pdf"'})
