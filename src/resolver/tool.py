"""Agent tool contract (6.3): one JSON request in, one JSON response out.

    {"action": "resolve", "clip": "clip.mp4", "prompt": "...", "reference": null, "session": "out/"}
    {"action": "refine",  "session": "out/", "instruction": "more comic, less brass"}
    {"action": "pick",    "session": "out/", "rank": 2, "overrides": {"hit": 7.2, "speech": [[1.0, 3.2]]}}
    {"action": "skip",    "session": "out/", "reason": "none fit"}
    {"action": "own",     "session": "out/", "track": "song.mp3", "rights_confirmed": true}
    {"action": "generate","session": "out/", "approved_cost": 0.5}

resolve, refine, skip and an unapproved generate answer with a ResolveResult;
pick, own and an approved generate with {"report": VerifyReport, "outputs": {...}}.
Errors never raise out of `handle`: they come back as status "error".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

from . import config as config_mod
from . import session as session_mod
from .review import write_review
from .schemas import ResolveResult

ACTIONS = ("resolve", "refine", "pick", "skip", "own", "generate")


def _need(req: dict, *keys: str) -> None:
    missing = [k for k in keys if req.get(k) in (None, "")]
    if missing:
        raise ValueError(f"missing field(s): {', '.join(missing)}")


def _overrides(raw: dict | None) -> dict:
    raw = dict(raw or {})
    out = {}
    if raw.get("hit") is not None:
        out["hit"] = float(raw["hit"])
    if raw.get("speech") is not None:
        out["speech"] = [(float(a), float(b)) for a, b in raw["speech"]]
    if raw.get("role") is not None:
        out["role"] = raw["role"]
    return out


def _report(report, paths: dict) -> dict:
    return {"status": report.status, "report": report.model_dump(mode="json"), "outputs": paths}


def _result(result: ResolveResult, session: str | None) -> dict:
    data = result.model_dump(mode="json")
    if session and (Path(session) / "session.json").exists():
        data["review_page"] = str(write_review(session))
    return data


def handle(req: dict, ctx: session_mod.Context | None = None) -> dict:
    try:
        action = req.get("action")
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; one of {', '.join(ACTIONS)}")
        if ctx is None:
            from .app import make_context
            cfg = config_mod.load(req.get("config"))
            ctx, _ = make_context(cfg, offline=bool(req.get("offline")), previews=req.get("previews", True))
        s = req.get("session")
        if action == "resolve":
            _need(req, "clip", "prompt", "session")
            return _result(session_mod.resolve(req["clip"], req["prompt"], s, ctx, reference=req.get("reference")), s)
        _need(req, "session")
        if action == "refine":
            _need(req, "instruction")
            return _result(session_mod.refine(s, req["instruction"], ctx), s)
        if action == "pick":
            _need(req, "rank")
            return _report(*session_mod.pick(s, int(req["rank"]), ctx, _overrides(req.get("overrides"))))
        if action == "skip":
            return _result(session_mod.skip(s, ctx, req.get("reason") or "skipped by user"), s)
        if action == "own":
            _need(req, "track")
            if req.get("rights_confirmed") is not True:
                return {"status": "needs_input", "message": "Confirm you have the rights to this file: "
                                                            "resend with \"rights_confirmed\": true (L7)."}
            return _report(*session_mod.own(s, req["track"], ctx, True, _overrides(req.get("overrides"))))
        # generate: never charges unless approved_cost equals the quote.
        out = session_mod.generate(s, ctx, approved_cost=req.get("approved_cost"))
        if isinstance(out, BaseModel):
            return _result(out, s)
        return _report(*out)
    except Exception as err:  # noqa: BLE001  (the contract: always answer with JSON)
        return {"status": "error", "message": f"{type(err).__name__}: {err}"}


def main(stdin=sys.stdin, stdout=sys.stdout) -> int:
    try:
        req = json.loads(stdin.read())
    except json.JSONDecodeError as err:
        stdout.write(json.dumps({"status": "error", "message": f"invalid JSON: {err}"}) + "\n")
        return 3
    resp = handle(req)
    stdout.write(json.dumps(resp, indent=2) + "\n")
    return {"ok": 0, "verify_failed": 1, "no_match": 2, "error": 3}.get(resp.get("status"), 0)
