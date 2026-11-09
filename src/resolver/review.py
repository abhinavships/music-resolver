"""Static review page per session (7.7): brief, three previews, reasons, licences, ratings.

No server: ratings are saved by a "Download ratings" button as JSON. The eval
mode hides condition labels and shuffles the order per rater.
"""

from __future__ import annotations

import html
import json
import os
import random
from pathlib import Path

from .schemas import Brief, Candidate, ResolveResult

CSS = """
:root{--bg:#f7f6f3;--fg:#1d1d1b;--muted:#6b6a65;--card:#fff;--line:#e3e1db;--accent:#2f6f4f;--warn:#9a5b00}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a19f97;--card:#20201e;--line:#34332f;--accent:#6fc79a;--warn:#e0a84a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}
.muted{color:var(--muted)}.brief{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}
.brief dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:8px 0 0}.brief dt{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:8px}
video{width:100%;border-radius:6px;background:#000;aspect-ratio:16/9}.title{font-weight:600}
.lic{font-size:13px;color:var(--accent)}.warn{font-size:13px;color:var(--warn)}
.stars{display:flex;gap:6px}.stars label{border:1px solid var(--line);border-radius:6px;padding:2px 10px;cursor:pointer}
.stars input{display:none}.stars input:checked+span{font-weight:700;color:var(--accent)}
button{background:var(--accent);color:var(--bg);border:0;border-radius:8px;padding:10px 16px;font-size:15px;cursor:pointer;margin-top:16px}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}.note{margin:4px 0}
"""


def _e(x) -> str:
    return html.escape(str(x))


def _brief_block(brief: Brief | None, refine_cmd: str) -> str:
    if brief is None:
        return ""
    rows = []
    ref = brief.named_reference
    if ref:
        conf = f" (confidence {ref.confidence:.2f})" if ref.confidence < 0.6 else ""
        rows.append(("Reference", f"{_e(ref.canonical or ref.raw)}{conf}, original not available"))
    for cue in brief.cues():
        tempo = f", {cue.tempo_bpm[0]}-{cue.tempo_bpm[1]} BPM" if cue.tempo_bpm else ""
        rows.append((_e(cue.label), f"{_e(', '.join(cue.mood))}; {_e(', '.join(cue.instruments))}; "
                                    f"{cue.energy} energy{tempo}; {cue.mix_role}"))
    if brief.deferred_cues:
        rows.append(("Deferred", _e("; ".join(brief.deferred_cues))))
    if brief.ignored_instructions:
        rows.append(("Not music", _e("; ".join(brief.ignored_instructions))))
    dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    notes = "".join(f'<p class="note muted">{_e(n)}</p>' for n in brief.notes_for_user)
    return (f'<section class="brief"><strong>Brief</strong> <span class="muted">({brief.mode})</span>'
            f"<dl>{dl}</dl>{notes}<p class=\"muted\">Change it with <code>{_e(refine_cmd)}</code></p></section>")


def _card(c: Candidate, slot: str, page_dir: Path, hide_labels: bool) -> str:
    src = os.path.relpath(c.preview_path, page_dir) if c.preview_path else ""
    video = f'<video controls preload="metadata" src="{_e(src)}"></video>' if src else '<p class="muted">no preview</p>'
    label = "" if hide_labels else f'<div class="muted">#{c.rank} &middot; {_e(c.cue_label)}</div>'
    head = (f'<div class="title">Option {_e(slot)}</div>' if hide_labels else
            f'<div class="title">{_e(c.track.title)} <span class="muted">by {_e(c.track.artist)}</span></div>')
    details = "" if hide_labels else (
        f'<div>{_e(c.reason)}</div><div class="lic">{_e(c.track.licence.upper())}'
        f'{" &middot; " + _e(c.attribution) if c.attribution else ""}</div>'
        + "".join(f'<div class="warn">{_e(w)}</div>' for w in c.warnings))
    stars = "".join(f'<label><input type="radio" name="r-{_e(slot)}" value="{v}"><span>{v}</span></label>'
                    for v in range(1, 6))
    return (f'<article class="card" data-slot="{_e(slot)}" data-track="{_e(c.track.track_id)}">'
            f"{head}{label}{video}{details}<div class=\"stars\" aria-label=\"rating 1 to 5\">{stars}</div></article>")


SCRIPT = """
const meta = JSON.parse(document.getElementById('meta').textContent);
document.getElementById('dl').addEventListener('click', () => {
  const ratings = [...document.querySelectorAll('.card')].map(card => {
    const picked = card.querySelector('input:checked');
    return {slot: card.dataset.slot, track_id: card.dataset.track, rating: picked ? Number(picked.value) : null};
  });
  const rater = document.getElementById('rater').value.trim() || null;
  const data = {...meta, rater, saved_at: new Date().toISOString(), ratings};
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `ratings-${meta.session}${rater ? '-' + rater : ''}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  window.__lastRatings = data;
});
"""


def render_page(result: ResolveResult, page_dir: Path, session: str, prompt: str,
                hide_labels: bool = False, order: list[int] | None = None, extra_meta: dict | None = None) -> str:
    cands = result.candidates
    if order is not None:
        cands = [cands[i] for i in order]
    slots = [chr(ord("A") + i) for i in range(len(cands))]
    cards = "".join(_card(c, s, page_dir, hide_labels) for c, s in zip(cands, slots))
    meta = {"session": session, "prompt": prompt, "slots": {s: c.track.track_id for s, c in zip(slots, cands)},
            **(extra_meta or {})}
    brief = "" if hide_labels else _brief_block(result.brief, f'resolver refine {session} "more comic"')
    message = f'<p class="muted">{_e(result.message)}</p>' if not hide_labels else ""
    warn = "".join(f'<p class="warn">{_e(w)}</p>' for w in result.coverage_warnings) if not hide_labels else ""
    empty = '<p>No candidates to review.</p>' if not cands else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Music review</title><style>{CSS}</style></head>
<body><main><h1>Music review</h1><p class="muted">&ldquo;{_e(prompt)}&rdquo;</p>{message}{warn}{brief}
<section class="grid">{cards}{empty}</section>
<p><label>Your name or initials <input id="rater" autocomplete="off"></label></p>
<button id="dl" type="button">Download ratings</button>
<script type="application/json" id="meta">{json.dumps(meta)}</script><script>{SCRIPT}</script></main></body></html>"""


def write_review(out: str | Path) -> Path:
    out = Path(out)
    state = json.loads((out / "session.json").read_text())
    result = ResolveResult.model_validate(state["last_result"]) if state.get("last_result") else ResolveResult(
        status="error", message="no result yet")
    page = out / "review.html"
    page.write_text(render_page(result, out, out.name, state["prompt"]))
    return page


def write_blind_review(results: list[tuple[str, ResolveResult]], out: Path, prompt: str, rater_seed: int) -> Path:
    """Eval mode: candidates from several conditions, labels hidden, order shuffled per rater."""
    pooled: list[Candidate] = []
    origin: list[str] = []
    for condition, res in results:
        for c in res.candidates:
            pooled.append(c)
            origin.append(condition)
    order = list(range(len(pooled)))
    random.Random(rater_seed).shuffle(order)
    merged = ResolveResult(status="ok", candidates=pooled, message="")
    slots = [chr(ord("A") + i) for i in range(len(order))]
    key = {s: origin[i] for s, i in zip(slots, order)}
    page = out / f"review-rater{rater_seed}.html"
    page.write_text(render_page(merged, out, out.name, prompt, hide_labels=True, order=order,
                                extra_meta={"rater_seed": rater_seed}))
    (out / f"key-rater{rater_seed}.json").write_text(json.dumps(key, indent=2))
    return page
