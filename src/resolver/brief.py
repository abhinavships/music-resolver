"""Prompt to `Brief` (7.1): online (Claude + web search) or offline (keyword map), rights guard, cache.

The model call is injectable (`call`), so tests replay recorded outputs and
never need a key. Only the prompt text leaves the machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from importlib import resources
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from .config import Config
from .schemas import Brief, CueSpec, NamedReference

# call(system, user_text) -> (text, source_urls)
ModelCall = Callable[[str, str], tuple[str, list[str]]]

BANNED = ("melody", "melodies", "hook", "lyrics", "lyric", "same tune", "exactly like", "riff", "chorus line")
KEEP_TOKENS = {
    "music", "theme", "style", "song", "sound", "audio", "the", "and", "of", "vs", "ka", "type", "reel",
    "trending", "studio", "power", "entry",
}
MUSIC_WORDS = {
    "music", "song", "track", "bgm", "background", "beat", "audio", "sound", "score", "theme", "tune",
    "vibe", "feel", "piano", "violin", "dhol", "ukulele", "lo-fi", "trap", "build", "ending", "intro",
}
NON_MUSIC = ("caption", "subtitle", "title", "text", "cut ", "trim", "crop", "colour", "color", "grade",
             "zoom in", "transition", "font", "sticker", "speed up the video")


class BriefError(RuntimeError):
    pass


# --- prompt and keyword data ---------------------------------------------------

def system_prompt(version: str) -> str:
    return resources.files("resolver").joinpath(f"data/brief_prompt_{version}.txt").read_text()


def keyword_map() -> dict:
    return tomllib.loads(resources.files("resolver").joinpath("data/brief_keywords.toml").read_text())


def normalise(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())


def cache_key(prompt: str, model: str, version: str) -> str:
    return hashlib.sha256(f"{normalise(prompt)}\n{model}\n{version}".encode()).hexdigest()[:24]


# --- rights guard (L8, R5) -----------------------------------------------------

def name_tokens(names: list[str]) -> set[str]:
    toks: set[str] = set()
    for n in names:
        for t in re.findall(r"[a-z0-9']+", (n or "").lower()):
            if len(t) >= 3 and t not in KEEP_TOKENS:
                toks.add(t)
    return toks


def _strip(text: str, tokens: set[str]) -> str:
    out = " ".join(w for w in text.split() if re.sub(r"[^a-z0-9']", "", w.lower()) not in tokens)
    return re.sub(r"\s+([,.;])", r"\1", re.sub(r"\s+", " ", out)).strip(" ,;-")


def has_banned(text: str) -> bool:
    t = text.lower()
    return any(b in t for b in BANNED)


def rights_guard(brief: Brief, extra_names: list[str] | None = None,
                 fallback_queries: Callable[[CueSpec], list[str]] | None = None) -> Brief:
    """Remove reference and artist names from every query, drop melody/hook/lyric
    language, and keep at least two queries per cue (else fall back)."""
    names = list(extra_names or [])
    if brief.named_reference:
        names += [brief.named_reference.raw, brief.named_reference.canonical or ""]
    tokens = name_tokens(names)
    notes = list(brief.notes_for_user)

    def clean(cue: CueSpec) -> CueSpec:
        queries = []
        for q in cue.queries:
            if has_banned(q):
                continue
            q2 = _strip(q, tokens)
            if len(q2.split()) >= 2 and q2.lower() not in {x.lower() for x in queries}:
                queries.append(q2)
        if len(queries) < 2 and fallback_queries is not None:
            for q in fallback_queries(cue):
                q2 = _strip(q, tokens)
                if not has_banned(q2) and len(q2.split()) >= 2 and q2 not in queries:
                    queries.append(q2)
        words = lambda xs: [x for x in (_strip(v, tokens) for v in xs if not has_banned(v)) if x]  # noqa: E731
        queries = [re.sub(r"^(?:(?:a|an|the|ka|ki|ke|famous|some)\s+)+", "", q) for q in queries]
        return cue.model_copy(update={
            "queries": queries[:5], "mood": words(cue.mood), "instruments": words(cue.instruments),
            "label": _strip(cue.label, tokens) or cue.cue_type,
        })

    return brief.model_copy(update={
        "primary": clean(brief.primary), "alternatives": [clean(c) for c in brief.alternatives][:2],
        "notes_for_user": notes,
    })


# --- offline brief (D1, D2) ----------------------------------------------------

def _find_reference(text: str, refs: dict) -> tuple[str, dict, str] | None:
    best = None
    for key, ref in refs.items():
        for alias in sorted([key, *ref.get("aliases", [])], key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", text):
                if best is None or len(alias) > len(best[2]):
                    best = (key, ref, alias)
                break
    return best


def _matches(text: str, kw: dict) -> list[dict]:
    return [v for k, v in kw.items() if re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", text)]


def _cue_from(text: str, kw: dict, ref: dict | None, label_hint: str) -> CueSpec:
    hits = _matches(text, kw)
    if ref:
        hits = [*hits, ref]   # the words of this reading lead; the reference fills in
    mood: list[str] = []
    inst: list[str] = []
    energy, tempo, cue_type, role = "medium", None, "bed", None
    for h in hits:
        mood += [m for m in h.get("mood", []) if m not in mood]
        inst += [i for i in h.get("instruments", []) if i not in inst]
        if h.get("energy") == "high" or (h.get("energy") == "low" and energy == "medium"):
            energy = h["energy"]
        tempo = h.get("tempo", tempo)
        if h.get("cue_type") and cue_type == "bed":
            cue_type = h["cue_type"]
        role = h.get("role", role)
    vocals = "none"
    if re.search(r"\b(no vocals|instrumental)\b", text):
        vocals = "none"
    elif re.search(r"\b(vocals|singing|song with lyrics)\b", text):
        vocals = "ok"
    avoid = [m.group(1) for m in re.finditer(r"\bno ([a-z-]+)", text) if m.group(1) not in {"vocals"}]
    if "no vocals" in text:
        avoid.append("vocals")
    label = " ".join(mood[:2] + [cue_type if cue_type != "bed" else "music"]) or label_hint
    queries = []
    core = " ".join(mood[:3])
    if inst:
        queries.append(f"{core} music with {', '.join(inst[:3])}".strip())
    if mood:
        queries.append(f"{energy} energy {core} {cue_type if cue_type != 'bed' else 'background music'}".strip())
    if inst and len(inst) > 1:
        queries.append(f"{inst[0]} and {inst[1]} {mood[0] if mood else ''} instrumental".strip())
    return CueSpec(
        label=label, cue_type=cue_type, mood=mood, energy=energy,
        tempo_bpm=tuple(tempo) if tempo else None, instruments=inst, avoid=avoid, vocals=vocals,
        mix_role=role or ("featured" if energy == "high" else "underscore"), queries=queries,
    )


def _split_clauses(text: str) -> list[str]:
    parts = re.split(r",|\band then\b|\band\b(?= (?:add|put|make|insert|place|caption|cut|trim)\b)|\. ", text)
    return [p.strip() for p in parts if p.strip()]


def offline_brief(prompt: str, cfg: Config, note: str | None = None) -> Brief:
    data = keyword_map()
    kw, refs = data["keywords"], data["references"]
    text = normalise(prompt)
    ignored = []
    clauses = _split_clauses(text)
    music_clauses = []
    for c in clauses:
        if any(n in c for n in NON_MUSIC) and not any(re.search(rf"\b{re.escape(w)}\b", c) for w in MUSIC_WORDS):
            ignored.append(c)
        else:
            music_clauses.append(c)
    # "... music in the background and put caption when ..." : split the tail off.
    for i, c in enumerate(music_clauses):
        m = re.search(r"\band (put|add|insert) (caption|subtitle|text|title)s?\b.*", c)
        if m:
            ignored.append(m.group(0).removeprefix("and ").strip())
            music_clauses[i] = c[: m.start()].strip()
    music_text = " ".join(music_clauses) or text

    found = _find_reference(music_text, refs)
    named = None
    notes = [note] if note else []
    if found:
        key, ref, alias = found
        named = NamedReference(raw=alias, canonical=ref.get("canonical"), kind=ref.get("kind", "unknown"),
                               available=False, confidence=0.9 if alias == key else 0.7, sources=[])
        notes.append(f"The original {named.kind} music is not available; matching its sound instead.")
    if re.search(r"\b(exactly like|same tune|exact)\b", text):
        notes.append("An exact copy of a melody is not offered; this is a genre-level match instead (L8).")

    deferred: list[str] = []
    then = re.split(r"\b(?:first )?then\b|\bfollowed by\b", music_text)
    if len(then) > 1:
        music_text = then[0].replace(" first", "").strip()
        deferred = [t.strip(" ,") for t in then[1:] if t.strip(" ,")]

    readings = [r.strip() for r in re.split(r"\bor\b|\beither\b|\bwhichever\b", music_text) if r.strip()]
    readings = [r for r in readings if _matches(r, kw)] or [music_text]
    ref = found[1] if found else None
    primary = _cue_from(readings[0] if len(readings) > 1 else music_text, kw, ref, "background music")
    alternatives = [_cue_from(r, kw, ref, "alternative") for r in readings[1:3]] if len(readings) > 1 else []

    explicit = any(re.search(rf"\b{re.escape(w)}\b", music_text) for w in MUSIC_WORDS) or bool(named)
    requested = "explicit" if explicit else ("implied" if primary.mood else "none")
    # The raw request is one extra query, with names stripped by the guard below.
    raw_query = re.sub(r"\b(take this video|put|add|in the background|this clip|is clip mein|daalo|make it)\b", " ", music_text)
    primary = primary.model_copy(update={"queries": [*primary.queries, re.sub(r"\s+", " ", raw_query).strip()]})

    brief = Brief(
        version=cfg.brief.prompt_version, mode="offline", music_requested=requested, named_reference=named,
        primary=primary, alternatives=alternatives, deferred_cues=deferred,
        sfx_wanted=bool(re.search(r"\b(sfx|sound effects?|fight|slapstick|fails?)\b", text)),
        ignored_instructions=ignored, notes_for_user=notes,
    )
    if requested == "implied":
        brief.notes_for_user.append("Music was implied, not asked for; it is proposed, not applied (R10).")
    if deferred:
        brief.notes_for_user.append(f"One cue per clip in v1; deferred: {'; '.join(deferred)} (R15).")
    return rights_guard(brief, fallback_queries=_fallback_queries)


def _fallback_queries(cue: CueSpec) -> list[str]:
    base = " ".join(cue.mood[:3]) or "background"
    out = [f"{base} {cue.cue_type if cue.cue_type != 'bed' else 'music'}", f"{cue.energy} energy {base} instrumental"]
    if cue.instruments:
        out.append(f"{base} music with {' and '.join(cue.instruments[:2])}")
    return out


# --- online brief ----------------------------------------------------------------

def extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    if not raw:
        raise ValueError("no JSON object in the model output")
    return json.loads(raw)


def parse_brief(text: str, cfg: Config, sources: list[str]) -> Brief:
    data = extract_json(text)
    data.update({"version": cfg.brief.prompt_version, "mode": "online"})
    brief = Brief.model_validate(data)
    if brief.named_reference is not None:
        ref = brief.named_reference
        ref.available = False
        ref.sources = list(dict.fromkeys([*ref.sources, *sources]))
    for cue in brief.cues():
        if len(cue.queries) < 1:
            raise ValueError(f"cue {cue.label!r} has no queries")
    return brief


def anthropic_call(cfg: Config) -> ModelCall:
    """Claude with the server web search tool. Raises BriefError when unavailable."""
    try:
        import anthropic
    except ImportError as exc:
        raise BriefError("anthropic package not installed (pip install -e '.[api]')") from exc
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise BriefError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic()

    def call(system: str, user: str) -> tuple[str, list[str]]:
        messages: list = [{"role": "user", "content": user}]
        kwargs = {}
        if cfg.brief.server_fallback:
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        sources: list[str] = []
        texts: list[str] = []
        for _ in range(4):   # pause_turn: a long web search turn is resumed, not restarted
            resp = client.beta.messages.create(
                model=cfg.brief.model, max_tokens=16000, system=system, messages=messages,
                tools=[{"type": cfg.brief.web_search_tool, "name": "web_search",
                        "max_uses": cfg.brief.web_search_max_uses}],
                **kwargs,
            )
            if resp.stop_reason == "refusal":
                raise BriefError("model declined the request")
            for block in resp.content:
                if block.type == "text":
                    texts.append(block.text)
                elif block.type == "web_search_tool_result" and isinstance(block.content, list):
                    sources += [r.url for r in block.content if getattr(r, "url", None)]
            if resp.stop_reason != "pause_turn":
                break
            messages = [*messages, {"role": "assistant", "content": resp.content}]
        return "".join(texts), sources

    return call


def online_brief(prompt: str, cfg: Config, call: ModelCall) -> Brief:
    system = system_prompt(cfg.brief.prompt_version)
    user = f"Editing request:\n{prompt}"
    text, sources = call(system, user)
    try:
        brief = parse_brief(text, cfg, sources)
    except (ValueError, ValidationError) as err:
        # D5: one retry with the validation error appended.
        retry = f"{user}\n\nYour previous output was invalid: {str(err)[:800]}\nReturn only the corrected JSON object."
        text, more = call(system, retry)
        brief = parse_brief(text, cfg, sources + more)
    return rights_guard(brief, fallback_queries=_fallback_queries)


def make_brief(prompt: str, cfg: Config, call: ModelCall | None = None, use_cache: bool = True,
               offline: bool = False) -> Brief:
    """Online when a model is available, else offline (D1, D2). Cached by (prompt, model, version) (D8)."""
    cache_dir = Path(cfg.library.cache_dir) / "briefs"
    key = cache_key(prompt, cfg.brief.model, cfg.brief.prompt_version)
    path = cache_dir / f"{key}.json"
    if use_cache and not offline and path.exists():
        return Brief.model_validate_json(path.read_text())
    if offline:
        return offline_brief(prompt, cfg, note="Offline brief: keyword map only (no model).")
    try:
        call = call or anthropic_call(cfg)
    except BriefError as err:
        return offline_brief(prompt, cfg, note=f"Offline brief: {err} (D1).")
    try:
        brief = online_brief(prompt, cfg, call)
    except (ValueError, ValidationError) as err:
        return offline_brief(prompt, cfg, note=f"Offline brief: model output invalid twice ({str(err)[:120]}) (D5).")
    except BriefError as err:
        return offline_brief(prompt, cfg, note=f"Offline brief: {err}.")
    except Exception as err:  # noqa: BLE001  (network or API errors: D2)
        return offline_brief(prompt, cfg, note=f"Offline brief: model unreachable ({type(err).__name__}) (D2).")
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(brief.model_dump_json(indent=2))
    return brief


# --- refine (R11 to R13) -------------------------------------------------------

def directives(instruction: str) -> dict:
    t = normalise(instruction)
    like = re.search(r"\blike #?(\d)\b|\bmore like #?(\d)\b|\bsimilar to #?(\d)\b", t)
    return {
        "like_rank": int(next(g for g in like.groups() if g)) if like else None,
        "different": bool(re.search(r"completely different|something else entirely|totally different", t)),
        "include_shown": bool(re.search(r"\b(again|include (the )?(shown|previous)|show (them|those) again)\b", t)),
    }


def refine_offline(brief: Brief, instruction: str) -> Brief:
    """Apply a delta to the primary cue, never a rewrite."""
    t = normalise(instruction)
    kw = keyword_map()["keywords"]
    cue = brief.primary
    mood, inst, avoid = list(cue.mood), list(cue.instruments), list(cue.avoid)
    energy, tempo = cue.energy, cue.tempo_bpm
    for m in re.finditer(r"\b(?:no|less|without) ([a-z-]+)", t):
        word = m.group(1)
        avoid.append(word)
        mood = [x for x in mood if word not in x]
        inst = [x for x in inst if word not in x]
    for m in re.finditer(r"\bmore ([a-z-]+)", t):
        word = m.group(1)
        entry = kw.get(word, {})
        add = entry.get("mood", [word] if word not in {"like"} else [])
        mood = [*add, *[x for x in mood if x not in add]]
        inst = [*[i for i in entry.get("instruments", []) if i not in inst], *inst]
    if re.search(r"\bfaster\b", t):
        lo, hi = tempo or (100, 130)
        tempo, energy = (lo + 20, hi + 20), "high"
    if re.search(r"\bslower\b", t):
        lo, hi = tempo or (90, 120)
        tempo, energy = (max(50, lo - 20), max(70, hi - 20)), "low" if energy == "medium" else "medium"
    core = " ".join(mood[:3])
    queries = [f"{core} music with {', '.join(inst[:3])}" if inst else f"{core} music",
               f"{energy} energy {core} {cue.cue_type if cue.cue_type != 'bed' else 'background music'}"]
    if avoid:
        queries = [f"{q}, no {' no '.join(avoid[:2])}" for q in queries]
    queries += [q for q in cue.queries if not any(a in q.lower() for a in avoid)][:2]
    new = cue.model_copy(update={"mood": mood, "instruments": inst, "avoid": list(dict.fromkeys(avoid)),
                                 "energy": energy, "tempo_bpm": tempo, "queries": queries[:5]})
    return brief.model_copy(update={"primary": new})


REFINE_SYSTEM = """You edit an existing music brief. Apply only the change the user asks for, as a small delta;
keep every other field as it is. Never add melody, hook or lyric descriptions or proper nouns to queries.
Return only the full edited brief as one JSON object with the same fields."""


def refine(brief: Brief, instruction: str, cfg: Config, call: ModelCall | None = None) -> Brief:
    if brief.mode == "online" and call is not None:
        try:
            text, _ = call(REFINE_SYSTEM, f"Brief:\n{brief.model_dump_json()}\n\nChange: {instruction}")
            new = parse_brief(text, cfg, [])
            return rights_guard(new.model_copy(update={"named_reference": brief.named_reference}),
                                fallback_queries=_fallback_queries)
        except Exception:  # noqa: BLE001  (fall through to the offline delta)
            pass
    return rights_guard(refine_offline(brief, instruction), fallback_queries=_fallback_queries)
