"""Optional MCP server wrapping the same handler as `resolver tool` (M8).

Run with `python -m resolver.mcp_server` (needs the `mcp` extra). Tools:
resolve_music, refine_music, pick_music, skip_music, own_music.
"""

from __future__ import annotations

from .tool import handle


def build():
    try:
        from mcp.server.mcpserver import MCPServer as Server   # mcp 2.x
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server       # mcp 1.x

    app = Server("music-resolver")

    @app.tool()
    def resolve_music(clip: str, prompt: str, session: str, reference: str | None = None) -> dict:
        """Find three licensed tracks for a music request on a local clip; writes a review page."""
        return handle({"action": "resolve", "clip": clip, "prompt": prompt, "session": session,
                       "reference": reference})

    @app.tool()
    def refine_music(session: str, instruction: str) -> dict:
        """Adjust the search ("more comic", "no drums", "like #2", "something completely different")."""
        return handle({"action": "refine", "session": session, "instruction": instruction})

    @app.tool()
    def pick_music(session: str, rank: int, hit: float | None = None) -> dict:
        """Fit, render and verify candidate `rank`; optional hit time override in seconds."""
        return handle({"action": "pick", "session": session, "rank": rank,
                       "overrides": {"hit": hit} if hit is not None else {}})

    @app.tool()
    def skip_music(session: str, reason: str = "none fit") -> dict:
        """Skip the current option and move to the next fallback."""
        return handle({"action": "skip", "session": session, "reason": reason})

    @app.tool()
    def own_music(session: str, track: str, rights_confirmed: bool) -> dict:
        """Use the user's own track; only after they confirm they have the rights."""
        return handle({"action": "own", "session": session, "track": track, "rights_confirmed": rights_confirmed})

    return app


if __name__ == "__main__":
    build().run()
