# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Header Image Generator
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Generates the "Coherence Map" header visualisation for Director AI.

Central orchestrator overseeing multiple verification streams,
rendered as a professional 1280x640 GitHub-ready banner.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, ConnectionPatch
from pathlib import Path

# Configuration for 1280x640px header
WIDTH, HEIGHT = 12.8, 6.4
DPI = 100


def generate_director_header(output_path: str = "header.png") -> None:
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI, facecolor="#0d1117")
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_xticks([])
    ax.set_yticks([])

    # 1. Subtle background grid (The Foundation)
    grid_res = 1.0
    for x in np.arange(0, WIDTH, grid_res):
        ax.axvline(x, color="#1b2129", lw=0.5, alpha=0.5)
    for y in np.arange(0, HEIGHT, grid_res):
        ax.axhline(y, color="#1b2129", lw=0.5, alpha=0.5)

    # 2. The Core (Director) - Center-Right
    core_pos = (WIDTH * 0.7, HEIGHT * 0.5)
    core = RegularPolygon(
        core_pos, 6, radius=1.2,
        color="#58a6ff", alpha=0.1, ec="#58a6ff", lw=2,
    )
    ax.add_patch(core)
    # Mid ring
    ax.add_patch(RegularPolygon(
        core_pos, 6, radius=0.7,
        color="#58a6ff", alpha=0.05, ec="#58a6ff", lw=1,
    ))
    # Inner core
    ax.add_patch(RegularPolygon(
        core_pos, 6, radius=0.4,
        color="#ffffff", alpha=0.8,
    ))

    # 3. Agent Streams (Verification Paths)
    agent_y = np.linspace(HEIGHT * 0.2, HEIGHT * 0.8, 5)
    agent_x = WIDTH * 0.2
    stream_labels = ["NLI", "RAG", "SEC", "LYA", "PLV"]

    for i, y in enumerate(agent_y):
        # Agent Node
        ax.add_patch(plt.Circle(
            (agent_x, y), 0.15,
            color="#30363d", ec="#8b949e", lw=1,
        ))
        # Node label
        ax.text(
            agent_x, y, stream_labels[i],
            color="#c9d1d9", fontsize=6, fontfamily="monospace",
            ha="center", va="center",
        )

        # Coherence wave path to core
        path_x = np.linspace(agent_x + 0.5, core_pos[0] - 1.5, 80)
        path_y = y + 0.08 * np.sin(path_x * 3) * (path_x - agent_x) / 5
        ax.plot(path_x, path_y, color="#58a6ff", lw=1.2, alpha=0.25)

        # Straight connection line
        con = ConnectionPatch(
            xyA=(agent_x + 0.2, y),
            xyB=(core_pos[0] - 1.3, core_pos[1]),
            coordsA="data", coordsB="data",
            arrowstyle="-", color="#58a6ff", alpha=0.08, lw=0.5,
        )
        ax.add_artist(con)

    # 4. Branding Text
    ax.text(
        1.0, HEIGHT * 0.56, "DIRECTOR-AI",
        color="#ffffff", fontsize=42, fontweight="bold",
        fontfamily="sans-serif", alpha=0.9,
    )
    ax.text(
        1.0, HEIGHT * 0.44, "REAL-TIME LLM HALLUCINATION GUARDRAIL",
        color="#58a6ff", fontsize=16, fontfamily="monospace", alpha=0.8,
    )
    ax.text(
        1.0, HEIGHT * 0.37, "v1.0.0 // VERIFICATION_ACTIVE",
        color="#3fb950", fontsize=10, fontfamily="monospace", alpha=0.6,
    )

    # 5. Accent Data Spikes (deterministic seed for reproducibility)
    rng = np.random.default_rng(42)
    for _ in range(25):
        sx = rng.uniform(0, WIDTH)
        sy = rng.uniform(0, HEIGHT)
        ax.plot([sx, sx + 0.05], [sy, sy], color="#58a6ff", alpha=0.2, lw=2)

    # 6. Faint outer glow rings around core
    for r, a in [(1.8, 0.03), (2.4, 0.015)]:
        circle = plt.Circle(core_pos, r, fill=False, ec="#58a6ff", lw=0.8, alpha=a)
        ax.add_patch(circle)

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=DPI)
    plt.close(fig)
    print(f"Generated {output_path} ({WIDTH*DPI:.0f}x{HEIGHT*DPI:.0f})")


if __name__ == "__main__":
    out = Path(__file__).parent / "header.png"
    generate_director_header(str(out))
