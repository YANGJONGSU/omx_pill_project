#!/usr/bin/env python3
"""처방 중간 집기 20개의 배치표를 그림으로 만든다 (2026-08-10).

촬영 중에는 터미널 글자보다 그림이 빠르다. 32세트 때 쓰던 BATCH_SHEET.pdf 와
같은 형식으로, 격자 위에 알약 위치·각도와 약통에 든 것을 그린다.

    python make_midpick_sheet.py        # ~/Desktop/MIDPICK_SHEET.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

COL = {"빨강": "#d94040", "초록": "#3aa65a", "노랑": "#e8c33a"}
EN = {"빨강": "RED", "초록": "GREEN", "노랑": "YELLOW"}
CELLS_X = list("ABCDE")


def draw_one(ax, x):
    ax.set_xlim(-0.4, 5.4)
    ax.set_ylim(-0.9, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # 트레이(매트) + 5x4 격자
    ax.add_patch(mp.Rectangle((0, 0), 5, 4, fc="#5a5f5c", ec="#2b2e2c", lw=2))
    for i in range(6):
        ax.plot([i, i], [0, 4], color="#8a908c", lw=0.6)
    for j in range(5):
        ax.plot([0, 5], [j, j], color="#8a908c", lw=0.6)
    for i, c in enumerate(CELLS_X):
        ax.text(i + 0.5, -0.28, c, ha="center", va="top", fontsize=8, color="#444")
    for j in range(4):
        ax.text(-0.15, j + 0.5, str(j + 1), ha="right", va="center", fontsize=8, color="#444")

    # 트레이 위 알약
    goal = x["목표"]
    for color, (cell, ang) in x["트레이"].items():
        cx = CELLS_X.index(cell[0]) + 0.5
        cy = int(cell[1]) - 0.5
        th = np.radians(ang)
        dx, dy = 0.30 * np.cos(th), 0.30 * np.sin(th)
        # 캡슐: 색깔 끝이 화살표 방향
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color="white", lw=9,
                solid_capstyle="round", zorder=3)
        ax.plot([cx, cx + dx], [cy, cy + dy], color=COL[color], lw=9,
                solid_capstyle="round", zorder=4)
        ax.annotate("", xy=(cx + dx * 2.1, cy + dy * 2.1), xytext=(cx + dx * 1.2, cy + dy * 1.2),
                    arrowprops=dict(arrowstyle="-|>", color=COL[color], lw=1.6), zorder=5)
        if color == goal:
            ax.add_patch(mp.Circle((cx, cy), 0.46, fill=False, ec="#1565c0", lw=2.6, zorder=6))
            ax.text(cx, cy - 0.62, "집을 것", ha="center", fontsize=8,
                    color="#1565c0", fontweight="bold", zorder=7)

    # 약통
    bx = 5.55
    ax.add_patch(mp.FancyBboxPatch((bx, 0.6), 0.75, 2.8, boxstyle="round,pad=0.05",
                                   fc="#efe0bd", ec="#b9a575", lw=1.5))
    ax.text(bx + 0.38, 3.62, "약통", ha="center", fontsize=8, color="#6b5a33")
    for k, c in enumerate(x["약통"]):
        yy = 2.7 - k * 0.75
        ax.plot([bx + 0.2, bx + 0.56], [yy, yy], color=COL[c], lw=9,
                solid_capstyle="round")
    ax.set_xlim(-0.4, 6.6)


def main():
    L = json.load(open(os.path.expanduser("~/omx_pill_project/midpick_layouts.json")))
    out = os.path.expanduser("~/Desktop/MIDPICK_SHEET.pdf")
    per = 4
    with PdfPages(out) as pdf:
        for p in range(0, len(L), per):
            chunk = L[p:p + per]
            fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
            fig.suptitle("처방 중간 집기 — 약통에 담긴 게 있는 상태에서 목표 집기",
                         fontsize=14, fontweight="bold")
            for ax, x in zip(axes.ravel(), chunk):
                draw_one(ax, x)
                bottle = " + ".join(x["약통"])
                ax.set_title(f"{x['번호']}번  ({x['유형']})   약통: {bottle}   →  {x['목표']} 집기",
                             fontsize=11, pad=10)
            for ax in axes.ravel()[len(chunk):]:
                ax.axis("off")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
    print(f"저장: {out}  ({len(L)}개, {-(-len(L) // per)}쪽)")


if __name__ == "__main__":
    main()
