#!/usr/bin/env python
"""발표용 그림 생성. report/ 에 PNG로 떨어진다.

  fig1_position_coverage.png  집는 자세 2차원 분포 (파일럿 vs 재수집) — 핵심 근거
  fig2_pan_gaps.png           pan 1차원 분포와 빈 구간
  fig3_loss.png               학습 loss 곡선 (v1 20K vs v2 80K)

색은 dataviz 스킬의 검증된 기본 팔레트에서 슬롯 1(blue)/2(orange)를 값 변경 없이 사용.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from analyze_positions import cluster_split, grasp_table  # noqa: E402

OUT = Path("/home/user/omx_pill_project/report")
LOG_V1 = Path("/home/user/omx_pill_project/train/act_yellow.log")
LOG_V2 = Path("/home/user/omx_pill_project/train/act_yellow_v2.log")

# --- 팔레트 (dataviz 기본 팔레트, light 모드) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"
S1 = "#2a78d6"  # 슬롯1 blue  — 파일럿
S2 = "#eb6834"  # 슬롯2 orange — 재수집

# 한글 폰트. 설치된 건 .ttc 컬렉션이라 matplotlib이 대표 이름을 'Noto Sans CJK JP'로
# 잡는다 — Hangul 글리프는 같은 파일에 들어 있으므로 이름만 맞춰 쓰면 된다.
KFONT = "DejaVu Sans"
for cand in ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Medium.ttc"):
    p = Path("/usr/share/fonts/opentype/noto") / cand
    if p.exists():
        font_manager.fontManager.addfont(str(p))
        KFONT = font_manager.FontProperties(fname=str(p)).get_name()
        break
plt.rcParams.update({
    "font.family": KFONT,
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.edgecolor": GRID,
    "font.size": 11,
})


def style(ax):
    """축은 물러나고 데이터가 앞으로."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig1(pilot, new):
    """집는 자세 2차원 분포. 파일럿은 한 줄, 재수집은 면으로 퍼진다."""
    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=160)
    P = np.stack(list(pilot.values()))
    N = np.stack(list(new.values()))
    ax.scatter(P[:, 0], P[:, 1], s=110, c=S1, edgecolors=SURFACE, linewidths=2,
               label=f"파일럿 ep0~29 (n={len(P)})", zorder=3)
    ax.scatter(N[:, 0], N[:, 1], s=110, c=S2, edgecolors=SURFACE, linewidths=2,
               marker="s", label=f"재수집 ep30~59 (n={len(N)})", zorder=4)
    style(ax)
    ax.set_xlabel("shoulder_pan  (좌우 방향)")
    ax.set_ylabel("shoulder_lift  (전후·높이 방향)")
    ax.set_title("집는 순간의 팔 자세 — 알약이 놓였던 위치의 대리 지표",
                 fontsize=13, fontweight="bold", pad=14, loc="left")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
    ax.annotate("파일럿은 알약을 가로 한 줄에 놓아\nlift 폭이 13밖에 안 된다",
                xy=(P[:, 0].mean(), P[:, 1].mean()), xytext=(0.03, 0.28),
                textcoords="axes fraction", color=INK2, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=GRID, linewidth=1.5))
    fig.tight_layout()
    fig.savefig(OUT / "fig1_position_coverage.png")
    plt.close(fig)


def fig2(pilot, new):
    """pan 1차원 분포 — 덩어리와 빈 구간."""
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=160)
    pp = np.array([v[0] for v in pilot.values()])
    nn = np.array([v[0] for v in new.values()])
    allp = np.concatenate([pp, nn])

    rng = np.random.default_rng(0)
    ax.scatter(pp, np.full_like(pp, 1.0) + rng.uniform(-0.07, 0.07, len(pp)),
               s=90, c=S1, edgecolors=SURFACE, linewidths=1.6, zorder=3)
    ax.scatter(nn, np.full_like(nn, 0.0) + rng.uniform(-0.07, 0.07, len(nn)),
               s=90, c=S2, edgecolors=SURFACE, linewidths=1.6, marker="s", zorder=3)

    # 파일럿의 최대 빈 구간을 표시
    sp = np.sort(pp)
    gaps = np.diff(sp)
    i = int(np.argmax(gaps))
    ax.axvspan(sp[i], sp[i + 1], color=S1, alpha=0.10, zorder=1)
    ax.annotate(f"데이터가 한 프레임도 없는 구간\n{gaps[i]:.1f} (≈{gaps[i] * 1.8:.1f}°)",
                xy=((sp[i] + sp[i + 1]) / 2, 1.32), ha="center", color=INK2, fontsize=9.5)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["재수집\nep30~59", "파일럿\nep0~29"], fontsize=10)
    ax.set_ylim(-0.55, 1.75)
    ax.set_xlim(allp.min() - 1.5, allp.max() + 1.5)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("shoulder_pan  (집는 순간)")
    ax.set_title(f"파일럿은 덩어리 {len(cluster_split(pp))}개, 합치면 {len(cluster_split(allp))}개",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_pan_gaps.png")
    plt.close(fig)


def read_loss(path: Path):
    steps, losses = [], []
    if not path.exists():
        return np.array([]), np.array([])
    for line in path.open(errors="ignore"):
        m = re.search(r"step:([\d.]+)K smpl:\S+ ep:\d+ epch:[\d.]+ loss:([\d.]+)", line)
        if m:
            steps.append(float(m.group(1)) * 1000)
            losses.append(float(m.group(2)))
    return np.array(steps), np.array(losses)


def fig3():
    s1, l1 = read_loss(LOG_V1)
    s2, l2 = read_loss(LOG_V2)
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    if len(s1):
        ax.plot(s1, l1, color=S1, linewidth=2, label="v1 — 30개 / 20K 스텝")
        ax.annotate(f"{l1[-1]:.3f}", xy=(s1[-1], l1[-1]), xytext=(6, 6),
                    textcoords="offset points", color=INK2, fontsize=10)
    ax.plot(s2, l2, color=S2, linewidth=2, label="v2 — 60개 / 80K 스텝")
    ax.annotate(f"{l2[-1]:.3f}", xy=(s2[-1], l2[-1]), xytext=(6, 6),
                textcoords="offset points", color=INK2, fontsize=10)
    ax.set_yscale("log")
    style(ax)
    ax.set_xlabel("학습 스텝")
    ax.set_ylabel("loss (로그 척도)")
    ax.set_title("학습 곡선 — v1은 상한에 걸려 끊겼고, v2는 평평해졌다",
                 fontsize=13, fontweight="bold", pad=14, loc="left")
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_loss.png")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    poses, _, _ = grasp_table()
    pilot = {k: v for k, v in poses.items() if k < 30}
    new = {k: v for k, v in poses.items() if k >= 30}
    fig1(pilot, new)
    fig2(pilot, new)
    fig3()
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
