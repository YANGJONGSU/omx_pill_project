#!/usr/bin/env python3
"""발표용 그림을 만든다 — 지금까지 tsv 표로만 있던 지표를 그림으로.

  python make_figs.py          # report/fig_*.png 세 장
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from pathlib import Path

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("report")
INK, MUTED, LINE = "#16211c", "#6d7a73", "#e4e1d8"
RX, BAD, OK = "#0e5537", "#a8321f", "#0e7a4a"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=LINE, lw=0.8)
    ax.set_axisbelow(True)


def fig1():
    """목표 조건화 6가지 — 전부 기준선 근처였다."""
    names = ["원-핫\n토큰", "원-핫\n+FiLM", "CVAE\n제거", "Diffusion", "SmolVLA\n(언어)", "좌표\n(xy)"]
    vals = [1.00, 1.22, 1.13, 0.97, 1.02, None]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = range(len(names))
    bars = ax.bar(xs, [v if v else 0 for v in vals], color=[BAD] * 5 + [LINE], width=.62)
    ax.axhline(1.02, color=INK, ls="--", lw=1.6)
    ax.text(5.42, 1.035, "기준선 1.02\n(화면을 보지 않는 예측기)", va="bottom", ha="right",
            fontsize=9.5, color=INK)
    for x, v in zip(xs, vals):
        if v:
            ax.text(x, v + .012, f"{v:.2f}", ha="center", fontsize=11, color=INK, weight="bold")
    ax.text(5, .55, "실기에서\n3색 성공", ha="center", fontsize=10.5, color=RX, weight="bold")
    ax.set_xticks(list(xs)); ax.set_xticklabels(names, fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.35); ax.set_ylabel("자기색 대비", fontsize=11, color=MUTED)
    ax.set_title("목표를 어떤 형태로 줘도 정책은 무시했다", fontsize=14, color=INK,
                 weight="bold", pad=14, loc="left")
    style(ax); fig.tight_layout(); fig.savefig(OUT / "fig_goal_methods.png", dpi=160)
    print("  fig_goal_methods.png")


def fig2():
    """체크포인트별 지표 — 마지막이 최선이 아니다."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for c, color, lab in (("red", BAD, "빨강"), ("green", OK, "초록"), ("yellow", "#b8860b", "노랑")):
        f = Path(f"percolor_{c}.tsv")
        if not f.exists():
            continue
        d = pd.read_csv(f, sep="\t")
        x = d["스텝"].astype(int) / 1000
        ax.plot(x, d["자기색대비"], color=color, lw=2.2, marker="o", ms=4, label=lab)
        i = d["자기색대비"].idxmax()
        ax.plot(x[i], d["자기색대비"][i], "o", ms=11, mfc="none", mec=color, mew=2)
    ax.set_xlabel("학습 스텝 (천)", fontsize=11, color=MUTED)
    ax.set_ylabel("자기색 대비", fontsize=11, color=MUTED)
    ax.set_title("정점은 마지막 체크포인트가 아니다", fontsize=14, color=INK,
                 weight="bold", pad=14, loc="left")
    ax.legend(frameon=False, fontsize=10.5)
    style(ax); fig.tight_layout(); fig.savefig(OUT / "fig_checkpoints.png", dpi=160)
    print("  fig_checkpoints.png")


def fig3():
    """실기 결과 — 오프라인 지표가 실기를 예측하지 못했다."""
    rows = [("색별 정책 3개", "—", "성공", OK),
            ("원-핫 + FiLM", "1.22", "3색 연속 성공", OK),
            ("좌표 (xy)", "선택률 51%", "3색 연속 성공", OK),
            ("Diffusion", "0.97", "미검증", MUTED),
            ("SmolVLA", "2~5% 반응", "실패", BAD)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")
    ax.text(.02, .93, "오프라인 지표", fontsize=10.5, color=MUTED, transform=ax.transAxes)
    ax.text(.44, .93, "실기 결과", fontsize=10.5, color=MUTED, transform=ax.transAxes)
    for i, (n, off, real, c) in enumerate(rows):
        y = .78 - i * .155
        ax.text(.02, y, n, fontsize=12, color=INK, weight="bold", transform=ax.transAxes)
        ax.text(.30, y, off, fontsize=11.5, color=MUTED, transform=ax.transAxes)
        ax.text(.44, y, real, fontsize=12, color=c, weight="bold", transform=ax.transAxes)
    ax.text(.02, .06, "오프라인이 나빠도 실기에서 된 사례가 네 건 — 열린 루프와 닫힌 루프의 차이",
            fontsize=10.5, color=RX, transform=ax.transAxes)
    ax.set_title("오프라인 지표는 실기를 예측하지 못했다", fontsize=14, color=INK,
                 weight="bold", pad=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_offline_vs_real.png", dpi=160)
    print("  fig_offline_vs_real.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
