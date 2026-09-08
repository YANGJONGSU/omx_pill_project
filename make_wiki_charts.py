#!/usr/bin/env python3
"""위키 문서에 넣을 그래프를 만든다 (2026-08-10).

수치는 전부 실측값이며, 출처는 각 함수 주석에 적었다.
GitHub 위키는 Mermaid 를 렌더링하므로 구조도는 문서에 직접 쓰고,
여기서는 **수치 그래프만** PNG 로 만든다.

    python make_wiki_charts.py [출력폴더]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 한글 폰트 — 시스템에 있는 Noto CJK 를 쓴다. 없으면 네모로 깨진다.
for cand in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Noto Serif CJK KR"):
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

INK, MUTED, LINE = "#1f2328", "#59636e", "#d1d9e0"
RED, GREEN, YELLOW = "#c94a42", "#3d9160", "#d9a824"
ACCENT, BASE = "#12566b", "#9aa7b1"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=LINE, lw=.6)
    ax.set_axisbelow(True)


def chart_methods(out: Path):
    """방식별 자기색 대비. 출처: probe_policy.py --tests 5 실측."""
    labels = ["색별 정책 3개\n(목표 입력 없음)", "원-핫 토큰", "원-핫 + FiLM\n(학습 장면)",
              "원-핫 + FiLM\n(미학습 장면)", "SmolVLA\n(언어 지시)"]
    vals = [4.36, 1.00, 1.33, 1.03, 1.02]
    err = [(4.36 - 3.26, 5.45 - 4.36), (0, 0), (0, 0), (0, 0), (0, 0)]
    colors = [ACCENT, BASE, BASE, BASE, BASE]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, vals, color=colors, width=.6)
    ax.errorbar(0, 4.36, yerr=[[err[0][0]], [err[0][1]]], fmt="none",
                ecolor=ACCENT, capsize=5, lw=1.4)
    ax.axhline(1.02, color=RED, ls="--", lw=1.3)
    # 기준선 설명은 막대가 없는 위쪽 여백에 두고 화살표로 선을 가리킨다
    ax.annotate("기준선 1.02 — 화면을 참조하지 않고\n색별 평균 궤적만 출력할 때의 값",
                xy=(3.4, 1.02), xytext=(2.6, 2.6), color=RED, fontsize=8.5,
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .12, f"{v:.2f}",
                ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_ylabel("자기색 대비", fontsize=10, color=INK)
    ax.set_ylim(0, 6)
    ax.set_title("목표 전달 방식별 자기색 대비  —  값이 클수록 목표를 잘 따릅니다",
                 fontsize=11.5, color=INK, pad=14, loc="left")
    style(ax)
    fig.tight_layout(); fig.savefig(out / "chart_methods.png"); plt.close(fig)


def chart_curve(out: Path):
    """학습 스텝별 자기색 대비. 출처: goal_metric_curve_vae.tsv 및 구 FiLM 실측."""
    new_s = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    new_v = [1.01, 1.02, 1.05, 1.08, 1.11, 1.13, 1.15, 1.23, 1.23, 1.22]
    old_s = [40, 64, 80, 96, 128]
    old_v = [1.11, 1.11, 1.21, 1.33, 1.33]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(old_s, old_v, "o-", color=YELLOW, lw=2, ms=5, label="96 에피소드 (기존 데이터)")
    ax.plot(new_s, new_v, "o-", color=ACCENT, lw=2, ms=5, label="224 에피소드 (동일 장면 × 3목표)")
    ax.axhline(1.02, color=RED, ls="--", lw=1.2)
    ax.text(126, 1.035, "기준선 1.02", color=RED, fontsize=8.5, ha="right")
    ax.annotate("96K 이후 정체", xy=(96, 1.33), xytext=(103, 1.28),
                fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=.9))
    ax.annotate("80K 이후 정체", xy=(85, 1.23), xytext=(60, 1.29),
                fontsize=8.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=.9))
    ax.set_xlabel("학습 스텝 (천)", fontsize=10, color=INK)
    ax.set_ylabel("자기색 대비", fontsize=10, color=INK)
    ax.set_ylim(0.95, 1.42)
    ax.set_title("학습 진행에 따른 자기색 대비  —  두 경우 모두 정체합니다",
                 fontsize=11.5, color=INK, pad=14, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    fig.tight_layout(); fig.savefig(out / "chart_curve.png"); plt.close(fig)


def chart_tokens(out: Path):
    """ACT 인코더 입력 토큰 구성. 출처: modeling_act.py 구조 + 480x640 stride 32."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    parts = [600, 1, 1, 1]
    labels = ["카메라 2대\n600 (99.5%)", "잠재변수 1", "관절 상태 1", "목표 1 (0.17%)"]
    colors = [BASE, "#c9d3da", "#b6c2cb", RED]
    wedges, _ = ax.pie(parts, colors=colors, startangle=90,
                       wedgeprops=dict(width=.42, edgecolor="white", lw=1.5))
    ax.text(0, 0, "603\n토큰", ha="center", va="center", fontsize=13,
            color=INK, fontweight="bold")
    ax.legend(wedges, labels, frameon=False, fontsize=9,
              loc="center left", bbox_to_anchor=(.98, .5))
    ax.set_title("ACT 인코더 입력 토큰 구성", fontsize=11.5, color=INK, pad=10, loc="left")
    fig.tight_layout(); fig.savefig(out / "chart_tokens.png", bbox_inches="tight"); plt.close(fig)


def chart_dataset(out: Path):
    """데이터셋 구성. 출처: meta/episodes 집계 + 초기 96개 육안 확인(48개 표본).

    주의: 초기 수집 96개는 트레이 알약 개수가 1~3개로 섞여 있다. 신규 128개의
    집기만 트레이 3개로 고정이다. 둘을 묶어 '트레이 3개'로 표기하면 틀린다.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    a1.bar(["빨강", "초록", "노랑"], [79, 73, 72], color=[RED, GREEN, YELLOW], width=.55)
    for i, v in enumerate([79, 73, 72]):
        a1.text(i, v + 1.5, str(v), ha="center", fontsize=10, color=INK, fontweight="bold")
    a1.set_ylabel("에피소드 수", fontsize=10, color=INK)
    a1.set_ylim(0, 92)
    a1.set_title("색상별 에피소드", fontsize=10.5, color=INK, loc="left", pad=10)
    style(a1)

    kinds = ["초기 수집\n집기 96\n(트레이 1~3개 혼합)",
             "동일 장면 세트\n집기 96\n(트레이 3개)",
             "정지 시연 32\n(트레이 0~2개)"]
    vals = [96, 96, 32]
    bars = a2.bar(kinds, vals, color=[BASE, ACCENT, YELLOW], width=.5)
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width()/2, v + 2.5, str(v),
                ha="center", fontsize=10.5, color=INK, fontweight="bold")
    a2.set_ylabel("에피소드 수", fontsize=10, color=INK)
    a2.set_ylim(0, 120)
    a2.set_title("구간별 구성  (총 224)", fontsize=10.5, color=INK, loc="left", pad=10)
    a2.tick_params(axis="x", labelsize=8)
    style(a2)

    fig.tight_layout(); fig.savefig(out / "chart_dataset.png"); plt.close(fig)


def chart_traycount(out: Path):
    """트레이 알약 개수별 정답 분포. 개수만으로 정답이 갈리지 않음을 보입니다.

    초기 96개의 개수 분포는 48개 표본을 육안 확인해 얻은 근사치입니다
    (1개 약 10, 2개 약 22, 3개 약 16 → 96개로 환산). 정확한 전수 계수는
    색 검출기 신뢰도 문제로 보류했습니다.
    """
    counts = ["0개", "1개", "2개", "3개"]
    pick   = [0,  20,  44,  128]   # 초기 96 환산(20/44/32) + 신규 집기 96(3개)
    stop   = [8,  11,  21,  0]     # 빈트레이 8 / 두개담김 3+보강 / 목표만없음 21
    x = range(len(counts)); w = .36

    fig, ax = plt.subplots(figsize=(7.4, 4))
    ax.bar([i - w/2 for i in x], pick, w, label="집기 (목표가 트레이에 있음)", color=ACCENT)
    ax.bar([i + w/2 for i in x], stop, w, label="정지 (목표가 약통에 있음)", color=YELLOW)
    for i, (p, q) in enumerate(zip(pick, stop)):
        if p: ax.text(i - w/2, p + 2.5, str(p), ha="center", fontsize=9.5, color=INK)
        if q: ax.text(i + w/2, q + 2.5, str(q), ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(counts)
    ax.set_xlabel("에피소드 시작 시 트레이의 알약 개수", fontsize=10, color=INK)
    ax.set_ylabel("에피소드 수", fontsize=10, color=INK)
    ax.set_ylim(0, 150)
    ax.set_title("트레이 알약 개수별 정답 분포  —  개수만으로는 집기·정지가 갈리지 않습니다",
                 fontsize=11, color=INK, pad=14, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    # 설명은 막대가 없는 왼쪽 위 여백에 둔다 (범례 아래)
    ax.text(.02, .70, "1·2개 구간에 집기와 정지가 모두 존재합니다.\n"
                      "따라서 \"개수로 판단\"하는 지름길은 성립하지 않습니다.",
            transform=ax.transAxes, ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(); fig.savefig(out / "chart_traycount.png"); plt.close(fig)


def chart_drift(out: Path):
    """촬영 품질 — 동일 세트 내 알약 위치 편차. 출처: 81개 표본 실측."""
    bins = ["0~5mm", "5~10mm", "10~15mm"]
    vals = [58, 18, 5]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(bins, vals, color=[GREEN, GREEN, YELLOW], width=.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v}건 ({v/81*100:.0f}%)",
                ha="center", fontsize=9.5, color=INK)
    ax.axvline(1.5, color=RED, ls="--", lw=1.2)
    ax.text(1.55, 50, "그리퍼 여유 한계 (10mm)", color=RED, fontsize=8.5, va="center")
    ax.set_ylabel("표본 수", fontsize=10, color=INK)
    ax.set_ylim(0, 70)
    ax.set_title("동일 세트 3장에서 같은 알약의 위치 편차  —  중앙값 3.1mm",
                 fontsize=11, color=INK, pad=14, loc="left")
    style(ax)
    fig.tight_layout(); fig.savefig(out / "chart_drift.png"); plt.close(fig)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for fn in (chart_methods, chart_curve, chart_tokens, chart_dataset, chart_drift,
               chart_memorization, chart_step_vs_epoch, chart_delta, chart_hue,
               chart_traycount):
        fn(out)
        print(f"  {fn.__name__} 완료")
    print(f"\n  → {out}")



def chart_memorization(out: Path):
    """학습 장면 vs 미학습 장면 비교. 출처: 구 FiLM 128K, probe --tests 5 실측."""
    colors_ko = ["노랑", "빨강", "초록", "평균"]
    seen   = [1.34, 1.14, 1.22, 1.33]   # 학습에 쓴 에피소드
    unseen = [1.23, 0.99, 0.87, 1.03]   # 한 번도 안 본 에피소드
    x = range(len(colors_ko)); w = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    b1 = ax.bar([i - w/2 for i in x], seen,   w, label="학습에 사용한 장면", color=BASE)
    b2 = ax.bar([i + w/2 for i in x], unseen, w, label="처음 보는 장면",     color=ACCENT)
    ax.axhline(1.02, color=RED, ls="--", lw=1.2)
    ax.text(-0.48, 1.03, "기준선 1.02", color=RED, fontsize=8.5, ha="left", va="bottom")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + .015,
                    f"{b.get_height():.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(colors_ko)
    ax.set_ylabel("자기색 대비", fontsize=10, color=INK)
    ax.set_ylim(0.8, 1.5)
    ax.set_title("동일 모델을 학습 장면과 미학습 장면에서 측정한 결과\n"
                 "미학습 장면에서는 기준선 수준으로, 상승분은 암기에 기인합니다",
                 fontsize=11, color=INK, pad=14, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style(ax)
    fig.tight_layout(); fig.savefig(out / "chart_memorization.png"); plt.close(fig)


def chart_step_vs_epoch(out: Path):
    """같은 데이터를 스텝축과 에폭축으로 각각 그린다. 에폭 정렬의 필요성을 보입니다."""
    old_s = [40, 64, 80, 96, 128];  old_v = [1.11, 1.11, 1.21, 1.33, 1.33]
    new_s = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    new_v = [1.01, 1.02, 1.05, 1.08, 1.11, 1.13, 1.15, 1.23, 1.23, 1.22]
    old_e = [s*1000*8/115332 for s in old_s]
    new_e = [s*1000*8/245529 for s in new_s]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (ox, nx, xl) in zip((a1, a2),
                                [(old_s, new_s, "학습 스텝 (천)"),
                                 (old_e, new_e, "에폭 (데이터를 몇 번 봤는가)")]):
        ax.plot(ox, old_v, "o-", color=YELLOW, lw=2, ms=4.5, label="96 에피소드")
        ax.plot(nx, new_v, "o-", color=ACCENT, lw=2, ms=4.5, label="224 에피소드")
        ax.axhline(1.02, color=RED, ls="--", lw=1.1)
        ax.set_xlabel(xl, fontsize=9.5, color=INK)
        ax.set_ylim(0.97, 1.40)
        style(ax)
    a1.set_ylabel("자기색 대비", fontsize=10, color=INK)
    a1.set_title("스텝 기준 — 두 곡선이 겹칩니다", fontsize=10.5, color=INK, loc="left", pad=10)
    a2.set_title("에폭 기준 — 224개가 절반의 노출로 같은 값에 도달합니다",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    a2.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("동일한 데이터를 두 축으로 그린 결과  —  비교 기준에 따라 해석이 달라집니다",
                 fontsize=11.5, color=INK, x=0.01, ha="left", y=0.99)
    fig.tight_layout(); fig.savefig(out / "chart_step_vs_epoch.png"); plt.close(fig)


def chart_delta(out: Path):
    """목표 변경 시 행동 변화량 Δ. 주입 방식별 비교입니다."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9))

    a1.bar(["원-핫 토큰", "원-핫 + FiLM"], [0.02, 0.63], color=[BASE, ACCENT], width=.45)
    for i, v in enumerate([0.02, 0.63]):
        a1.text(i, v + .02, f"{v:.2f}", ha="center", fontsize=10, color=INK, fontweight="bold")
    a1.text(0.5, 0.42, "31배", ha="center", fontsize=12, color=ACCENT, fontweight="bold")
    a1.annotate("", xy=(1, 0.60), xytext=(0, 0.05),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))
    a1.set_ylabel("Δ (목표 변경 시 행동 변화량)", fontsize=9.5, color=INK)
    a1.set_ylim(0, 0.78)
    a1.set_title("주입 방식별 Δ — FiLM 은 신호를 통과시킵니다",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    style(a1)

    steps = [40, 128]
    a2.plot(steps, [0.05, 0.02], "o-", color=RED, lw=2, ms=6, label="원-핫 토큰")
    a2.plot([10, 20, 40, 60, 80, 90, 100], [0.23, 0.29, 0.39, 0.47, 0.50, 0.54, 0.51],
            "o-", color=ACCENT, lw=2, ms=4.5, label="원-핫 + FiLM (224개)")
    a2.set_xlabel("학습 스텝 (천)", fontsize=9.5, color=INK)
    a2.set_ylabel("Δ", fontsize=9.5, color=INK)
    a2.set_ylim(0, 0.62)
    a2.set_title("학습 진행에 따른 Δ — 토큰 방식은 오히려 감소합니다",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    a2.legend(frameon=False, fontsize=8.5, loc="center right")
    style(a2)

    fig.tight_layout(); fig.savefig(out / "chart_delta.png"); plt.close(fig)


def chart_hue(out: Path):
    """색상 증강이 색 간격을 얼마나 잠식하는지. 출처: 실측 HSV + 증강 설정."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = ["빨강 354°", "노랑 50°", "초록 134°"]
    pos = [354 - 360, 50, 134]          # 빨강을 -6도로 펴서 직선 배치
    cols = [RED, YELLOW, GREEN]
    for p, c, n in zip(pos, cols, names):
        ax.errorbar(p, 1, xerr=18, fmt="o", color=c, ms=13, capsize=6,
                    elinewidth=2.4, ecolor=c, alpha=.95)
        ax.text(p, 1.22, n, ha="center", fontsize=9.5, color=INK)
    ax.annotate("", xy=(-6, 0.72), xytext=(50, 0.72),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
    ax.text(22, 0.62, "56°", ha="center", fontsize=9, color=MUTED)
    ax.annotate("", xy=(50, 0.72), xytext=(134, 0.72),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
    ax.text(92, 0.62, "84°", ha="center", fontsize=9, color=MUTED)
    ax.text(22, 1.55, "증강 ±18° 씩 흔들면 두 색이 최대 36° 근접\n= 56° 간격의 64% 잠식",
            ha="center", fontsize=9, color=RED)
    ax.set_xlim(-40, 175); ax.set_ylim(0.45, 1.85)
    ax.set_yticks([]); ax.set_xlabel("색상(hue) 각도", fontsize=9.5, color=INK)
    ax.set_title("색상 증강이 색 간격에 미치는 영향  —  가로 막대가 증강 범위입니다",
                 fontsize=11, color=INK, pad=14, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout(); fig.savefig(out / "chart_hue.png"); plt.close(fig)

if __name__ == "__main__":
    main()
