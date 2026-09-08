#!/usr/bin/env python3
"""좌표 조건화 결과 그래프 — 발표자료용 (2026-08-13).

두 장을 만든다.

  ① xy_curve.png     자기목표 대비 + 학습 손실을 한 그래프에
       원-핫에서 가장 설득력 있던 근거가 "손실은 0.067까지 내려갔는데 목표 추종은
       1.22 에서 멈췄다" 였다. 좌표에서 둘이 같이 오르면 대비가 선명해진다.

  ② xy_gain.png      응답 이득 · 방향 일치
       "목표를 옮긴 만큼 따라갔는가". Diffusion 은 크게 움직였지만(Δ 1.05)
       방향이 무작위여서 비율이 1.00 이었다. 이 그래프가 그 차이를 보여준다.

    python make_xy_chart.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

for cand in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Noto Serif CJK KR"):
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "wiki" / "images"
BLUE, RED, GREY, GREEN = "#1f4e79", "#a8342d", "#8a8f98", "#1f7a4d"

# 비교군 — 원-핫 + FiLM (act_film_224). goal_metric_curve_vae.tsv 의 실측값
ONEHOT = [(10000, 1.01), (40000, 1.08), (50000, 1.11), (70000, 1.15),
          (80000, 1.23), (100000, 1.22), (110000, 1.25)]
BASELINE = 1.02


def read_metrics():
    rows = []
    f = PROJECT / "goal_metric_xy.tsv"
    for r in csv.DictReader(f.open(), delimiter="\t"):
        try:
            rows.append({k: r[k] for k in r})
        except Exception:
            pass
    out = []
    for r in rows:
        try:
            out.append(dict(step=int(r["ckpt"]), ratio=float(r["ratio"]),
                            gain=float(r["gain_pct"]) if r.get("gain_pct") else None,
                            sign=float(r["sign_pct"]) if r.get("sign_pct") else None))
        except (ValueError, KeyError):
            continue
    return sorted(out, key=lambda x: x["step"])


def read_loss(log=PROJECT / "train" / "act_xy_224.log"):
    """로그에서 (스텝, 손실) 을 뽑는다. step 은 'step:52K' 형식이라 구간별 평균을 쓴다."""
    pat = re.compile(r"step:(\d+)K smpl:\d+K ep:\d+ epch:[\d.]+ loss:([\d.]+)")
    acc = {}
    for m in pat.finditer(log.read_text(errors="replace")):
        k = int(m.group(1)) * 1000
        acc.setdefault(k, []).append(float(m.group(2)))
    return sorted((k, float(np.mean(v))) for k, v in acc.items())


def chart_curve(met, loss):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    s = [m["step"] / 1000 for m in met]
    r = [m["ratio"] for m in met]

    ax.axhline(BASELINE, color=GREY, ls=":", lw=1.2)
    ax.annotate(f"기준선 {BASELINE} — 목표를 보지 않는 예측기", xy=(0, 0),
                xytext=(0.985, 0.055), xycoords="axes fraction", ha="right",
                color=GREY, fontsize=8.5)

    ox = [x / 1000 for x, _ in ONEHOT]
    oy = [y for _, y in ONEHOT]
    ax.plot(ox, oy, "o-", color=GREY, lw=1.6, ms=4, label="원-핫 + FiLM (기존 방식)")
    ax.plot(s, r, "o-", color=BLUE, lw=2.2, ms=5, label="좌표 목표 (신규)")

    ax.set_xlabel("학습 스텝 (천)")
    ax.set_ylabel("자기목표 대비", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_ylim(0.7, max(3.2, max(r) * 1.15))
    ax.grid(alpha=0.25, lw=0.6)

    if loss:
        ax2 = ax.twinx()
        lx = [x / 1000 for x, _ in loss]
        ly = [y for _, y in loss]
        ax2.plot(lx, ly, "-", color=RED, lw=1.4, alpha=0.8, label="학습 손실 (로그축)")
        ax2.set_yscale("log")
        ax2.set_ylabel("학습 손실 (로그)", color=RED)
        ax2.tick_params(axis="y", labelcolor=RED)
        ax2.set_ylim(min(ly) * 0.7, max(ly) * 1.4)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if loss else ([], []))
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.set_title("목표 표현을 기호에서 좌표로 바꾼 효과", fontsize=11, pad=10)
    fig.tight_layout()
    p = OUT / "chart_xy_curve.png"
    fig.savefig(p); plt.close(fig)
    print(f"  {p}")


def chart_gain(met):
    have = [m for m in met if m["gain"] is not None]
    if not have:
        print("  응답 이득 데이터가 아직 없습니다 — 건너뜁니다")
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    s = [m["step"] / 1000 for m in have]
    ax.plot(s, [m["sign"] for m in have], "o-", color=GREEN, lw=2, ms=5,
            label="방향 일치율  (목표 쪽으로 움직였는가)")
    ax.plot(s, [m["gain"] for m in have], "o-", color=BLUE, lw=2.2, ms=5,
            label="응답 이득  (옮긴 만큼 따라갔는가)")
    ax.axhline(50, color=GREY, ls=":", lw=1.2)
    ax.annotate("50% = 무작위", xy=(s[0], 50), xytext=(s[0], 53), color=GREY, fontsize=8.5)
    ax.set_xlabel("학습 스텝 (천)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="center right", fontsize=8.5)
    ax.set_title("방향은 처음부터 맞았고, 크기가 자란다", fontsize=11, pad=10)
    fig.tight_layout()
    p = OUT / "chart_xy_gain.png"
    fig.savefig(p); plt.close(fig)
    print(f"  {p}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    met = read_metrics()
    print(f"측정점 {len(met)}개")
    chart_curve(met, read_loss())
    chart_gain(met)


def chart_pick():
    """실전 질문 — 지정한 알약을 고르는가. 학습에 안 쓴 장면에서만 잰 값."""
    import csv as _csv
    rows = {}
    f = PROJECT / "goal_metric_xy_pick.tsv"
    if not f.exists():
        return
    for r in _csv.DictReader(f.open(), delimiter="\t"):
        try:
            rows.setdefault(r["run"], []).append((int(r["ckpt"]), float(r["pick_pct"])))
        except (ValueError, KeyError):
            continue
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    label = {"act_xy_224": "좌표 목표 (학습 장면)",
             "act_marker2": "좌표 + 화면 속 표식 (미학습 장면)",
             "act_marker_INVALID_affine": "표식 — 증강 버그 (무효)"}
    color = {"act_xy_224": BLUE, "act_marker2": GREEN,
             "act_marker_INVALID_affine": GREY}
    for run, pts in rows.items():
        pts = sorted(set(pts))
        ax.plot([p[0] / 1000 for p in pts], [p[1] for p in pts], "o-", ms=5, lw=2,
                color=color.get(run, GREY), label=label.get(run, run))
    ax.axhline(50, color=RED, ls=":", lw=1.4)
    ax.annotate("50% = 무작위 (둘 중 하나를 찍는 것과 같음)", xy=(0, 0),
                xytext=(0.98, 0.08), xycoords="axes fraction", ha="right",
                color=RED, fontsize=8.5)
    ax.set_xlabel("학습 스텝 (천)")
    ax.set_ylabel("지정한 알약을 고른 비율 (%)")
    ax.set_ylim(35, 100)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("목표를 어떻게 주든 지정한 알약을 고르지 못했다", fontsize=11, pad=10)
    fig.tight_layout()
    p = OUT / "chart_xy_pick.png"
    fig.savefig(p); plt.close(fig)
    print(f"  {p}")


chart_pick()
