#!/usr/bin/env python3
"""좌표 조건화 라벨 생성 — 목표 알약의 탑뷰 픽셀 좌표 (2026-08-12).

원-핫 [1,0,0] 대신 목표 알약의 위치 [u, v] 를 목표로 준다.
목표가 출력과 같은 좌표계에 있으므로 무시하면 손실이 즉시 오른다 — 지름길이 없다.

로봇 본체(초록 PCB·케이블·빨간 표시)가 색 검출에 걸린다. 손으로 박스를 그리면
그 안의 진짜 알약까지 버리게 되므로(ep3), **데이터에서 정적 마스크를 뽑는다**:
모든 에피소드 첫 프레임에서 같은 자리에 같은 색이 계속 잡히면 그건 알약이 아니다.

    python make_xy_labels.py                 # 전체 처리
    python make_xy_labels.py --dump 8        # 검출 결과 이미지도 저장
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT = Path(__file__).resolve().parent
COLORS = ["red", "green", "yellow"]
STOP_LEN = 449                 # 정지 시연 길이. 집는 동작이 없어 목표 좌표가 존재하지 않는다

# 실측 캡슐 화소값 (10/50/90 분위)
#   빨강 H 169~178 S  73~165 V 148~244
#   초록 H  63~ 71 S  77~110 V 134~226
#   노랑 H  26~ 29 S  46~ 82 V 165~255   ← 채도가 낮다
RANGES = {
    "red":    [((0, 60, 90), (6, 255, 255)), ((165, 60, 90), (179, 255, 255))],
    "yellow": [((20, 35, 120), (34, 255, 255))],
    "green":  [((55, 55, 90), (80, 255, 255))],
}
MIN_AREA, MAX_AREA = 15, 1200  # 캡슐은 절반만 유색이라 색 영역이 35~85px
STATIC_FRAC = 0.35             # 에피소드 35% 이상에서 잡히는 화소 = 로봇/배경
RETRY_FRAMES = (0, 6, 12, 20)  # 첫 프레임에서 가려지면 조금 뒤를 본다


def color_mask(bgr, color, area):
    x0, y0, x1, y1 = area["x0"], area["y0"], area["x1"], area["y1"]
    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RANGES[color]:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


# 로봇 본체가 트레이 영역 안에 들어온다 (초록 PCB·케이블·빨간 표시).
# 라벨은 팔이 홈에 있는 첫 프레임에서 만들므로 영향이 적지만, **추론 중에는
# 팔이 트레이 위를 지나며 자기 부품이 알약으로 잡힌다.** 2026-08-17 실기 영상에서
# 목표가 로봇 본체 쪽으로 166px 튀어 팔이 그리로 내려갔다.
ROBOT_BOX = (225, 235, 340, 310)      # x0, y0, x1, y1 (화면 좌표)


def blobs_from(mask, area, static=None, exclude_robot=False):
    if static is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(static))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if not (MIN_AREA <= a <= MAX_AREA):
            continue
        cx, cy = float(cent[i][0] + area["x0"]), float(cent[i][1] + area["y0"])
        if exclude_robot:
            rx0, ry0, rx1, ry1 = ROBOT_BOX
            if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
                continue
        out.append((cx, cy, a))
    return out


def frame_bgr(ds, idx):
    a = ds[int(idx)]["observation.images.top"].numpy()
    return cv2.cvtColor((a.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="1unasy/pill_v3")
    ap.add_argument("--dump", type=int, default=0, help="검출 결과 이미지 N개 저장")
    a = ap.parse_args()

    area = json.loads((PROJECT / "area.json").read_text())
    ds = LeRobotDataset(a.repo_id)
    fr = list(ds.meta.episodes["dataset_from_index"])
    to = list(ds.meta.episodes["dataset_to_index"])
    H, W = area["y1"] - area["y0"], area["x1"] - area["x0"]

    picks = []
    for e in range(ds.meta.total_episodes):
        t = ds.meta.episodes["tasks"][e]
        t = t.as_py() if hasattr(t, "as_py") else t
        t = t[0] if isinstance(t, (list, tuple)) else t
        c = next((c for c in COLORS if c in str(t)), None)
        if c and int(to[e]) - int(fr[e]) != STOP_LEN:
            picks.append((e, c))
    print(f"집기 에피소드 {len(picks)}개 (정지 시연 {ds.meta.total_episodes - len(picks)}개 제외)")

    # ── 1차: 정적 마스크 만들기 ────────────────────────────────────────────
    acc = {c: np.zeros((H, W), np.int32) for c in COLORS}
    cnt = Counter()
    cache = {}
    for e, c in picks:
        bgr = frame_bgr(ds, fr[e])
        cache[e] = bgr
        m = color_mask(bgr, c, area)
        acc[c] += (m > 0).astype(np.int32)
        cnt[c] += 1

    static = {}
    for c in COLORS:
        s = ((acc[c] >= max(2, int(cnt[c] * STATIC_FRAC))) * 255).astype(np.uint8)
        static[c] = cv2.dilate(s, np.ones((3, 3), np.uint8))
        print(f"  {c:7s} 정적 화소 {int((static[c] > 0).sum()):5d}개 제외 (로봇·배경)")

    # ── 2차: 알약만 남기고 검출 ───────────────────────────────────────────
    labels, fails, counts = {}, [], Counter()
    for e, c in picks:
        got = None
        for off in RETRY_FRAMES:
            idx = fr[e] + off
            if idx >= to[e]:
                break
            bgr = cache[e] if off == 0 else frame_bgr(ds, idx)
            b = blobs_from(color_mask(bgr, c, area), area, static[c])
            if len(b) == 1:
                got = (b[0], off)
                break
            if got is None and len(b) > 1:
                got = ("multi", off, len(b))
        if got and got[0] != "multi":
            (u, v, ar), off = got
            labels[e] = {"color": c, "u": round(u / 640, 5), "v": round(v / 480, 5),
                         "px": [round(u, 1), round(v, 1)], "area": ar, "frame_off": off}
            counts["ok"] += 1
        else:
            fails.append((e, c, got[2] if got else 0))
            counts["fail"] += 1

    n = len(picks)
    print(f"\n라벨 생성 결과")
    print("=" * 60)
    print(f"  성공 {counts['ok']:3d}개 ({counts['ok']/n*100:.1f}%)   실패 {counts['fail']:3d}개")
    if fails:
        print(f"  실패 목록: " + ", ".join(f"ep{e}({c},{k}개)" for e, c, k in fails[:20]))

    # 좌표 분포 — 격자를 고르게 덮는지 (원-핫에서는 목표가 3종류뿐이었다)
    if labels:
        us = np.array([d["u"] for d in labels.values()])
        vs = np.array([d["v"] for d in labels.values()])
        print(f"\n  목표 좌표 종류 {len(labels)}가지")
        print(f"    u 범위 {us.min():.3f} ~ {us.max():.3f}   표준편차 {us.std():.3f}")
        print(f"    v 범위 {vs.min():.3f} ~ {vs.max():.3f}   표준편차 {vs.std():.3f}")

    out = PROJECT / "xy_labels.json"
    out.write_text(json.dumps({"labels": {str(k): v for k, v in labels.items()},
                               "fails": fails, "area": area}, indent=1, ensure_ascii=False))
    print(f"\n  저장: {out}")

    if a.dump:
        d = Path("/home/user/.claude/jobs/fd2b9c3d/tmp/xy")
        d.mkdir(parents=True, exist_ok=True)
        for e in list(labels)[:: max(1, len(labels) // a.dump)][: a.dump]:
            bgr = cache[e].copy()
            u, v = labels[e]["px"]
            cv2.circle(bgr, (int(u), int(v)), 14, (0, 255, 255), 2)
            cv2.rectangle(bgr, (area["x0"], area["y0"]), (area["x1"], area["y1"]), (120, 120, 120), 1)
            cv2.imwrite(str(d / f"ep{e}_{labels[e]['color']}.png"), bgr)
        print(f"  검출 이미지: {d}")


if __name__ == "__main__":
    main()
