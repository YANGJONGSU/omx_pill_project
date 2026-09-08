#!/usr/bin/env python3
"""좌표 조건화가 성립하는지 학습 전에 확인한다 (2026-08-12).

핵심 질문 두 가지:
  ① 탑뷰 첫 프레임에서 목표 색 알약을 HSV 로 검출할 수 있는가
  ② 목표 색 덩어리가 **한 개로 특정되는가** (두 개면 어느 쪽이 목표인지 모른다)

②가 깨지면 좌표 라벨을 만들 수 없다. 그 경우 파지 시점 위치와 대조해야 하는데
그건 픽셀↔로봇 호모그래피가 필요하다. 이 스크립트는 그 필요 여부를 판정한다.

    python check_hsv_labels.py --n 60
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

# OpenCV HSV: H 는 0~179 (실제 각도의 절반). 빨강 354도 → 177, 노랑 50도 → 25, 초록 134도 → 67
# 실측값 (ep0/ep60 캡슐 화소 10/50/90 분위):
#   빨강 H 169~178  S  73~165  V 148~244
#   초록 H  63~ 71  S  77~110  V 134~226
#   노랑 H  26~ 29  S  46~ 82  V 165~255   ← 채도가 낮다. 90 을 요구하면 전부 놓친다
RANGES = {
    "red":    [((0, 60, 90), (6, 255, 255)), ((165, 60, 90), (179, 255, 255))],
    "yellow": [((20, 35, 120), (34, 255, 255))],
    "green":  [((55, 55, 90), (80, 255, 255))],
}
# 캡슐은 절반만 유색이라 색 영역이 35~85px 에 불과하다
MIN_AREA = 15
MAX_AREA = 1200

# 로봇 본체(초록 PCB·케이블)가 ROI 하단 중앙에 들어와 초록 오검출을 낸다.
# 알약을 놓지 않는 영역이므로 제외한다.
ROBOT_BOX = (225, 235, 340, 310)      # x0, y0, x1, y1


def detect(bgr, color, area):
    x0, y0, x1, y1 = area["x0"], area["y0"], area["x1"], area["y1"]
    roi = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    rx0, ry0, rx1, ry1 = ROBOT_BOX
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        cx, cy = float(cent[i][0] + x0), float(cent[i][1] + y0)
        if not (MIN_AREA <= a <= MAX_AREA):
            continue
        if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:      # 로봇 본체 영역
            continue
        out.append((cx, cy, int(a)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="1unasy/pill_v3")
    ap.add_argument("--n", type=int, default=60, help="검사할 에피소드 수 (고르게 표본)")
    a = ap.parse_args()

    area = json.loads((PROJECT / "area.json").read_text())
    ds = LeRobotDataset(a.repo_id)
    total = ds.meta.total_episodes
    fr = list(ds.meta.episodes["dataset_from_index"])
    to = list(ds.meta.episodes["dataset_to_index"])

    eps = list(range(0, total, max(1, total // a.n)))[: a.n]
    counts, rows, fails = Counter(), [], []

    for e in eps:
        task = ds.meta.episodes["tasks"][e]
        task = task.as_py() if hasattr(task, "as_py") else task
        task = task[0] if isinstance(task, (list, tuple)) else task
        color = next((c for c in COLORS if c in str(task)), None)
        if color is None:
            continue
        if int(to[e]) - int(fr[e]) == 449:      # 정지 시연은 집지 않으므로 제외
            continue

        img = ds[int(fr[e])]["observation.images.top"]
        rgb = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        blobs = detect(bgr, color, area)
        counts[len(blobs)] += 1
        rows.append((e, color, len(blobs)))
        if len(blobs) != 1:
            fails.append((e, color, len(blobs), [(round(x), round(y), s) for x, y, s in blobs[:4]]))

    n = len(rows)
    print(f"\n좌표 라벨 가능성 점검 — {n}개 에피소드 표본 (전체 {total})")
    print("=" * 66)
    for k in sorted(counts):
        tag = " ← 라벨 확정 가능" if k == 1 else (" ← 검출 실패" if k == 0 else " ← 모호. 어느 쪽이 목표인지 불명")
        print(f"  목표색 덩어리 {k}개 : {counts[k]:3d}개 ({counts[k]/n*100:5.1f}%){tag}")
    print("-" * 66)
    ok = counts.get(1, 0) / n * 100
    print(f"  **단일 검출률 {ok:.1f}%**")
    print()
    if ok >= 90:
        print("  판정: 좌표 라벨을 바로 만들 수 있습니다. 호모그래피 불필요.")
    elif ok >= 70:
        print("  판정: 대부분 가능. 모호한 것만 파지 위치로 판별하면 됩니다.")
    else:
        print("  판정: 검출이 불안정합니다. HSV 범위를 먼저 조정해야 합니다.")

    if fails:
        print("\n  문제 에피소드 (최대 15개):")
        for e, c, k, b in fails[:15]:
            print(f"    ep{e:3d} {c:7s} 덩어리 {k}개  {b}")

    (PROJECT / "hsv_label_check.json").write_text(
        json.dumps({"rows": rows, "counts": dict(counts), "single_rate": ok},
                   indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
