#!/usr/bin/env python3
"""probe_xy 가 매번 다시 계산하던 것들을 한 번만 계산해 캐시한다 (2026-08-12).

체크포인트마다 188 에피소드의 프레임을 전부 디코딩해 파지 시점을 찾느라
30분 타임아웃에 걸렸다. 이 값들은 **정책과 무관하고 데이터에만 의존**하므로
한 번 구해 두면 된다.

캐시 내용
  - 에피소드별 파지 프레임 인덱스와 그때의 정답 행동
  - u→shoulder_pan, v→shoulder_lift 회귀 계수
  - 같은 장면의 방해 목표 좌표 (다른 색 알약)

    python make_xy_cache.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from make_xy_labels import blobs_from, color_mask, frame_bgr

PROJECT = Path(__file__).resolve().parent
COLORS = ["red", "green", "yellow"]


def main():
    lab = json.loads((PROJECT / "xy_labels.json").read_text())
    labels = {int(k): v for k, v in lab["labels"].items()}
    area = lab["area"]

    ds = LeRobotDataset("1unasy/pill_v3_xy")
    fr = list(ds.meta.episodes["dataset_from_index"])
    to = list(ds.meta.episodes["dataset_to_index"])

    cache, U, V, P, L = {}, [], [], [], []
    eps = sorted(labels)
    for n, e in enumerate(eps, 1):
        a, b = int(fr[e]), int(to[e])
        act = np.array([ds[i]["action"].numpy() for i in range(a, b, 3)])
        g = act[:, 5]
        idx = np.where(g < (g.max() + g.min()) / 2)[0]
        if len(idx) == 0:
            continue
        c = int(idx[0])
        gi = a + c * 3

        # 같은 장면의 다른 색 알약 = 방해 목표
        bgr = frame_bgr(ds, a)
        tgt = np.array([labels[e]["u"], labels[e]["v"]])
        cand = []
        for col in COLORS:
            if col == labels[e]["color"]:
                continue
            for u, v, _ in blobs_from(color_mask(bgr, col, area), area):
                cand.append([u / 640, v / 480])
        wrong = None
        if cand:
            wrong = max(cand, key=lambda p: float(np.hypot(*(np.array(p) - tgt))))

        cache[str(e)] = {"from": a, "to": b, "grasp": gi,
                         "act": [round(float(x), 4) for x in act[c]],
                         "u": labels[e]["u"], "v": labels[e]["v"],
                         "color": labels[e]["color"],
                         "wrong": [round(x, 5) for x in wrong] if wrong else None}
        U.append(labels[e]["u"]); V.append(labels[e]["v"])
        P.append(float(act[c, 0])); L.append(float(act[c, 1]))
        if n % 25 == 0:
            print(f"  {n}/{len(eps)}")

    pu = np.polyfit(U, P, 1).tolist()
    pv = np.polyfit(V, L, 1).tolist()
    n_wrong = sum(1 for v in cache.values() if v["wrong"])
    print(f"\n  에피소드 {len(cache)}개   방해 목표 있는 것 {n_wrong}개")
    print(f"  pan  = {pu[0]:+.1f}·u {pu[1]:+.1f}   상관 {np.corrcoef(U, P)[0,1]:+.3f}")
    print(f"  lift = {pv[0]:+.1f}·v {pv[1]:+.1f}   상관 {np.corrcoef(V, L)[0,1]:+.3f}")

    out = PROJECT / "xy_cache.json"
    out.write_text(json.dumps({"episodes": cache, "pan_fit": pu, "lift_fit": pv},
                              indent=1, ensure_ascii=False))
    print(f"  저장: {out}")


if __name__ == "__main__":
    main()
