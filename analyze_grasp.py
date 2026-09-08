#!/usr/bin/env python3
"""파지 순간 사진에서 보정값을 계산한다.

왜 (2026-08-07): 화살표로 영점을 못 맞췄다. 파지 반사는 드물게 걸려 표본이 1~5건이었고,
학습 데이터로 낸 편향은 실기와 부호가 어긋났다(깊이). 그래서 **실기에서 실제로 집는
순간의 사진**을 남기고, 그것을 학습 때의 파지 순간과 비교한다.

기준점은 학습 데이터에서 뽑았다 (성공 시연 75건, wrist 화면에서 알약 중심 위치):
    빨강 (0.56, 0.44)   초록 (0.60, 0.54)   노랑 (0.59, 0.34)
wrist 는 그리퍼에 붙어 있으므로 이 자리가 곧 "제대로 물린 위치"다.

축척은 캡슐 색깔 절반(9mm)을 자로 써서 프레임마다 새로 잡는다 — 높이가 매번 달라
고정 mm/px 를 못 쓴다.

    python analyze_grasp.py --color red
    python analyze_grasp.py --color red --apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# 학습 성공 시연에서 잰 "제대로 물렸을 때" 알약 중심 (wrist 화면, 0~1)
TARGET = {"red": (0.56, 0.44), "green": (0.60, 0.54), "yellow": (0.59, 0.34)}
PILL_HSV = {
    "red":    [((0, 90, 60), (10, 255, 255)), ((170, 90, 60), (180, 255, 255))],
    "green":  [((38, 60, 40), (85, 255, 255))],
    "yellow": [((20, 90, 90), (35, 255, 255))],
}
CAPSULE_HALF_MM = 9.0
RUN = {"red": "act_v3_red29", "green": "act_v3_green40k", "yellow": "act_v3_yellow40k"}


def measure(path: Path, color: str):
    """사진 하나에서 (좌우 mm, 깊이 mm) 어긋남을 잰다. + = 알약이 목표보다 오른쪽/아래."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for lo, hi in PILL_HSV[color]:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    if st[i, 4] < 1500:
        return None
    major = max(st[i, 2], st[i, 3])
    if major < 20:
        return None
    # 색 절반만 잡으면 캡슐이 눕는 각도에 따라 중심이 흔들린다 (2026-08-07 실측:
    # 성공한 파지인데도 사분위 12.5mm 가 나왔다). 색 덩어리에 붙어 있는 **흰 절반**까지
    # 합쳐 캡슐 전체의 중심을 쓴다.
    white = cv2.inRange(hsv, np.array((0, 0, 150)), np.array((180, 60, 255)))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    band = np.zeros_like(m)
    cv2.drawContours(band, [np.array([[max(0, st[i,0]-60), max(0, st[i,1]-60)],
                                      [min(w, st[i,0]+st[i,2]+60), max(0, st[i,1]-60)],
                                      [min(w, st[i,0]+st[i,2]+60), min(h, st[i,1]+st[i,3]+60)],
                                      [max(0, st[i,0]-60), min(h, st[i,1]+st[i,3]+60)]])], -1, 255, -1)
    full = cv2.bitwise_or(m, cv2.bitwise_and(white, band))
    n2, _, st2, cen2 = cv2.connectedComponentsWithStats(full)
    if n2 >= 2:
        j = 1 + int(np.argmax(st2[1:, 4]))
        if st2[j, 4] > st[i, 4]:
            i, st, cen = j, st2, cen2
    major = max(st[i, 2], st[i, 3])
    mm_per_px = CAPSULE_HALF_MM / major
    tx, ty = TARGET[color]
    dx = (cen[i][0] / w - tx) * w * mm_per_px
    dy = (cen[i][1] / h - ty) * h * mm_per_px
    return float(dx), float(dy), int(st[i, 4])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", required=True, choices=list(TARGET))
    ap.add_argument("--dir", default=None, help="기본 grasp_shots/<색>")
    ap.add_argument("--apply", action="store_true", help="grasp_offset.json 에 저장")
    args = ap.parse_args()

    d = Path(args.dir or f"grasp_shots/{args.color}")
    shots = sorted(d.glob("*_wrist.png"))
    if not shots:
        raise SystemExit(f"사진 없음: {d}/*_wrist.png\n"
                         f"  먼저 --dump-grasp {d} 로 돌려서 파지 순간을 찍으세요.")

    print(f"{args.color} — 파지 순간 사진 {len(shots)}장\n")
    print(f"{'사진':<12}{'좌우 mm':>10}{'깊이 mm':>10}{'알약 px':>10}")
    rows = []
    for s in shots:
        r = measure(s, args.color)
        if r is None:
            print(f"  {s.stem[:8]:<10}{'알약 못 찾음':>20}")
            continue
        dx, dy, ar = r
        rows.append((dx, dy))
        print(f"  {s.stem[:8]:<10}{dx:>+10.1f}{dy:>+10.1f}{ar:>10}")

    if not rows:
        raise SystemExit("\n잰 게 없다 — 사진에 알약이 안 보인다")

    a = np.array(rows)
    lat = float(np.median(a[:, 0]))
    dep = float(np.median(a[:, 1]))
    # 알약이 목표보다 오른쪽(+dx)에 찍혔다 = 손이 왼쪽에 있었다 → 오른쪽(+)으로 보낸다
    # 알약이 목표보다 아래(+dy)에 찍혔다  = 손이 뒤(먼 쪽)에 있었다 → 가까이(-)로 보낸다
    rec_lat, rec_dep = lat, -dep

    print("-" * 44)
    print(f"  중앙값        좌우 {lat:+.1f}mm   깊이 {dep:+.1f}mm   (표본 {len(rows)})")
    if len(rows) > 1:
        print(f"  퍼짐(사분위)  좌우 {np.subtract(*np.percentile(a[:,0],[75,25])):.1f}mm"
              f"   깊이 {np.subtract(*np.percentile(a[:,1],[75,25])):.1f}mm")
    print()
    print(f"  ▶ 권장 보정   좌우 {rec_lat:+.0f}mm   깊이 {rec_dep:+.0f}mm")

    if args.apply:
        from datetime import datetime
        p = Path("grasp_offset.json")
        j = json.loads(p.read_text())
        k = f"{RUN[args.color]}|pick {args.color} pill"
        old = j.get(k, {})
        nd = round(float(old.get("depth_mm", 0)) + rec_dep)
        nl = round(float(old.get("lateral_mm", 0)) + rec_lat)
        j[k] = {"depth_mm": float(nd), "lateral_mm": float(nl), "z_mm": 0.0,
                "zones": old.get("zones", {}),
                "saved": datetime.now().isoformat(timespec="seconds"),
                "note": f"analyze_grasp.py 사진 {len(rows)}장에서 계산 "
                        f"(이전 깊이{old.get('depth_mm',0):+.0f} 좌우{old.get('lateral_mm',0):+.0f})"}
        p.write_text(json.dumps(j, indent=1, ensure_ascii=False))
        print(f"\n  ✅ 저장:  깊이 {old.get('depth_mm',0):+.0f} → {nd:+d}"
              f"   좌우 {old.get('lateral_mm',0):+.0f} → {nl:+d}")


if __name__ == "__main__":
    main()
