#!/usr/bin/env python
"""트레이·약통을 학습 때 위치로 되돌린다 (구역별 실시간 측정).

`align_top.py` 는 **카메라**를 맞추는 도구다 — 세 구역의 이동량 중앙값을 쓰기 때문에
카메라가 그대로인데 매트만 밀린 경우에는 중앙값에 묻혀서 안 보인다.
이 스크립트는 구역을 **따로따로** 보여준다. 그래야 무엇이 움직였는지 구분된다.

  로봇 베이스만 0 이 아니다  → 로봇이 밀렸다
  매트 구역만 0 이 아니다    → 매트가 밀렸다 (이 스크립트로 되돌린다)
  전부 같은 값이다           → 카메라가 밀렸다 (align_top.py 로 맞춘다)

2026-08-03 밤에 실제로 이 경우가 나왔다. 카메라·모터·통신·제어주기가 전부 정상인데
정책이 알약 앞에서 튀었고, 재보니 **로봇 베이스는 0/-1 인데 매트 아래쪽이 -16px(≈11mm),
약통이 -9px(≈6mm) 위로 밀려** 있었다. 팔이 집으면서 매트를 밀어올린 것으로 보인다.
평가 중간에 갑자기 증상이 생겨서 그 뒤로 계속됐다는 관찰과 성격이 맞는다.

  python ~/omx_pill_project/align_scene.py

  화면 : 빨강=학습 때, 청록=지금. 구역마다 화살표와 숫자가 뜬다.
  매트와 약통을 밀어서 각 구역의 dx, dy 를 0 으로 만든 뒤 고정한다.
  q/ESC : 종료
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path("/home/user/omx_pill_project")
TOP = "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"

# (y0, y1, x0, x1) — 알약이 놓이는 자리를 피해서, 각 물체를 하나씩만 걸치도록 잡았다
REGIONS = {
    "robot": (400, 478, 190, 400),
    "mat_top": (95, 135, 160, 460),
    "mat_bottom": (355, 385, 160, 460),
    "mat_left": (150, 330, 115, 155),
    "mat_right": (150, 330, 480, 520),
    "pillbox": (130, 350, 520, 610),
}
LABEL = {
    "robot": "로봇",
    "mat_top": "매트 위",
    "mat_bottom": "매트 아래",
    "mat_left": "매트 왼",
    "mat_right": "매트 오른",
    "pillbox": "약통",
}
OK_PX = 4.0


def measure(ref_gray: np.ndarray, now_gray: np.ndarray) -> dict[str, tuple[int, int, float]]:
    out = {}
    for name, (y0, y1, x0, x1) in REGIONS.items():
        res = cv2.matchTemplate(now_gray, ref_gray[y0:y1, x0:x1], cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        out[name] = (loc[0] - x0, loc[1] - y0, float(score))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="트레이·약통을 학습 위치로 되돌린다")
    ap.add_argument("--ref", default=str(PROJECT / "preflight_ref" / "top_train.png"))
    args = ap.parse_args()

    ref = cv2.imread(args.ref)
    if ref is None:
        raise SystemExit(f"기준 이미지를 못 읽습니다: {args.ref}")
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture(TOP, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("top 카메라를 못 엽니다 (평가·녹화가 돌고 있으면 끄세요)")

    print("빨강=학습 때, 청록=지금. 매트와 약통을 밀어 각 구역을 0 으로 만드세요. q 로 종료.")
    win = "align scene  (RED=TRAIN  CYAN=NOW)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 960)

    last = {}
    while True:
        ok, now = cap.read()
        if not ok:
            continue
        g_now = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY)
        last = measure(g_ref, g_now)

        view = cv2.merge([g_now, g_now, g_ref])
        for name, (y0, y1, x0, x1) in REGIONS.items():
            dx, dy, score = last[name]
            good = abs(dx) < OK_PX and abs(dy) < OK_PX
            color = (0, 255, 0) if good else (0, 165, 255)
            cv2.rectangle(view, (x0, y0), (x1, y1), color, 1)
            # 어느 쪽으로 밀어야 하는지: 지금이 (dx,dy) 만큼 갔으니 반대로 되돌린다
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if not good:
                cv2.arrowedLine(view, (cx, cy), (cx - dx * 4, cy - dy * 4), color, 2, tipLength=0.3)
            txt = f"{LABEL[name]} {dx:+d},{dy:+d}"
            cv2.putText(view, txt, (x0 + 2, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(view, txt, (x0 + 2, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        moved = [LABEL[n] for n, (dx, dy, _) in last.items()
                 if max(abs(dx), abs(dy)) >= OK_PX]
        head = "OK - 전부 제자리" if not moved else "밀린 곳: " + ", ".join(moved)
        cv2.putText(view, head, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(view, head, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if not moved else (0, 165, 255), 2)
        cv2.putText(view, "arrow = push this way", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(view, "arrow = push this way", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(win, view)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n마지막 측정:")
    for name, (dx, dy, score) in last.items():
        print(f"  {LABEL[name]:<8} dx={dx:+3d} dy={dy:+3d} (일치도 {score:.2f})")


if __name__ == "__main__":
    main()
