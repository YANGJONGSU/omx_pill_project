#!/usr/bin/env python
"""카메라 자리 잡기용 실시간 뷰어 (처음 세팅할 때 쓴다).

`camera_live.py` 는 **이미 찍어둔 학습 데이터와 화각을 맞출 때** 쓰는 도구라
데이터셋이 있어야 한다. 이건 그 전 단계 — 데이터가 아직 없을 때 트레이와 약통이
화면에 잘 들어오게 카메라를 놓는 용도다.

화면에 그려주는 것:
  · 중앙 십자선과 중앙 영역 박스 — 트레이를 여기에 맞춘다
  · 50픽셀 격자와 좌표 — `--set-area` 로 지정할 범위를 눈으로 가늠할 수 있다
  · 가장자리 여백선 — 알약이 이 밖으로 나가면 화면에서 잘린다

사용법:
  python ~/omx_pill_project/camera_setup.py            # top 카메라
  python ~/omx_pill_project/camera_setup.py --cam wrist
  python ~/omx_pill_project/camera_setup.py --both     # 두 대 나란히

키: s = 지금 화면 저장,  q 또는 ESC = 종료

끝나면 반드시 창을 닫을 것 — 이 창이 카메라를 잡고 있으면 녹화가 안 된다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path("/home/user/omx_pill_project")
CAMS = {
    "top": "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0",
    "wrist": "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0",
}


def open_cam(path: str):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"카메라를 열 수 없습니다: {path}\n(녹화·평가가 돌고 있으면 먼저 끄세요)")
    return cap


def annotate(frame: np.ndarray, name: str) -> np.ndarray:
    """격자·중앙선·여백선을 그린다. 원본은 건드리지 않는다."""
    img = frame.copy()
    h, w = img.shape[:2]

    # 50픽셀 격자 (연한 회색)
    for x in range(0, w, 50):
        cv2.line(img, (x, 0), (x, h), (70, 70, 70), 1)
        cv2.putText(img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    for y in range(0, h, 50):
        cv2.line(img, (0, y), (w, y), (70, 70, 70), 1)
        cv2.putText(img, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

    # 가장자리 여백 (이 밖은 잘릴 위험)
    m = 40
    cv2.rectangle(img, (m, m), (w - m, h - m), (0, 140, 255), 1)

    # 중앙 영역 — 트레이를 여기 맞춘다
    cx, cy = w // 2, h // 2
    bw, bh = int(w * 0.30), int(h * 0.30)
    cv2.rectangle(img, (cx - bw, cy - bh), (cx + bw, cy + bh), (0, 230, 0), 2)

    # 중앙 십자선
    cv2.line(img, (cx, cy - 18), (cx, cy + 18), (0, 230, 0), 1)
    cv2.line(img, (cx - 18, cy), (cx + 18, cy), (0, 230, 0), 1)

    # 밝기 (조명이 너무 어둡거나 날아가지 않았는지)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    label = f"{name}  mean {g.mean():.0f}  std {g.std():.0f}"
    if g.mean() < 60:
        label += "  <- 너무 어둡다"
    elif g.mean() > 200:
        label += "  <- 너무 밝다"
    cv2.putText(img, label, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, label, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description="카메라 자리 잡기 실시간 뷰어")
    ap.add_argument("--cam", choices=list(CAMS), default="top")
    ap.add_argument("--both", action="store_true", help="두 대를 나란히 본다")
    args = ap.parse_args()

    names = list(CAMS) if args.both else [args.cam]
    caps = {n: open_cam(CAMS[n]) for n in names}

    print("=" * 58)
    print(" 초록 박스  = 화면 중앙 영역. 트레이를 여기에 맞춘다")
    print(" 주황 테두리 = 가장자리 여백. 트레이·약통이 이 안에 다 들어와야 한다")
    print(" 회색 격자   = 50픽셀 (--set-area 로 지정할 범위 가늠용)")
    print("-" * 58)
    print(" 확인할 것: 트레이가 중앙에, 약통도 화면 안에,")
    print("            케이블·공구 같은 게 화면에 안 들어오게")
    print("-" * 58)
    print(" s = 화면 저장    q 또는 ESC = 종료")
    print("=" * 58)

    try:
        while True:
            shots = []
            for n in names:
                ok, frame = caps[n].read()
                if not ok:
                    continue
                shots.append(annotate(frame, n))
            if not shots:
                continue

            view = np.hstack(shots) if len(shots) > 1 else shots[0]
            cv2.imshow("camera setup  (s=save, q=quit)", view)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                out = PROJECT / "report" / "camera_setup.png"
                out.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(out), view)
                print(f"저장: {out}")
    except KeyboardInterrupt:
        pass
    finally:
        for c in caps.values():
            c.release()
        cv2.destroyAllWindows()
        print("\n종료했습니다. 카메라를 놓았으니 이제 녹화·set-area 를 할 수 있습니다.")


if __name__ == "__main__":
    main()
