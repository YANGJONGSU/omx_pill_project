#!/usr/bin/env python
"""top·wrist 두 카메라를 학습 때 화면과 나란히 띄운다 (눈으로 확인용).

위 줄 = 지금, 아래 줄 = 학습 때. 왼쪽이 top, 오른쪽이 wrist.

wrist 는 팔에 달려 있어서 **팔이 홈 자세일 때만** 비교가 성립한다
(학습 기준도 에피소드 시작 프레임 = 홈 자세로 만들었다).
팔이 홈이 아니면 wrist 쪽 비교는 무의미하니 자세부터 맞출 것.

  python ~/omx_pill_project/cams_view.py

  b : 겹쳐보기 토글 (빨강=학습, 청록=지금). 어긋나면 색이 갈라져 보인다
  q : 종료
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT = Path("/home/user/omx_pill_project")
rot180 = False
show_targets = True
REF = PROJECT / "preflight_ref"
TOP = "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"
WRIST = "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0"


def open_cam(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"카메라를 못 엽니다: {path}\n(평가·녹화가 돌고 있으면 먼저 끄세요)")
    return cap


# top 화면에 배치 목표를 겹쳐 그린다 (top_view.py 와 같은 좌표).
# 카메라 하나를 두 프로그램이 동시에 못 열기 때문에 목표선을 여기로 합쳤다.
from top_view import _targets


def draw_targets(img: np.ndarray) -> np.ndarray:
    if not show_targets:
        return img
    out = img.copy()
    for (x0, y0), (x1, y1), name, col in _targets(rot180):
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        cv2.putText(out, name, (x0 + 5, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(out, name, (x0 + 5, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return out


def tag(img: np.ndarray, text: str, color=(0, 255, 255)) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="top·wrist 동시 확인 + 배치 목표선")
    ap.add_argument("--rot180", action="store_true", help="카메라가 반대편에서 볼 때")
    ap.add_argument("--no-targets", action="store_true", help="배치 목표선 끄기")
    args = ap.parse_args()
    global rot180, show_targets
    rot180, show_targets = args.rot180, not args.no_targets

    refs = {}
    for cam, name in (("top", "top_train.png"), ("wrist", "wrist_train.png")):
        p = REF / name
        refs[cam] = cv2.imread(str(p)) if p.exists() else None
        if refs[cam] is None:
            print(f"기준 없음: {p.name} — 아래 줄은 검게 나옵니다")
            refs[cam] = np.zeros((480, 640, 3), np.uint8)

    caps = {"top": open_cam(TOP), "wrist": open_cam(WRIST)}
    win = "cameras   (top | wrist)   b=blend  q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 960)
    blend = False

    while True:
        live = {}
        for cam, cap in caps.items():
            ok, fr = cap.read()
            live[cam] = fr if ok else np.zeros((480, 640, 3), np.uint8)

        if blend:
            tiles = []
            for cam in ("top", "wrist"):
                g_now = cv2.cvtColor(live[cam], cv2.COLOR_BGR2GRAY)
                g_ref = cv2.cvtColor(refs[cam], cv2.COLOR_BGR2GRAY)
                tiles.append(tag(cv2.merge([g_now, g_now, g_ref]), f"{cam}  RED=TRAIN CYAN=NOW"))
            view = np.hstack(tiles)
        else:
            top_row = np.hstack([tag(draw_targets(live["top"]), "NOW  top"),
                                 tag(live["wrist"], "NOW  wrist")])
            bot_row = np.hstack([tag(refs["top"], "TRAIN  top", (0, 165, 255)),
                                 tag(refs["wrist"], "TRAIN  wrist", (0, 165, 255))])
            view = np.vstack([top_row, bot_row])

        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("b"):
            blend = not blend

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
