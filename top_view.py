#!/usr/bin/env python
"""top 카메라 생중계 — 벽에 달면서 화면을 보기 위한 것.

설치 조건: **트레이 전체와 약통이 한 화면에** 들어와야 한다. 각도는 자유다
(화각 기준은 녹화가 끝난 뒤 학습 영상에서 새로 뽑으므로 지금 기준과 맞출 필요가 없다).

화면 위 숫자:
  sharp  — 선명도. 초점이 맞았는지. 트레이 무늬가 보이는 상태에서 높을수록 좋다
  bright — 밝기 평균. 너무 어둡거나(80 이하) 날아가면(200 이상) 조명을 손볼 것

  q / ESC : 종료
"""

from __future__ import annotations

import cv2

TOP = "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"

# v3 배치 목표 (640x480 화면 좌표).
# 기본(v2 구도): 로봇이 아래, 매트가 앞, 약통이 오른쪽.
# 벽에 단 카메라가 반대편에서 보면 화면이 180도 돌아간다 → --rot180 으로 목표도 뒤집는다.
# **정책은 화면 방향을 신경 쓰지 않는다.** 학습과 평가가 같은 방향이기만 하면 된다.
_BASE = [
    (((55, 70), (470, 390)), "MAT (pills here)", (0, 200, 255)),
    (((490, 110), (620, 400)), "PILL BOX", (255, 180, 0)),
    (((230, 395), (430, 476)), "ROBOT BASE", (180, 180, 180)),
]


def _targets(rot180: bool):
    out = []
    for ((x0, y0), (x1, y1)), name, col in _BASE:
        if rot180:
            x0, x1 = 640 - x1, 640 - x0
            y0, y1 = 480 - y1, 480 - y0
        out.append(((x0, y0), (x1, y1), name, col))
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="top 카메라 생중계 + v3 배치 목표")
    ap.add_argument("--rot180", action="store_true",
                    help="카메라가 반대편에서 보고 있을 때 (로봇이 화면 위, 약통이 왼쪽)")
    args = ap.parse_args()
    global TARGETS, MAT_BOX
    TARGETS = _targets(args.rot180)
    MAT_BOX = (TARGETS[0][0], TARGETS[0][1])

    cap = cv2.VideoCapture(TOP, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("top 카메라를 못 엽니다 (평가·녹화가 돌고 있으면 먼저 끄세요)")

    print("벽에 다는 중 확인용. 트레이 전체 + 약통이 한 화면에 들어오게 하세요. q 로 종료.")
    win = "top view (wall mount)   q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 960)

    while True:
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        txt = f"sharp {cv2.Laplacian(g, cv2.CV_64F).var():.0f}   bright {g.mean():.0f}"
        cv2.putText(fr, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(fr, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        h, w = fr.shape[:2]
        # 안내선: 이 안쪽에 검은 매트가 통째로 들어오면 된다.
        # 가장자리에 붙으면 알약이 화면 밖으로 나가고, 너무 작으면 알약이 몇 픽셀 안 된다.
        # ---- v3 배치 목표 (640x480 기준) -------------------------------
        # v2 와 같은 구도다: 로봇이 아래(가까운 쪽), 매트가 그 앞, 약통이 오른쪽.
        # 그 구도로 20개를 찍어봤고 팔이 닿는 범위·가림 문제가 없다는 게 확인됐다.
        # 카메라나 물건을 옮겨서 실물이 이 네모 안에 들어오게 한다.
        for (tx0, ty0), (tx1, ty1), name, col in TARGETS:
            cv2.rectangle(fr, (tx0, ty0), (tx1, ty1), col, 2)
            cv2.putText(fr, name, (tx0 + 5, ty0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(fr, name, (tx0 + 5, ty0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        # 매트(어두운 큰 덩어리)가 지금 화면 밖으로 잘렸는지 자동 판정
        dark = (cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) < 110).astype("uint8")
        n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
        if n > 1:
            i = 1 + int(st[1:, 4].argmax())
            mx, my, mw, mh, ma = st[i]
            touch = mx <= 2 or my <= 2 or mx + mw >= w - 2 or my + mh >= h - 2
            fill = 100 * ma / (w * h)
            (gx0, gy0), (gx1, gy1) = MAT_BOX
            ox = max(0, min(mx + mw, gx1) - max(mx, gx0))
            oy = max(0, min(my + mh, gy1) - max(my, gy0))
            iou = (ox * oy) / max(mw * mh + (gx1 - gx0) * (gy1 - gy0) - ox * oy, 1)
            msg = ("MAT CUT OFF - move camera back" if touch
                   else ("MAT TOO SMALL - move closer" if fill < 15
                         else (f"OK  (iou {iou:.2f})" if iou > 0.6 else f"align to MAT box (iou {iou:.2f})")))
            col = (0, 0, 255) if msg != "OK" else (0, 255, 0)
            cv2.rectangle(fr, (mx, my), (mx + mw, my + mh), col, 2)
            cv2.putText(fr, f"{msg}   mat {fill:.0f}%% of frame", (10, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(fr, f"{msg}   mat {fill:.0f}%% of frame", (10, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.imshow(win, fr)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
