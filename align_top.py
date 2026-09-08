#!/usr/bin/env python3
"""top 카메라를 예전 위치로 되돌리는 도구 (2026-08-09).

21:30~21:33 사이에 카메라가 왼쪽으로 밀려서 화면이 +96px 오른쪽으로 이동했다.
기존 145개 중 144개는 예전 위치에서 찍혔으므로, 새로 찍는 것도 같은 위치여야 섞인다.

기준 프레임을 반투명으로 겹쳐 보여주고, 위상 상관으로 어긋난 픽셀 수를 실시간 표시한다.
|dx|,|dy| 가 3px 아래로 내려오면 초록색으로 바뀐다 — 그때 카메라를 고정하고 q 를 누른다.

    python align_top.py
"""
import cv2
import numpy as np

TOP = "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"
REF = "/tmp/claude-1000/-home-user/50ba30d5-878b-404a-8c7f-7b9c0e2c8e75/scratchpad/top_old.png"
TOL = 3.0

ref = cv2.imread(REF)
if ref is None:
    raise SystemExit(f"기준 프레임을 못 읽었다: {REF}")
ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

cap = cv2.VideoCapture(TOP, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    raise SystemExit("top 카메라를 못 열었다. 녹화 창이 켜져 있는지 확인해라.")

print("카메라를 조금씩 움직여 dx, dy 를 0 에 맞춰라. 초록색이 되면 고정하고 q.")
mode = 0  # 0=겹쳐보기, 1=차이보기, 2=현재만
while True:
    ok, frame = cap.read()
    if not ok:
        continue
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(ref_g, g)
    good = abs(dx) < TOL and abs(dy) < TOL

    if mode == 0:
        view = cv2.addWeighted(frame, 0.5, ref, 0.5, 0)
    elif mode == 1:
        view = cv2.absdiff(frame, ref)
    else:
        view = frame.copy()

    color = (0, 220, 0) if good else (0, 80, 255)
    cv2.rectangle(view, (0, 0), (639, 62), (0, 0, 0), -1)
    cv2.putText(view, f"dx {dx:+6.1f}  dy {dy:+6.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(view, "OK - fix camera and press q" if good else "move camera",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    # 어느 쪽으로 밀어야 하는지 화살표
    if not good:
        cx, cy = 480, 30
        ax = int(np.clip(-dx, -60, 60))
        cv2.arrowedLine(view, (cx, cy), (cx + ax, cy), color, 3, tipLength=0.3)
    cv2.putText(view, "m: view mode   q: quit", (400, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imshow("top 정렬 (기준=예전 위치)", view)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("m"):
        mode = (mode + 1) % 3

cap.release()
cv2.destroyAllWindows()
print(f"마지막 어긋남  dx={dx:+.1f}  dy={dy:+.1f}")
print("카메라를 놓았습니다. 이제 녹화를 시작해도 됩니다.")
