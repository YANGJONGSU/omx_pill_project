#!/usr/bin/env python
"""wrist 카메라 초점을 학습 때 수준으로 맞춘다.

2026-08-03 밤에 wrist 선명도가 **184** 로 떨어져 있었다 (학습 때 중앙값 ~2900, 범위 1287~5369).
15배 흐린 상태였고, 팔을 세워둔 정지 상태에서도 같은 값이라 흔들림이 아니라 렌즈 문제다.
Innomaker U20CAM 은 렌즈를 돌려 맞추는 수동 초점이라 팔 진동에 조금씩 풀릴 수 있다.

wrist 는 그리퍼가 알약에 닿는 순간을 보는 카메라다. 흐려지면 색과 대략 위치는 맞히지만
마지막 1~2cm 조준을 잃는다 — 실제로 그 증상이 났다.

  python ~/omx_pill_project/focus_wrist.py

렌즈를 천천히 돌리면서 숫자가 최대가 되는 지점을 찾는다. 1300 이상이면 학습 범위 안이다.
팔은 아무 자세나 상관없지만, 트레이처럼 무늬가 있는 걸 보고 있어야 숫자가 의미 있다.
q 로 종료.
"""
import cv2, numpy as np

WRIST = "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0"
TRAIN_MIN, TRAIN_MED = 1287.0, 2900.0

cap = cv2.VideoCapture(WRIST, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("wrist 카메라를 못 엽니다 (평가·녹화가 돌고 있으면 끄세요)")

win = "wrist focus   (target >= 1300)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 1280, 960)
hist, best = [], 0.0
while True:
    ok, fr = cap.read()
    if not ok:
        continue
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    v = float(cv2.Laplacian(g, cv2.CV_64F).var())
    hist.append(v); hist[:] = hist[-120:]
    best = max(best, v)

    color = (0, 255, 0) if v >= TRAIN_MIN else ((0, 165, 255) if v >= TRAIN_MIN / 2 else (0, 0, 255))
    for txt, y, sc in ((f"{v:7.0f}", 60, 1.6),
                       (f"best {best:.0f}   train {TRAIN_MIN:.0f}~5369 (med {TRAIN_MED:.0f})", 100, 0.7),
                       ("렌즈를 천천히 돌려 숫자를 최대로", 130, 0.6)):
        cv2.putText(fr, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 5)
        cv2.putText(fr, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, sc, color, 2)
    # 최근 추이 그래프 (돌리는 방향이 맞는지 바로 보인다)
    h, w = fr.shape[:2]
    if len(hist) > 2:
        mx = max(max(hist), TRAIN_MIN)
        pts = [(w - len(hist) * 4 + i * 4, h - 10 - int(x / mx * 120)) for i, x in enumerate(hist)]
        cv2.polylines(fr, [np.array(pts, np.int32)], False, color, 2)
        y_t = h - 10 - int(TRAIN_MIN / mx * 120)
        cv2.line(fr, (w - 480, y_t), (w - 5, y_t), (255, 255, 255), 1)
    cv2.imshow(win, fr)
    if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
        break
cap.release()
cv2.destroyAllWindows()
print(f"마지막 {v:.0f} / 최고 {best:.0f}  (학습 최소 {TRAIN_MIN:.0f})")
print("맞췄으면:  python ~/omx_pill_project/check_wrist.py  로 다시 확인")
