#!/usr/bin/env python
"""카메라 2대가 실제로 영상을 주는지 확인한다 (eval.sh / record.sh 돌리기 전에).

lerobot 과 똑같은 방식으로 연다 (V4L2 백엔드 + MJPG). 확인하는 것:
  - 장치가 열리는지
  - 프레임이 오는지, fps 가 나오는지
  - 화면이 검정/단색이 아닌지 (픽셀 평균과 표준편차로 판정)
  - 미리보기 JPG 저장 (눈으로 확인)

사용법: python ~/omx_pill_project/check_cameras.py
"""

import time
from pathlib import Path

import cv2

OUT = Path("/tmp/cam_check")
CAMS = {
    "top": "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0",
    "wrist": "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0",
}
# 학습 데이터의 정상 프레임 기준값 (0~255 환산): top 평균 169, wrist 평균 87
SECONDS = 3.0


def check(name: str, path: str) -> bool:
    print(f"\n[{name}] {path}")
    if not Path(path).exists():
        print("  X 장치 경로가 없습니다 — USB 가 빠졌거나 다른 포트에 꽂혀 있습니다")
        return False

    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("  X 열리지 않습니다 (다른 프로그램이 쓰고 있을 수 있음)")
        return False
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    n, last = 0, None
    means, stds = [], []
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        ok, frame = cap.read()
        if ok:
            n += 1
            last = frame
            means.append(float(frame.mean()))
            stds.append(float(frame.std()))
    cap.release()

    if n == 0 or last is None:
        print("  X 프레임이 하나도 안 옵니다")
        return False

    fps = n / SECONDS
    mean, std = sum(means) / len(means), max(stds)
    OUT.mkdir(exist_ok=True)
    jpg = OUT / f"{name}.jpg"
    cv2.imwrite(str(jpg), last)
    print(f"  프레임 {n}개 ({fps:.1f} fps), 해상도 {last.shape[1]}x{last.shape[0]}")
    print(f"  픽셀 평균 {mean:.1f} / 편차 최대 {std:.1f}  -> 미리보기 {jpg}")

    if mean < 1.0:
        print("  X 완전히 검은 화면입니다 (영상이 안 들어옴)")
        return False
    if std < 3.0:
        print("  X 단색 화면입니다 (렌즈가 막혔거나 영상이 안 들어옴)")
        return False
    if fps < 20:
        print(f"  △ fps 가 낮습니다 ({fps:.1f}) — USB 대역폭이나 접점 문제일 수 있습니다")
        return True
    print("  O 정상")
    return True


if __name__ == "__main__":
    results = {name: check(name, path) for name, path in CAMS.items()}
    print("\n" + "=" * 50)
    bad = [n for n, ok in results.items() if not ok]
    if bad:
        print(f"문제 있는 카메라: {', '.join(bad)}")
        print("정책은 top/wrist 두 화면을 다 입력으로 받는다. 하나라도 검정이면")
        print("eval.sh 를 돌려도 팔이 엉뚱하게 움직인다 — 고치고 나서 돌릴 것.")
        print("먼저 해볼 것: 해당 USB 를 뽑았다 다시 꽂고 이 스크립트를 다시 실행")
    else:
        print("두 대 다 정상 — eval.sh 돌려도 된다")
        print(f"미리보기 이미지로 화각도 확인할 것: {OUT}/top.jpg, {OUT}/wrist.jpg")
