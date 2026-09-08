#!/usr/bin/env python
"""top 카메라 실시간 정렬 뷰어 (rerun).

카메라를 손으로 움직이면서 dx/dy가 0에 가까워지는 걸 실시간으로 본다.
찍고-확인하고-다시 옮기는 왕복이 필요 없다.

rerun 창의 왼쪽 트리에서 볼 것을 고른다:
  align/blend   빨강=학습 화각, 청록=지금. 색이 사라지고 회색이 되면 정렬 완료
  align/live    지금 화면 그대로
  align/dx, dy  어긋난 픽셀 (0에 가까울수록 좋다, 5 이내면 통과)

cv2.imshow 를 쓰지 않는 이유: OpenCV 번들 Qt 플러그인이 libxcb-cursor0 없이는
못 뜬다 (sudo 필요). rerun 은 lerobot 이 이미 쓰고 있어 확실히 동작한다.

끝나면 Ctrl+C. 이 창이 카메라를 잡고 있으면 녹화·평가를 못 돌리니 반드시 끌 것.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rerun as rr

D = Path("/home/user/.cache/huggingface/lerobot/1unasy/pill_v2")
TOP = "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"


def train_frame():
    from torchcodec.decoders import VideoDecoder

    files = sorted((D / "videos/observation.images.top/chunk-000").glob("*.mp4"))
    dec = VideoDecoder(str(files[-1]))
    rgb = dec.get_frames_at([5]).data[0].permute(1, 2, 0).numpy().astype(np.uint8)
    return rgb, files[-1].name


def main():
    train_rgb, src = train_frame()
    g_train = cv2.cvtColor(train_rgb, cv2.COLOR_RGB2GRAY)
    a = g_train.astype(np.float32)[:240]
    win_f = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    aw = a * win_f

    cap = cv2.VideoCapture(TOP, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("top 카메라를 열 수 없습니다 (녹화·평가가 돌고 있으면 먼저 끄세요)")

    rr.init("top_align", spawn=True)
    rr.log("align/train", rr.Image(train_rgb), static=True)
    print(f"학습 프레임: {src}")
    print("rerun 창에서 align/blend 를 보세요. 끝나면 Ctrl+C")

    dx = dy = 0.0
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            g_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            n += 1
            if n % 5 == 0:  # 매 프레임 위상상관은 무겁다
                b = g_now.astype(np.float32)[:240] * win_f
                (dx, dy), _ = cv2.phaseCorrelate(aw, b)

            rr.set_time_sequence("frame", n)
            # 빨강=학습, 청록=지금. 겹치면 회색.
            blend = np.dstack([g_train, g_now, g_now])
            rr.log("align/blend", rr.Image(blend))
            rr.log("align/live", rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            rr.log("align/dx", rr.Scalar(dx))
            rr.log("align/dy", rr.Scalar(dy))

            if n % 30 == 0:
                mark = "OK" if abs(dx) < 5 and abs(dy) < 5 else "어긋남"
                print(f"  dx={dx:+6.1f}  dy={dy:+6.1f}   {mark}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print(f"\n종료. 마지막 어긋남: dx={dx:+.1f}, dy={dy:+.1f}")


if __name__ == "__main__":
    main()
