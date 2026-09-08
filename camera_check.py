#!/usr/bin/env python
"""top 카메라가 학습 때 화각과 같은지 확인한다.

ACT는 픽셀에서 바로 관절 각도를 뽑는다. 화각이 밀리면 같은 자리의 알약이 다른
좌표로 보이고, 모델은 그만큼 어긋난 곳으로 간다. 학습 품질과 무관하게 실패한다.
2026-07-31에 실제로 이 문제가 났다 (dx=-25px, dy=-18px) — 자세한 건 v1_pilot/STATUS.md.

두 장을 따로 저장하므로 이미지 뷰어에서 번갈아 보며 맞추면 된다.

사용법:
  python ~/omx_pill_project/camera_check.py           # 저장 + 이동량 측정
  python ~/omx_pill_project/camera_check.py --ep 55   # 다른 학습 프레임과 비교
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

D = Path("/home/user/.cache/huggingface/lerobot/1unasy/pill_v2")
OUT = Path("/home/user/omx_pill_project/report")
TOP = "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"


def grab_top():
    cap = cv2.VideoCapture(TOP, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("top 카메라를 열 수 없습니다 (녹화·평가가 돌고 있으면 먼저 끄세요)")
    ok = False
    for _ in range(15):  # 자동 노출 안정화
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("프레임 읽기 실패")
    return frame  # BGR


def train_frame(ep: int):
    """학습 데이터에서 해당 에피소드가 들어 있는 영상의 한 프레임."""
    from torchcodec.decoders import VideoDecoder

    # file-005 = ep30~31, file-006 = ep32~. 없으면 가장 최근 파일을 쓴다.
    files = sorted((D / "videos/observation.images.top/chunk-000").glob("*.mp4"))
    path = files[-1]
    dec = VideoDecoder(str(path))
    rgb = dec.get_frames_at([5]).data[0].permute(1, 2, 0).numpy().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), path.name


def main():
    ap = argparse.ArgumentParser(description="top 화각 확인")
    ap.add_argument("--ep", type=int, default=30, help="비교할 학습 에피소드 (참고용 표시)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    now = grab_top()
    train, src = train_frame(args.ep)

    p_train = OUT / "top_TRAIN.png"
    p_now = OUT / "top_NOW.png"
    cv2.imwrite(str(p_train), train)
    cv2.imwrite(str(p_now), now)

    a = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY).astype(np.float32)[:240]
    b = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY).astype(np.float32)[:240]
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a * win, b * win)

    # 정렬용 겹친 이미지: 학습=빨강, 지금=청록. 어긋나면 색 테두리가 갈라져 보이고
    # 정확히 맞으면 색이 사라지고 회색으로 합쳐진다. 두 장을 번갈아 보는 것보다 쉽다.
    g_train = cv2.cvtColor(train, cv2.COLOR_BGR2GRAY)
    g_now = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY)
    blend = cv2.merge([g_now, g_now, g_train])  # BGR: B=지금, G=지금, R=학습
    cv2.putText(blend, "RED=TRAIN  CYAN=NOW  (색 사라지면 정렬 완료)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(blend, "RED=TRAIN  CYAN=NOW  (align until gray)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    p_blend = OUT / "top_ALIGN.png"
    cv2.imwrite(str(p_blend), blend)

    print(f"학습 프레임 출처: {src}")
    print(f"  {p_train}")
    print(f"  {p_now}")
    print(f"  {p_blend}   ← 이걸 보고 맞추는 게 제일 쉽다")
    print(f"\n카메라 이동 추정: dx={dx:+.1f}px, dy={dy:+.1f}px  (신뢰도 {resp:.3f})")
    if abs(dx) < 5 and abs(dy) < 5:
        print("판정: 화각 일치 ✓  — 평가를 돌려도 된다")
    else:
        move_x = "오른쪽" if dx < 0 else "왼쪽"
        move_y = "아래" if dy < 0 else "위"
        print(f"판정: 어긋남 ✗ — 화면이 {move_x}으로 {abs(dx):.0f}px, {move_y}로 {abs(dy):.0f}px 밀렸다")
        print(f"      카메라를 반대로 (즉 {move_x} 화면이 되도록) 조금씩 옮기고 다시 실행할 것")
    print("\n※ 배경도 봐야 한다. 학습에 없던 물체(종이·공구)와 케이블 위치 변화도 실패 원인이다.")


if __name__ == "__main__":
    main()
