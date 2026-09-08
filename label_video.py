#!/usr/bin/env python3
"""녹화 영상에 설명 띠를 붙인다 — 발표에서 오해가 없도록.

분홍 원은 **정책 입력이 아니다.** run_policy.py 가 녹화용 복사본에만 그리는
표시다. 아무 설명 없이 발표하면 "화면에 원을 그려 넣어 학습시켰다" 로
읽힐 수 있어서, 영상 자체에 그렇게 적어 둔다.

  python label_video.py report/run_xy_191507.mp4 --out report/발표_xy.mp4 \
      --title "좌표 조건화 (act_xy_224)" \
      --note "분홍 원 = 검출기가 찾은 목표. 정책에는 숫자 2개(u,v)로 들어간다 — 화면에 그린 게 아니다"
"""
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def band(width: int, title: str, note: str, height: int = 96) -> np.ndarray:
    """제목과 주석이 담긴 띠를 만든다 (BGR)."""
    img = Image.new("RGB", (width, height), (14, 85, 55))       # 약국 초록
    d = ImageDraw.Draw(img)
    d.text((22, 16), title, font=ImageFont.truetype(BOLD, 30), fill=(255, 255, 255))
    d.text((22, 58), note, font=ImageFont.truetype(REG, 19), fill=(190, 226, 208))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--scale", type=float, default=1.0, help="가로 축소 배율 (용량 줄이기)")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.src)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * a.scale)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * a.scale)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = band(W, a.title, a.note)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + bar.shape[0]))

    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if a.scale != 1.0:
            f = cv2.resize(f, (W, H))
        vw.write(np.vstack([bar, f]))
        n += 1
        if n % 300 == 0:
            print(f"  {n}/{N}", flush=True)
    cap.release()
    vw.release()
    print(f"  완료 — {a.out}  ({n} 프레임, {n / fps:.0f}초)")


if __name__ == "__main__":
    main()
