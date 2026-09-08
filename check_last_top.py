#!/usr/bin/env python3
"""방금 찍은 에피소드의 top 화면이 기준 위치와 맞는지 검사한다 (2026-08-10).

왜 필요한가:
    2026-08-09 촬영에서 ep190·ep199 두 개가 화면이 +96px 밀린 채로 저장됐다.
    카메라를 물리적으로 건드린 게 아니고, 미리보기는 늘 정상이었다.
    재현되는 조건은 "프레임을 읽은 뒤에 해상도를 설정" 하나인데(그러면 +96.5px),
    lerobot 의 어느 경로가 그 상태가 되는지는 아직 못 짚었다.
    원인을 못 짚었으므로 **결과를 검사한다** — 밀린 채로 저장되면 즉시 알려준다.

카메라를 열지 않는다. 방금 저장된 영상 파일만 읽으므로 녹화에 아무 영향이 없다.

    python check_last_top.py            # 종료코드 0=정상, 1=밀림, 2=검사불가
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.expanduser("~/.cache/huggingface/lerobot/1unasy/pill_v3")
REF = os.path.expanduser("~/omx_pill_project/top_reference.png")
TOL = 5.0


def newest_top_video():
    fs = glob.glob(f"{ROOT}/videos/observation.images.top/chunk-000/*.mp4")
    return max(fs, key=os.path.getmtime) if fs else None


def first_frame(path):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
        out = t.name
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", "0.5", "-i", path, "-vframes", "1", out],
        capture_output=True, timeout=120,
    )
    if r.returncode != 0:
        return None
    im = cv2.imread(out)
    os.unlink(out)
    return im


def main():
    if not os.path.exists(REF):
        print(f"  [검사불가] 기준 이미지가 없다: {REF}")
        return 2
    v = newest_top_video()
    if v is None:
        print("  [검사불가] top 영상을 못 찾았다")
        return 2
    im = first_frame(v)
    if im is None:
        print(f"  [검사불가] 영상을 못 읽었다: {os.path.basename(v)}")
        return 2
    ref = cv2.imread(REF)
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(g(ref), g(im))
    if abs(dx) < TOL and abs(dy) < TOL:
        print(f"  ✅ 화면 정상 (dx {dx:+.1f}, dy {dy:+.1f})")
        return 0
    print()
    print("  ██████████████████████████████████████████████████████████")
    print(f"  ██  화면이 밀렸습니다 — dx {dx:+.1f}px, dy {dy:+.1f}px")
    print(f"  ██  파일: {os.path.basename(v)}")
    print("  ██")
    print("  ██  이 에피소드는 다른 화면으로 찍혔습니다. 버려야 합니다.")
    print("  ██  → 스크립트를 끄고 클로드에게 '밀렸다' 고 말하세요.")
    print("  ██████████████████████████████████████████████████████████")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
