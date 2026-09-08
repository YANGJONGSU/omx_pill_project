#!/usr/bin/env python
"""wrist 카메라가 학습 때와 같은 자리에 붙어 있는지 확인한다.

`preflight.py` 는 wrist 화각 비교를 **건너뛴다** — 팔에 달린 카메라라 팔 자세가 바뀌면
화면이 당연히 달라지기 때문이다(2026-08-02 에 그것 때문에 오탐이 나서 녹화가 막혔다).
그래서 wrist 는 오늘까지 한 번도 검증된 적이 없다.

하지만 **홈 자세는 항상 같다.** 학습 데이터의 모든 에피소드가 같은 홈에서 시작하므로,
팔을 홈으로 보낸 뒤 찍으면 학습 때 wrist 화면과 직접 비교할 수 있다.
카메라가 브래킷에서 돌아갔거나 밀렸다면 여기서 잡힌다 —
wrist 가 돌아가 있으면 정책은 그리퍼와 알약의 상대 위치를 잘못 보고 빗나간다.

  python ~/omx_pill_project/check_wrist.py            # 기준 만들고 비교
  python ~/omx_pill_project/check_wrist.py --no-home  # 팔을 안 움직이고 지금 화면만 비교

기준은 학습 영상의 에피소드 시작 프레임(팔=홈) 중앙값으로 만들어
`preflight_ref/wrist_train.png` 에 저장한다. top 은 `top_train.png` 와 같은 방식이다.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

PROJECT = Path("/home/user/omx_pill_project")
DATA = Path("/home/user/.cache/huggingface/lerobot/1unasy/pill_v2")
REF = PROJECT / "preflight_ref"


def build_reference(cam: str, out: Path, skip_first: int = 2) -> np.ndarray:
    """학습 영상의 에피소드 시작(팔=홈) 프레임 중앙값."""
    from torchcodec.decoders import VideoDecoder

    rows = []
    for f in sorted(glob.glob(str(DATA / "meta/episodes/**/*.parquet"), recursive=True)):
        for r in pq.read_table(f).to_pylist():
            rows.append((r["episode_index"],
                         r[f"videos/observation.images.{cam}/file_index"],
                         r[f"videos/observation.images.{cam}/from_timestamp"]))
    rows.sort()
    frames = []
    for ep, fi, ts in rows:
        if ep < skip_first:      # ep0~1 은 화각이 나머지와 다르다 (top_train 과 같은 기준)
            continue
        dec = VideoDecoder(str(DATA / f"videos/observation.images.{cam}/chunk-000/file-{fi:03d}.mp4"))
        fr = dec.get_frames_played_at([ts + 0.3]).data[0].permute(1, 2, 0).numpy().astype(np.uint8)
        frames.append(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    med = np.median(np.stack(frames), axis=0).astype(np.uint8)
    cv2.imwrite(str(out), med)
    print(f"기준 생성: {out.name}  ({len(frames)}개 프레임 중앙값)")
    return med


def compare(ref: np.ndarray, now: np.ndarray, label: str) -> None:
    a = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a * win, b * win)

    # 회전·확대는 평행이동으로 안 잡힌다. 사분면별로 따로 재서 서로 다르면 돌아간 것이다.
    h, w = a.shape
    quad = {}
    for name, (ys, xs) in {"좌상": (slice(0, h // 2), slice(0, w // 2)),
                           "우상": (slice(0, h // 2), slice(w // 2, w)),
                           "좌하": (slice(h // 2, h), slice(0, w // 2)),
                           "우하": (slice(h // 2, h), slice(w // 2, w))}.items():
        ta, tb = a[ys, xs], b[ys, xs]
        wq = cv2.createHanningWindow((ta.shape[1], ta.shape[0]), cv2.CV_32F)
        (qx, qy), qr = cv2.phaseCorrelate(ta * wq, tb * wq)
        quad[name] = (qx, qy, qr)

    print(f"\n[{label}]")
    print(f"  전체 이동  dx={dx:+.1f}  dy={dy:+.1f}  (신뢰도 {resp:.2f})")
    print(f"  {'사분면':<6}{'dx':>8}{'dy':>8}{'신뢰도':>9}")
    for k, (qx, qy, qr) in quad.items():
        print(f"  {k:<6}{qx:>8.1f}{qy:>8.1f}{qr:>9.2f}")
    good = [v for v in quad.values() if v[2] > 0.3]
    if len(good) >= 3:
        spread = max(abs(v[0] - good[0][0]) + abs(v[1] - good[0][1]) for v in good)
        if spread > 8:
            print(f"  → 사분면끼리 {spread:.0f}px 어긋난다: **카메라가 돌았거나 기울었다** (평행이동 아님)")
        elif abs(dx) < 5 and abs(dy) < 5:
            print("  → 학습 때와 같다 ✓")
        else:
            print(f"  → 통째로 {np.hypot(dx, dy):.0f}px 밀렸다 (평행이동)")
    else:
        print("  → 신뢰도가 낮아 판정 불가 (화면이 많이 달라졌거나 특징이 없다)")


def main() -> None:
    ap = argparse.ArgumentParser(description="wrist·top 카메라를 학습 때와 비교")
    ap.add_argument("--no-home", action="store_true", help="팔을 홈으로 보내지 않는다 (비교는 부정확해진다)")
    args = ap.parse_args()

    REF.mkdir(exist_ok=True)
    refs = {}
    for cam, name in (("top", "top_train.png"), ("wrist", "wrist_train.png")):
        p = REF / name
        refs[cam] = cv2.imread(str(p)) if p.exists() else build_reference(cam, p)

    from lerobot.cameras.configs import Cv2Backends
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
    from run_policy import TOP, WRIST, go_home, load_home

    cams = {
        "top": OpenCVCameraConfig(index_or_path=TOP, width=640, height=480, fps=30,
                                  fourcc="MJPG", backend=Cv2Backends.V4L2),
        "wrist": OpenCVCameraConfig(index_or_path=WRIST, width=640, height=480, fps=30,
                                    fourcc="MJPG", backend=Cv2Backends.V4L2),
    }
    robot = OmxFollower(OmxFollowerConfig(port="/dev/omx_follower", id="omx_follower_arm",
                                          cameras=cams, disable_torque_on_disconnect=False))
    for attempt in range(1, 4):
        try:
            robot.connect()
            break
        except Exception as e:
            if attempt == 3:
                raise
            print(f"연결 실패 {attempt}/3 ({type(e).__name__}) — 3초 뒤 재시도")
            for dev in [getattr(robot, "bus", None), *getattr(robot, "cameras", {}).values()]:
                try:
                    if dev is not None and getattr(dev, "is_connected", False):
                        dev.disconnect()
                except Exception:
                    pass
            import time
            time.sleep(3)

    try:
        if not args.no_home:
            print("홈 자세로 보냅니다 (학습 에피소드 시작 자세와 같아야 비교가 성립한다)")
            go_home(robot, load_home(), 3.0)
        obs = robot.get_observation()
        for cam in ("top", "wrist"):
            now = cv2.cvtColor(np.ascontiguousarray(obs[cam]), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(PROJECT / "report" / f"{cam}_NOW_home.png"), now)
            compare(refs[cam], now, cam)
        blend = cv2.merge([cv2.cvtColor(cv2.cvtColor(np.ascontiguousarray(obs["wrist"]), cv2.COLOR_RGB2BGR),
                                        cv2.COLOR_BGR2GRAY)] * 2
                          + [cv2.cvtColor(refs["wrist"], cv2.COLOR_BGR2GRAY)])
        cv2.imwrite(str(PROJECT / "report" / "wrist_ALIGN.png"), blend)
        print("\n겹친 그림: report/wrist_ALIGN.png  (빨강=학습, 청록=지금)")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
