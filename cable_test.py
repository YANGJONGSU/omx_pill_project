#!/usr/bin/env python
"""케이블 고정이 실제로 됐는지 검증한다. 팔을 왕복시키면서 USB 끊김을 센다.

**통과 기준: 왕복 30회 동안 USB disconnect 0회.**

이 검사가 필요한 이유는 고장 방식 때문이다. wrist(Innomaker) USB 는 가만히 두면
몇 시간도 멀쩡하다가 **팔이 내려갈 때 케이블이 당겨지면서** 빠진다. 그래서 카메라를
켜놓고 쳐다보는 것으로는 고쳐졌는지 알 수 없고, 팔을 실제로 움직여봐야 한다.
2026-08-04 저녁에는 이 고장으로 커널 disconnect 가 32회 넘게 났고 간격이 21초까지
줄어, 진행 중이던 에피소드가 매번 통째로 날아갔다.

카메라는 열지 않는다 — 커널 이벤트(`journalctl -k`)만 본다. 그래야 녹화·평가 프로세스와
장치를 두고 다투지 않고, 판정도 OpenCV 디코드 사정에 안 휘둘린다.

사용법:
  python ~/omx_pill_project/cable_test.py            # 30회 왕복
  python ~/omx_pill_project/cable_test.py --n 10     # 짧게
  python ~/omx_pill_project/cable_test.py --dry-run  # 안 움직이고 30초만 감시(기준선)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.utils.robot_utils import precise_sleep

PROJECT = Path("/home/user/omx_pill_project")
HOME_FILE = PROJECT / "home_pose.json"

# 케이블이 가장 팽팽해지는 자세로 간다. 고장은 "팔이 내려갈 때" 났으므로
# 어깨를 낮추고 팔꿈치를 펴는 조합을 훑는다 (홈 기준 상대 오프셋, 도).
SWEEP = [
    {"shoulder_lift.pos": +0.0,  "elbow_flex.pos": +0.0},   # 홈
    {"shoulder_lift.pos": -25.0, "elbow_flex.pos": +20.0},  # 앞으로 뻗어 내려감
    {"shoulder_lift.pos": -35.0, "elbow_flex.pos": +30.0},  # 가장 낮게
    {"shoulder_lift.pos": -25.0, "elbow_flex.pos": +20.0},
]
STEP_SECONDS = 1.2
USB_PAT = re.compile(r"usb [\d.-]+: USB disconnect|Cannot enable|error -71")


def kernel_usb_events(since: str) -> list[str]:
    out = subprocess.run(
        ["journalctl", "-k", "--since", since, "--no-pager"],
        capture_output=True, text=True,
    ).stdout
    return [ln for ln in out.splitlines() if USB_PAT.search(ln)]


def load_home() -> dict[str, float]:
    import json

    if not HOME_FILE.exists():
        raise SystemExit(f"홈 자세 파일이 없습니다: {HOME_FILE}")
    return json.loads(HOME_FILE.read_text())


def move_to(robot, target: dict[str, float], seconds: float) -> None:
    """smoothstep 보간. run_policy.go_home 과 같은 방식으로 서보가 안 튀게 한다."""
    start = {f"{m}.pos": float(v) for m, v in robot.bus.sync_read("Present_Position").items()}
    fps, n = 30, max(int(seconds * 30), 1)
    for i in range(1, n + 1):
        u = i / n
        w = u * u * (3 - 2 * u)
        robot.send_action(
            {k: start.get(k, target[k]) + (target[k] - start.get(k, target[k])) * w
             for k in target}
        )
        precise_sleep(1 / fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="왕복 횟수 (기본 30)")
    ap.add_argument("--dry-run", action="store_true", help="안 움직이고 30초만 감시")
    args = ap.parse_args()

    since = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    before = len(kernel_usb_events(since))

    if args.dry_run:
        print("기준선 측정 — 팔을 안 움직이고 30초 감시합니다.")
        time.sleep(30)
        ev = kernel_usb_events(since)
        print(f"\n가만히 있을 때 USB 이벤트 {len(ev)}건")
        for ln in ev:
            print(f"  {ln}")
        print("\n여기서 0이 아니면 케이블이 아니라 커넥터·허브·전원 쪽이다.")
        return

    home = load_home()
    robot = OmxFollower(
        OmxFollowerConfig(port="/dev/omx_follower", id="omx_follower_arm",
                          cameras={}, disable_torque_on_disconnect=False)
    )
    print("팔로워 연결 중… (카메라는 열지 않습니다)")
    robot.connect()

    print(f"홈으로 이동 후 {args.n}회 왕복합니다. 케이블이 걸리는 소리·장력을 같이 보세요.")
    print("중단은 Ctrl+C.\n")
    events = []
    try:
        move_to(robot, home, 3.0)
        for i in range(1, args.n + 1):
            for pose in SWEEP:
                target = {k: home[k] + pose.get(k, 0.0) for k in home}
                move_to(robot, target, STEP_SECONDS)
            now = kernel_usb_events(since)
            new = now[len(events):]
            events = now
            mark = "" if not new else f"  ← USB 이벤트 {len(new)}건!"
            print(f"  {i:>3}/{args.n} 왕복   누적 이벤트 {len(events)}{mark}")
            for ln in new:
                print(f"        {ln}")
        move_to(robot, home, 3.0)
    except KeyboardInterrupt:
        print("\n중단됨 — 홈으로 보냅니다")
        try:
            move_to(robot, home, 3.0)
        except Exception:
            pass
    finally:
        robot.disconnect()

    total = len(kernel_usb_events(since)) - before
    print("\n" + "=" * 60)
    if total == 0:
        print(f"통과 — {args.n}회 왕복 동안 USB 이벤트 0건.")
        print("케이블 고정이 됐다. 이제 초록 20개를 찍어도 같은 손실이 안 난다.")
    else:
        print(f"실패 — USB 이벤트 {total}건.")
        print("아직 당겨지고 있다. 확인할 것:")
        print("  ① 가장 낮은 자세에서 케이블이 팽팽해지는가 (여유 고리가 모자람)")
        print("  ② 당김을 케이블타이가 받는가, 커넥터가 받는가 (링크 2점 고정)")
        print("  ③ 이미 늘어난 커넥터면 케이블 교체")
        print("\n`--dry-run` 으로 가만히 있을 때도 나는지 보면 케이블인지 커넥터인지 갈린다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
