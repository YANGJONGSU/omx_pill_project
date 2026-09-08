#!/usr/bin/env python
"""약국 조제 시연 — 병명을 번호로 고르면 그 처방대로 알약을 담는다.

팀 프로젝트 "AI 기반 대학병원 환자 안내 및 약국 자동 조제 로봇 시스템" 의 조제 파트.
정형외과 처방 4종을 준비했다.

    번호 입력 → 트레이에 필요한 색이 있는지 확인 → 순서대로 집어 약통에 담기 → 다시 물어봄

**모델은 "지정된 알약 하나 집기" 만 안다** (CLAUDE.md 2번). 처방 조합 로직은 모델이
아니라 이 파일이 담당한다. 다만 목표 지정 방식은 설계와 달라졌다 — 3색 원-핫 조건화가
학습되지 않아서(2026-08-06 계측: 원-핫을 바꿔도 Δ액션 0.03) **색마다 정책을 갈아끼운다.**
밖에서 보는 동작은 같다.

사용법:
  python ~/omx_pill_project/pharmacy.py               # 카메라 창 같이 뜸
  python ~/omx_pill_project/pharmacy.py --no-show     # 화면 없이
  python ~/omx_pill_project/pharmacy.py --list        # 처방표만 보고 끝
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prescribe as P  # noqa: E402  (정책 로딩·pick 루프를 그대로 쓴다)
from run_policy import (  # noqa: E402
    PROJECT,
    TOP,
    WRIST,
    close_cameras_window,
    go_home,
    load_home,
    load_offset,
    load_policy,
    warmup,
)

# 색 ↔ 약효. 시연에서 "왜 이 색인가" 를 설명할 수 있게 실제 처방 조합으로 엮었다.
DRUG = {
    "red": "소염진통제",
    "yellow": "근이완제",
    "green": "위장보호제",
}

# 정형외과 처방 4종. 소염진통제에는 위장보호제를 같이 내는 게 실제 처방이라 그렇게 묶었다.
MENU = {
    1: ("단순 타박상", ["red"]),
    2: ("급성 요통", ["red", "green"]),
    3: ("근육 염좌", ["red", "yellow"]),
    4: ("만성 관절염", ["red", "yellow", "green"]),
}

# 트레이(배치 영역) 안에서만 알약을 찾는다 — 로봇 베이스의 초록 회로기판을 알약으로
# 오인한 적이 있다 (2026-08-06). 밝기로 후보를 잡고 색으로 분류한다.
AREA = (91, 102, 415, 305)
HSV = {   # (H,S,V) 하한, 상한
    "red":    [((0, 90, 80), (10, 255, 255)), ((170, 90, 80), (180, 255, 255))],
    "yellow": [((20, 80, 120), (35, 255, 255))],
    "green":  [((36, 60, 60), (85, 255, 255))],
}


def detect_colors(frame) -> dict[str, int]:
    """트레이 안에 각 색이 몇 개 보이는지.

    **make_xy_labels 의 검출기를 그대로 쓴다.** 224 에피소드에서 단일 검출률 97.9%,
    모호 0% 로 검증된 값이다. 여기 있던 옛 HSV 는 실측과 어긋나 있었다:

        노랑 채도 하한 80        실측 노랑 캡슐의 채도는 46~82 — 대부분 놓쳤다
        초록 색상 36~85          너무 넓어 로봇 PCB 의 초록도 잡혔다
        면적 40 이상, 상한 없음  캡슐은 절반만 유색이라 색 영역이 35~85px 이고,
                                 상한이 없으면 로봇 본체 같은 큰 덩어리도 알약으로 셌다
    """
    import json
    from pathlib import Path

    from make_xy_labels import blobs_from, color_mask

    area = json.loads((Path(__file__).resolve().parent / "area.json").read_text())
    return {c: len(blobs_from(color_mask(frame, c, area), area, exclude_robot=True))
            for c in ("red", "yellow", "green")}


def grab_top(warmup=30, min_bright=12):
    """top 한 장. 로봇에 연결하기 전에만 쓴다 (V4L2 는 두 번 안 열린다).

    **워밍업이 필요하다.** 카메라를 열자마자 읽으면 자동노출이 잡히기 전이라
    새까만 프레임이 온다 (2026-08-17 실기에서 트레이 개수가 전부 0 으로 나온 원인).
    30프레임(약 1초)을 버리고, 그래도 어두우면 더 기다린다. 끝까지 어두우면
    None 을 돌려 **0개로 착각하지 않게** 한다.
    """
    import cv2
    import numpy as np

    c = cv2.VideoCapture(TOP, cv2.CAP_V4L2)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    f = None
    try:
        for i in range(warmup + 45):
            ok, x = c.read()
            if not ok:
                continue
            f = x
            if i >= warmup and float(np.asarray(x).mean()) >= min_bright:
                break
    finally:
        c.release()
    if f is None or float(np.asarray(f).mean()) < min_bright:
        return None
    return f


def count_pills(frames=5) -> dict[str, int]:
    """여러 장을 찍어 색별 **최빈값**을 낸다.

    한 장만 보면 조명 흔들림이나 모션블러로 개수가 한둘 왔다갔다 한다.
    트레이는 정지해 있으므로 여러 장의 최빈값이 진짜 개수다.
    """
    from collections import Counter

    votes = {c: Counter() for c in ("red", "yellow", "green")}
    got = 0
    for _ in range(frames):
        f = grab_top()
        if f is None:
            continue
        got += 1
        for c, n in detect_colors(f).items():
            votes[c][n] += 1
    if got == 0:
        return {}
    return {c: (v.most_common(1)[0][0] if v else 0) for c, v in votes.items()}


def print_menu() -> None:
    print()
    print("╔" + "═" * 56 + "╗")
    print("║" + " 약국 자동 조제 — 정형외과".center(48) + "║")
    print("╠" + "═" * 56 + "╣")
    for k, (name, pills) in MENU.items():
        drugs = " + ".join(f"{DRUG[c]}({c})" for c in pills)
        print(f"║ {k}. {name:<10} {drugs:<38}║")
    print("║" + " 0. 종료".ljust(56) + "║")
    print("╚" + "═" * 56 + "╝")


def main():
    ap = argparse.ArgumentParser(description="병명 번호로 처방을 골라 조제한다")
    ap.add_argument("--list", action="store_true", help="처방표만 보고 끝")
    ap.add_argument("--no-show", action="store_true", help="카메라 창을 띄우지 않는다")
    ap.add_argument("--no-check", action="store_true", help="트레이 색 확인을 건너뛴다")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--between", type=float, default=2.0)
    ap.add_argument("--home-time", type=float, default=3.0)
    ap.add_argument("--min-carry", type=float, default=4.0)
    ap.add_argument("--cam-reconnect", type=int, default=30)
    ap.add_argument("--temporal-ensemble", type=float, default=0.01)
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--no-auto-correct", action="store_true")
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--show-every", type=int, default=3)
    ap.add_argument("--relax-on-exit", action="store_true", default=True)
    args = ap.parse_args()
    args.show = not args.no_show

    print_menu()
    if args.list:
        return

    # --- 정책 세 개를 미리 올린다 (처방 사이에 로딩으로 멈추지 않게) ---
    from lerobot.cameras.configs import Cv2Backends
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import get_safe_torch_device

    device = get_safe_torch_device(args.device)
    home = load_home()
    home_depth = home["elbow_flex.pos"] - home["shoulder_lift.pos"]

    print("\n정책 로드")
    loaded = {}
    for c in DRUG:
        run, repo = P.POLICY[c]
        ck = PROJECT / "train" / run / "checkpoints" / args.ckpt / "pretrained_model"
        if not ck.is_dir():
            raise SystemExit(f"{c} 정책이 없습니다: {ck}")
        pol, cfg, pre, post, meta = load_policy(
            ck, repo, args.device, args.n_action_steps, args.temporal_ensemble)
        corr = None if args.no_auto_correct else P.load_correction(run)
        warmup(pol, cfg, pre, post, meta, device, f"pick {c} pill")
        loaded[c] = (run, pol, cfg, pre, post, meta, corr)
        print(f"  {DRUG[c]:<8} {c:<7} ← {run}" + ("  (위치별 보정 적용)" if corr else ""))

    # --- 로봇 연결 ---
    cams = {
        n: OpenCVCameraConfig(index_or_path=p, width=640, height=480, fps=args.fps,
                              backend=Cv2Backends.V4L2, fourcc="MJPG")
        for n, p in (("top", TOP), ("wrist", WRIST))
    }
    robot = OmxFollower(OmxFollowerConfig(
        port="/dev/omx_follower", id="omx_follower_arm", cameras=cams,
        disable_torque_on_disconnect=args.relax_on_exit))
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
                    try:
                        dev.is_connected = False
                    except Exception:
                        pass
            precise_sleep(3.0)

    history = []
    try:
        go_home(robot, home, args.home_time)
        while True:
            print_menu()
            try:
                raw = input("  병명 번호를 입력하세요 > ").strip()
            except EOFError:
                break
            if raw in ("0", "q", ""):
                break
            if not raw.isdigit() or int(raw) not in MENU:
                print("  없는 번호입니다.")
                continue

            name, pills = MENU[int(raw)]
            print(f"\n■ {name} — " + " → ".join(f"{DRUG[c]}({c})" for c in pills))

            # 트레이 확인: 로봇이 카메라를 쥐고 있으므로 로봇 관측을 쓴다
            if not args.no_check:
                try:
                    import cv2

                    obs = robot.get_observation()
                    top = obs.get("top")
                    if top is not None:
                        bgr = cv2.cvtColor(np.asarray(top), cv2.COLOR_RGB2BGR)
                        seen = detect_colors(bgr)
                        print("  트레이: " + "  ".join(f"{DRUG[c]} {seen.get(c,0)}개" for c in DRUG))
                        missing = [c for c in pills if seen.get(c, 0) < 1]
                        if missing:
                            print("  !! 트레이에 없는 약: "
                                  + ", ".join(f"{DRUG[c]}({c})" for c in missing))
                            print("     학습 데이터에 '목표가 없는' 판이 없어서, 없는 색을")
                            print("     부르면 로봇이 다른 알약으로 갑니다. 놓고 다시 시도하세요.")
                            continue
                except Exception as e:
                    print(f"  (색 확인 실패 — 건너뜁니다: {e})")

            results = []
            t_all = time.perf_counter()
            for i, color in enumerate(pills, 1):
                bundle = loaded[color]
                offsets = load_offset(bundle[0], f"pick {color} pill")
                print(f"\n  [{i}/{len(pills)}] {DRUG[color]} ({color})", flush=True)
                ok = P.run_one_pick(robot, bundle, device, args,
                                    color, home, home_depth, offsets)
                results.append((color, ok))
                bundle[1].reset()
                go_home(robot, home, args.home_time)
                if i < len(pills):
                    precise_sleep(args.between)

            done = sum(ok for _, ok in results)
            dt = time.perf_counter() - t_all
            print(f"\n  ── {name} 완료 {done}/{len(pills)}  ({dt:.0f}초)")
            for c, ok in results:
                print(f"     {DRUG[c]:<8} {'담음' if ok else '실패'}")
            history.append((name, done, len(pills), dt))
            print("\n  약통을 비우고 트레이를 다시 채운 뒤 다음 번호를 입력하세요.")
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        try:
            go_home(robot, home, args.home_time)
        except Exception:
            pass
        robot.disconnect()
        if args.show:
            close_cameras_window()

    if history:
        print("\n" + "═" * 58)
        print(" 이번 세션 기록")
        for name, done, total, dt in history:
            print(f"   {name:<12} {done}/{total}  {dt:.0f}초")
        ok = sum(d for _, d, _, _ in history)
        tot = sum(t for _, _, t, _ in history)
        print(f"   합계 {ok}/{tot}")
        print("═" * 58)


if __name__ == "__main__":
    main()
