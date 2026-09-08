#!/usr/bin/env python
"""처방대로 알약을 순서대로 집어 약통에 담는다 (State Machine).

CLAUDE.md 2번의 설계 그대로다 — **모델은 "지정된 알약 하나 집기"만 안다.**
"감기약 = 빨강 + 노랑" 같은 처방 조합은 모델이 아니라 이 스크립트가 담당한다.

    처방 ["red", "yellow"]  →  pick(red) → 홈 복귀 → pick(yellow) → 홈 복귀 → 끝

한 번의 pick 이 끝났는지는 **그리퍼로** 판정한다 (사람이 세지 않는다):
    열림 → 닫힘(집음) → 열림(약통에 놓음) → 팔이 홈 근처로 복귀   = 1회 완료
학습 데이터의 한 에피소드가 정확히 이 모양이라(집기→담기→홈 복귀), 정책이
그 궤적을 따라오면 자연히 이 순서로 신호가 나온다.

**전제: 목표 조건화가 학습돼 있어야 한다.**
2026-08-06 현재 `act_v3_3color` 는 원-핫을 무시한다(Δ액션 0.03 —
`probe_policy.py --tests 5`). 그 상태로 이 스크립트를 돌리면 처방과 무관하게
같은 알약만 반복해서 집는다. 그래서 이 스크립트는 **집은 색을 확인하지 않는다** —
확인은 사람이 눈으로 하거나, 조건화가 고쳐진 뒤 top 영상으로 판정해야 한다.

사용법:
  python ~/omx_pill_project/prescribe.py --pills red yellow
  python ~/omx_pill_project/prescribe.py --pills red green yellow --run act_v3_3color
  python ~/omx_pill_project/prescribe.py --pills red --dry-run     # 로봇 없이 순서만 확인
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.datasets.utils import build_dataset_frame  # noqa: E402
from lerobot.policies.utils import make_robot_action  # noqa: E402
from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig  # noqa: E402
from lerobot.utils.constants import OBS_STR  # noqa: E402
from lerobot.utils.control_utils import predict_action  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import get_safe_torch_device  # noqa: E402

from run_policy import (  # noqa: E402
    GRIPPER_CLOSED,
    show_cameras,
    GRIPPER_OPEN,
    PROJECT,
    TOP,
    WRIST,
    approach_progress,
    go_home,
    goal_values,
    grasp_offset,
    load_home,
    apply_correction,
    load_correction,
    load_offset,
    load_policy,
    reconnect_cameras,
    warmup,
)

COLORS = ["yellow", "red", "green"]

# 색마다 **정책이 다르다.** 3색 원-핫 조건화는 학습되지 않았고(2026-08-06 계측:
# 원-핫을 바꿔도 Δ액션 0.03), 색별 단일 정책으로 전환했다. 각 정책은 자기 오프셋과
# 자기 위치별 보정식(train/<run>/correction.json)을 갖는다.
POLICY = {
    # red36 = ep0~28 + ep89~95(회복 시연 7개), 3색 베이스에서 10K 파인튜닝.
    # red29(맨바닥 40K, 9에폭)도 남아 있다 — 색 구분·정지가 더 확실하니 문제가 생기면 되돌릴 것.
    # 촬영용: red36(파인튜닝 10K)은 노랑을 집었다. red29(맨바닥 40K, 9에폭)가 유일하게
    # 색 구분이 확실하다. 회복 시연 7개는 빠져 있지만 시연에는 이쪽이 안전하다.
    "red":    ("act_v3_red40k", "1unasy/pill_v3_onehot"),
    "green":  ("act_v3_green40k",  "1unasy/pill_v3_onehot"),
    "yellow": ("act_v3_yellow40k", "1unasy/pill_v3_onehot"),
}

# 처방 예시 — 발표·데모용. 실제로는 --pills 로 준다.
PRESCRIPTIONS = {
    "감기": ["red", "yellow"],
    "몸살": ["red", "yellow", "green"],
    "소화": ["green"],
    "전체": ["yellow", "red", "green"],
}


class PickState:
    """한 번의 pick 이 어디까지 왔는지. 그리퍼 신호만으로 판정한다.

    대기 → 접근(그리퍼 열림) → 파지(닫힘) → 담기(다시 열림) → 복귀(홈 근처) → 완료
    """

    APPROACH, GRASPED, DONE = "접근", "파지", "완료"

    def __init__(self, home: dict[str, float], min_carry: float = 4.0):
        """min_carry: 집은 뒤 이만큼은 쥐고 있어야 '약통까지 옮겼다'로 본다.

        2026-08-06 실패 사례: 초록이 파지 15.2초 → '놓음' 17.0초로 **1.8초** 만에
        열렸다. 약통까지 간 게 아니라 집자마자 놓친 것이다(노랑은 11.2초). 시간
        조건이 없으면 그 실패를 성공으로 오인한다.

        그리고 **홈 복귀를 정책에 맡기지 않는다.** 빨강만 스스로 멈추고 초록·노랑은
        계속 움직인다 — 그러면 홈 판정이 영영 안 걸린다. 담은 것이 확인되면 이
        스크립트가 직접 홈으로 보낸다.
        """
        self.stage = self.APPROACH
        self.home = home
        self.min_carry = min_carry
        self.t0 = time.perf_counter()
        self.t_grasp = None
        self.t_close = None
        self.drops = 0

    def update(self, action: dict, obs: dict) -> str:
        g = action.get("gripper.pos")
        if g is None:
            return self.stage
        g = float(g)
        now = time.perf_counter()
        if self.stage == self.APPROACH and g < GRIPPER_CLOSED:
            self.stage = self.GRASPED
            self.t_close = now
            self.t_grasp = now - self.t0
            print(f"    ▸ 파지 ({self.t_grasp:.1f}초)", flush=True)
        elif self.stage == self.GRASPED and g > GRIPPER_OPEN:
            held = now - self.t_close
            if held < self.min_carry:
                # 너무 빨리 열렸다 = 놓친 것. 다시 접근 단계로 (회복 시연을 배웠으면 다시 잡는다)
                self.drops += 1
                self.stage = self.APPROACH
                print(f"    ▸ 놓침 ({held:.1f}초만 쥠) — 다시 시도 {self.drops}회", flush=True)
            else:
                self.stage = self.DONE
                print(f"    ▸ 약통에 놓음 ({now - self.t0:.1f}초, {held:.1f}초 운반) — 완료",
                      flush=True)
        return self.stage


def run_one_pick(robot, bundle, device, args,
                 color: str, home: dict, home_depth: float, offsets: tuple) -> bool:
    """알약 하나를 집어 약통에 담는다. 성공하면 True.

    정책은 계속 돌지만, **끝을 판정하는 것은 이 함수다** — 학습 데이터의 한
    에피소드가 홈 복귀까지 포함하므로 그 신호를 그대로 쓴다.
    """
    run, policy, cfg, pre, post, meta, corr = bundle
    features = meta.features
    task = f"pick {color} pill"
    goal = goal_values(meta, task)
    depth_mm, lateral_mm, z_mm = offsets
    st = PickState(home, args.min_carry)
    steps = 0
    obs_fail = 0
    reconnects = 0
    dt = 1.0 / args.fps
    deadline = time.perf_counter() + args.timeout

    on = [k for k, v in goal.items() if v == 1.0]
    print(f"  목표 원-핫: {on[0] if on else '(조건화 없는 정책)'}", flush=True)

    # **필수.** 워밍업이나 직전 pick 의 예측이 앙상블에 남아 있으면, 시간 앙상블
    # coeff=0.01 이 옛 예측에 가중을 더 주기 때문에 진짜 예측이 들어와도 한동안
    # 그 값이 지배한다 — 팔이 제자리에 머문다 (2026-08-06 에 이것 때문에 안 움직였다).
    # run_policy.py 는 루프 직전에 이걸 한다.
    policy.reset()

    import os
    _dbg = os.environ.get("PICK_DEBUG") == "1"
    _it = 0
    while time.perf_counter() < deadline:
        loop_t = time.perf_counter()
        _it += 1
        if _dbg and _it <= 2:
            print(f"      [dbg {_it}] get_observation 진입", flush=True)
        try:
            obs = robot.get_observation()
            if _dbg and _it <= 3:
                print(f"      [dbg {_it}] get_observation 완료", flush=True)
            obs_fail = 0
        except Exception as e:
            obs_fail += 1
            if obs_fail < 3:
                precise_sleep(0.1)
                continue
            if reconnects < args.cam_reconnect and reconnect_cameras(robot):
                reconnects += 1
                obs_fail = 0
                policy.reset()
                continue
            print(f"    !! 관측 실패로 중단: {e}", flush=True)
            return False

        obs.update(goal)
        if _dbg and _it <= 3:
            print(f"      [dbg {_it}] build_dataset_frame 진입", flush=True)
        frame = build_dataset_frame(features, obs, prefix=OBS_STR)
        values = predict_action(
            observation=frame, policy=policy, device=device,
            preprocessor=pre, postprocessor=post, use_amp=cfg.use_amp,
            task=task, robot_type=robot.robot_type,
        )
        action = make_robot_action(values, features)
        if _dbg and (_it <= 2 or _it % 30 == 0) and _it <= 1500:
            K = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "gripper.pos"]
            cmd = "  ".join(f"{float(action.get(k, 0)):+7.2f}" for k in K)
            nowv = "  ".join(f"{float(obs.get(k, 0)):+7.2f}" for k in K)
            print(f"      [{_it/30:5.1f}s] 명령 {cmd}", flush=True)
            print(f"      [{_it/30:5.1f}s] 실측 {nowv}", flush=True)

        if corr is not None:
            action = apply_correction(action, corr, approach_progress(obs, home_depth))

        if (depth_mm or lateral_mm or z_mm) and st.stage == PickState.APPROACH:
            r = approach_progress(obs, home_depth)
            pan = obs.get("shoulder_pan.pos")
            action = grasp_offset(action, depth_mm * r, lateral_mm * r, pan, z_mm * r)

        robot.send_action(action)
        steps += 1
        # 정책이 보는 화면을 그대로 띄운다 (시연에서 무엇을 보고 가는지 확인용)
        if getattr(args, 'show', False) and steps % max(1, getattr(args,'show_every',3)) == 0:
            try:
                show_cameras(obs, steps, True, 0.0)
            except Exception:
                pass
        if st.update(action, obs) == PickState.DONE:
            return True   # 홈 복귀는 호출부가 go_home 으로 직접 한다

        precise_sleep(max(0.0, dt - (time.perf_counter() - loop_t)))

    print(f"    !! {args.timeout:.0f}초 안에 못 끝냄 (단계: {st.stage})", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser(description="처방대로 알약을 순서대로 담는다")
    ap.add_argument("--pills", nargs="+", default=None, metavar="색",
                    help=f"담을 순서. 예: --pills red yellow   (가능: {COLORS})")
    ap.add_argument("--prescription", choices=list(PRESCRIPTIONS), default=None,
                    help=f"미리 정한 처방 {list(PRESCRIPTIONS)}")
    ap.add_argument("--no-auto-correct", action="store_true")
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=90.0, help="pick 하나의 제한 시간(초)")
    ap.add_argument("--between", type=float, default=2.0, help="pick 사이 대기(초)")
    ap.add_argument("--min-carry", type=float, default=4.0, metavar="초",
                    help="집은 뒤 이만큼 쥐고 있어야 '약통까지 옮겼다'로 본다. "
                         "짧게 열리면 놓친 것으로 보고 다시 시도한다")
    ap.add_argument("--home-time", type=float, default=3.0)
    ap.add_argument("--cam-reconnect", type=int, default=30)
    ap.add_argument("--temporal-ensemble", type=float, nargs="?", const=0.01, default=0.01,
                    help="기본 0.01 — **실기에서 검증된 값**. -0.05 는 계측상 지연이 "
                         "1.92→0.63초로 줄지만 2026-08-06 실기에서 알약 위에서 안 "
                         "내려오는 증상이 나왔다. 바꾸려면 한 번에 하나씩 검증할 것")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--relax-on-exit", action="store_true")
    ap.add_argument("--show", action="store_true",
                    help="정책이 보는 top·wrist 화면을 cv2 창으로 띄운다")
    ap.add_argument("--show-every", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="로봇 없이 순서만 출력")
    args = ap.parse_args()

    pills = args.pills or (PRESCRIPTIONS[args.prescription] if args.prescription else None)
    if not pills:
        raise SystemExit("--pills 또는 --prescription 중 하나를 주세요.")
    bad = [c for c in pills if c not in COLORS]
    if bad:
        raise SystemExit(f"모르는 색: {bad} (가능: {COLORS})")

    print("=" * 62)
    print(f" 처방: {' → '.join(pills)}   ({len(pills)}개)")
    print("=" * 62)
    for i, c in enumerate(pills, 1):
        print(f"  {i}. {c} 하나를 집어 약통에 담고 홈 복귀")
    print("=" * 62)
    if args.dry_run:
        print("\n--dry-run — 로봇을 건드리지 않았습니다.")
        return

    # predict_action 은 torch.device 를 요구한다 (문자열이면 .type 에서 죽는다)
    device = get_safe_torch_device(args.device)

    home = load_home()
    home_depth = home["elbow_flex.pos"] - home["shoulder_lift.pos"]

    # 쓸 색의 정책을 미리 다 올려둔다 (pick 사이에 로딩으로 멈추지 않게)
    loaded = {}
    for c in dict.fromkeys(pills):
        run, repo = POLICY[c]
        ck = PROJECT / "train" / run / "checkpoints" / args.ckpt / "pretrained_model"
        if not ck.is_dir():
            raise SystemExit(f"{c} 정책이 없습니다: {ck}\n  아직 학습 안 된 색입니다.")
        print(f"  {c:<7} ← {run}", flush=True)
        # load_policy 는 문자열, warmup/predict_action 은 torch.device 를 받는다
        pol, cfg, pre, post, meta = load_policy(
            ck, repo, args.device, args.n_action_steps, args.temporal_ensemble)
        corr = None if args.no_auto_correct else load_correction(run)
        if corr is not None:
            print(f"          위치별 보정 적용 (좌우 {corr['pan']['r2']*100:.0f}%)", flush=True)
        warmup(pol, cfg, pre, post, meta, device, f"pick {c} pill")
        loaded[c] = (run, pol, cfg, pre, post, meta, corr)

    from lerobot.cameras.configs import Cv2Backends
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

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

    results = []
    try:
        go_home(robot, home, args.home_time)
        for i, color in enumerate(pills, 1):
            bundle = loaded[color]
            offsets = load_offset(bundle[0], f"pick {color} pill")
            print(f"\n[{i}/{len(pills)}] {color}   "
                  f"오프셋 깊이 {offsets[0]:+.0f} 좌우 {offsets[1]:+.0f} 높이 {offsets[2]:+.0f} mm",
                  flush=True)
            ok = run_one_pick(robot, bundle, device, args,
                              color, home, home_depth, offsets)
            results.append((color, ok))
            bundle[1].reset()
            go_home(robot, home, args.home_time)
            if i < len(pills):
                precise_sleep(args.between)
    except KeyboardInterrupt:
        print("\n중단됨 — 홈으로 보냅니다")
        try:
            go_home(robot, home, args.home_time)
        except Exception:
            pass
    finally:
        robot.disconnect()

    print("\n" + "=" * 62)
    done = sum(ok for _, ok in results)
    for c, ok in results:
        print(f"  {c:<8} {'완료' if ok else '실패'}")
    print(f"  처방 {len(pills)}개 중 {done}개 완료")
    if done < len(pills):
        print("\n  못 끝낸 게 있습니다. 확인할 것:")
        print("   · 알약이 배치 영역 안에 있었는가")
        print("   · 그리퍼가 실제로 닫혔는가 (--timeout 을 늘려볼 것)")
        print("   · 오프셋이 맞는가 (run.sh 로 한 색씩 맞춘 뒤 여기서 쓴다)")
    print("=" * 62)
    print("\n주의: 트레이에 그 색이 없으면 정책이 다른 색으로 간다 —")
    print("학습 데이터에 '목표가 없는' 판이 하나도 없어서 정의가 안 된 상황이다.")
    print("처방에 넣기 전에 그 색이 트레이에 있는지 확인할 것.")


if __name__ == "__main__":
    main()
