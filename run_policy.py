#!/usr/bin/env python
"""학습된 정책을 실기에서 '그냥 계속' 돌린다 (데이터셋 저장 없음).

eval.sh(lerobot-record)와의 차이:

  - **저장하지 않는다.** 에피소드/리셋 구간도 없다. 평가 데이터셋을 만들지 않으므로
    `eval_` 접두어 규칙, `--resume`, 인코딩 대기 같은 게 전부 사라진다.
  - **시간 상한이 없다.** 켜면 바로 돌고, 내가 멈출 때까지 멈추지 않는다.
    한 번의 픽이 끝나도 알아서 끊지 않으므로 원하는 상태가 될 때까지 지켜보면 된다.
  - **홈 복귀 버튼이 있다.** 스페이스 한 번이면 정책을 끊고 지정한 안정 자세로
    부드럽게 돌아간다. 자세가 이상해졌을 때 전원을 내리거나 손으로 잡을 필요가 없다.

홈 자세는 `home_pose.json`에 저장된다. 파일이 없으면 학습 데이터셋 에피소드들의
시작 자세 중앙값을 쓴다 — 학습 데이터가 항상 이 자세에서 시작했으므로 정책
입장에서도 가장 자연스러운 출발점이다.
바꾸고 싶으면 `--set-home`으로 팔을 직접 잡고 원하는 자세를 가르치면 된다.

**주의: v2 환경(새 트레이·화각·약통)에서는 기존 `home_pose.json` 이 옛 세팅 값이다.**
파일럿과 작업 영역이 다르므로, v2 데이터를 찍기 전에 `--set-home` 으로 다시
가르치거나 파일을 지워서 v2 데이터 기준으로 다시 계산되게 할 것.

사용법:
  python ~/omx_pill_project/run_policy.py                 # last 체크포인트로 계속 실행
  python ~/omx_pill_project/run_policy.py --ckpt 010000    # 다른 체크포인트
  python ~/omx_pill_project/run_policy.py --n-action-steps 20   # open-loop 구간 줄이기
  python ~/omx_pill_project/run_policy.py --set-home       # 홈 자세를 새로 가르치기

실행 중 키 (터미널 포커스와 무관하게 동작한다):
  스페이스  홈 복귀 후 일시정지
  s         정책 시작 / 재개
  p         일시정지 (그 자리에서 정지, 자세 유지)
  ESC 또는 q  홈 복귀 후 종료
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
# SmolVLA 는 import 해야 설정 레지스트리에 등록된다 (없으면 draccus 가 'smolvla' 를
# 못 찾는다). 설치 안 된 환경에서도 ACT 는 돌아가야 하므로 실패를 삼킨다.
try:
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
except Exception:
    pass
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging

PROJECT = Path("/home/user/omx_pill_project")
HOME_FILE = PROJECT / "home_pose.json"
TOP = "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"
WRIST = "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0"

# 노랑 파일럿 30 에피소드의 시작 자세 중앙값 (관절별 std 2.5 이내로 일관됐다).
# 팔은 접힌 채 책상을 향하고 그리퍼는 열린 상태다 — 토크를 꺼도 무너지지 않는다.
DEFAULT_HOME = {
    "shoulder_pan.pos": 0.34,
    "shoulder_lift.pos": -61.34,
    "elbow_flex.pos": 55.02,
    "wrist_flex.pos": 44.13,
    "wrist_roll.pos": 0.93,
    "gripper.pos": 59.72,
}

# 홈 복귀 속도 상한(정규화 단위/초). 멀리 떨어져 있으면 시간을 늘려서 이 속도를 지킨다.
HOME_MAX_SPEED = 25.0

# 스텝당 명령 변화량 상한 (도/프레임).
# pill_v2 20개의 실제 명령 변화량 99.9 백분위 × 1.5 로 뽑았다 (전체 최대는 0.83~3.15).
# 2026-08-03 에 정책이 깊이축으로 **66도/프레임** 을 보내는 것이 기록으로 확인됐다 —
# 학습 최대의 100배다. 이건 어떤 경우에도 정상 명령이 아니라 "순간이동"으로 보이던 그것이고,
# 서보는 그 명령을 충실히 따르고 있었다(실측 최대가 명령 최대와 같았다).
# 상한을 걸면 팔은 같은 목표를 향해 가되 학습 때 속도로 간다.
STEP_LIMIT = {
    "shoulder_pan.pos": 0.95,
    "shoulder_lift.pos": 1.61,
    "elbow_flex.pos": 1.32,
    "wrist_flex.pos": 1.61,
    "wrist_roll.pos": 0.96,
    "gripper.pos": 1.87,
}

# 명령/실측 기록에 쓸 관절 키 순서
JOINT_KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

# --------------------------------------------------------------- 파지점 오프셋
#
# 화면 밀기(--top-shift-y)는 알약과 팔이 **같이** 움직여서 정책이 보는 상대 위치가
# 그대로다. 그래서 124mm 까지 밀어도 안 들었다. 대신 정책이 낸 관절값에 직접 더한다.
#
# 관절 방향은 pill_v3 파지 자세 29개에서 회귀로 뽑았다 (2026-08-05):
#   깊이축(elbow_flex - shoulder_lift) 1도당
#     shoulder_lift -0.427 (상관 -0.99)   elbow_flex +0.573 (상관 +1.00)
#     wrist_flex    -0.123 (상관 -0.37)   ← 그리퍼 각도를 유지시키는 성분
#   shoulder_pan 은 깊이축과 무관하다 (상관 -0.10) → 좌우와 전후를 따로 밀 수 있다
#
# mm 환산:
#   깊이 — 화면 1px = 0.31도(깊이축) = 0.65mm  →  1mm = 0.477도   (실측)
#   좌우 — 배치 영역 폭 324px(210mm)에 파지 pan 범위 43.5도  →  1mm = 0.207도
#          (추정값이다. 반경 277mm 로 환산되어 OMX 실제 도달거리와 맞는다)
# 그리퍼 판정값. 학습 데이터에서 열림 ~60, 파지 ~47.4 로 일정하다 (STATUS.md).
GRIPPER_CLOSED, GRIPPER_OPEN = 52.0, 57.0

# 조제 성공 판정값 — pill_v3 224 에피소드 실측 (2026-08-16).
#   ① 실제로 물었는가 — 알약을 물면 그 두께만큼 덜 닫힌다
#        성공 파지 192건  관측 그리퍼 최솟값 49.89~51.89 (중앙값 50.13)
#        실패 파지  25건  49.04~50.13 (중앙값 49.26)
#   ② 약통 위에서 놓았는가 — 놓는 자세가 2도 안에 모여 있다
#        놓는 순간 shoulder_pan  -23.6 ~ -25.5
# 둘 다 맞아야 성공이다. 트레이 위에서 놓는 재시도는 ②에서 걸러지므로
# **회복 동작을 방해하지 않는다.**
# 224 에피소드 전체로 검증: 성공 192건 모두 검출, 실패 25건 오판 0건, 놓침 0건.
# pan 조건만으로도 완벽하게 갈린다(실패 파지는 전부 트레이 위에서 일어난다).
# 그리퍼 조건은 **빈손으로 약통까지 가는** 경우를 잡는 보험이다 — 실기에서
# 노랑이 공중에서 닫고 그대로 옮기는 것을 관찰했다 (2026-08-15).
# 정지 복귀 지점 — 학습 시연 224개의 평균 시작 pan (홈보다 0.55도 위)
STALL_HOME_PAN = 0.29

HELD_MIN = 49.85
BOTTLE_PAN = (-28.0, -21.0)

DEPTH_DEG_PER_MM = 0.31 / 0.65
# 부호 주의: pill_v3 76개 대조 결과 **pan 이 커지면 화면 왼쪽**이다
# (배치표 좌우% vs 파지 pan 상관 -0.78, 기울기 -0.330 도/%).
# 그래서 '오른쪽으로 +mm' 를 pan 감소로 보내려면 환산 상수가 음수여야 한다.
LATERAL_DEG_PER_MM = -0.207
DEPTH_DIR = {           # 깊이축 +1도를 만드는 관절 변화
    "shoulder_lift.pos": -0.427,
    "elbow_flex.pos": +0.573,
    "wrist_flex.pos": -0.123,
}

# 손끝을 **위로** 올리는 방향 (+z_mm = 책상에서 멀어짐).
# pill_v3 29개의 파지 직전 0.5초 하강 방향을 뽑아(shoulder_lift -0.977, 부호 일치 86%)
# 부호를 뒤집고, 거기 섞여 있던 깊이 성분(+1.143)을 빼서 **순수 수직**으로 만들었다.
# 그래서 z 를 올려도 앞뒤 위치는 안 변한다.
#
# mm 환산은 깊이축과 같은 0.477도/mm 를 쓴다 — 두 방향의 관절벡터 크기가 0.73 vs 0.70 으로
# 비슷해서 1차 근사로는 맞지만, **정확한 실측은 아니다.** 눈으로 보고 1mm 씩 맞추는 용도다.
Z_DIR = {
    "shoulder_lift.pos": +0.510,
    "elbow_flex.pos": +0.511,
    "wrist_flex.pos": -0.073,
}


OFFSET_FILE = PROJECT / "grasp_offset.json"

# 파지 자세를 4구역으로 나눈다 (pill_v3 96개 파지 자세의 중앙값으로 경계).
# 위치별 자동보정은 **직선**(오차 = a + b·위치)이라 구역마다 다르게 어긋나는 것은 못 잡는다 —
# "어떤 자리는 되고 어떤 자리는 안 된다"(2026-08-07 사용자 관찰)가 그 증상이다.
# 구역마다 손보정 값을 따로 두면 그 성분을 잡을 수 있다.
# pan 은 **클수록 화면 왼쪽**이다 (배치표 76개 대조, 상관 -0.78).
PAN_MID, DEPTH_MID = 5.2, 28.6
ZONE_NAMES = ["오른쪽·가까움", "오른쪽·멂", "왼쪽·가까움", "왼쪽·멂"]


def zone_of(action: dict) -> int:
    """정책이 가려는 자세로 구역을 판정한다 (0~3). 실기에서는 정답을 모르니 예측을 쓴다."""
    p = action.get("shoulder_pan.pos")
    lf = action.get("shoulder_lift.pos")
    eb = action.get("elbow_flex.pos")
    if p is None or lf is None or eb is None:
        return 0
    left = float(p) >= PAN_MID
    far = (float(eb) - float(lf)) >= DEPTH_MID
    return (2 if left else 0) + (1 if far else 0)


def load_correction(run: str) -> dict | None:
    """위치별 자동 보정식 (fit_correction.py 가 만든다).

    **상수 오프셋으로는 못 잡는 성분이 있다.** 2026-08-06 계측에서 좌우 오차가
    pan 에 비례하는 것이 확인됐다(상관 +0.78, 기울기 +0.060 도/도) — 배율 오차라
    한 위치에서 맞추면 다른 위치가 어긋난다. 그래서 정책이 낸 예측 자세를 보고
    그 자리에 맞는 보정량을 계산해 뺀다.
    """
    f = PROJECT / "train" / run / "correction.json"
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text())
        return d if "pan" in d and "depth" in d else None
    except Exception:
        return None


def apply_correction(action: dict, corr: dict, scale: float = 1.0) -> dict:
    """예측 자세에서 위치별 오차를 계산해 빼준다.

    **scale 로 접근 진행도를 곱해서 넣어야 한다.** 이 식은 *파지 자세*에서 적합한
    것이라 (깊이축 -9.8~63.9), 홈 자세(116)처럼 적합 범위 밖에 그대로 쓰면 외삽이
    되어 접근 궤적까지 틀어진다 — 2026-08-06 에 그 상태로 돌려서 아예 못 집었다.
    """
    p = action.get("shoulder_pan.pos")
    lf = action.get("shoulder_lift.pos")
    eb = action.get("elbow_flex.pos")
    if p is None or lf is None or eb is None:
        return action
    pan = float(p)
    dep = float(eb) - float(lf)
    e_pan = (corr["pan"]["b0"] + corr["pan"]["b1"] * pan) * scale
    e_dep = (corr["depth"]["b0"] + corr["depth"]["b1"] * dep) * scale
    if scale <= 0.0:
        return action
    action["shoulder_pan.pos"] = pan - e_pan
    # 깊이축 보정은 DEPTH_DIR 비율로 나눠 넣는다 (파지 자세를 유지하기 위해)
    for k, c in DEPTH_DIR.items():
        if k in action:
            action[k] = float(action[k]) - c * e_dep
    return action


def _offset_key(run: str, task: str) -> str:
    """오프셋은 **정책과 목표 색마다 다르다**, 그래서 키를 나눠 저장한다.

    편향은 그 체크포인트가 학습한 결과물이라 재학습하면 무효가 되고(3색 원-핫으로
    다시 학습하면 act_v3_red29 값은 못 쓴다), 색마다 알약 크기·모양이 달라
    잡는 높이도 달라진다. 하나로 저장하면 서로 덮어써서 매번 다시 맞춰야 한다.
    """
    return f"{run}|{task}"


def load_offset(run: str = "", task: str = "", zone: int | None = None) -> tuple[float, float, float]:
    """이 정책·색으로 마지막에 쓴 오프셋. 없으면 0.

    로그로는 복구가 안 된다 — stdout 이 tee 로 파이프되면 블록 버퍼링이라
    print 가 안 찍히고, 비정상 종료하면 그대로 날아간다.
    """
    if not OFFSET_FILE.exists():
        return 0.0, 0.0, 0.0
    try:
        d = json.loads(OFFSET_FILE.read_text())
        # 옛 평면 형식(키 없이 depth_mm 만 있던 것)도 읽어준다
        e = d.get(_offset_key(run, task), d if "depth_mm" in d else {})
        if zone is not None:
            e = (e.get("zones") or {}).get(str(zone), e)
        return (float(e.get("depth_mm", 0.0)), float(e.get("lateral_mm", 0.0)),
                float(e.get("z_mm", 0.0)))
    except Exception:
        return 0.0, 0.0, 0.0


def save_offset(depth_mm: float, lateral_mm: float, z_mm: float = 0.0,
                run: str = "", task: str = "", zone: int | None = None) -> None:
    """바뀔 때마다 즉시 쓴다 (종료를 기다리지 않는다 — 카메라가 죽으면 못 쓴다).

    다른 정책·색의 값은 건드리지 않고 자기 키만 갱신한다.
    """
    from datetime import datetime

    try:
        d = {}
        if OFFSET_FILE.exists():
            try:
                old = json.loads(OFFSET_FILE.read_text())
                d = old if not (old and "depth_mm" in old) else {}
            except Exception:
                d = {}
        k = _offset_key(run, task)
        rec = {"depth_mm": depth_mm, "lateral_mm": lateral_mm, "z_mm": z_mm,
               "saved": datetime.now().isoformat(timespec="seconds")}
        if zone is None:
            rec["zones"] = (d.get(k) or {}).get("zones", {})
            d[k] = rec
        else:
            cur = d.get(k) or {}
            cur.setdefault("depth_mm", 0.0); cur.setdefault("lateral_mm", 0.0)
            cur.setdefault("z_mm", 0.0)
            cur.setdefault("zones", {})
            cur["zones"][str(zone)] = rec
            d[k] = cur
        OFFSET_FILE.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    except Exception as e:
        print(f"  오프셋 저장 실패: {e}")


# 접근 진행도를 재는 기준 (pill_v3 29개 실측, 2026-08-05).
# 깊이축(elbow_flex - shoulder_lift)은 홈 116도에서 파지 22도까지 94도 줄어든다.
# 에피소드 시작 108~121, 파지 -9.8~63.9 로 두 구간이 안 겹쳐서 진행도로 쓸 수 있다.
GRASP_DEPTH = 22.0


# 접근의 몇 %에서 오프셋이 100% 가 되는가.
# 이걸 1.0 으로 두면(= 파지 순간에 100%) **막바지까지 오프셋이 계속 자란다.**
# 자라는 오프셋은 곧 추가 속도라, 그리퍼가 닫히는 동안 손끝이 알약을 앞으로 민다
# ("잘 될 땐 그냥 집는데 이상할 땐 앞으로 밀면서 집는다" — 사용자 관찰 2026-08-05).
# 실측: 램프를 켠 판만 닫힘 2.70초/11mm 이동, 나머지는 1.07~1.37초/3mm (학습 1.13초).
# 0.5 로 두면 접근 절반에서 이미 100% 라 마지막 구간에서는 상수 = 미는 성분이 0 이다.
RAMP_FULL_AT = 0.5


def approach_progress(obs: dict, home_depth: float) -> float:
    """0 = 홈 자세, 1 = 오프셋 100%. 접근 절반(RAMP_FULL_AT)에서 이미 100% 가 된다.

    오프셋을 첫 스텝부터 상수로 걸면 **홈 자세까지 틀어진 채로 출발한다** —
    팔이 시작부터 꺾이고, 보정이 필요 없는 구간에서 관절만 축난다.
    그렇다고 파지까지 쭉 키우면 위에 적은 "미는" 문제가 생긴다. 그래서 초반에만 키우고
    **알약에 다가가는 구간에서는 상수**로 둔다.
    """
    now = obs.get("elbow_flex.pos")
    lift = obs.get("shoulder_lift.pos")
    if now is None or lift is None:
        return 1.0
    span = home_depth - GRASP_DEPTH
    if abs(span) < 1e-6:
        return 1.0
    p = (home_depth - (float(now) - float(lift))) / span
    return float(np.clip(p / RAMP_FULL_AT, 0.0, 1.0))


def grasp_offset(action: dict, depth_mm: float, lateral_mm: float,
                 pan_deg: float | None = None, z_mm: float = 0.0) -> dict:
    """정책이 낸 관절 명령에 손끝 오프셋을 더한다.

    depth_mm  + = **책상 기준** 로봇에서 먼 쪽(top 화면 위쪽),  - = 가까운 쪽
    lateral_mm + = 로봇/화면 기준 **오른쪽** (pan 은 감소한다 — 부호 주의)

    관절 하나만 건드리면 손끝이 호를 그리며 높이까지 바뀐다. 세 관절을 데이터에서
    나온 비율로 같이 움직여야 파지 자세를 유지한 채 앞뒤로만 옮겨진다.

    **pan_deg 를 주면 책상 좌표계로 해석한다.** 깊이 관절은 팔의 *반경* 방향으로만
    밀 수 있는데, 실제 편향은 카메라-작업대 정합에서 오므로 책상 기준 고정 방향이다.
    둘은 정면(pan=0)에서만 같고 좌우로 갈수록 벌어진다 — 그래서 정면에서 맞춘 값이
    왼쪽·오른쪽에서 서로 다르게 어긋난다(사용자 관찰, 2026-08-05).
    책상 벡터를 pan 으로 분해하면 값 하나로 좌우 전체가 맞는다.
    """
    if pan_deg is not None and (depth_mm or lateral_mm):
        th = np.radians(float(pan_deg))
        radial = lateral_mm * np.sin(th) + depth_mm * np.cos(th)
        tangential = lateral_mm * np.cos(th) - depth_mm * np.sin(th)
    else:
        radial, tangential = depth_mm, lateral_mm

    if radial:
        d = radial * DEPTH_DEG_PER_MM
        for k, c in DEPTH_DIR.items():
            if k in action:
                action[k] = float(action[k]) + c * d
    if tangential and "shoulder_pan.pos" in action:
        action["shoulder_pan.pos"] = (
            float(action["shoulder_pan.pos"]) + tangential * LATERAL_DEG_PER_MM
        )
    if z_mm:
        z = z_mm * DEPTH_DEG_PER_MM
        for k, c in Z_DIR.items():
            if k in action:
                action[k] = float(action[k]) + c * z
    return action


# --------------------------------------------------------------------------- 키 입력


class Keys:
    """pynput 전역 리스너. 눌린 키를 이벤트 플래그로 바꾼다.

    lerobot의 init_keyboard_listener는 →/←/ESC 세 개로 고정이라 쓰지 않는다.
    """

    # cv2.waitKeyEx 가 주는 코드 (리눅스/GTK)
    _CV = {27: "quit", 32: "go_home", ord("s"): "resume", ord("p"): "pause",
           ord("q"): "quit", ord("["): "shift-", ord("]"): "shift+",
           65362: "depth+", 65364: "depth-", 65361: "lat-", 65363: "lat+",
           ord("w"): "z+", ord("x"): "z-"}

    def feed(self, code: int) -> None:
        """cv2 창에서 받은 키 코드를 같은 플래그로 바꾼다 (--local-keys 경로)."""
        what = self._CV.get(code) or self._CV.get(code & 0xFF)
        if what == "quit":
            self.quit = True
        elif what == "go_home":
            self.go_home = True
        elif what == "resume":
            self.resume = True
        elif what == "pause":
            self.pause = True
        elif what == "shift+":
            self.shift_delta += 3.0
        elif what == "shift-":
            self.shift_delta -= 3.0
        elif what == "depth+":
            self.depth_delta += self.step
        elif what == "depth-":
            self.depth_delta -= self.step
        elif what == "lat+":
            self.lateral_delta += self.step
        elif what == "lat-":
            self.lateral_delta -= self.step
        elif what == "z+":
            self.z_delta += self.step
        elif what == "z-":
            self.z_delta -= self.step

    def __init__(self, local_only: bool = False, step: float = 1.0):
        self.step = step          # 화살표 한 번에 움직일 mm
        self.go_home = False
        self.pause = False
        self.resume = False
        self.quit = False
        self.shift_delta = 0.0   # [ / ] 로 top 화면 보정량을 실행 중에 조절한 누적값
        self.depth_delta = 0.0   # ↑ / ↓ 로 파지점을 앞뒤로 (mm)
        self.lateral_delta = 0.0 # ← / → 로 파지점을 좌우로 (mm)
        self.z_delta = 0.0       # w / x 로 파지점 높이 (mm, + = 위로)
        self.listener = None

        if local_only:
            # 키를 cv2 창에서만 받는다 (show_cameras 가 feed() 를 부른다).
            print("  키 입력: 카메라 창에서만 받습니다 (그 창을 클릭해 포커스를 두세요)")
            return

        try:
            from pynput import keyboard
        except Exception:
            logging.warning("pynput 없음 — 키 입력 없이 돌아갑니다. 중단은 Ctrl+C")
            return

        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    self.go_home = True
                elif key == keyboard.Key.esc:
                    self.quit = True
                elif getattr(key, "char", None) == "q":
                    self.quit = True
                elif getattr(key, "char", None) == "p":
                    self.pause = True
                elif getattr(key, "char", None) == "s":
                    self.resume = True
                # 깊이 보정을 실기에서 바로 훑기 위한 키. 재시작하면 카메라가 또 죽으므로
                # 값을 바꿀 때마다 프로세스를 새로 띄우지 않게 한다.
                elif getattr(key, "char", None) == "w":
                    self.z_delta += self.step
                elif getattr(key, "char", None) == "x":
                    self.z_delta -= self.step
                elif getattr(key, "char", None) == "]":
                    self.shift_delta += 3.0     # 더 크게 = 팔이 덜 뻗음
                elif getattr(key, "char", None) == "[":
                    self.shift_delta -= 3.0
                # 파지점 오프셋. 어느 쪽이 "위"인지 말로 합의가 안 됐으므로(WORKLOG §7.2)
                # 한 번의 실행에서 양쪽을 다 눌러보고 눈으로 정하는 게 빠르다. 5mm 단위.
                elif key == keyboard.Key.up:
                    self.depth_delta += self.step     # 로봇에서 먼 쪽 (화면 위쪽)
                elif key == keyboard.Key.down:
                    self.depth_delta -= self.step     # 로봇에 가까운 쪽 (화면 아래쪽)
                elif key == keyboard.Key.right:
                    self.lateral_delta += self.step
                elif key == keyboard.Key.left:
                    self.lateral_delta -= self.step
            except Exception as e:  # 리스너 스레드가 죽으면 키가 전부 먹통이 된다
                print(f"키 처리 오류: {e}")

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def stop(self):
        if self.listener is not None:
            self.listener.stop()


# ----------------------------------------------------------------------- 홈 자세


def load_home() -> dict[str, float]:
    if HOME_FILE.is_file():
        home = json.loads(HOME_FILE.read_text())
        missing = set(DEFAULT_HOME) - set(home)
        if missing:
            raise ValueError(f"{HOME_FILE} 에 빠진 관절이 있습니다: {sorted(missing)}")
        print(f"홈 자세: {HOME_FILE}")
        return {k: float(home[k]) for k in DEFAULT_HOME}
    print("홈 자세: 기본값 (노랑 파일럿 데이터셋의 에피소드 시작 자세)")
    return dict(DEFAULT_HOME)


def print_pose(label: str, pose: dict[str, float]) -> None:
    body = "  ".join(f"{k.removesuffix('.pos')}={v:+7.2f}" for k, v in pose.items())
    print(f"  {label}: {body}")


def set_home(robot: OmxFollower) -> None:
    """팔을 손으로 움직여 원하는 자세를 홈으로 저장한다."""
    print("\n[홈 자세 가르치기]")
    print("  팔을 한 손으로 받친 상태에서 엔터를 누르세요. 토크가 꺼지면 팔이 주저앉습니다.")
    input("  준비되면 엔터 > ")
    robot.bus.disable_torque()
    print("  토크 해제됨. 원하는 자세로 옮긴 뒤 엔터를 누르세요 (팔은 계속 잡고 있을 것).")
    input("  저장하려면 엔터 > ")
    pose = {f"{m}.pos": float(v) for m, v in robot.bus.sync_read("Present_Position").items()}
    HOME_FILE.write_text(json.dumps(pose, indent=2) + "\n")
    print(f"\n저장했습니다: {HOME_FILE}")
    print_pose("홈", pose)
    print("  팔을 계속 잡은 채로 토크를 다시 켭니다.")
    robot.bus.enable_torque()
    print("  토크 복구 완료.")


def go_home(robot: OmxFollower, home: dict[str, float], seconds: float) -> None:
    """현재 자세에서 홈까지 부드럽게 보간해 이동한다 (30Hz, smoothstep).

    한 번에 목표를 던지면 서보가 최대 속도로 튀어나가므로, 거리에 맞춰 시간을
    늘리고 가감속을 준다. 이동 중 키 입력은 무시한다 — 중간에 멈추면 오히려 위험하다.
    """
    start = {f"{m}.pos": float(v) for m, v in robot.bus.sync_read("Present_Position").items()}
    dist = max(abs(home[k] - start.get(k, home[k])) for k in home)
    duration = max(seconds, dist / HOME_MAX_SPEED)
    print(f"  홈 복귀 중… (최대 이동량 {dist:.1f}, {duration:.1f}초)")

    fps = 30
    n = max(int(duration * fps), 1)
    for i in range(1, n + 1):
        u = i / n
        w = u * u * (3 - 2 * u)  # smoothstep: 시작/끝 속도 0
        robot.send_action({k: start.get(k, home[k]) + (home[k] - start.get(k, home[k])) * w for k in home})
        precise_sleep(1 / fps)
    print("  홈 도착.")


# ------------------------------------------------------------------------- 정책


def _save_trace(rows: list[list[float]], path: str, with_load: bool = False) -> None:
    import csv
    from datetime import datetime

    if path == "auto":
        (PROJECT / "report").mkdir(exist_ok=True)
        path = str(PROJECT / "report" / f"trace_{datetime.now():%H%M%S}.csv")
    head = (["t"] + [f"cmd_{k.removesuffix('.pos')}" for k in JOINT_KEYS]
            + [f"now_{k.removesuffix('.pos')}" for k in JOINT_KEYS])
    if with_load:
        head += [f"load_{k.removesuffix('.pos')}" for k in JOINT_KEYS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    print(f"기록 저장: {path}  ({len(rows)} 스텝)")


def read_load(robot) -> list[float]:
    """모터 부하(Present_Load, 0.1% 단위, 부호 = 방향).

    **위치 오차로는 누르는 힘이 안 보인다.** 다이나믹셀은 위치 제어라 손끝이 책상에
    눌려 있어도 위치 오차는 1mm 남짓인데 전류만 계속 올라간다 — 실제로 2026-08-05 에
    12·13번이 과부하(0x20)로 래치됐다. 미는 증상을 판정하려면 이 값이 필요하다.
    """
    try:
        d = robot.bus.sync_read("Present_Load")
        return [float(d.get(k.removesuffix(".pos"), float("nan"))) for k in JOINT_KEYS]
    except Exception:
        return [float("nan")] * len(JOINT_KEYS)


def load_policy(ckpt_dir: Path, repo_id: str, device: str, n_action_steps: int | None,
                temporal_ensemble: float | None = None):
    """체크포인트와 전처리/후처리 파이프라인을 만든다.

    데이터셋 메타데이터(stats, features)만 필요하므로 비디오는 열지 않는다.

    temporal_ensemble 을 주면 ACT 원 논문의 추론 방식으로 바꾼다: 매 스텝 청크를
    새로 예측하고 겹치는 예측들을 지수가중으로 합친다. 기본 방식(n_action_steps=100)은
    한 번 예측한 뒤 3.3초를 관측 없이 밀어붙이므로, 헛집으면 그 3.3초 동안 복구할
    기회가 없다. 앙상블은 매 프레임 관측을 반영하므로 그 실패 모드를 직접 친다.
    재학습은 필요 없다 — 추론 설정만 바뀐다.

    주의: 앙상블러는 정책 생성 시점에 만들어지므로(modeling_act.py:66) make_policy
    전에 설정해야 하고, n_action_steps 는 1이어야 한다(configuration_act.py:139).
    """
    meta = LeRobotDatasetMetadata(repo_id)
    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    cfg.pretrained_path = str(ckpt_dir)
    cfg.device = device
    if n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    if temporal_ensemble is not None:
        # ACT 전용이다. SmolVLA·Diffusion 에는 temporal_ensemble_coeff 필드가 없어
        # 값이 무시되는데, 아래 n_action_steps=1 만 적용되어 **50스텝 청크에서 첫
        # 1스텝만 실행**하게 된다. 실기에서 팔이 제자리에 머문다 (2026-08-16 실측).
        if not hasattr(cfg, "temporal_ensemble_coeff"):
            print(f"  ⚠ {type(cfg).__name__} 은 시간 앙상블을 지원하지 않습니다 — 무시합니다\n"
                  f"    (적용하면 n_action_steps 가 1 로 떨어져 거의 움직이지 않습니다)")
        else:
            cfg.temporal_ensemble_coeff = temporal_ensemble
            cfg.n_action_steps = 1

    # 정책이 우리와 다른 카메라 이름을 기대하면 변환표를 만든다.
    # smolvla_base 는 camera1/2/3 을 쓰므로 학습 때 --rename_map 으로 맞췄고,
    # 추론에서도 같은 변환이 필요하다. 없으면 이미지가 통째로 빠진다.
    global CAM_RENAME
    want = [k for k in getattr(cfg, "input_features", {}) if ".images." in k]
    have = [k for k in meta.features if ".images." in k]
    orig_feats = None
    if want and sorted(want) != sorted(have):
        CAM_RENAME = {}
        orig_feats = dict(cfg.input_features)
        feats = {k: v for k, v in cfg.input_features.items() if k not in want}
        for src, dst in zip(sorted(have), sorted(want)):
            feats[src] = cfg.input_features[dst]
            CAM_RENAME[src] = dst
        # make_policy 의 검증은 데이터셋 이름으로 통과시키고,
        # 만든 뒤에는 정책이 기대하는 원래 이름으로 되돌린다.
        # 되돌리지 않으면 정책이 배치에서 이미지를 못 찾는다.
        cfg.input_features = feats
        print(f"  ↔ 카메라 이름 변환: "
              + ", ".join(f"{a.split('.')[-1]}→{b.split('.')[-1]}"
                          for a, b in CAM_RENAME.items()))

    policy = make_policy(cfg, ds_meta=meta)
    if orig_feats is not None:
        cfg.input_features = orig_feats
        policy.config.input_features = orig_feats
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(ckpt_dir),
        dataset_stats=meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, cfg, pre, post, meta


GOAL_FEATURE = "observation.environment_state"
LIVE_UV_COLOR = None      # 좌표 조건화 정책이면 목표 색 이름이 들어간다
TASK_COLOR = None         # 태스크 문자열에서 뽑은 목표 색 (정지 판단용)
UV_NAMES = None           # v10: observation.state 안의 좌표 칸 이름들
CAM_RENAME: dict[str, str] = {}   # 정책이 다른 카메라 이름을 기대할 때의 변환표
AREA = None
_LAST_UV = {"goal_u": 0.5, "goal_v": 0.5}


def goal_values(meta, task: str) -> dict[str, float]:
    """목표 원-핫을 관측에 끼워 넣을 값들. 목표 조건화 정책이 아니면 빈 dict.

    ACT 는 태스크 문자열을 읽지 않으므로(modeling_act.py 에 tokenizer 없음), 목표는
    `observation.environment_state` 로만 들어간다(modeling_act.py:466 이 이 키를 직접
    참조하므로 없으면 KeyError 로 죽는다). 로봇은 이 값을 만들어 줄 수 없으니
    여기서 태스크 문자열을 보고 만든다 — make_onehot.py 와 같은 색 순서를 쓴다.

    `build_dataset_frame` 이 피처의 names 를 키로 관측 dict 에서 꺼내가므로
    (datasets/utils.py:690), 이름을 맞춘 값만 obs 에 넣어주면 원-핫이 조립된다.

    같은 이유로 `lerobot-record`(eval.sh) 로는 목표 조건화 정책을 평가할 수 없다 —
    거기서는 관측에 값을 끼워 넣을 자리가 없다. 평가도 이 스크립트로 한다.
    """
    sys.path.insert(0, str(PROJECT))
    from make_onehot import color_of_task

    # smolvla_v10 (2026-08-21): 좌표가 environment_state 가 아니라
    # **observation.state 안에** 있다. SmolVLA 는 environment_state 를 읽지
    # 않으므로(OBS_STATE 와 이미지만 본다) 좌표를 상태에 접어 넣었고, 상태
    # 이름이 [관절 6개] + [goal_u0, goal_v0, ... goal_u12, goal_v12] 다.
    # build_dataset_frame 이 이름으로 상태를 조립하므로 그 이름만 채우면 된다.
    #
    # **이 검사를 environment_state 보다 먼저 한다.** pill_v3_uvstate 는
    # pill_v3_xy 를 복사해 만들어 environment_state 가 남아 있는데(정책은 안
    # 읽는다), 그걸 먼저 보면 좌표를 엉뚱한 자리에 넣어 상태 칸이 0 으로 남는다.
    st = meta.features.get("observation.state") or {}
    _st_names = [str(n) for n in (st.get("names") or [])]

    # smolvla_v11 (2026-08-21): 색 **원-핫**이 상태 안에 반복돼 있다
    # (goal_yellow0, goal_red0, goal_green0, goal_yellow1, ...).
    # v10 과 달리 검출기가 좌표를 주지 않는다 — 정책이 화면에서 색을 찾는다.
    # film224 가 원-핫만으로 실기 3색 연속에 성공했으므로 이 데이터로 학습
    # 가능하다는 것은 확인돼 있다.
    oh_names = [n for n in _st_names
                if any(n.startswith(f"goal_{c}") for c in ("yellow", "red", "green"))]
    if oh_names:
        color = color_of_task(task)
        out = {n: (1.0 if n.startswith(f"goal_{color}") else 0.0) for n in oh_names}
        if (meta.features.get(GOAL_FEATURE) or {}).get("names"):
            out.update({n: (1.0 if n == f"goal_{color}" else 0.0)
                        for n in meta.features[GOAL_FEATURE]["names"]})
        return out

    uv_names = [n for n in _st_names if n.startswith(("goal_u", "goal_v"))]
    if uv_names:
        out = {n: 0.0 for n in uv_names}
        # pill_v3_uvstate 는 pill_v3_xy 를 복사해 만들어 environment_state 가
        # 그대로 남아 있다. 정책은 그 키를 읽지 않지만 build_dataset_frame 은
        # 데이터셋 피처를 전부 조립하므로 goal_u/goal_v 가 없으면 KeyError 로
        # 죽는다. 값은 상태 쪽과 같게 채운다 (쓰이지 않으므로 무해하다).
        if (meta.features.get(GOAL_FEATURE) or {}).get("names"):
            out.update({n: 0.0 for n in meta.features[GOAL_FEATURE]["names"]})
        out["__live_uv__"] = color_of_task(task)
        out["__uv_names__"] = uv_names
        return out

    ft = meta.features.get(GOAL_FEATURE)
    if ft is None:
        return {}

    names = list(ft["names"])
    if names == ["goal_u", "goal_v"]:
        # 좌표 조건화 정책(act_xy_224 / act_marker2). 목표가 원-핫이 아니라 탑뷰의
        # 픽셀 좌표이므로 매 프레임 화면에서 목표 색을 찾아 넣어야 한다.
        # 값은 live_goal_uv() 가 채운다 — 여기서는 자리만 만든다.
        return {"goal_u": 0.0, "goal_v": 0.0, "__live_uv__": color_of_task(task)}

    color = color_of_task(task)
    return {name: (1.0 if name == f"goal_{color}" else 0.0) for name in names}


def live_goal_uv(top_bgr, color: str, area: dict, last: dict) -> dict[str, float]:
    """탑뷰에서 목표 색 알약을 찾아 정규화 좌표로 돌려준다 (좌표 조건화 정책 전용).

    학습 라벨을 만든 make_xy_labels.py 와 **같은 HSV 범위·같은 면적 조건**을 쓴다.
    다르면 학습 때와 다른 좌표가 들어가 정책이 엉뚱한 곳으로 간다.

    못 찾으면 직전 값을 유지한다 — 알약을 집어 든 순간에는 화면에서 사라지므로
    거기서 좌표가 0 으로 튀면 팔이 원점으로 끌려간다.

    **한 번도 못 찾았으면 None 을 돌려준다.** 목표 색이 트레이에 아예 없다는 뜻이고,
    그때 화면 중앙(0.5, 0.5)을 목표로 주면 팔이 엉뚱한 데로 간다 (2026-08-18 실기:
    초록이 약통에 있는데 초록을 시키자 팔이 움직였다). 그 경우 호출한 쪽이
    정지 상태를 유지한다.
    """
    sys.path.insert(0, str(PROJECT))
    from make_xy_labels import color_mask, blobs_from

    blobs = blobs_from(color_mask(top_bgr, color, area), area, exclude_robot=True)
    if blobs:
        u, v, _ = max(blobs, key=lambda b: b[2])      # 가장 큰 덩어리
        last["goal_u"], last["goal_v"] = u / 640.0, v / 480.0
        last["seen"] = True
    elif not last.get("seen"):
        return None                                    # 한 번도 못 봤다 = 트레이에 없다
    return {"goal_u": last["goal_u"], "goal_v": last["goal_v"]}


def warmup(policy, cfg, pre, post, meta, device, task: str) -> None:
    """첫 추론은 CUDA 커널 컴파일 때문에 0.4초쯤 걸린다.

    로봇이 움직이기 시작한 뒤에 그 지연을 맞으면 첫 액션이 한 주기를 통째로
    넘겨버리므로, 연결 전에 가짜 관측으로 한 번 돌려서 미리 데워둔다.
    """
    obs = {}
    for key, ft in meta.features.items():
        if key == "observation.state":
            obs.update({name: 0.0 for name in ft["names"]})
        elif ft["dtype"] in ("image", "video"):
            h, w, c = ft["shape"]
            obs[key.removeprefix(f"{OBS_STR}.images.")] = np.zeros((h, w, c), dtype=np.uint8)
    _g = dict(goal_values(meta, task)); _g.pop("__live_uv__", None)
    obs.update(_g)

    t = time.perf_counter()
    frame = build_dataset_frame(meta.features, obs, prefix=OBS_STR)
    predict_action(
        observation=frame,
        policy=policy,
        device=device,
        preprocessor=pre,
        postprocessor=post,
        use_amp=cfg.use_amp,
        task=task,
        robot_type="omx_follower",
    )
    policy.reset()  # 가짜 관측이 액션 큐에 남지 않게 비운다
    print(f"  워밍업 완료 ({(time.perf_counter() - t) * 1e3:.0f} ms)")


def shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """top 화면을 픽셀 단위로 밀어서 정책에 넣는다 (화각 어긋남 보정).

    pill_v3 는 ep11 과 ep12 사이에 카메라-작업대 상대 자세가 36px + 0.8도 바뀐 채로
    수집됐다. 두 화각이 섞여 학습돼서 정책의 "화면 y -> 깊이" 매핑이 그 중간값으로
    치우쳐 있고, 실기에서 깊이 방향으로 일정하게 빗나간다.

    화면을 반대로 밀면 그 편향이 상쇄된다. dy 를 +로 주면 물체가 화면 아래쪽
    (=로봇에 가까운 쪽)에 있는 것처럼 보이므로 팔이 덜 뻗는다.
    학습 데이터에서 잰 환산: 화면 1px ~= 0.31도(elbow-lift) ~= 0.65mm.

    재학습 없는 임시 보정이다. 위치에 따라 편향이 조금씩 다르므로(두 그룹의 기울기가
    0.355 vs 0.277 도/px) 상수 하나로는 평균만 지워진다.
    """
    if dx == 0 and dy == 0:
        return img
    import cv2  # 이 파일의 다른 cv2 사용처와 같게 지연 임포트한다 (--show 없이도 동작해야 한다)

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def reconnect_cameras(robot) -> bool:
    """카메라를 닫았다 다시 연다. 전부 성공하면 True.

    wrist(Innomaker)가 팔 움직임에 케이블이 당겨져 USB 에서 빠지는 고장이 있다.
    커널은 곧바로 다시 잡지만(`usb ...: new high-speed USB device`) lerobot 에는
    재시도가 없어(`camera_opencv.py:462`) 열려 있던 핸들이 죽는다. 여기서 다시 연다.
    """
    ok = True
    for name, cam in robot.cameras.items():
        try:
            try:
                cam.disconnect()
            except Exception:
                pass
            time.sleep(0.3)
            cam.connect()
        except Exception as e:
            logging.warning(f"{name} 카메라 재연결 실패: {e}")
            ok = False
    return ok


# ─── 파지 반사 (--grasp-reflex) ──────────────────────────────────────────
# 아이디어(사용자, 2026-08-07): "wrist 카메라에 알약이 그리퍼 안으로 들어오면
# 그대로 집게 하면 되지 않나."  맞다. 다만 신호가 **크기가 아니다.**
#
# 성공 시연 96개의 wrist 영상을 파지 순간 기준으로 재봤다 (2026-08-07):
#
#   덩어리 넓이   접근 -2.0s 4.05%  →  -1.0s 4.46%  →  파지 3.87%
#     wrist 가 이미 코앞이라 크기가 안 변한다. 크기로는 구분이 안 된다.
#
#   양쪽 그리퍼 패드가 화면에 보임
#     -3.0s  0%  /  -2.0s  0%  /  -1.0s  3%  /  -0.5s 24%  /  파지 78%
#     **접근 중 오경보 0%.** 마지막 0.5초에만 켜진다.
#
# 패드는 팔이 파지 높이까지 내려와야 화각에 들어온다. 그래서 "패드가 보인다"가
# 곧 "손끝이 알약 높이에 왔다"이다. 여기에 "알약이 두 패드 안쪽 사이에 있다"를
# 더하면 좌우 정렬까지 확인된다.
#
# 성공 시연에서는 두 조건이 정확히 같이 켜졌다(78% = 78%) — 사람이 몰 때는 패드가
# 보이는 순간 이미 정렬돼 있었다. 정책이 빗나갈 때는 갈린다. 그래서 둘 다 본다.
PAD_HSV = ((115, 18, 80), (165, 110, 235))   # 그리퍼 패드(연보라, 검은 점무늬)
PAD_MIN_PX = 2000
CAPSULE_HALF_MM = 9.0        # 캡슐 색깔 절반의 길이 — 화면 축척을 재는 자로 쓴다
# 패드 간격 안에서 알약이 있어야 할 자리 (성공 시연 75건 실측, 2026-08-07)
REL_CENTER, REL_TOL = 0.04, 0.20
PILL_HSV = {
    "red":    [((0, 90, 60), (10, 255, 255)), ((170, 90, 60), (180, 255, 255))],
    "green":  [((38, 60, 40), (85, 255, 255))],
    "yellow": [((20, 90, 90), (35, 255, 255))],
}


def _largest(mask):
    n, _, st, cen = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return 0, None, None
    i = 1 + int(np.argmax(st[1:, 4]))
    return int(st[i, 4]), st[i], cen[i]


def grasp_reflex(img_rgb, color: str):
    """wrist 프레임에서 '지금 닫으면 잡힌다'를 판정한다.

    반환: (닫을까, 설명문). img_rgb 는 로봇이 준 RGB 프레임 그대로.
    """
    if img_rgb is None or color not in PILL_HSV:
        return False, ""
    bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    k5 = np.ones((5, 5), np.uint8)

    # ① 양쪽 패드 — 화면 좌/우 절반에서 각각 찾는다
    pm = cv2.morphologyEx(cv2.inRange(hsv, np.array(PAD_HSV[0]), np.array(PAD_HSV[1])),
                          cv2.MORPH_OPEN, k5)
    la, lst, _ = _largest(pm[:, :w // 2])
    ra, rst, _ = _largest(pm[:, w // 2:])
    if la < PAD_MIN_PX or ra < PAD_MIN_PX:
        return False, ""
    lx = (lst[0] + lst[2]) / w              # 좌패드의 안쪽(오른쪽) 가장자리
    rx = (rst[0] + w // 2) / w              # 우패드의 안쪽(왼쪽) 가장자리

    # ② 목표색 알약이 두 패드 사이에 있는가
    im = np.zeros((h, w), np.uint8)
    for lo, hi in PILL_HSV[color]:
        im |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    ar, _, cen = _largest(cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    if ar < 1500 or cen is None:
        return False, ""
    # 그냥 "패드 사이"는 너무 헐렁하다 — 그리퍼가 열려 있어 간격이 화면의 62%나 된다.
    # 실제로 2026-08-07 실기에서 x 0.90(오른쪽 끝)에도 반사가 걸렸다.
    # 성공 시연 75건의 파지 순간을 재보니 간격 안 상대위치가 +0.040 ± 0.25 였다
    # (세 색 모두 +0.03~+0.06 — 카메라와 그리퍼의 실제 정렬 오차). 그 범위만 인정한다.
    px = cen[0] / w
    mid, gap = (lx + rx) / 2, rx - lx
    rel = (px - mid) / gap if gap > 0.05 else 9.9
    if abs(rel - REL_CENTER) < REL_TOL:
        return True, (f"알약이 파지 위치 (x {px:.2f}, 간격내 {rel:+.2f} "
                      f"[{REL_CENTER-REL_TOL:+.2f}~{REL_CENTER+REL_TOL:+.2f}], {ar}px)")

    # --- 빗나감을 mm 로 환산해서 알려준다 -----------------------------------
    # 반사는 정렬을 **고치지 못한다** (사이에 들어와야 켜진다). 대신 얼마나 빗나갔는지
    # 재서 알려주면 화살표로 감 잡는 대신 숫자를 보고 영점을 맞출 수 있다.
    #
    # 축척은 알약 자신으로 잡는다 — wrist 높이가 매 프레임 달라서 고정 mm/px 를
    # 못 쓴다. 캡슐의 색깔 절반은 길이가 일정하다(약 9mm).
    _, st, _ = _largest(cv2.morphologyEx(im, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    major = max(st[2], st[3])                       # 색깔 절반의 긴 쪽 (px)
    mm_per_px = CAPSULE_HALF_MM / major if major > 20 else None
    # 목표는 패드 정중앙이 아니라 +0.04 쪽이다 (성공 시연 실측). 거기까지의 거리를 잰다.
    target = mid + REL_CENTER * gap
    off_px = (px - target) * w                      # +면 알약이 목표보다 화면 오른쪽
    if mm_per_px is None:
        return False, f"파지 위치 벗어남 (알약 {px:.2f}, 목표 {target:.2f})"
    off_mm = off_px * mm_per_px
    # 화면 오른쪽 = pan 이 작아지는 쪽 (LATERAL_DEG_PER_MM 이 음수인 것과 같은 관계).
    # 알약이 오른쪽에 있으면 손을 오른쪽으로 보내야 하니 좌우 보정을 +로 올린다.
    return False, (f"빗나감 {off_mm:+.1f}mm "
                   f"({'화면 오른쪽' if off_mm > 0 else '화면 왼쪽'}) — "
                   f"좌우 보정을 {off_mm:+.0f}mm 쪽으로")


SHOW_WIN = "policy view  (top | wrist)"


def show_cameras(obs: dict, step: int, running: bool, hz: float, keys=None) -> None:
    """정책이 방금 읽은 관측 이미지를 그대로 한 창에 붙여 띄운다.

    obs 의 이미지 키는 카메라 이름 그대로다 (top, wrist) — run_policy.py:261 참고.
    로봇에서 온 프레임은 RGB 라 BGR 로 뒤집어야 색이 맞는다.
    """
    import cv2

    tiles = []
    for name in ("top", "wrist"):
        img = obs.get(name)
        if img is None:
            continue
        img = np.ascontiguousarray(img)
        if img.ndim == 3 and img.shape[2] == 3:
            img = img[:, :, ::-1]  # RGB -> BGR
        img = np.ascontiguousarray(img)
        cv2.putText(img, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(img)
    if not tiles:
        return

    h = min(t.shape[0] for t in tiles)
    tiles = [t[:h] for t in tiles]
    canvas = np.hstack(tiles)

    state = "RUN" if running else "PAUSED"
    color = (0, 255, 0) if running else (0, 165, 255)
    txt = f"{state}   step {step}   {hz:.1f} Hz"
    cv2.putText(canvas, txt, (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
    cv2.putText(canvas, txt, (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.imshow(SHOW_WIN, canvas)
    # waitKeyEx 는 **이 창에 포커스가 있을 때만** 키를 준다. 전역 리스너(pynput)와
    # 달리 다른 창 타이핑이 안 새어 들어온다 — 한글 두벌식은 ㅂ이 물리 q 키라
    # 채팅창에 한글만 쳐도 종료(q)가 나갔다. 2026-08-05 에 실행이 두 번 이렇게 죽었다.
    code = cv2.waitKeyEx(1)
    if keys is not None and code != -1:
        keys.feed(code)


def close_cameras_window() -> None:
    try:
        import cv2

        cv2.destroyWindow(SHOW_WIN)
        cv2.waitKey(1)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="정책을 실기에서 계속 실행 (저장 없음)")
    ap.add_argument("--run", default="act_v2")
    ap.add_argument("--ckpt", default="last")
    ap.add_argument(
        "--repo-id",
        default="1unasy/pill_v2_onehot",
        help="정규화 통계·피처를 가져올 학습 데이터셋 (학습에 쓴 것과 같아야 한다)",
    )
    ap.add_argument(
        "--task",
        default="pick yellow pill",
        help="목표. 목표 조건화 정책이면 이 문자열의 색으로 원-핫이 만들어진다",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="관측 1회로 실행할 액션 수. 기본은 체크포인트 값(100=3.3초). 헛집고 회복을 못 하면 20으로",
    )
    ap.add_argument(
        "--temporal-ensemble",
        type=float,
        nargs="?",
        const=0.01,
        default=None,
        metavar="COEFF",
        help="ACT 시간 앙상블 추론 (매 스텝 재예측). 헛집고 회복 못 하는 문제에 쓴다. "
        "값 생략 시 0.01 (원 논문 기본값). n_action_steps 는 자동으로 1이 된다",
    )
    ap.add_argument("--hold-home", type=float, default=8.0, metavar="도",
                    help="**정책 성능을 볼 때는 0 으로 꺼야 한다.** 이것은 학습이 못 하는 "
                         "일을 코드로 막는 안전장치이지 정책의 능력이 아니다. "
                         "홈에서 이 거리 안(기본 8도)에 --hold-after 초 넘게 머물면 "
                         "정지하려는 것으로 보고 홈 자세로 되돌린다. 0 이면 끔. "
                         "실기에서 정지가 풀릴 때 5초에 걸쳐 조금씩 새어 나갔으므로, "
                         "이미 몇 도 벗어난 뒤에도 잡히도록 넉넉히 잡는다.")
    ap.add_argument("--hold-after", type=float, default=4.0, metavar="초",
                    help="홈 근처에 이 시간 넘게 머물러야 붙잡기 시작한다 (기본 4초). "
                         "pill_v3 192개 실측: 정상 집기는 홈 2.5도 안을 중앙값 2.68초, "
                         "90%%가 3.98초 안에 떠난다. 실기에서 정지가 풀린 것은 5.1초였다. "
                         "그 사이인 4초로 가른다. "
                         "정지 시연은 15초 동안 0.05도만 움직이는데, 정책은 홈에서도 "
                         "미세한 값을 내고 그것이 쌓여 5초 만에 3도를 넘어 팔이 들린다. "
                         "한 번 들리면 정지 판단이 풀려 없는 알약을 집으러 간다.")
    ap.add_argument("--stall-secs", type=float, default=20.0, metavar="초",
                    help="이 시간 동안 팔이 거의 안 움직이면 홈으로 복귀해 다시 시작한다 "
                         "(기본 20초, 0이면 끔). 정책이 특정 자세에서 정지를 출력하면 자세가 "
                         "안 바뀌고 그래서 계속 정지하는 자기 유지 루프에 빠진다 — "
                         "2026-08-18 실기에서 137초 동안 0.02도만 움직였다.")
    ap.add_argument("--stall-move", type=float, default=0.5, metavar="도",
                    help="정지로 판정할 이동량 문턱 (기본 0.5도). pill_v3 실측: 정상 집기 중에도 "
                         "5초 창에서 하위 1%%가 1.86도, 하위 0.1%%가 0.61도까지 내려간다. "
                         "1.5도로 잡으면 **알약에 접근하는 중을 멈춤으로 오인해 끊는다.** "
                         "실기의 진짜 갇힘은 137초 동안 0.02도였으므로 0.5도면 충분히 가른다.")
    ap.add_argument("--goal-update-lift", type=float, default=-35.0, metavar="도",
                    help="좌표 정책에서 shoulder_lift 가 이 값 이하일 때만 목표를 갱신한다 "
                         "(기본 -35). 팔이 트레이 위로 올라오면 검출기가 로봇 부품을 알약으로 "
                         "잡으므로 그때는 마지막 값을 유지한다. 홈은 -60.5 다.")
    ap.add_argument("--seq-home", action="store_true",
                    help="--sequence 에서 색을 바꾸기 전에 홈으로 복귀한다. 담은 직후 팔은 "
                         "약통 위에 있는데, 학습 시연은 전부 홈에서 시작하므로 그 자세는 "
                         "학습 분포 밖이다.")
    ap.add_argument("--sequence", default=None, metavar="색,색,...",
                    help="여러 색을 한 프로세스 안에서 순서대로 조제한다 (원-핫 정책 전용). "
                         "알약을 약통에 놓을 때마다 다음 색으로 목표를 바꾼다. 색별 정책과 "
                         "달리 정책·카메라를 다시 열지 않으므로 색 전환이 즉시 이뤄진다.")
    ap.add_argument("--home-time", type=float, default=3.0, help="홈 복귀 최소 소요 시간(초)")
    ap.add_argument("--set-home", action="store_true", help="팔을 손으로 옮겨 홈 자세를 새로 저장하고 종료")
    ap.add_argument("--no-start-home", action="store_true", help="시작할 때 홈으로 가지 않는다")
    ap.add_argument("--start-paused", action="store_true", help="정지 상태로 시작 (s 를 눌러야 움직임)")
    ap.add_argument(
        "--relax-on-exit",
        action="store_true",
        help="종료할 때 토크를 끈다. 홈이 책상에 닿는 자세일 때만 쓸 것 (기본은 토크 유지)",
    )
    ap.add_argument("--display", action="store_true", help="rerun 으로 카메라/액션 시각화")
    ap.add_argument("--top-shift-y", type=float, default=0.0,
                    help="top 화면을 세로로 미는 양(px). +면 팔이 덜 뻗는다 (1px≈0.65mm). "
                         "화각 두 개가 섞인 pill_v3 의 깊이 편향 임시 보정용")
    ap.add_argument("--grasp-offset-depth", type=float, default=None, metavar="MM",
                    help="파지점을 앞뒤로 민다. + = 로봇에서 먼 쪽(화면 위), - = 가까운 쪽(화면 아래). "
                         "실행 중에는 ↑ ↓ 로 5mm 씩 조절된다. 집은 뒤(약통으로 옮기는 구간)에는 안 걸린다. "
                         "생략하면 grasp_offset.json 의 마지막 값을 이어받는다")
    ap.add_argument("--grasp-offset-lateral", type=float, default=None, metavar="MM",
                    help="파지점을 좌우로 민다. + = 로봇 기준 오른쪽(화면 오른쪽). 실행 중 ← → 로 조절")
    ap.add_argument("--grasp-offset-z", type=float, default=0.0, metavar="MM",
                    help="파지점 높이. + = 위로(책상에서 멀어짐), - = 아래로. 실행 중 w / x 로 조절. "
                         "너무 낮으면 손끝이 책상에 눌려 힘이 쌓이고, 그게 풀리면서 앞으로 튄다")
    ap.add_argument("--top-shift-x", type=float, default=0.0,
                    help="top 화면을 가로로 미는 양(px). +면 화면 오른쪽으로 민다")
    ap.add_argument("--cam-reconnect", type=int, default=30,
                    help="카메라 관측 실패 시 재연결을 몇 번까지 시도할지 (0이면 종전처럼 바로 종료)")
    ap.add_argument("--zones", action="store_true",
                    help="파지 자세를 4구역으로 나눠 **구역마다 다른 손보정**을 쓴다. "
                         "화살표는 지금 가려는 구역의 값을 바꾼다. "
                         "'어떤 자리는 되고 어떤 자리는 안 된다'일 때 쓴다")
    ap.add_argument("--no-auto-correct", action="store_true",
                    help="train/<run>/correction.json 의 위치별 보정을 쓰지 않는다. "
                         "기본은 있으면 적용 — 좌우 오차가 pan 에 비례하는 배율 오차라 "
                         "상수 오프셋으로는 위치마다 어긋난다")
    ap.add_argument("--grasp-reflex", default=None, metavar="색",
                    choices=["red", "green", "yellow"],
                    help="wrist 에서 알약이 그리퍼 패드 사이에 들어오면 정책을 기다리지 "
                         "않고 닫는다 (팔 축은 안 건드림). 성공 시연 96개 계측 결과 "
                         "접근 중 오경보 0%%, 파지 순간 78%% 검출")
    ap.add_argument("--grasp-reflex-frames", type=int, default=3, metavar="N",
                    help="반사가 N 프레임 연속 걸려야 인정한다 (기본 3 = 0.1초)")
    ap.add_argument("--record-video", default=None, metavar="파일",
                    help="정책이 보는 화면(top+wrist)과 목표를 mp4 로 남긴다. 실패 순간을 "
                         "나중에 프레임 단위로 돌려볼 수 있다. 화면 녹화와 달리 **정책의 입력** 이다.")
    ap.add_argument("--dump-grasp", default=None, metavar="폴더",
                    help="파지 순간의 top·wrist 화면과 자세를 저장한다. 나중에 그리퍼와 "
                         "알약이 얼마나 어긋났는지 재서 보정을 계산하는 데 쓴다")
    ap.add_argument("--no-freeze-on-grasp", action="store_true",
                    help="그리퍼가 닫히는 동안 팔을 그 자리에 고정하지 않는다(옛 동작). "
                         "기본은 고정 — 닫는 데 1~2.7초가 걸리는 동안 팔이 전진하면 "
                         "손끝이 알약을 앞으로 민다")
    ap.add_argument("--offset-step", type=float, default=1.0, metavar="MM",
                    help="화살표 한 번에 움직일 파지점 보정량(mm). 기본 1mm. "
                         "거칠게 훑을 때는 5 로 올린다")
    ap.add_argument("--radial-offset", action="store_true",
                    help="오프셋을 책상 좌표계가 아니라 팔의 반경 방향으로 해석한다(옛 동작). "
                         "기본은 책상 좌표계 — pan 으로 분해해서 왼쪽·오른쪽이 같은 값으로 맞는다")
    ap.add_argument("--no-offset-ramp", action="store_true",
                    help="파지점 오프셋을 접근 진행도에 비례해 넣지 않고 첫 스텝부터 상수로 건다. "
                         "기본은 램프 ON — 홈에서 0, 파지 자세에서 100%%. 상수로 걸면 홈 자세까지 "
                         "틀어진 채로 출발해 관절이 꺾인다")
    ap.add_argument("--local-keys", action="store_true",
                    help="키를 전역이 아니라 카메라 창에서만 받는다 (--show 필요). "
                         "한글 두벌식은 ㅂ이 물리 q 키라, 전역 리스너면 다른 창에 한글만 쳐도 "
                         "종료가 나간다. 실기 중에는 켜는 것을 권한다")
    ap.add_argument("--show", action="store_true",
                    help="정책이 보는 top·wrist 화면을 cv2 창으로 띄운다 (rerun 설치 불필요)")
    ap.add_argument("--limit", nargs="?", type=float, const=1.0, default=None, metavar="배수",
                    help="스텝당 명령 변화량을 학습 데이터 수준으로 제한한다 (STEP_LIMIT × 배수). "
                         "값 생략 시 1.0. 정책이 학습에 없는 크기로 튀는 것을 막는다")
    ap.add_argument("--trace", nargs="?", const="auto", default=None, metavar="PATH",
                    help="매 스텝의 명령·실측 관절값을 CSV 로 남긴다 (기본 report/trace_<시각>.csv). "
                         "'순간이동' 같은 증상이 명령에서 오는지 팔에서 오는지 가른다")
    ap.add_argument("--show-every", type=int, default=3,
                    help="--show 일 때 몇 스텝마다 그릴지 (기본 3 = 약 10Hz, 제어 주기 보호)")
    args = ap.parse_args()

    init_logging()

    cams = {
        "top": OpenCVCameraConfig(
            index_or_path=TOP, width=640, height=480, fps=30, fourcc="MJPG", backend=Cv2Backends.V4L2
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=WRIST, width=640, height=480, fps=30, fourcc="MJPG", backend=Cv2Backends.V4L2
        ),
    }
    robot = OmxFollower(
        OmxFollowerConfig(
            port="/dev/omx_follower",
            id="omx_follower_arm",
            cameras={} if args.set_home else cams,
            disable_torque_on_disconnect=args.relax_on_exit,
        )
    )

    if args.set_home:
        robot.connect()
        try:
            set_home(robot)
        finally:
            robot.config.disable_torque_on_disconnect = False  # 가르친 자세 그대로 잡고 있게
            robot.disconnect()
        return

    ckpt_dir = PROJECT / "train" / args.run / "checkpoints" / args.ckpt / "pretrained_model"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"체크포인트가 없습니다: {ckpt_dir}")

    home = load_home()
    print_pose("목표", home)
    print(f"정책 로드: {ckpt_dir}")
    policy, cfg, pre, post, meta = load_policy(
        ckpt_dir, args.repo_id, args.device, args.n_action_steps, args.temporal_ensemble
    )
    device = get_safe_torch_device(cfg.device)
    features = meta.features
    # --sequence 를 주면 색 목록을 만들고, 담을 때마다 다음 색으로 넘어간다.
    seq = [c.strip() for c in args.sequence.split(",")] if args.sequence else []
    if seq:
        from make_onehot import COLORS as _OH
        bad = [c for c in seq if c not in _OH]
        if bad:
            raise SystemExit(f"모르는 색: {bad} (가능: {_OH})")
        args.task = f"pick {seq[0]} pill"
    seq_i = 0

    goal = goal_values(meta, args.task)  # 목표값. 목표 조건화 정책이 아니면 빈 dict
    # SmolVLA 는 목표를 **언어(task 문자열)로** 받으므로 environment_state 가 없다.
    # 그래도 목표를 바꿀 수 있으니 --sequence 를 쓸 수 있다. 색마다 모델이 다른
    # 색별 정책만 한 프로세스로 이어갈 수 없다.
    lang_goal = type(policy).__name__.lower().startswith("smolvla")
    if seq and not goal and not lang_goal:
        raise SystemExit("--sequence 는 목표 조건화(원-핫·좌표·언어) 정책에만 쓸 수 있습니다.\n"
                         "색별 정책은 색마다 정책이 달라 한 프로세스로 이어갈 수 없습니다 "
                         "— dispense.sh 를 쓰십시오.")
    global LIVE_UV_COLOR, AREA, _LAST_UV, TASK_COLOR, UV_NAMES
    # 정지 판단용 — 태스크 문자열의 색. AREA 는 좌표 정책이 아니어도 필요하다.
    try:
        from make_onehot import color_of_task
        TASK_COLOR = color_of_task(args.task)
    except Exception:
        TASK_COLOR = None
    if AREA is None:
        AREA = json.loads((PROJECT / "area.json").read_text())
    global CAM_RENAME
    CAM_RENAME = {}
    LIVE_UV_COLOR = goal.pop("__live_uv__", None)
    UV_NAMES = goal.pop("__uv_names__", None)   # v10: 상태 안의 좌표 칸 이름들
    if LIVE_UV_COLOR:
        AREA = json.loads((PROJECT / "area.json").read_text())
        _LAST_UV = {"goal_u": 0.5, "goal_v": 0.5}
        print(f" 목표 좌표     : 매 프레임 탑뷰에서 {LIVE_UV_COLOR} 검출 (HSV)")
    warmup(policy, cfg, pre, post, meta, device, args.task)

    if args.display:
        from lerobot.utils.visualization_utils import init_rerun

        init_rerun(session_name="run_policy")

    print("=" * 62)
    print(f" 태스크        : {args.task}")
    if goal:
        on = [k for k, v in goal.items() if v == 1.0]
        print(f" 목표 원-핫    : {on[0] if on else '?'}  {[goal[n] for n in features[GOAL_FEATURE]['names']]}")
    else:
        print(" 목표 원-핫    : 없음 (목표 조건화가 아닌 정책 — 아무 알약이나 집습니다)")
    if args.temporal_ensemble is not None:
        print(f" 추론          : 시간 앙상블 (coeff={args.temporal_ensemble}, 매 스텝 재예측)")
    else:
        print(f" open-loop     : n_action_steps={cfg.n_action_steps} ({cfg.n_action_steps / args.fps:.1f}초)")
    print(" 저장          : 하지 않음 (데이터셋 생성 없음)")
    print(" 종료 조건     : 없음 — 내가 멈출 때까지 계속 돈다")
    print("-" * 62)
    print(" 스페이스  홈 복귀 후 정지")
    print(" s         시작 / 재개      p  그 자리에서 정지")
    print(" ESC / q   홈 복귀 후 종료")
    print("=" * 62)

    keys = Keys(local_only=args.local_keys and args.show, step=args.offset_step)

    # 이전 프로세스를 죽이고 곧바로 다시 열면 핸드셰이크가 자주 깨진다
    # ("Failed to sync read 'Homing_Offset' ... no status packet" / 모터 하나가 빠진 것으로 보임).
    # 2026-08-03 에 세 번 났고 전부 재실행 직후였다. USB 시리얼이 정리되기 전에
    # 스캔이 들어가는 것이라 잠깐 쉬었다 다시 하면 붙는다 — 모터 고장이 아니다.
    for attempt in range(1, 4):
        try:
            robot.connect()
            break
        except Exception as e:
            if attempt == 3:
                raise
            print(f"연결 실패 {attempt}/3 ({type(e).__name__}) — 3초 뒤 다시 시도합니다")
            # robot.connect() 는 버스를 먼저 열고 카메라를 나중에 연다. 중간에 실패하면
            # 버스만 열린 어중간한 상태로 남아서, 그냥 다시 connect() 하면
            # DeviceAlreadyConnectedError 가 난다. 열린 것만 골라 닫고 다시 시작한다.
            # 조용히 삼키면 안 된다 — 2026-08-05 에 여기서 버스 닫기가 실패했는데
            # 아무 말이 없어서, 2·3차가 DeviceAlreadyConnectedError 로 연쇄로 죽고
            # 원인이 카메라였다는 게 로그에 안 남았다.
            for dev in [getattr(robot, "bus", None), *getattr(robot, "cameras", {}).values()]:
                try:
                    if dev is not None and getattr(dev, "is_connected", False):
                        dev.disconnect()
                except Exception as ce:
                    print(f"  정리 실패 ({type(dev).__name__}): {ce}", flush=True)
                    # 닫기가 실패해도 플래그만 내려두면 다음 connect() 가 다시 연다.
                    # 실제 핸들은 프로세스 종료 때 OS 가 회수한다.
                    try:
                        dev.is_connected = False
                    except Exception:
                        pass
            precise_sleep(3.0)

    running = not args.start_paused
    trace: list[list[float]] | None = [] if args.trace else None
    last_cmd: dict[str, float] = {}
    clipped = 0
    recent: deque[float] = deque(maxlen=31)   # 최근 스텝 시각 (실제 Hz 계산용)
    obs_fail = 0
    reconnects = 0
    last_shift_y = args.top_shift_y
    last_off = None
    # 값을 안 주면 지난 실행에서 화살표로 맞춘 값을 이어받는다
    home_depth = home['elbow_flex.pos'] - home['shoulder_lift.pos']
    cur_zone = -1
    corr = None if args.no_auto_correct else load_correction(args.run)
    if corr is not None:
        print(f"  위치별 보정 적용: 좌우 설명력 {corr['pan']['r2']*100:.0f}%, "
              f"깊이 {corr['depth']['r2']*100:.0f}%  (표본 {corr.get('n','?')}개)", flush=True)
    saved_depth, saved_lateral, saved_z = load_offset(args.run, args.task)
    base_depth = saved_depth if args.grasp_offset_depth is None else args.grasp_offset_depth
    base_lateral = saved_lateral if args.grasp_offset_lateral is None else args.grasp_offset_lateral
    base_z = saved_z if args.grasp_offset_z == 0.0 else args.grasp_offset_z
    if args.grasp_offset_depth is None and (saved_depth or saved_lateral):
        print(f"  지난 오프셋 이어받음: 깊이 {saved_depth:+.0f}mm  좌우 {saved_lateral:+.0f}mm",
              flush=True)
    # 그리퍼가 한 번 닫히면 그때부터는 약통으로 옮기는 구간이다. 약통 위치는 고정이라
    # 거기까지 밀면 놓기가 어긋난다 → 오프셋은 **집기 전에만** 건다.
    carrying = False
    stall_ref, stall_since, stall_n = None, time.perf_counter(), 0
    holding, target_absent = False, False   # 홈 유지 판단용
    vw = None            # --record-video 用
    grip_min = 999.0     # 무는 동안 관측된 그리퍼 최솟값 (빈손 판별용)
    reflex_latched, reflex_hits, reflex_last_msg = False, 0, -999  # 파지 반사
    fade = 0.0            # 오프셋 on/off 를 0.5초에 걸쳐 (한 프레임 계단이면 팔이 튄다)
    freeze_pose = None    # 그리퍼가 닫히는 동안 팔을 붙잡아 둘 자세
    step = 0
    slow = 0
    t0 = time.perf_counter()
    last_report = t0

    try:
        if not args.no_start_home:
            go_home(robot, home, args.home_time)
        if running:
            print("3초 뒤 정책이 시작됩니다. 손 치우세요.")
            precise_sleep(3.0)
            print("▶ 실행 중")
        else:
            print("⏸ 정지 상태 — s 를 누르면 시작합니다")
        policy.reset()

        while True:
            loop_t = time.perf_counter()

            # --- 키 처리 -------------------------------------------------
            if keys.quit:
                keys.quit = False
                print("\n종료 요청 — 홈으로 돌아갑니다")
                go_home(robot, home, args.home_time)
                break

            if keys.go_home:
                keys.go_home = False
                running = False
                print("\n■ 홈 복귀 (정책 정지)")
                go_home(robot, home, args.home_time)
                policy.reset()
                print("⏸ 정지 상태 — s 를 누르면 다시 시작합니다")
                continue

            if keys.pause:
                keys.pause = False
                if running:
                    running = False
                    # 지금 자세를 목표로 다시 써서 즉시 멈춰 세운다
                    hold = {f"{m}.pos": float(v) for m, v in robot.bus.sync_read("Present_Position").items()}
                    robot.send_action(hold)
                    print("⏸ 정지 (그 자리에서 자세 유지)")

            if keys.resume:
                keys.resume = False
                if not running:
                    running = True
                    policy.reset()
                    print("▶ 재개")

            if not running:
                # 정지 중에도 카메라 창은 계속 돌린다. 알약을 놓는 동안 top 화면을 보면서
                # 배치 영역 안에 들어갔는지 확인해야 하기 때문이다. 관측만 읽고 정책은
                # 돌리지 않으므로 팔은 움직이지 않는다.
                if args.show:
                    try:
                        show_cameras(robot.get_observation(), step, running, 0.0, keys)
                    except Exception:
                        pass
                precise_sleep(0.05)
                continue

            # --- 관측 ----------------------------------------------------
            try:
                obs = robot.get_observation()
                obs_fail = 0
                shift_y = args.top_shift_y + keys.shift_delta
                if (args.top_shift_x or shift_y) and "top" in obs:
                    obs["top"] = shift_image(obs["top"], args.top_shift_x, shift_y)
                if shift_y != last_shift_y:
                    print(f"\n  top 보정 {shift_y:+.0f}px ({shift_y * 0.65:+.1f}mm) — "
                          f"+일수록 팔이 덜 뻗는다")
                    last_shift_y = shift_y
                    policy.reset()   # 입력이 바뀌었으니 남은 액션 chunk 를 버린다
            except Exception as e:
                # wrist USB가 빠지는 알려진 고장. 저장할 게 없으니 잃는 것도 없다.
                # 팔을 세워둔 채로 죽지 않게, 홈으로 보내고 정지시킨다.
                obs_fail += 1
                logging.warning(f"관측 실패 {obs_fail}회: {e}")
                if obs_fail < 3:
                    precise_sleep(0.1)
                    continue

                # wrist USB 이탈(알려진 고장). 커널이 곧바로 장치를 다시 잡고 by-id
                # 경로도 그대로라, 카메라만 다시 열면 실행을 이어갈 수 있다.
                # 재연결이 한도까지 계속 실패할 때만 홈으로 보내고 끝낸다.
                if reconnects < args.cam_reconnect:
                    reconnects += 1
                    print(f"\n⚠ 카메라 관측 실패 — 재연결 {reconnects}/{args.cam_reconnect}")
                    try:
                        hold = {f"{m}.pos": float(v)
                                for m, v in robot.bus.sync_read("Present_Position").items()}
                        robot.send_action(hold)  # 재연결하는 동안 그 자리에 서 있게
                    except Exception:
                        pass
                    if reconnect_cameras(robot):
                        obs_fail = 0
                        policy.reset()  # 끊긴 구간이 있으니 액션 chunk 를 새로 시작한다
                        print("  재연결 성공 — 이어서 실행합니다")
                        continue
                    precise_sleep(1.0)
                    continue

                print("\n카메라 재연결에 계속 실패했습니다 — 홈으로 보냅니다")
                go_home(robot, home, args.home_time)
                break

            # --- 정책이 보는 화면 기록 -----------------------------------
            if args.record_video:
                _t = obs.get("top"); _w = obs.get("wrist")
                if _t is not None and _w is not None:
                    if vw is None:
                        import cv2 as _c
                        Path(args.record_video).parent.mkdir(parents=True, exist_ok=True)
                        vw = _c.VideoWriter(args.record_video, _c.VideoWriter_fourcc(*"mp4v"),
                                            float(args.fps), (_t.shape[1] + _w.shape[1], _t.shape[0]))
                    import cv2 as _c
                    _tb = np.ascontiguousarray(_t)[:, :, ::-1].copy()
                    _wb = np.ascontiguousarray(_w)[:, :, ::-1].copy()
                    if _wb.shape[0] != _tb.shape[0]:
                        _wb = _c.resize(_wb, (_wb.shape[1], _tb.shape[0]))
                    # 목표 표시 — 좌표 정책이면 검출 위치에 원, 그 외엔 목표색 글자
                    if LIVE_UV_COLOR and goal.get("goal_u") is not None:
                        _c.circle(_tb, (int(_LAST_UV["goal_u"] * _tb.shape[1]),
                                        int(goal["goal_v"] * _tb.shape[0])), 14, (255, 0, 255), 2)
                    _c.putText(_tb, f"{args.task}  step {step}", (8, 22),
                               _c.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    _c.putText(_wb, f"grip {obs.get('gripper.pos', 0):.1f}", (8, 22),
                               _c.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    vw.write(np.hstack([_tb, _wb]))

            # --- 정지 상태 유지 ------------------------------------------
            # 목표가 트레이에 없으면 정책은 홈에서 "정지" 를 낸다. 다만 완전한 0 이
            # 아니라 프레임마다 미세한 값이 나오고, 그것이 누적돼 5초 만에 홈에서
            # 3도를 벗어난다. 한 번 들리면 그 자세는 정지 시연에 없는 자세라 판단이
            # 풀리고, 없는 알약을 집으러 간다 (2026-08-18 실기: 스텝 152 에서 이탈).
            # 시연은 15초 동안 0.05도만 움직였다 — 그 수준을 지키게 한다.
            if args.hold_home > 0 and not carrying and TASK_COLOR:
                # **시간으로는 못 가른다.** 접근 중과 정지 중은 홈 근처에 머무는
                # 시간이 겹친다 (정상 집기 중앙값 2.68초, 90% 3.98초 / 실기 정지가
                # 풀린 시각 5.1초). 어떤 문턱을 잡아도 정상 집기의 28% 이상을 막았다.
                #
                # 가를 수 있는 것은 하나다 — **트레이에 목표 색이 있는가.**
                # 없으면 정지가 맞고, 있으면 어떤 자세든 붙잡으면 안 된다.
                # 검출은 팔이 화면을 가리지 않는 홈 근처에서만 믿는다.
                _ks = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos")
                _cur = np.array([float(obs.get(k, 0.0)) for k in _ks])
                _hm = np.array([float(home[k]) for k in _ks])
                if float(np.abs(_cur - _hm).max()) <= args.hold_home:
                    _top = obs.get("top")
                    if _top is not None and step % 5 == 0:      # 6Hz 로 확인
                        import cv2 as _cv
                        from make_xy_labels import blobs_from, color_mask
                        # 여기서는 로봇 영역을 **제외하지 않는다.** 목표가 팔 바로 옆에
                        # 있으면 제외 영역에 걸려 "없다" 가 되고 정상 집기를 막는다
                        # (pill_v3 실측 6.2%: ep117·128·133 이 전부 팔 옆이었다).
                        # 반대 방향 오판(로봇 부품을 알약으로 봄)은 "안 막는" 쪽이라 안전하다.
                        _b = blobs_from(color_mask(_cv.cvtColor(_top, _cv.COLOR_RGB2BGR),
                                                   TASK_COLOR, AREA), AREA)
                        target_absent = len(_b) == 0
                    if target_absent:
                        if not holding:
                            holding = True
                            print(f"\n  ⏸ 트레이에 {TASK_COLOR} 이(가) 없습니다 — "
                                  f"홈 자세를 유지합니다", flush=True)
                        # 실제로 붙잡는 것은 action 이 만들어진 뒤에 한다.
                        # 여기서 action 을 건드리면 첫 프레임에 아직 없어서
                        # UnboundLocalError 가 난다 (2026-08-19 실기).
                    elif holding:
                        holding = False
                        print(f"\n  ▶ {TASK_COLOR} 을(를) 찾았습니다 — 집으러 갑니다", flush=True)
                else:
                    holding = False

            # --- 막다른 자세에서 빠져나오기 ------------------------------
            # 정책이 어떤 자세에서 "정지" 를 출력하면 자세가 안 바뀌고, 자세가 그대로니
            # 다음 예측도 정지다. 한 번 걸리면 스스로는 못 빠져나온다.
            # 홈으로 되돌려 다른 자세에서 다시 보게 한다.
            if args.stall_secs > 0:
                _now = np.array([float(obs.get(k, 0.0)) for k in
                                 ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos")])
                if stall_ref is None or float(np.abs(_now - stall_ref).max()) > args.stall_move:
                    stall_ref, stall_since = _now, time.perf_counter()
                elif (time.perf_counter() - stall_since) > args.stall_secs and not carrying:
                    stall_n += 1
                    # **홈 그 자리로 돌아가는 것이 정답이다.**
                    # pill_v3 224개 실측: 색별 시작 자세가 pan +0.12~+0.46, lift -60.2~-61.0
                    # 으로 사실상 한 곳이고, 정지 시연 32개도 같은 자리다(pan +0.67~+1.04).
                    # 시연의 자연스러운 흔들림은 표준편차 1.0~1.5도뿐이다.
                    # 그래서 앞서 넣었던 2~8도 흔들기는 그 2~5배로 과했고, 부자연스럽게
                    # 튀면서 오히려 학습 분포에서 멀어졌다.
                    #
                    # 같은 자리로 돌아가도 벗어난다 — 갇힘은 "접근 도중의 어정쩡한 자세"
                    # 에서 생기지 홈에서는 생기지 않는다(홈 화면으로는 7~48도 움직였다).
                    # 회차가 늘 때만 시연의 산포(±1.5도) 안에서 아주 조금 달리 준다.
                    # 복귀 지점은 **학습 시연의 평균 시작 자세**로 잡는다.
                    # 실측 시작 pan: red +0.28 / green +0.46 / yellow +0.12 (평균 +0.29)
                    # home_pose.json 은 -0.26 으로 그보다 0.55도 낮다. 직접 가르친 값이라
                    # 파일은 건드리지 않고 여기서만 보정한다.
                    sign = 1 if stall_n % 2 else -1
                    jitter = min(stall_n - 1, 2) * 0.7 * sign      # 0, ±0.7, ±1.4도
                    h2 = dict(home)
                    h2["shoulder_pan.pos"] = STALL_HOME_PAN + jitter
                    print(f"\n  ⟳ {args.stall_secs:.0f}초째 제자리 — 홈으로 되돌립니다"
                          f" ({stall_n}회{f', pan {jitter:+.1f}도' if abs(jitter) > 0.05 else ''})",
                          flush=True)
                    go_home(robot, h2, args.home_time)
                    policy.reset()
                    stall_ref, stall_since = None, time.perf_counter()
                    continue

            # --- 추론 및 전송 --------------------------------------------
            if LIVE_UV_COLOR:
                # 좌표 조건화 정책 — 목표를 화면에서 찾되, **팔이 트레이를 가리지 않을
                # 때만 갱신한다.**
                #
                # 매 프레임 갱신하면 팔이 알약을 덮는 순간 검출기가 로봇의 빨간 부품을
                # 목표로 잡고 팔이 자기 자신을 쫓는다 (2026-08-17 영상: 목표가 166px
                # 아래로 튀어 팔이 내려갔다). 반대로 아예 고정해 버리면 알약이 굴러가거나
                # 집기 실패로 밀려나도 옛 자리로 계속 간다.
                #
                # 그래서 팔 높이로 가른다. 목표가 튄 구간은 lift -21.5~-11.5 였고
                # 홈은 -60.5 다. 여유를 둬 lift <= -35 일 때만 갱신한다 — 팔이 물러나면
                # 다시 반영되므로 실패 후 재시도에서도 새 위치를 잡는다.
                _lift = obs.get("shoulder_lift.pos")
                _safe = _lift is not None and float(_lift) <= args.goal_update_lift
                if _safe:
                    _top = obs.get("top")
                    if _top is not None:
                        import cv2 as _cv
                        _uv = live_goal_uv(_cv.cvtColor(_top, _cv.COLOR_RGB2BGR),
                                           LIVE_UV_COLOR, AREA, _LAST_UV)
                        if _uv is None:
                            # 목표 색이 트레이에 없다. 억지 좌표를 만들지 않고 멈춘다 —
                            # 정지 시연이 가르친 것과 같은 상황이다.
                            if not _LAST_UV.get("absent"):
                                _LAST_UV["absent"] = True
                                print(f"\n  ⏸ 트레이에 {LIVE_UV_COLOR} 이(가) 없습니다 — "
                                      f"집으러 가지 않고 대기합니다", flush=True)
                            robot.send_action({k: float(home[k]) for k in home})
                            precise_sleep(1.0 / args.fps)
                            continue
                        _LAST_UV["absent"] = False
                        if not _LAST_UV.get("shown"):
                            _LAST_UV["shown"] = True
                            print(f"\n  🎯 목표: ({_uv['goal_u']:.3f}, {_uv['goal_v']:.3f})"
                                  f"  — 팔이 트레이를 가리기 전까지 계속 갱신합니다",
                                  flush=True)
                if UV_NAMES:
                    # v10: 좌표가 상태 안에 (u,v)×13 으로 반복돼 있다. 학습 때와
                    # 같은 순서(goal_u0, goal_v0, goal_u1, ...)로 전부 채운다.
                    for _n in UV_NAMES:
                        goal[_n] = _LAST_UV["goal_u" if _n.startswith("goal_u") else "goal_v"]
                    # 남아 있는 environment_state 자리도 같이 채운다 (정책은 안
                    # 읽지만 프레임 조립이 요구한다).
                    if "goal_u" in goal:
                        goal["goal_u"] = _LAST_UV["goal_u"]; goal["goal_v"] = _LAST_UV["goal_v"]
                else:
                    goal["goal_u"] = _LAST_UV["goal_u"]; goal["goal_v"] = _LAST_UV["goal_v"]
            obs.update(goal)  # 목표값 (로봇 관측에는 없는 값이라 여기서 넣는다)
            frame = build_dataset_frame(features, obs, prefix=OBS_STR)
            # smolvla_base 는 camera1/2 를 기대하는데 우리 데이터는 top/wrist 다.
            # 학습 때 --rename_map 으로 맞췄으므로 추론에서도 같게 바꾼다.
            if CAM_RENAME:
                for src, dst in CAM_RENAME.items():
                    if src in frame:
                        frame[dst] = frame.pop(src)
            action_values = predict_action(
                observation=frame,
                policy=policy,
                device=device,
                preprocessor=pre,
                postprocessor=post,
                use_amp=cfg.use_amp,
                task=args.task,
                robot_type=robot.robot_type,
            )
            action = make_robot_action(action_values, features)

            # 목표 색이 트레이에 없다고 판단됐으면 홈 자세로 덮어쓴다.
            # 판단은 위에서 하고 적용은 여기서 — action 이 여기서 처음 생긴다.
            if holding:
                for k in home:
                    if k in action:
                        action[k] = float(home[k])

            # --- 위치별 자동 보정 (있으면) --------------------------------
            if corr is not None:
                # 파지에 가까울수록 100% (홈에서는 0) — 적합 범위 밖 외삽을 막는다
                action = apply_correction(action, corr,
                                          approach_progress(obs, home_depth))

            # --- 파지점 오프셋 -------------------------------------------
            # 정책은 알약 위치를 제대로 읽는다(probe_policy.py [4]: 사람 0.71 vs 정책 0.71)
            # 그런데 실기에서 일정하게 빗나간다 → 남은 것은 상수 편향이고, 상수로 뺀다.
            if args.zones:
                z = zone_of(action)
                if z != cur_zone:
                    cur_zone = z
                    zd, zl, zz = load_offset(args.run, args.task, z)
                    base_depth, base_lateral, base_z = zd, zl, zz
                    keys.depth_delta = keys.lateral_delta = keys.z_delta = 0.0
                    last_off = None
                    print(f"\n  ▸ 구역 {z} ({ZONE_NAMES[z]})  저장값 "
                          f"깊이 {zd:+.0f} 좌우 {zl:+.0f} 높이 {zz:+.0f}mm", flush=True)
            depth_mm = base_depth + keys.depth_delta
            lateral_mm = base_lateral + keys.lateral_delta
            z_mm = base_z + keys.z_delta
            # --- 파지 반사 ------------------------------------------------
            # 알약이 두 그리퍼 패드 사이에 들어오면 정책을 기다리지 않고 닫는다.
            # **팔 축은 건드리지 않는다** — 그리퍼 명령 하나만 덮어쓴다.
            # 한 번 걸리면 carrying 이 될 때까지 계속 닫아둔다(깜빡이면 놓친다).
            if args.grasp_reflex and not carrying:
                if reflex_latched:
                    action["gripper.pos"] = GRIPPER_CLOSED - 2.0
                else:
                    ok, why = grasp_reflex(obs.get("wrist"), args.grasp_reflex)
                    if ok:
                        reflex_hits += 1
                        # 몇 프레임 연속으로 걸려야 인정한다 (한 프레임 튐 방지)
                        if reflex_hits >= args.grasp_reflex_frames:
                            reflex_latched = True
                            action["gripper.pos"] = GRIPPER_CLOSED - 2.0
                            print(f"\n  ✊ 파지 반사 — {why}", flush=True)
                    else:
                        reflex_hits = 0
                        # 패드는 보이는데 알약이 사이에 없다 = 지금 못 잡는 그 상황.
                        # 얼마나 빗나갔는지 1초에 한 번만 찍는다 (매 프레임이면 도배).
                        if why and step - reflex_last_msg > args.fps:
                            reflex_last_msg = step
                            print(f"\n  ✋ {why}", flush=True)
            elif carrying:
                reflex_latched, reflex_hits = False, 0

            # 실제(관측) 그리퍼 위치. 알약을 물면 그 두께만큼 덜 닫힌다 —
            # 빈손으로 닫으면 끝까지 닫힌다. 둘을 가르는 값을 실측하려고 기록한다.
            _gm = obs.get("gripper.pos")
            if _gm is not None:
                grip_min = min(grip_min, float(_gm))

            g = action.get("gripper.pos")
            if g is not None and float(g) < GRIPPER_CLOSED:
                if not carrying and args.dump_grasp:
                    # 파지 순간의 화면을 남긴다. 나중에 그리퍼와 알약이 실제로 얼마나
                    # 어긋났는지 재서 보정을 계산하기 위함 (2026-08-07).
                    # 화살표로 감을 잡는 대신 사진을 측정해서 값을 낸다.
                    _d = Path(args.dump_grasp)
                    _d.mkdir(parents=True, exist_ok=True)
                    _t = time.strftime("%H%M%S")
                    for _n in ("top", "wrist"):
                        _im = obs.get(_n)
                        if _im is not None:
                            cv2.imwrite(str(_d / f"{_t}_{_n}.png"),
                                        np.ascontiguousarray(_im)[:, :, ::-1])
                    (_d / f"{_t}_pose.json").write_text(json.dumps(
                        {"action": {k: float(v) for k, v in action.items()},
                         "state": {k: float(v) for k, v in obs.items()
                                   if isinstance(v, (int, float))},
                         "offset": {"depth": depth_mm, "lateral": lateral_mm, "z": z_mm}},
                        indent=1))
                    print(f"\n  📷 파지 순간 저장: {_t}", flush=True)
                if not carrying:
                    grip_min = 999.0     # 이번 파지 구간의 최솟값을 새로 잰다
                carrying = True          # 집었다 → 여기서부터 약통까지는 원래 명령대로
            elif g is not None and float(g) > GRIPPER_OPEN:
                if carrying:
                    seq_done = False
                    # 집었다가 놓았다고 곧바로 성공이 아니다 — 빈손으로 닫아도, 회복
                    # 중 재시도로 놓아도 같은 신호가 난다. 두 조건을 함께 본다.
                    _pan = obs.get("shoulder_pan.pos")
                    at_bottle = _pan is not None and BOTTLE_PAN[0] <= float(_pan) <= BOTTLE_PAN[1]
                    held = grip_min >= HELD_MIN
                    if not (at_bottle and held):
                        why = []
                        if not held:
                            why.append(f"빈손(그리퍼 {grip_min:.1f} < {HELD_MIN})")
                        if not at_bottle:
                            why.append(f"약통 밖(pan {float(_pan) if _pan is not None else 0:.1f})")
                        # continue 를 쓰면 안 된다 — 이 아래에 robot.send_action 이 있어
                        # 그 프레임의 명령이 통째로 사라진다. 판정이 반복되는 동안
                        # 팔이 멈춘다 (2026-08-16 실기에서 두 번째 색이 안 움직인 원인).
                        print(f"\n  ✋ 놓쳤습니다 — {', '.join(why)}. 정책이 다시 시도합니다.",
                              flush=True)
                        grip_min = 999.0
                    else:
                        print(f"\n  ✅ 담기 완료 — 그리퍼 {grip_min:.1f}, "
                              f"pan {float(_pan):.1f} (약통)", flush=True)
                        seq_done = True
                    if seq and seq_done:
                        seq_i += 1
                        if seq_i >= len(seq):
                            print(f"\n  🎉 처방 조제 완료 — {' → '.join(seq)}", flush=True)
                            go_home(robot, home, args.home_time)
                            break
                        # 목표만 바꾼다. 정책도 카메라도 그대로다 — 이것이 원-핫의 이점이다.
                        args.task = f"pick {seq[seq_i]} pill"

                        # 정지 판단(--hold-home)이 보는 색도 같이 바꾼다.
                        # 2026-08-20: 여기서 안 바꿔 초록을 담은 뒤에도 TASK_COLOR 가
                        # "green" 으로 남았다. 트레이에 초록이 없으니 안전장치가
                        # "목표 색이 없다" 며 홈에 붙잡았고 노랑으로 영영 못 갔다.
                        # (웹 조제에서 재현 — 후반 pan 이 ±1.3도로 굳었다.)
                        # 8/17 의 3색 성공은 --hold-home 이 생기기 전(8/18)이라
                        # 이 버그에 걸리지 않았다. 연속 조제에서만 터진다 — 단색은
                        # 시작할 때 그 색이 트레이에 있으므로 막히지 않는다.
                        try:
                            TASK_COLOR = color_of_task(args.task)
                        except Exception:
                            TASK_COLOR = None

                        # 담은 직후 팔은 약통 위에 있다. 학습 시연은 전부 홈에서
                        # 시작하므로 그 자세는 학습 분포 밖이다. --seq-home 으로 켠다.
                        # (앞서 이 조치를 시험했을 때는 continue 버그가 함께 있어
                        #  로봇 명령이 죽는 상태였다 — 그 실험은 무효다.)
                        print(f"\n  ▶ 다음 목표: {seq[seq_i]} ({seq_i + 1}/{len(seq)})",
                              flush=True)
                        if args.seq_home:
                            print("  … 홈으로 복귀", flush=True)
                            go_home(robot, home, args.home_time)
                        policy.reset()
                        goal = goal_values(meta, args.task)
                        # 좌표 정책은 매 프레임 화면에서 목표 색을 검출한다. 여기서
                        # 갱신하지 않으면 색을 바꿔도 **첫 색만 계속 찾는다**.
                        UV_NAMES = goal.pop("__uv_names__", None)
                        _lv = goal.pop("__live_uv__", None)
                        if _lv:
                            LIVE_UV_COLOR = _lv
                            _LAST_UV = {"goal_u": 0.5, "goal_v": 0.5}   # 새 색은 처음부터 다시 찾는다
                        saved_depth, saved_lateral, saved_z = load_offset(args.run, args.task)
                        grip_min = 999.0
                carrying = False         # 놓았다 → 다음 집기를 위해 다시 건다
            # 켜고 끄는 것을 한 프레임에 하면 오프셋만큼(20mm ≈ 깊이축 9.5도) 계단이 생겨
            # 팔이 툭 튄다. 0.5초에 걸쳐 빼고 넣는다.
            fade = np.clip(fade + (-1.0 if carrying else 1.0) / (0.5 * args.fps), 0.0, 1.0)
            if (depth_mm or lateral_mm or z_mm) and fade > 0.0:
                # 홈에서는 0, 접근 절반부터는 100% (그 뒤로 상수 — 미는 성분이 없다)
                r = 1.0 if args.no_offset_ramp else approach_progress(obs, home_depth)
                pan = None if args.radial_offset else obs.get("shoulder_pan.pos")
                action = grasp_offset(action, depth_mm * r * fade,
                                      lateral_mm * r * fade, pan, z_mm * r * fade)

            # --- 집는 동안 팔 고정 ---------------------------------------
            # 그리퍼가 닫히기 시작하면 팔을 그 자리에 붙잡고 **그리퍼만** 닫는다.
            # 닫히는 데 1~2.7초가 걸리는데(학습 1.13초) 그동안 팔이 계속 전진하면
            # 손끝이 알약을 앞으로 밀어버린다 — "이상할 땐 앞으로 밀면서 집는다".
            # 실측으로 닫는 동안 팔이 최대 11mm 움직이고 있었다 (2026-08-05).
            if not args.no_freeze_on_grasp and g is not None:
                closing = float(g) < GRIPPER_OPEN and not carrying
                if closing:
                    if freeze_pose is None:
                        freeze_pose = {k: float(v) for k, v in action.items()
                                       if k != "gripper.pos"}
                        print(f"\n  파지 — 팔 고정 (그리퍼만 닫힘)", flush=True)
                    action.update(freeze_pose)   # 그리퍼 명령은 그대로 통과시킨다
                else:
                    freeze_pose = None
            if (depth_mm, lateral_mm, z_mm) != last_off:
                # flush 필수 — tee 로 파이프되면 블록 버퍼링이라 안 그러면 로그에 안 남는다
                zt = f"  [구역 {cur_zone} {ZONE_NAMES[cur_zone]}]" if args.zones else ""
                print(f"\n  파지점 보정  깊이 {depth_mm:+.0f}mm (+먼쪽/-가까운쪽)   "
                      f"좌우 {lateral_mm:+.0f}mm   높이 {z_mm:+.0f}mm (+위로){zt}", flush=True)
                save_offset(depth_mm, lateral_mm, z_mm, args.run, args.task,
                            cur_zone if args.zones else None)
                last_off = (depth_mm, lateral_mm, z_mm)

            # 스텝당 변화량 제한. 이전에 **보낸 명령** 기준으로 자른다(실측 기준으로 자르면
            # 추종 지연이 누적돼 팔이 계속 뒤처진다). 목표는 그대로 두고 가는 속도만 학습 수준으로 낮춘다.
            if args.limit is not None:
                for k, lim in STEP_LIMIT.items():
                    if k not in action:
                        continue
                    prev = last_cmd.get(k)
                    if prev is None:
                        continue
                    d = float(action[k]) - prev
                    cap = lim * args.limit
                    if abs(d) > cap:
                        action[k] = prev + (cap if d > 0 else -cap)
                        clipped += 1
                last_cmd.update({k: float(v) for k, v in action.items()})

            robot.send_action(action)

            # 명령과 실제 위치를 그대로 남긴다. "잡기 직전에 순간이동한다" 같은 증상은
            # 눈으로 보면 원인을 못 가른다 — 명령이 튀는 것(정책 문제)과 명령은 매끄러운데
            # 팔만 튀는 것(서보/기구 문제)이 전혀 다른데 겉보기는 같기 때문이다.
            # 스텝당 리스트 append 뿐이라 제어 주기에 영향이 없다 (종료할 때 한 번에 쓴다).
            if trace is not None:
                trace.append([loop_t - t0]
                             + [float(action.get(k, float("nan"))) for k in JOINT_KEYS]
                             + [float(obs.get(k, float("nan"))) for k in JOINT_KEYS])

            if args.display:
                from lerobot.utils.visualization_utils import log_rerun_data

                log_rerun_data(observation=obs, action=action)

            # 정책이 보고 있는 화면을 그대로 띄운다. 카메라는 이 프로세스가 독점하고
            # 있어서 별도 뷰어로는 못 연다 (V4L2 가 두 번 열리지 않는다).
            # --display 는 rerun 뷰어(별도 설치)를 요구하므로 여기서는 cv2 를 쓴다.
            # 최근 30 스텝의 실제 주기로 Hz 를 낸다. 예전에는 step/(지금-시작) 이었는데
            # **정지 상태로 대기한 시간까지 분모에 들어가서** s 를 늦게 누를수록 낮게 나왔다.
            # 2026-08-03 에 이 숫자를 보고 "Hz 가 낮아서 실패한다"고 오진할 뻔했다.
            recent.append(loop_t)
            hz = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6) if len(recent) > 1 else 0.0

            if args.show and step % args.show_every == 0:
                show_cameras(obs, step, running, hz, keys)

            # --- 주기 유지 ------------------------------------------------
            step += 1
            dt = time.perf_counter() - loop_t
            if dt > 1 / args.fps:
                slow += 1
            precise_sleep(max(1 / args.fps - dt, 0.0))

            now = time.perf_counter()
            if now - last_report >= 10.0:
                el = now - t0
                print(
                    f"  [{int(el // 60):02d}:{int(el % 60):02d}] {step} 스텝, "
                    f"실제 {hz:.1f} Hz, 목표({args.fps}Hz) 미달 {slow}회 "
                    f"({100 * slow / max(step, 1):.1f}%)"
                    + (f", 명령 제한 {clipped}회" if args.limit is not None else "")
                )
                last_report = now

    except KeyboardInterrupt:
        print("\nCtrl+C — 홈으로 돌아갑니다")
        try:
            go_home(robot, home, args.home_time)
        except Exception as e:
            logging.warning(f"홈 복귀 실패: {e}")
    finally:
        # 어떻게 끝나든(정상 종료·Ctrl+C·예외) 기록은 남긴다.
        # 끊어서 볼 때가 많은데 정상 종료에만 저장하면 매번 다시 돌려야 한다.
        if trace:
            _save_trace(trace, args.trace)
        keys.stop()
        if vw is not None:
            vw.release(); print(f"  영상 저장: {args.record_video}")
        close_cameras_window()
        robot.disconnect()
        if not args.relax_on_exit:
            print("토크는 켜둔 채로 종료합니다 (팔이 자세를 유지). 힘을 빼려면 --relax-on-exit")
        el = time.perf_counter() - t0
        print(f"총 {int(el // 60)}분 {int(el % 60)}초, {step} 스텝 실행")


if __name__ == "__main__":
    main()
