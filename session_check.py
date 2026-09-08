#!/usr/bin/env python
"""세션 시작 전 점검 — 녹화·평가 전에 매번 돌린다. 하나라도 FAIL 이면 시작하지 않는다.

왜 만들었나 — 2026-08-03 에 하루를 통째로 날린 원인이 전부 "어긋났는데 몰랐다" 였다.

  · top 화각 기준 파일이 **녹화 전** 저장본이라 실제 학습 화각과 6.4px 달랐다
  · 세션 도중 매트가 11mm, 약통이 6mm 밀렸다 (팔이 밀어올림)
  · **wrist 초점이 15배 흐려져 있었다** (선명도 184, 학습 때 1287~5369)
  · 홈 자세가 v1 파일럿 값이었다 (wrist_roll 2.7도 차이)
  · 카메라 USB 가 빠졌는데 검은 프레임만 나와서 한참 다른 걸 의심했다

이 중 어느 것도 평가 결과만 봐서는 알 수 없다. 그래서 매번 기계적으로 잰다.

  python ~/omx_pill_project/session_check.py              # 전체 점검
  python ~/omx_pill_project/session_check.py --no-robot   # 카메라만 (로봇 안 켰을 때)

기준값은 `preflight_ref/session_ref.json` 에 저장한다. **녹화가 끝난 뒤**
`make_reference.py` 로 학습 영상에서 다시 만드는 것이 원칙이다 — 녹화 전에 저장한
기준은 실제 학습 화각과 다를 수 있고, 실제로 v2 에서 그것 때문에 평가 두 번을 날렸다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path("/home/user/omx_pill_project")
REF = PROJECT / "preflight_ref"
REF_META = REF / "session_ref.json"
TOP = "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"
WRIST = "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0"

SHIFT_WARN, SHIFT_FAIL = 5.0, 12.0
MOTOR_IDS = [11, 12, 13, 14, 15, 16]

# 화각을 구역별로 잰다. 중앙값 하나만 보면 "카메라는 멀쩡한데 매트만 밀린" 경우를 놓친다.
# 2026-08-03 에 실제로 놓쳤다 — align_top.py 는 세 구역의 중앙값을 쓴다.
REGIONS = {
    "robot": (400, 478, 190, 400),
    "mat_left": (150, 330, 115, 155),
    "mat_right": (150, 330, 480, 520),
    "pillbox": (130, 350, 520, 610),
}


class Result:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, level: str, name: str, detail: str) -> None:
        self.rows.append((level, name, detail))

    @property
    def failed(self) -> bool:
        return any(r[0] == "FAIL" for r in self.rows)

    def show(self) -> None:
        mark = {"OK": "  OK ", "WARN": " WARN", "FAIL": " FAIL"}
        print()
        for level, name, detail in self.rows:
            print(f"[{mark[level]}] {name:<18} {detail}")
        print()
        if self.failed:
            print("FAIL 이 있습니다 — 고치고 다시 돌리세요. 이 상태로 찍으면 기존 데이터와 못 섞입니다.")
        elif any(r[0] == "WARN" for r in self.rows):
            print("WARN 만 있습니다 — 내용을 보고 판단하세요.")
        else:
            print("전부 통과. 시작해도 됩니다.")


def grab(path: str, n: int = 20) -> np.ndarray | None:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        return None
    ok, fr = False, None
    for _ in range(n):
        ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def sharpness(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def color_spread(img: np.ndarray) -> float:
    """채널 간 평균 차이. 0 이면 흑백 — 카메라가 야간모드로 넘어갔거나 고장난 것이다."""
    b, g, r = (img[:, :, i].astype(int) for i in range(3))
    return float((np.abs(b - g).mean() + np.abs(g - r).mean()) / 2)


def check_cameras(res: Result, ref_meta: dict) -> dict:
    """연결·컬러·밝기·선명도, 그리고 기준이 있으면 구역별 화각."""
    now = {}
    for name, path in (("top", TOP), ("wrist", WRIST)):
        img = grab(path)
        if img is None:
            res.add("FAIL", f"{name} 카메라", "장치를 열 수 없습니다 — USB 연결을 확인하세요")
            continue
        now[name] = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bright, sharp, spread = gray.mean(), sharpness(img), color_spread(img)

        if bright < 5:
            res.add("FAIL", f"{name} 프레임", f"검은 화면 (밝기 {bright:.0f}) — USB 가 빠졌거나 물려 있습니다")
            continue
        if spread < 0.5:
            res.add("FAIL", f"{name} 색", "흑백으로 나옵니다 — 색으로 알약을 고르는 과제라 치명적입니다")
        else:
            res.add("OK", f"{name} 색", f"채널차이 {spread:.1f}")

        lo, hi = ref_meta.get(f"{name}_bright", (60, 210))
        lvl = "OK" if lo <= bright <= hi else "WARN"
        res.add(lvl, f"{name} 밝기", f"{bright:.0f} (기준 {lo:.0f}~{hi:.0f})")

        rng = ref_meta.get(f"{name}_sharp")
        if rng:
            lvl = "OK" if sharp >= rng[0] else "FAIL"
            detail = f"{sharp:.0f} (학습 {rng[0]:.0f}~{rng[1]:.0f})"
            if lvl == "FAIL":
                detail += "  ← 렌즈 초점이 풀렸습니다. focus_wrist.py 로 맞추세요"
            res.add(lvl, f"{name} 선명도", detail)
        else:
            res.add("WARN", f"{name} 선명도", f"{sharp:.0f} (기준 없음 — 녹화 후 make_reference.py)")
    return now


def check_framing(res: Result, top_now: np.ndarray | None) -> None:
    ref_path = REF / "top_train.png"
    if top_now is None or not ref_path.exists():
        res.add("WARN", "화각", "기준(top_train.png)이 없습니다 — 녹화 후 만듭니다")
        return
    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    gnow = cv2.cvtColor(top_now, cv2.COLOR_BGR2GRAY)
    moved = []
    for name, (y0, y1, x0, x1) in REGIONS.items():
        tpl = ref[y0:y1, x0:x1]
        r = cv2.matchTemplate(gnow, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(r)
        dx, dy = loc[0] - x0, loc[1] - y0
        shift = float(np.hypot(dx, dy))
        if score < 0.5:
            res.add("WARN", f"화각 {name}", f"일치도 {score:.2f} — 판정 불가")
            continue
        lvl = "OK" if shift < SHIFT_WARN else ("WARN" if shift < SHIFT_FAIL else "FAIL")
        if lvl != "OK":
            moved.append(name)
        res.add(lvl, f"화각 {name}", f"dx={dx:+d} dy={dy:+d} ({shift:.0f}px, 일치도 {score:.2f})")

    if len(moved) >= 3:
        res.add("WARN", "화각 해석", "여러 구역이 같이 밀렸습니다 → **카메라**가 움직인 것 (align_top.py)")
    elif moved:
        res.add("WARN", "화각 해석", f"{', '.join(moved)} 만 밀렸습니다 → **그 물건**이 움직인 것 (align_scene.py)")


def check_bus(res: Result) -> None:
    try:
        from dynamixel_sdk import PacketHandler, PortHandler
    except Exception as e:
        res.add("WARN", "모터 버스", f"dynamixel_sdk 없음: {e}")
        return
    # 장치가 아예 없으면 openPort() 가 False 를 돌려주는 게 아니라 예외를 던진다.
    # 2026-08-04 에 팔로워 USB 가 빠진 상태로 돌려서 점검 도구 자체가 죽었다.
    port = PortHandler("/dev/omx_follower")
    try:
        opened = port.openPort()
    except Exception as e:
        res.add("FAIL", "모터 버스", f"/dev/omx_follower 없음 — USB 커넥터를 확인하세요 ({type(e).__name__})")
        return
    if not opened:
        res.add("FAIL", "모터 버스", "/dev/omx_follower 를 열 수 없습니다 (전원·USB 확인)")
        return
    port.setBaudRate(1_000_000)
    ph = PacketHandler(2.0)
    found = set()
    for _ in range(3):
        res_ping, _ = ph.broadcastPing(port)
        found |= set(res_ping)
    missing = [i for i in MOTOR_IDS if i not in found]
    temps = []
    for i in MOTOR_IDS:
        v, comm, _ = ph.read1ByteTxRx(port, i, 146)
        if comm == 0:
            temps.append(v)
    port.closePort()
    if missing:
        res.add("FAIL", "모터 버스", f"응답 없는 id {missing} — 케이블·커넥터 확인")
    else:
        res.add("OK", "모터 버스", f"6개 응답, 온도 {min(temps)}~{max(temps)}C")


def check_home(res: Result, repo_id: str) -> None:
    home_file = PROJECT / "home_pose.json"
    if not home_file.exists():
        res.add("WARN", "홈 자세", "home_pose.json 없음 — 녹화 후 데이터셋에서 만드세요")
        return
    import glob

    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"/home/user/.cache/huggingface/lerobot/{repo_id}/data/chunk-000/*.parquet"))
    if not files:
        res.add("WARN", "홈 자세", f"{repo_id} 데이터가 없어 대조 못 함")
        return
    starts = []
    for f in files:
        t = pq.read_table(f)
        ei = np.array(t.column("episode_index").to_pylist())
        st = np.array([np.asarray(x) for x in t.column("observation.state").to_pylist()], dtype=np.float32)
        for e in sorted(set(ei.tolist())):
            starts.append(st[ei == e][:10].mean(axis=0))
    med = np.median(np.array(starts), axis=0)
    home = json.loads(home_file.read_text())
    names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
             "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    diff = max(abs(float(home[n]) - med[i]) for i, n in enumerate(names))
    lvl = "OK" if diff < 1.0 else "FAIL"
    detail = f"데이터셋 시작 자세와 최대 {diff:.2f}도 차이"
    if lvl == "FAIL":
        detail += "  ← 다른 데이터셋 값입니다. 지금 쓰는 데이터셋에서 다시 만드세요"
    res.add(lvl, "홈 자세", detail)


def main() -> None:
    ap = argparse.ArgumentParser(description="세션 시작 전 점검")
    ap.add_argument("--no-robot", action="store_true", help="모터·홈 자세 점검을 건너뛴다")
    ap.add_argument("--repo-id", default="1unasy/pill_v3", help="홈 자세를 대조할 데이터셋")
    args = ap.parse_args()

    ref_meta = json.loads(REF_META.read_text()) if REF_META.exists() else {}
    res = Result()
    now = check_cameras(res, ref_meta)
    check_framing(res, now.get("top"))
    if not args.no_robot:
        check_bus(res)
        check_home(res, args.repo_id)

    res.show()
    raise SystemExit(1 if res.failed else 0)


if __name__ == "__main__":
    main()
