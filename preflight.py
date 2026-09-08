#!/usr/bin/env python
"""녹화·평가 전에 '지난번과 같은 환경인가'를 30초 만에 확인한다.

모방학습에서 되돌릴 수 없는 실수는 관측 분포가 바뀐 채로 데이터를 찍는 것이다.
학습 설정은 언제든 다시 돌리면 되지만, 카메라가 밀린 채 찍은 60개는 이전 데이터와
섞을 수 없다. 2026-07-31에 top 화각이 dx=-25px 밀린 채로 평가를 돌린 적이 있다.

기준 프레임을 한 번 저장해 두고(--save-ref), 이후 매 세션 시작 때 비교한다.

사용법:
  python ~/omx_pill_project/preflight.py --save-ref   # 세팅 확정 후 최초 1회
  python ~/omx_pill_project/preflight.py              # 매 세션 시작 때
  python ~/omx_pill_project/preflight.py --no-motors  # 팔 없이 카메라만

종료 코드: 0 = 통과, 1 = FAIL 있음 (녹화하면 안 됨)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/user/omx_pill_project")
REF = ROOT / "preflight_ref"
CALIB = Path("/home/user/.cache/huggingface/lerobot/calibration")

CAMS = {
    "top": "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0",
    "wrist": "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0",
}
ARMS = {
    # 이름: (장치 링크, 기대 모터 ID, 캘리브레이션 JSON)
    "follower": ("/dev/omx_follower", [11, 12, 13, 14, 15, 16],
                 CALIB / "robots/omx_follower/omx_follower_arm.json"),
    "leader": ("/dev/omx_leader", [1, 2, 3, 4, 5, 6],
               CALIB / "teleoperators/omx_leader/omx_leader_arm.json"),
}
BAUD = 1_000_000
W, H, FPS = 640, 480, 30

# 화각이 이보다 밀리면 이전 데이터와 섞을 수 없다고 본다.
SHIFT_WARN, SHIFT_FAIL = 5.0, 12.0
# 조명 변화 허용치 (0~255 평균 기준)
BRIGHT_WARN, BRIGHT_FAIL = 12.0, 25.0
# 180 에피소드 예상 용량 여유
DISK_MIN_GB = 20.0

results: list[tuple[str, str, str]] = []  # (상태, 항목, 메시지)


def add(status: str, item: str, msg: str) -> None:
    results.append((status, item, msg))
    mark = {"OK": "✓", "WARN": "!", "FAIL": "✗"}[status]
    print(f"  {mark} {item:22s} {msg}")


# ────────────────────────────────────────────── 1. 장치 경로
def check_paths(no_motors: bool) -> None:
    print("\n[1] 장치 경로")
    for name, path in CAMS.items():
        if Path(path).exists():
            add("OK", f"{name} 카메라", "연결됨")
        else:
            add("FAIL", f"{name} 카메라", f"경로 없음 — USB 확인: {path}")
    if no_motors:
        return
    for name, (dev, _, _) in ARMS.items():
        if Path(dev).exists():
            add("OK", f"{name} 팔", f"{dev}")
        else:
            add("FAIL", f"{name} 팔", f"{dev} 없음 — USB 또는 udev 규칙 확인")


# ────────────────────────────────────────────── 2. 모터 + 캘리브레이션
def check_motors() -> None:
    print("\n[2] 모터 응답 · 캘리브레이션 일치")
    try:
        from dynamixel_sdk import PacketHandler, PortHandler
    except ImportError:
        add("WARN", "dynamixel_sdk", "설치 안 됨 — 모터 점검 건너뜀")
        return

    for name, (dev, want_ids, calib_path) in ARMS.items():
        if not Path(dev).exists():
            continue
        port = PortHandler(dev)
        ph = PacketHandler(2.0)
        try:
            if not port.openPort() or not port.setBaudRate(BAUD):
                add("FAIL", f"{name} 포트", "열기 실패 (다른 프로그램이 쓰는 중일 수 있음)")
                continue
            found, comm = ph.broadcastPing(port)
            ids = sorted(found.keys()) if found else []
            missing = [i for i in want_ids if i not in ids]
            if missing:
                add("FAIL", f"{name} 모터", f"응답 {ids} — {missing} 없음")
            else:
                add("OK", f"{name} 모터", f"ID {ids} 전부 응답")

            if not calib_path.exists():
                add("WARN", f"{name} 캘리브", f"파일 없음: {calib_path.name}")
                continue
            calib = json.loads(calib_path.read_text())
            bad = []
            for joint, c in calib.items():
                mid = c["id"]
                if mid not in ids:
                    continue
                dm, _, _ = ph.read1ByteTxRx(port, mid, 10)       # Drive_Mode
                ho, _, _ = ph.read4ByteTxRx(port, mid, 20)       # Homing_Offset (signed)
                if ho > 2**31:
                    ho -= 2**32
                mx, _, _ = ph.read4ByteTxRx(port, mid, 48)       # Max_Position_Limit
                mn, _, _ = ph.read4ByteTxRx(port, mid, 52)       # Min_Position_Limit
                got = (dm, ho, mn, mx)
                want = (c["drive_mode"], c["homing_offset"], c["range_min"], c["range_max"])
                if got != want:
                    bad.append(f"{joint}: 모터{got} ≠ 파일{want}")
            if bad:
                add("FAIL", f"{name} 캘리브", f"{len(bad)}개 관절 불일치")
                for b in bad:
                    print(f"      {b}")
                print("      → 이대로 녹화하면 lerobot 이 모터에 파일 값을 덮어씁니다")
            else:
                add("OK", f"{name} 캘리브", "모터 EEPROM 과 일치")
        finally:
            try:
                port.closePort()
            except Exception:
                pass


# ────────────────────────────────────────────── 3. 카메라 실측
def grab(path: str):
    """lerobot 과 동일하게 연다 (V4L2 + MJPG). 실패하면 None."""
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, {}
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    frame = None
    for _ in range(15):  # 자동 노출 안정화
        ok, f = cap.read()
        if ok:
            frame = f
    cc = int(cap.get(cv2.CAP_PROP_FOURCC))
    meta = {
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": "".join(chr((cc >> (8 * i)) & 0xFF) for i in range(4)),
    }
    cap.release()
    return frame, meta


def check_cameras(save_ref: bool) -> dict:
    print("\n[3] 카메라 설정 · 화면")
    frames = {}
    for name, path in CAMS.items():
        if not Path(path).exists():
            continue
        frame, meta = grab(path)
        if frame is None:
            add("FAIL", f"{name} 열기", "프레임을 못 받음 (녹화·평가가 돌고 있으면 끄세요)")
            continue
        frames[name] = frame
        if (meta["w"], meta["h"]) != (W, H) or meta["fourcc"] != "MJPG":
            add("FAIL", f"{name} 설정",
                f'{meta["w"]}x{meta["h"]} {meta["fourcc"]} — 기대값 {W}x{H} MJPG')
        else:
            add("OK", f"{name} 설정", f'{W}x{H} MJPG {meta["fps"]:.0f}fps')
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(g.std()) < 8.0:
            add("FAIL", f"{name} 화면", f"거의 단색 (std {g.std():.1f}) — 렌즈 가림·먹통 의심")
        else:
            add("OK", f"{name} 화면", f"평균 {g.mean():.0f} / 편차 {g.std():.0f}")
    return frames


# ────────────────────────────────────────────── 4. 화각 · 조명 비교
def check_alignment(frames: dict, save_ref: bool) -> None:
    print("\n[4] 화각 · 조명 (기준 프레임 대비)")
    REF.mkdir(exist_ok=True)
    if save_ref:
        for name, f in frames.items():
            cv2.imwrite(str(REF / f"{name}.png"), f)
        (REF / "meta.json").write_text(json.dumps(
            {n: {"mean": float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean())}
             for n, f in frames.items()}, indent=1))
        add("OK", "기준 저장", f"{REF} — 이후 이 화면을 기준으로 비교합니다")
        return

    meta_p = REF / "meta.json"
    if not meta_p.exists():
        add("WARN", "기준 프레임", "없음 — 세팅 확정 후 --save-ref 로 한 번 저장하세요")
        return
    ref_meta = json.loads(meta_p.read_text())

    for name, now in frames.items():
        rp = REF / f"{name}.png"
        # top 은 **학습 영상에서 뽑은 화각**(top_train.png)이 있으면 그쪽을 기준으로 쓴다.
        # top.png 는 녹화 *전에* 저장한 것이라 실제 학습 화각과 다르다 — 2026-08-03 에
        # 녹화 20개를 전수로 재보니 ep2 에서 +6.4/+3.2 px 옮겨진 뒤 ep19 까지 고정이었다.
        # 새로 찍을 데이터는 기존 20개와 섞여야 하므로 맞춰야 할 쪽은 학습 화각이다.
        if name == "top" and (REF / "top_train.png").exists():
            rp = REF / "top_train.png"
        if not rp.exists():
            continue

        # wrist 는 팔에 달려 있어서 팔 자세가 바뀌면 화면이 당연히 달라진다.
        # 기준 프레임과 비교하는 건 팔이 그때와 똑같은 자세일 때만 의미가 있는데,
        # 그럴 보장이 없다 (기준 저장 이후 텔레옵만 해도 팔이 어디로든 간다).
        # 실제로 2026-08-02 에 이것 때문에 dy=+65px / 신뢰도 0.01 로 오탐이 나서
        # 녹화가 막혔다. wrist 는 [3]번 항목(설정·화면 정상)으로 충분히 걸러진다.
        if name == "wrist":
            add("OK", "wrist 화각", "비교 안 함 (팔에 달린 카메라라 자세따라 화면이 바뀜)")
            continue

        ref = cv2.imread(str(rp))
        a = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY).astype(np.float32)
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a * win, b * win)
        shift = float(np.hypot(dx, dy))
        txt = f"dx={dx:+.1f} dy={dy:+.1f} px (신뢰도 {resp:.2f})"
        if shift >= SHIFT_FAIL:
            add("FAIL", f"{name} 화각", f"{txt} — 이 상태로 찍으면 기존 데이터와 못 섞습니다")
        elif shift >= SHIFT_WARN:
            add("WARN", f"{name} 화각", f"{txt} — 맞춰 두는 게 좋습니다")
        else:
            add("OK", f"{name} 화각", txt)

        d = b.mean() - ref_meta.get(name, {}).get("mean", b.mean())
        if abs(d) >= BRIGHT_FAIL:
            add("FAIL", f"{name} 조명", f"평균 밝기 {d:+.0f} — 조명이 크게 달라졌습니다")
        elif abs(d) >= BRIGHT_WARN:
            add("WARN", f"{name} 조명", f"평균 밝기 {d:+.0f}")
        else:
            add("OK", f"{name} 조명", f"평균 밝기 {d:+.0f}")

        # 정렬용 겹친 이미지: 기준=빨강, 지금=청록. 색이 사라지면 맞은 것.
        g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        g_now = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY)
        blend = cv2.merge([g_now, g_now, g_ref])
        cv2.putText(blend, "RED=REF  CYAN=NOW", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(blend, "RED=REF  CYAN=NOW", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(ROOT / f"report/preflight_{name}_ALIGN.png"), blend)


# ────────────────────────────────────────────── 5. 저장공간 · GPU
def check_system() -> None:
    print("\n[5] 저장공간 · GPU")
    free = shutil.disk_usage("/home/user").free / 1e9
    if free < DISK_MIN_GB:
        add("FAIL", "디스크", f"{free:.0f}GB 남음 — 180 에피소드에 최소 {DISK_MIN_GB:.0f}GB 필요")
    else:
        add("OK", "디스크", f"{free:.0f}GB 남음")
    try:
        import torch
        if torch.cuda.is_available():
            add("OK", "CUDA", torch.cuda.get_device_name(0))
        else:
            add("WARN", "CUDA", "사용 불가 — 녹화는 되지만 학습을 못 합니다")
    except ImportError:
        add("WARN", "CUDA", "torch 없음 — 확인 못 함")


# ────────────────────────────────────────────── 손으로 확인할 것
MANUAL = [
    "★ 로봇 베이스 클램프가 표시해 둔 자리에 그대로 있는가 "
    "(베이스가 밀리면 같은 위치의 알약이 다른 관절값이 되어 전 데이터가 무효)",
    "★ top 카메라 붐암 클램프가 표시해 둔 자리에 그대로 있는가 "
    "(붐암은 처지고 부딪히면 밀린다 — 7/31 dx=-25px 사고)",
    "트레이가 바닥에 표시해 둔 자리에 놓여 있는가",
    "트레이 검은 도화지가 깔려 있고 가장자리 경사가 살아 있는가",
    "배치 영역 네 귀퉁이에 팔이 무리 없이 닿는가 (bash ~/il_ws/teleop.sh 로 확인)",
    "화면 안에 학습에 없던 물체(공구·종이·컵·인형)가 들어와 있지 않은가",
    "wrist 카메라 USB 케이블이 팔 움직임에 당겨지지 않게 고정돼 있는가",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="녹화 전 환경 점검")
    ap.add_argument("--save-ref", action="store_true",
                    help="지금 화면을 기준으로 저장 (세팅 확정 후 최초 1회)")
    ap.add_argument("--no-motors", action="store_true", help="팔 점검 건너뛰기")
    args = ap.parse_args()

    (ROOT / "report").mkdir(exist_ok=True)
    print("=" * 66)
    print(" 녹화 전 환경 점검")
    print("=" * 66)

    check_paths(args.no_motors)
    if not args.no_motors:
        check_motors()
    frames = check_cameras(args.save_ref)
    check_alignment(frames, args.save_ref)
    check_system()

    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    n_warn = sum(1 for s, _, _ in results if s == "WARN")

    print("\n" + "=" * 66)
    if n_fail:
        print(f" 결과: FAIL {n_fail}개 / WARN {n_warn}개 — 녹화를 시작하지 마세요")
        for s, item, msg in results:
            if s == "FAIL":
                print(f"   ✗ {item}: {msg}")
    else:
        print(f" 결과: 통과 ✓  (WARN {n_warn}개)")
    print("=" * 66)
    print("\n[눈으로 확인할 것 — 자동으로는 못 봅니다]")
    for m in MANUAL:
        print(f"  □ {m}")
    print()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
