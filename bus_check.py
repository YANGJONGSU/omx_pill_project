#!/usr/bin/env python
"""모터 버스가 간헐적으로 끊기는지 계측한다 (케이블 장력 의심).

왜 만들었나 — 2026-08-03 평가 중에 `run_policy.py` 가 뜨지 못했고, 원인이
**id15(wrist_roll)가 버스 스캔에서 빠진 것**이었다. 바로 다시 핑을 때리니 6개가
전부 응답했다. 즉 죽은 게 아니라 **간헐적으로 끊긴다**. 이런 끊김은 온도·화각 같은
정적인 계측에는 절대 안 잡히는데, 실기에서는 팔이 "툭" 하고 튀는 것으로 나타난다
(명령/응답 패킷이 한 번 씹히면 정확히 그렇게 움직인다).

같은 세션에서 사용자가 **"로봇 기준 왼쪽에 알약을 뒀을 때 더 자주 난다"** 고 관찰했다.
그 자세가 손목 쪽 케이블을 당긴다면 장력이 원인이고, 그러면 **끊김이 특정 자세 구간에
몰려서** 나타난다. 이 스크립트는 그걸 직접 잰다 — 자세(pan)별 실패율 표를 뽑는다.

  python ~/omx_pill_project/bus_check.py                 # 60초 감시 (팔 안 움직임)
  python ~/omx_pill_project/bus_check.py --seconds 180
  python ~/omx_pill_project/bus_check.py --sweep         # pan 을 좌우로 쓸며 자세별로 잰다

`--sweep` 은 팔을 실제로 움직인다. 트레이 위 물건을 치우고 돌릴 것.
읽기 전용인 기본 모드는 팔을 건드리지 않으므로 정책을 돌리다 멈춘 직후에 그대로 써도 된다.

주의: 이 스크립트가 도는 동안에는 `run_policy.py`/녹화를 같이 돌릴 수 없다
(시리얼 포트를 두 프로세스가 못 연다).
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict

from dynamixel_sdk import COMM_SUCCESS, GroupSyncRead, PacketHandler, PortHandler

PORT = "/dev/omx_follower"
BAUD = 1_000_000
IDS = {11: "shoulder_pan", 12: "shoulder_lift", 13: "elbow_flex",
       14: "wrist_flex", 15: "wrist_roll", 16: "gripper"}

ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
ADDR_GOAL_POSITION = 116
ADDR_HARDWARE_ERROR = 70
ADDR_TORQUE_ENABLE = 64

# pan(11번) 원시 틱을 도로 바꾸는 값. XL430 은 4096틱 = 360도, 2048 이 중앙.
TICK_TO_DEG = 360.0 / 4096.0
CENTER = 2048


def to_deg(tick: int) -> float:
    return (tick - CENTER) * TICK_TO_DEG


class Bus:
    def __init__(self) -> None:
        self.port = PortHandler(PORT)
        if not self.port.openPort():
            raise SystemExit(f"포트를 못 엽니다: {PORT} (녹화·평가가 돌고 있으면 먼저 끄세요)")
        if not self.port.setBaudRate(BAUD):
            raise SystemExit("보레이트 설정 실패")
        self.ph = PacketHandler(2.0)
        self.reader = GroupSyncRead(self.port, self.ph, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        for i in IDS:
            self.reader.addParam(i)

    def read_positions(self) -> tuple[dict[int, int], list[int], bool]:
        """(응답한 모터의 위치, 응답 없는 id 목록, 패킷 자체 성공 여부)."""
        comm = self.reader.txRxPacket()
        ok = comm == COMM_SUCCESS
        values, missing = {}, []
        for i in IDS:
            if self.reader.isAvailable(i, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                values[i] = self.reader.getData(i, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            else:
                missing.append(i)
        return values, missing, ok

    def hardware_errors(self) -> dict[int, int]:
        out = {}
        for i in IDS:
            v, comm, _ = self.ph.read1ByteTxRx(self.port, i, ADDR_HARDWARE_ERROR)
            out[i] = v if comm == COMM_SUCCESS else -1
        return out

    def write_goal(self, motor_id: int, tick: int) -> bool:
        # write*TxRx 는 (comm, error) 두 개만 돌려준다. read*TxRx 만 (value, comm, error) 다.
        comm, _ = self.ph.write4ByteTxRx(self.port, motor_id, ADDR_GOAL_POSITION, int(tick))
        return comm == COMM_SUCCESS

    def torque_on(self, motor_id: int) -> bool:
        v, comm, _ = self.ph.read1ByteTxRx(self.port, motor_id, ADDR_TORQUE_ENABLE)
        return comm == COMM_SUCCESS and v == 1

    def close(self) -> None:
        self.port.closePort()


def report(total: int, fail_packets: int, per_motor: Counter, by_pan: dict, elapsed: float) -> None:
    print("\n" + "=" * 62)
    print(f"읽기 {total}회 / {elapsed:.0f}초 ({total / max(elapsed, 1e-6):.1f} Hz)")
    if total == 0:
        return
    print(f"패킷 실패 {fail_packets}회 ({100 * fail_packets / total:.2f}%)")
    print("\n모터별 무응답:")
    if not per_motor:
        print("  없음 — 이 구간에서는 끊김이 재현되지 않았습니다")
    for i, n in per_motor.most_common():
        print(f"  id{i} {IDS[i]:<14} {n:>6}회  ({100 * n / total:.2f}%)")

    if by_pan:
        print("\npan 자세별 무응답 (케이블 장력이면 특정 구간에 몰린다):")
        print(f"  {'pan 구간':<14}{'읽기':>8}{'무응답':>8}{'비율':>8}")
        for lo in sorted(by_pan):
            n, bad = by_pan[lo]
            if n == 0:
                continue
            flag = "  ←" if bad and 100 * bad / n > 1.0 else ""
            print(f"  {lo:>+4}~{lo + 5:>+4}도{'':<4}{n:>8}{bad:>8}{100 * bad / n:>7.2f}%{flag}")
    print("=" * 62)


def watch(bus: Bus, seconds: float, hz: float) -> None:
    print(f"{seconds:.0f}초 동안 {hz:.0f}Hz 로 읽습니다. 팔은 움직이지 않습니다.")
    print("케이블 장력을 의심한다면, 도는 동안 손으로 케이블을 살살 당겨보세요.\n")
    run(bus, lambda: None, seconds, hz)


def sweep(bus: Bus, cycles: int, hz: float, lo_deg: float, hi_deg: float, speed: float) -> None:
    """pan 을 좌우로 천천히 쓸면서 읽는다. 자세별 실패율을 얻는 게 목적이다."""
    if not bus.torque_on(11):
        raise SystemExit("11번 토크가 꺼져 있습니다. 팔이 힘을 받는 상태에서만 쓸 수 있습니다.")
    vals, missing, _ = bus.read_positions()
    if 11 in missing:
        raise SystemExit("11번을 못 읽습니다 — 이미 끊긴 상태입니다. 케이블부터 확인하세요.")
    start = vals[11]
    print(f"pan 을 {lo_deg:+.0f}~{hi_deg:+.0f}도 사이에서 {cycles}회 왕복합니다 "
          f"({speed:.0f}도/초). 트레이 위 물건을 치우세요.")
    input("준비되면 엔터 > ")

    targets = []
    for _ in range(cycles):
        targets += [hi_deg, lo_deg]
    targets.append(to_deg(start))

    state = {"i": 0, "cur": to_deg(start), "t": time.perf_counter()}

    def step():
        now = time.perf_counter()
        dt, state["t"] = now - state["t"], now
        goal = targets[state["i"]]
        d = goal - state["cur"]
        move = min(abs(d), speed * dt)
        state["cur"] += move if d > 0 else -move
        if abs(goal - state["cur"]) < 0.5:
            state["i"] = min(state["i"] + 1, len(targets) - 1)
        bus.write_goal(11, CENTER + state["cur"] / TICK_TO_DEG)

    run(bus, step, seconds=None, hz=hz, done=lambda: state["i"] >= len(targets) - 1)


def run(bus: Bus, step, seconds: float | None, hz: float, done=None) -> None:
    total = fail_packets = 0
    per_motor: Counter = Counter()
    by_pan: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    first_fail = None
    t0 = time.perf_counter()
    period = 1.0 / hz
    try:
        while True:
            loop = time.perf_counter()
            el = loop - t0
            if seconds is not None and el >= seconds:
                break
            if done is not None and done():
                break

            values, missing, ok = bus.read_positions()
            total += 1
            if not ok:
                fail_packets += 1
            if 11 in values:
                b = int(to_deg(values[11]) // 5 * 5)
                by_pan[b][0] += 1
                if missing:
                    by_pan[b][1] += 1
            for i in missing:
                per_motor[i] += 1
            if missing and first_fail is None:
                pan = to_deg(values[11]) if 11 in values else float("nan")
                first_fail = (el, missing, pan)
                print(f"  ⚠ {el:6.1f}초  무응답 id={missing}  pan={pan:+.1f}도")
            elif missing and total % 30 == 0:
                pan = to_deg(values[11]) if 11 in values else float("nan")
                print(f"  ⚠ {el:6.1f}초  무응답 id={missing}  pan={pan:+.1f}도")

            step()
            time.sleep(max(period - (time.perf_counter() - loop), 0.0))
    except KeyboardInterrupt:
        print("\n중단됨")

    report(total, fail_packets, per_motor, by_pan, time.perf_counter() - t0)

    errs = bus.hardware_errors()
    bad = {i: v for i, v in errs.items() if v not in (0, -1)}
    unread = [i for i, v in errs.items() if v == -1]
    print("\n하드웨어 에러 레지스터(0x70):")
    print(f"  이상 없음: {[i for i, v in errs.items() if v == 0]}")
    if bad:
        print(f"  **에러 비트 있음**: {bad}  (1=입력전압, 4=과열, 8=엔코더, 16=과부하)")
    if unread:
        print(f"  읽기 실패: {unread}  ← 지금도 끊겨 있는 모터")
    if first_fail:
        print(f"\n첫 끊김: {first_fail[0]:.1f}초, id={first_fail[1]}, pan={first_fail[2]:+.1f}도")
    print("\n판정 기준")
    print("  무응답이 0 이면 이 구간에서는 재현 안 됨 — 더 오래, 또는 --sweep 으로 다시")
    print("  특정 pan 구간에 몰리면 → 케이블 장력 (그 자세에서 당겨진다)")
    print("  자세와 무관하게 흩어지면 → 커넥터 접촉 불량이나 전원 쪽")


def main() -> None:
    ap = argparse.ArgumentParser(description="모터 버스 간헐 끊김 계측")
    ap.add_argument("--seconds", type=float, default=60.0, help="감시 모드 지속 시간")
    ap.add_argument("--hz", type=float, default=30.0, help="읽기 주기 (제어 루프와 같은 30Hz 가 기본)")
    ap.add_argument("--sweep", action="store_true", help="pan 을 좌우로 쓸며 자세별로 잰다 (팔이 움직인다)")
    ap.add_argument("--cycles", type=int, default=3, help="--sweep 왕복 횟수")
    ap.add_argument("--lo", type=float, default=-25.0, help="--sweep pan 최소 각도")
    ap.add_argument("--hi", type=float, default=20.0, help="--sweep pan 최대 각도")
    ap.add_argument("--speed", type=float, default=12.0, help="--sweep 속도 (도/초)")
    args = ap.parse_args()

    bus = Bus()
    try:
        if args.sweep:
            sweep(bus, args.cycles, args.hz, args.lo, args.hi, args.speed)
        else:
            watch(bus, args.seconds, args.hz)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
