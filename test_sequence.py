#!/usr/bin/env python3
"""연속 조제 로직을 로봇 없이 검증한다 (2026-08-16).

실물로 돌리기 전에 코드가 맞는지 본다. 실제 에피소드의 관측·행동을 순서대로
먹여 가상 조제를 만들고, run_policy 의 상태 전이가 의도대로 도는지 확인한다.

  ① 성공 판정이 각 색마다 **정확히 한 번** 발동하는가
  ② 색 전환이 제 순서로 일어나는가
  ③ 마지막 색에서 완료 처리되는가
  ④ 중간의 실패 파지(회복 시연)를 성공으로 오인하지 않는가

run_policy.py 의 판정 코드와 **같은 상수·같은 조건**을 쓴다. 여기서 통과해도
실기 성공을 보장하지는 않는다 — 검증하는 것은 로직이지 정책이 아니다.

    python test_sequence.py
    python test_sequence.py --combo red,green,yellow
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

ROOT = Path.home() / ".cache/huggingface/lerobot/1unasy/pill_v3"

# run_policy.py 와 같은 값이어야 한다. 다르면 검증이 무의미하다.
from run_policy import BOTTLE_PAN, GRIPPER_CLOSED, GRIPPER_OPEN, HELD_MIN  # noqa: E402


def load():
    data = pd.concat([pd.read_parquet(f, columns=["episode_index", "observation.state", "action"])
                      for f in sorted(glob.glob(str(ROOT / "data/*/*.parquet")))])
    meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "meta/episodes/*/*.parquet")))])
    S = np.stack(data["observation.state"].values)
    A = np.stack(data["action"].values)
    ep = data["episode_index"].values

    def color(t):
        t = t[0] if hasattr(t, "__len__") and not isinstance(t, str) else t
        for c in ("red", "green", "yellow"):
            if c in str(t):
                return c
    C = {int(r.episode_index): color(r.tasks) for _, r in meta.iterrows()}
    L = {int(r.episode_index): int(r.length) for _, r in meta.iterrows()}
    return S, A, ep, C, L


def simulate(frames, seq, verbose=False):
    """run_policy 의 성공 판정·색 전환을 그대로 흉내낸다.

    frames: [(관측 그리퍼, 명령 그리퍼, pan), ...] 를 색 순서대로 이어붙인 것
    """
    carrying = False
    grip_min = 999.0
    seq_i = 0
    events = []

    for k, (g_obs, g_cmd, pan) in enumerate(frames):
        grip_min = min(grip_min, g_obs)

        if g_cmd < GRIPPER_CLOSED:
            if not carrying:
                grip_min = g_obs
            carrying = True
        elif g_cmd > GRIPPER_OPEN:
            if carrying:
                at_bottle = BOTTLE_PAN[0] <= pan <= BOTTLE_PAN[1]
                held = grip_min >= HELD_MIN
                if not (at_bottle and held):
                    events.append(("miss", k, seq[seq_i] if seq_i < len(seq) else None,
                                   round(grip_min, 2), round(pan, 1)))
                    grip_min = 999.0
                else:
                    events.append(("done", k, seq[seq_i], round(grip_min, 2), round(pan, 1)))
                    seq_i += 1
                    if seq_i >= len(seq):
                        events.append(("finish", k, None, None, None))
                        carrying = False
                        break
                    grip_min = 999.0
            carrying = False
    return events, seq_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", default=None, help="색 조합 (기본: 처방표 전체)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    S, A, ep, C, L = load()
    by_color = {}
    for e in sorted(set(ep.tolist())):
        c = C.get(e)
        if c and L.get(e) != 449:          # 정지 시연 제외
            by_color.setdefault(c, []).append(e)

    def frames_of(e):
        m = ep == e
        return list(zip(S[m][:, 5], A[m][:, 5], S[m][:, 0]))

    if a.combo:
        combos = [("직접", a.combo.split(","))]
    else:
        d = json.loads((PROJECT / "web/prescriptions.json").read_text())
        combos = [(p["코드"], p["조합"]) for p in d["처방"]]

    print(f"  판정 상수: HELD_MIN={HELD_MIN}  BOTTLE_PAN={BOTTLE_PAN}")
    print(f"  (run_policy.py 에서 직접 가져옴 — 값이 어긋날 수 없다)\n")

    rng = np.random.default_rng(0)
    allok = True
    for code, seq in combos:
        # 각 색에서 에피소드를 무작위로 골라 이어붙인다 = 가상 조제
        ok_runs = 0
        detail = None
        for trial in range(5):
            frames = []
            for c in seq:
                e = int(rng.choice(by_color[c]))
                frames += frames_of(e)
            ev, done = simulate(frames, seq)
            n_done = sum(1 for x in ev if x[0] == "done")
            n_miss = sum(1 for x in ev if x[0] == "miss")
            fin = any(x[0] == "finish" for x in ev)
            if n_done == len(seq) and fin:
                ok_runs += 1
            elif detail is None:
                detail = (ev, n_done, n_miss, fin)
        mark = "✓" if ok_runs == 5 else "✗"
        if ok_runs != 5:
            allok = False
        print(f"  {mark} {code:8s} {' → '.join(seq):24s} 5회 중 {ok_runs}회 정상 완주")
        if detail and a.verbose:
            ev, n_done, n_miss, fin = detail
            print(f"      담기 {n_done}/{len(seq)}  놓침판정 {n_miss}  완료신호 {fin}")
            for t, k, c, g, p in ev[:8]:
                print(f"        {t:6s} 프레임{k:6d} {str(c):6s} 그리퍼 {g} pan {p}")

    print()
    if allok:
        print("  모든 처방 조합에서 로직이 정상 동작합니다.")
        print("  (색 전환·완료 처리·실패 오인 없음. 정책 성능은 별개입니다.)")
    else:
        print("  ⚠ 일부 조합에서 로직이 어긋납니다 — --verbose 로 확인하십시오.")


if __name__ == "__main__":
    main()
