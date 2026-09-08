#!/usr/bin/env python3
"""직전 실행이 **지시한 색을 집었는지** 본다.

  python check_run.py            # 가장 최근 기록
  python check_run.py --n 3      # 최근 3개

파지 순간의 shoulder_pan 을 트레이 위 알약 위치와 견줘 판정한다. 색 이름은
grasp_shots 에 저장된 사진과 pose.json 으로 확인하는 것이 가장 확실하다.
"""
import argparse, glob, os
import numpy as np
import pandas as pd

HELD, PAN = 49.85, (-28.0, -21.0)


def summarize(f: str) -> dict:
    d = pd.read_csv(f)
    g = d["cmd_gripper"].values
    ng = d["now_gripper"].values
    p = d["now_shoulder_pan"].values
    tries = ok = 0
    gmin = 999.0
    grabs = []
    for i in range(1, len(g)):
        if g[i - 1] >= 52 and g[i] < 52:
            tries += 1
            grabs.append(float(p[i]))          # 집으려 한 순간의 pan
        if g[i] < 52:
            gmin = min(gmin, ng[i])
        if g[i - 1] < 52 and g[i] >= 52:
            if gmin >= HELD and PAN[0] <= p[i] <= PAN[1]:
                ok += 1
            gmin = 999.0
    return {"파일": os.path.basename(f), "초": len(d) / 30, "시도": tries,
            "성공": ok, "파지pan": grabs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    fs = sorted(glob.glob("report/trace_*.csv"), key=os.path.getmtime)[-a.n:]
    for f in fs:
        r = summarize(f)
        print(f"  {r['파일']:22s} {r['초']:5.0f}초  시도 {r['시도']:2d}  성공 {r['성공']:2d}")
        if r["파지pan"]:
            print(f"      집으려 한 위치 (pan): "
                  + ", ".join(f"{v:+.1f}" for v in r["파지pan"]))
    # 파지 사진이 있으면 같이 알려 준다 — 어느 색을 집었는지는 사진이 가장 확실하다
    shots = sorted(glob.glob("grasp_shots/*/*_top.png"), key=os.path.getmtime)[-3:]
    if shots:
        print("\n  최근 파지 사진 (어느 색을 집었는지 확인용)")
        for s in shots:
            print(f"    {s}")


if __name__ == "__main__":
    main()
