#!/usr/bin/env python3
"""정책의 **상수 편향**을 학습 데이터로 잰다 — 로봇 없이.

왜 만들었나 (2026-08-07): 초록 영점을 실기에서 화살표로 못 맞췄다. 파지 반사가
드물게 걸려 측정이 1~5건씩밖에 안 쌓였고, 그 값들이 서로 어긋났다
(좌우 -5 → -18mm, -6 → -11mm, +3 → -8mm). 표본이 부족해서 결론이 안 났다.

대신 학습 장면을 정책에 그대로 보여주고 **사람이 간 위치와 얼마나 다른지**를 재면
에피소드마다 표본이 나온다. 카메라 불량과 무관하고, 로봇을 안 돌려도 된다.

읽는 법:
    좌우 편향이 +면 정책이 사람보다 pan 을 크게 준다는 뜻이고,
    pan 이 크면 화면 왼쪽으로 가므로(LATERAL_DEG_PER_MM 이 음수) 오른쪽으로 당겨야 한다.
    출력의 "권장 좌우 보정"은 그 환산까지 끝낸 값이다.

    bash: python fit_bias.py --run act_v3_green40k --episodes 29 58
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from probe_policy import I_PAN, TOP, WRIST, depth_axis, predict_chunk
from validate_policy import (TRAIN_DIR, gripper_close_frame, gripper_threshold,
                             load_policy)

# run_policy.py 와 같은 환산 상수를 쓴다 (거기서 실측으로 뽑은 값)
LATERAL_DEG_PER_MM = -0.207          # pan 이 커지면 화면 왼쪽
DEPTH_DEG_PER_MM = 0.31 / 0.65       # 깊이축 도/mm
PAN_MID, DEPTH_MID = 5.2, 28.6       # run_policy.zone_of 와 같은 구역 경계


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="act_v3_green40k")
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--repo-id", default="1unasy/pill_v3_onehot")
    ap.add_argument("--episodes", nargs=2, type=int, metavar=("시작", "끝"),
                    default=[29, 58], help="쓸 에피소드 범위 (양끝 포함)")
    ap.add_argument("--extra", nargs="+", type=int, default=[],
                    help="추가 에피소드 (빨강의 회복 시연처럼 떨어져 있는 것)")
    ap.add_argument("--offsets", nargs="+", type=int, default=[-30, -15, -5, 0],
                    help="파지 시점 기준 몇 프레임 전을 볼지")
    ap.add_argument("--task", default=None, help="grasp_offset.json 키 (기본은 run 에서 색을 추정)")
    ap.add_argument("--apply-zones", action="store_true", help="구역별 값을 grasp_offset.json 에 저장")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.task is None:
        c = next((c for c in ('red', 'green', 'yellow') if c in args.run), 'red')
        args.task = f'pick {c} pill'

    print(f"데이터셋 로드: {args.repo_id}")
    ds = LeRobotDataset(args.repo_id)
    ep_index = np.asarray(ds.hf_dataset["episode_index"])
    all_actions = np.asarray(ds.hf_dataset["action"], dtype=np.float32)

    ckpt = TRAIN_DIR / args.run / "checkpoints" / args.ckpt / "pretrained_model"
    if not ckpt.is_dir():
        raise SystemExit(f"체크포인트 없음: {ckpt}")
    print(f"정책 로드: {ckpt}")
    policy, cfg, pre, post = load_policy(ckpt, ds, args.device)

    eps = list(range(args.episodes[0], args.episodes[1] + 1)) + list(args.extra)
    print(f"에피소드 {len(eps)}개, 파지 기준 {args.offsets} 프레임\n")

    rows = []
    for ep in eps:
        gidx = np.where(ep_index == ep)[0]
        if not len(gidx):
            continue
        gt = all_actions[gidx]
        thr, low = gripper_threshold(gt)
        close = gripper_close_frame(gt, thr, low)
        if close is None:
            continue
        for off in args.offsets:
            f = close + off
            if not (0 <= f < len(gidx)):
                continue
            gi = int(gidx[f])
            # 정책이 이 장면에서 내놓는 첫 액션 vs 사람이 실제로 준 액션
            pred = predict_chunk(policy, cfg, pre, post, ds[gi])[0]
            true = gt[f]
            rows.append((ep, off,
                         float(pred[I_PAN] - true[I_PAN]),
                         float(depth_axis(pred) - depth_axis(true))))
        print(f"\r  진행 {len(rows)}건 (ep{ep})", end="", flush=True)
    print()

    if not rows:
        raise SystemExit("측정된 게 없다")

    a = np.array([(r[2], r[3]) for r in rows])
    offs = np.array([r[1] for r in rows])

    print("\n" + "=" * 66)
    print(f"  {args.run} 의 상수 편향  (표본 {len(rows)}건)")
    print("=" * 66)
    print(f"{'시점':>10}{'좌우(pan) 도':>16}{'깊이축 도':>14}{'표본':>7}")
    for off in args.offsets:
        m = offs == off
        if not m.any():
            continue
        print(f"{f'파지{off:+d}':>10}{np.median(a[m, 0]):>+16.3f}{np.median(a[m, 1]):>+14.3f}{m.sum():>7}")

    # 파지 직전(-15~0)만 쓴다. 멀리서는 아직 접근 중이라 편향이 섞인다.
    near = offs >= -15
    pan_bias = float(np.median(a[near, 0]))
    dep_bias = float(np.median(a[near, 1]))
    lat_mm = pan_bias / LATERAL_DEG_PER_MM
    dep_mm = dep_bias / DEPTH_DEG_PER_MM

    print("-" * 66)
    print(f"  파지 직전(-15~0) 중앙값   좌우 {pan_bias:+.3f}도   깊이 {dep_bias:+.3f}도")
    print(f"  사분위 폭                 좌우 {np.subtract(*np.percentile(a[near,0],[75,25])):.3f}도"
          f"   깊이 {np.subtract(*np.percentile(a[near,1],[75,25])):.3f}도")
    print()
    print(f"  ▶ 권장 좌우 보정  {-lat_mm:+.1f} mm")
    print(f"  ▶ 권장 깊이 보정  {-dep_mm:+.1f} mm")
    print()
    print("  * 정책이 사람보다 치우친 만큼을 반대로 빼는 값이다.")

    # --- 구역별 편향 ---------------------------------------------------
    # 사용자 관찰(2026-08-07): "오른쪽에서 맞추고 왼쪽에 두면 차이가 너무 심함."
    # 상수 하나로는 못 잡는다. 직선 보정(correction.json)도 초록 깊이는 설명력이
    # 25% 뿐이라 위치에 따라 남는 오차가 크다. 구역을 나눠 따로 낸다.
    print("\n" + "=" * 66)
    print("  구역별 편향  (오른쪽/왼쪽 × 가까움/멂)")
    print("=" * 66)
    zrows = []
    for ep, off, dpan, ddep in rows:
        if off < -15:
            continue
        gidx = np.where(ep_index == ep)[0]
        gt = all_actions[gidx]
        thr, low = gripper_threshold(gt)
        close = gripper_close_frame(gt, thr, low)
        if close is None:
            continue
        f = close + off
        if not (0 <= f < len(gidx)):
            continue
        act = gt[f]
        left = float(act[I_PAN]) >= PAN_MID
        far = float(depth_axis(act)) >= DEPTH_MID
        zrows.append(((2 if left else 0) + (1 if far else 0), dpan, ddep))

    ZN = ["오른쪽·가까움", "오른쪽·멂", "왼쪽·가까움", "왼쪽·멂"]
    print(f"{'구역':<16}{'좌우 mm':>10}{'깊이 mm':>10}{'표본':>7}")
    zone_out = {}
    for z in range(4):
        v = [(p, d) for zz, p, d in zrows if zz == z]
        if not v:
            print(f"  {ZN[z]:<14}{'표본 없음':>10}")
            continue
        lp = -float(np.median([x[0] for x in v])) / LATERAL_DEG_PER_MM
        ld = -float(np.median([x[1] for x in v])) / DEPTH_DEG_PER_MM
        zone_out[z] = (round(ld, 1), round(lp, 1))
        print(f"  {ZN[z]:<14}{lp:>+10.1f}{ld:>+10.1f}{len(v):>7}")
    if zone_out:
        sp = max(v[1] for v in zone_out.values()) - min(v[1] for v in zone_out.values())
        sd = max(v[0] for v in zone_out.values()) - min(v[0] for v in zone_out.values())
        print(f"\n  구역 간 편차   좌우 {sp:.1f}mm   깊이 {sd:.1f}mm")
        print("  → 이 폭만큼은 상수 하나로 절대 못 잡는다. --zones 로 구역별로 둬야 한다.")
        print("\n  grasp_offset.json 에 넣으려면:  --apply-zones")
        if args.apply_zones:
            import json as _j
            from datetime import datetime as _dt
            p = Path("grasp_offset.json")
            d = _j.loads(p.read_text())
            key = f"{args.run}|{args.task}"
            e = d.setdefault(key, {"depth_mm": 0.0, "lateral_mm": 0.0, "z_mm": 0.0})
            e["zones"] = {str(z): {"depth_mm": v[0], "lateral_mm": v[1], "z_mm": 0.0,
                                   "saved": _dt.now().isoformat(timespec="seconds")}
                          for z, v in zone_out.items()}
            p.write_text(_j.dumps(d, indent=1, ensure_ascii=False))
            print(f"  ✅ {key} 의 구역 {sorted(zone_out)} 저장됨")

    # --- 이 정책이 wrist 를 실제로 보는가 -------------------------------
    # 사용자 관찰(2026-08-07): "wrist 로 초록을 보면서 가게 학습했는데 그렇게 가는
    # 느낌이 안 든다." 카메라를 하나씩 가려서 예측이 얼마나 망가지는지로 확인한다.
    # 가려도 안 변하면 그 카메라를 안 보고 있다는 뜻이다.
    print("\n" + "=" * 66)
    print("  카메라를 가리면 예측이 얼마나 망가지나")
    print("=" * 66)
    probes = []
    for ep in eps[:12]:                      # 12개면 충분하고 빠르다
        gidx = np.where(ep_index == ep)[0]
        if not len(gidx):
            continue
        gt = all_actions[gidx]
        thr, low = gripper_threshold(gt)
        close = gripper_close_frame(gt, thr, low)
        if close is None:
            continue
        probes += [int(gidx[close + o]) for o in (-30, -15, 0) if 0 <= close + o < len(gidx)]

    base_err, res = None, {}
    for label, blackout in (("정상", None), ("top 가림", TOP), ("wrist 가림", WRIST)):
        errs = []
        for gi in probes:
            pred = predict_chunk(policy, cfg, pre, post, ds[gi], blackout=blackout)[0]
            errs.append(np.abs(pred - all_actions[gi]).mean())
        e = float(np.mean(errs))
        res[label] = e
        if blackout is None:
            base_err = e
        print(f"  {label:<12} 오차 {e:6.3f}" +
              (f"   정상 대비 {e/base_err:4.2f}배" if blackout is not None else ""))
    print()
    w = res["wrist 가림"] / base_err
    if w < 1.3:
        print(f"  ▶ wrist 를 거의 안 본다 ({w:.2f}배). 가려도 예측이 안 바뀐다.")
        print("    → 손끝 정렬을 화면으로 안 잡고 관절값만 재생하고 있을 수 있다.")
    else:
        print(f"  ▶ wrist 를 보고 있다 ({w:.2f}배). 가리면 예측이 그만큼 망가진다.")


if __name__ == "__main__":
    main()
