#!/usr/bin/env python3
"""좌표를 **상태에** 넣은 정책이 좌표를 따르는지 잰다 (2026-08-20).

probe_xy_free.py 와 판정 방법이 같다 — 학습에 한 번도 나온 적 없는 임의 좌표를
주고, 요구 방향과 예측 방향의 **상관**을 본다. 암기면 상관이 안 나온다.

다른 점은 좌표를 넣는 자리뿐이다.

    act_xy_224 (ACT)     observation.environment_state = (u, v)
    smolvla_v10          observation.state 의 6번 칸부터 (u,v)×13 반복

SmolVLA 는 environment_state 를 읽지 않으므로(OBS_STATE 와 이미지만 본다)
좌표를 상태에 접어 넣었다. 판정도 같은 자리에 써 넣어야 한다.

**지시문을 바꾸는 probe_vla.py 로는 이 정책을 못 잰다.** v10 의 목표는 언어가
아니라 상태에 있으므로 문장을 바꿔도 예측이 안 변한다 (그게 정상이다).

    python probe_uvstate.py --run smolvla_v10 --ckpt 002500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT / "train"
sys.path.insert(0, str(PROJECT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from validate_policy import load_policy  # noqa: E402

N_JOINT = 6


def predict_end(policy, cfg, pre, post, sample, uv, repeat: int):
    """좌표를 상태에 써 넣고 청크 끝의 pan 각도를 얻는다."""
    obs = {}
    for k, v in sample.items():
        if k == "task":
            continue
        obs[k] = v.clone() if torch.is_tensor(v) else v
    obs["task"] = sample.get("task", "")

    st = obs["observation.state"]
    st = st.clone() if torch.is_tensor(st) else torch.as_tensor(np.asarray(st))
    wide = torch.as_tensor(np.tile(np.asarray(uv, dtype=np.float32), repeat))
    st[..., N_JOINT:N_JOINT + 2 * repeat] = wide.to(st.dtype)
    obs["observation.state"] = st

    with torch.no_grad():
        ch = post(policy.predict_action_chunk(pre(obs))).float().cpu().numpy()
    policy.reset()
    return np.asarray(ch).reshape(-1, ch.shape[-1])[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--repo-id", default="1unasy/pill_v3_uvstate")
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=13)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    C = json.loads((PROJECT / "xy_cache.json").read_text())
    cache = {int(k): v for k, v in C["episodes"].items()}
    pu = C["pan_fit"]      # 좌표 u → 요구 pan 각도 (실측 회귀식)

    ds = LeRobotDataset(a.repo_id)
    ckpt = TRAIN_DIR / a.run / "checkpoints" / a.ckpt / "pretrained_model"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트가 없습니다: {ckpt}")
    policy, cfg, pre, post = load_policy(ckpt, ds, a.device)

    used = np.array([[v["u"], v["v"]] for v in cache.values()])
    lo, hi = used.min(0), used.max(0)

    rng = np.random.default_rng(0)
    eps = sorted(cache)[:: max(1, len(cache) // a.scenes)][: a.scenes]

    want, got, seps = [], [], []
    for e in eps:
        c = cache[e]
        t = c["grasp"] - 100
        if t < c["from"]:
            continue
        sample = ds[t]
        scene_pts = [np.array([c["u"], c["v"]])]
        if c["wrong"]:
            scene_pts.append(np.array(c["wrong"]))

        n = 0
        for _ in range(a.grid * 6):
            if n >= a.grid:
                break
            p = rng.uniform(lo, hi)
            if min(float(np.hypot(*(p - q))) for q in scene_pts) < 0.12:
                continue
            pred = predict_end(policy, cfg, pre, post, sample, p.astype(np.float32), a.repeat)
            want.append(float(np.polyval(pu, p[0])))
            got.append(float(pred[0]))
            seps.append(min(float(np.hypot(*(p - q))) for q in scene_pts))
            n += 1

    want, got = np.array(want), np.array(got)
    if len(want) < 10:
        raise SystemExit("표본이 부족합니다.")

    corr = float(np.corrcoef(want, got)[0, 1])
    slope = float(np.polyfit(want, got, 1)[0])

    print(f"\n임의 좌표 추종 시험 — {a.run} / {a.ckpt}")
    print("=" * 66)
    print(f"  장면 {len(eps)}개 × 임의 좌표 {a.grid}개 = 표본 {len(want)}개")
    print(f"  알약과의 최소 거리 중앙값 {np.median(seps):.3f}")
    print()
    print(f"  요구 각도 범위 {want.min():+.1f} ~ {want.max():+.1f}")
    print(f"  예측 각도 범위 {got.min():+.1f} ~ {got.max():+.1f}")
    print(f"  **상관 {corr:+.3f}**   기울기 {slope:+.2f}")
    print()
    # **기준은 +0.484 다.** act_xy_224 를 이 시험(자유 좌표)으로 쟀을 때의 값이고
    # 그 정책이 실기 3색 연속 조제에 성공했다. goal_metric_xy.tsv 의 +0.981 은
    # **학습 장면에서** 잰 다른 시험이므로 여기에 갖다 대면 안 된다 — 그 착오로
    # 2500스텝의 +0.252 를 실패로 판정할 뻔했다 (실제로는 1/60 스텝에서 절반).
    print(f"  기준: act_xy_224 @150K = +0.484  (같은 시험, 실기 3색 연속 성공)")
    print()
    if corr >= 0.45:
        print("  판정: 되는 정책과 같은 수준입니다.")
    elif corr >= 0.25:
        print("  판정: 좌표를 쓰기 시작했습니다. 더 돌릴 값어치가 있습니다.")
    else:
        print("  판정: 아직 좌표를 안 씁니다.")

    out = PROJECT / "goal_metric_uvstate.tsv"
    new = not out.exists()
    with open(out, "a", encoding="utf-8") as f:
        if new:
            f.write("run\tckpt\tcorr\tslope\tn\n")
        f.write(f"{a.run}\t{a.ckpt}\t{corr:.3f}\t{slope:.2f}\t{len(want)}\n")
    print(f"\n  기록 -> {out.name}")


if __name__ == "__main__":
    main()
