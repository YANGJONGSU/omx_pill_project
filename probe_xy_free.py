#!/usr/bin/env python3
"""암기인지 학습인지 가른다 — 알약이 없는 임의 좌표를 준다 (2026-08-13).

문제
----
probe_xy.py 는 학습에 쓴 188 에피소드에서 잰다. 학습 장면에서 잰 값은 성능이 아니다
(트러블슈팅 B-2: 원-핫 FiLM 이 학습 장면 1.33, 미학습 장면 1.03 이었다).
따로 떼어 둔 에피소드가 없으므로 다른 방법으로 확인해야 한다.

방법
----
**학습에서 한 번도 등장한 적 없는 좌표**를 목표로 준다. 격자로 만든 임의의 점이라
알약이 놓인 적도, 정답 궤적이 존재한 적도 없다.

    암기했다면   본 적 없는 좌표 → 대응하는 궤적이 없으므로 아무 데나 간다
    배웠다면     좌표가 요구하는 방향으로 팔이 간다

판정은 요구 각도와 예측 각도의 상관으로 한다. 이 시험은 "집기에 성공하는가"가
아니라 **"좌표를 공간 명령으로 이해하는가"** 를 본다.

    python probe_xy_free.py --run act_xy_224 --ckpt 150000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from probe_xy import predict_end  # noqa: E402
from validate_policy import load_policy  # noqa: E402

PROJECT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT / "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--repo-id", default="1unasy/pill_v3_xy")
    ap.add_argument("--scenes", type=int, default=12, help="장면 수")
    ap.add_argument("--grid", type=int, default=5, help="장면당 임의 좌표 수")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    C = json.loads((PROJECT / "xy_cache.json").read_text())
    cache = {int(k): v for k, v in C["episodes"].items()}
    pu = C["pan_fit"]

    ds = LeRobotDataset(a.repo_id)
    ckpt = TRAIN_DIR / a.run / "checkpoints" / a.ckpt / "pretrained_model"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트가 없습니다: {ckpt}")
    policy, cfg, pre, post = load_policy(ckpt, ds, a.device)

    # 학습에 등장한 좌표들 — 이것과 멀리 떨어진 점만 쓴다
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
            # 이 장면의 실제 알약들과 충분히 떨어진 점만 (알약 위 좌표는 제외)
            if min(float(np.hypot(*(p - q))) for q in scene_pts) < 0.12:
                continue
            pred = predict_end(policy, cfg, pre, post, sample, p.astype(np.float32))
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
    print(f"  알약과의 최소 거리 중앙값 {np.median(seps):.3f} (정규화 좌표)")
    print()
    print(f"  요구 각도 범위 {want.min():+.1f} ~ {want.max():+.1f}")
    print(f"  예측 각도 범위 {got.min():+.1f} ~ {got.max():+.1f}")
    print(f"  **상관 {corr:+.3f}**   기울기 {slope:+.2f}  (1.00 = 요구한 만큼 정확히 움직임)")
    print()
    if corr >= 0.8:
        print("  판정: 학습입니다. 본 적 없는 좌표에도 그 방향으로 갑니다.")
        print("        암기로는 이 결과가 나올 수 없습니다.")
    elif corr >= 0.5:
        print("  판정: 부분적으로 일반화했습니다. 좌표를 쓰되 화면에도 크게 의존합니다.")
    else:
        print("  판정: 암기 가능성이 큽니다. 본 적 없는 좌표에는 반응하지 않습니다.")
        print("        학습 장면에서 잰 자기목표 대비 값을 성능으로 해석하면 안 됩니다.")

    out = PROJECT / "goal_metric_xy_free.tsv"
    new = not out.exists()
    with out.open("a") as f:
        if new:
            f.write("run\tckpt\tcorr\tslope\tn\n")
        f.write(f"{a.run}\t{a.ckpt}\t{corr:+.3f}\t{slope:+.2f}\t{len(want)}\n")
    print(f"  기록: {out}")


if __name__ == "__main__":
    main()
