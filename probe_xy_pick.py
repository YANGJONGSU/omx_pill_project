#!/usr/bin/env python3
"""실전 질문에 직접 답한다 — 지정한 알약을 골라내는가 (2026-08-13).

앞선 지표들의 한계
------------------
  자기목표 대비 3.68   틀린 목표를 주면 오차가 커진다     → 목표를 쓰긴 한다
  응답 이득     20%    옮긴 만큼은 안 따라간다             → 30% 정도만 쓴다
  임의좌표 상관 0.48   본 적 없는 좌표에도 부분 반응       → 암기는 아니다

어느 것도 **"알약 세 개 중 지정한 것을 집는가"** 에 답하지 않는다. 실전에서는
좌표와 화면이 같은 곳을 가리키므로 30% 가중이어도 결과가 같을 수 있다. 문제는
화면만으로 구분이 안 되는 상황에서 좌표가 결정을 내려주느냐다.

방법
----
같은 장면에서 **각 알약의 좌표를 차례로** 목표로 준다. 예측한 마지막 자세의
shoulder_pan 을 u 로 역환산해, 어느 알약에 가장 가까운지 본다.

    지시한 알약이 가장 가까움  →  성공
    다른 알약이 더 가까움      →  실패 (다른 알약을 집는다)

이것이 실기 성공률의 오프라인 대리 지표다. 완전한 대체는 아니지만, 실기 없이
얻을 수 있는 것 중 가장 실전에 가깝다.

    python probe_xy_pick.py --run act_xy_224 --ckpt 150000
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
    ap.add_argument("--n", type=int, default=60, help="장면 수")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--offsets", default="60,100,140,180",
                    help="파지 몇 프레임 전에서 볼지. 여러 개면 시행 수가 배로 는다")
    ap.add_argument("--holdout", action="store_true",
                    help="marker_episodes.json 의 떼어 둔 장면에서만 잰다 (학습 미사용)")
    a = ap.parse_args()

    C = json.loads((PROJECT / "xy_cache.json").read_text())
    cache = {int(k): v for k, v in C["episodes"].items()}
    pu = C["pan_fit"]                      # pan = pu[0]*u + pu[1]
    inv = lambda pan: (pan - pu[1]) / pu[0]   # noqa: E731  pan → u

    ds = LeRobotDataset(a.repo_id)
    ckpt = TRAIN_DIR / a.run / "checkpoints" / a.ckpt / "pretrained_model"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트가 없습니다: {ckpt}")
    policy, cfg, pre, post = load_policy(ckpt, ds, a.device)

    eps = [e for e in sorted(cache) if cache[e]["wrong"]]
    if a.holdout:
        hold = set(json.loads((PROJECT / "marker_episodes.json").read_text())["holdout"])
        eps = [e for e in eps if e in hold]
        print(f"  떼어 둔 장면에서만 측정 — {len(eps)}개 (학습에 쓰지 않음)")
    eps = eps[:: max(1, len(eps) // a.n)][: a.n]

    offs = [int(x) for x in a.offsets.split(",")]
    ok, tot, margins, rows = 0, 0, [], []
    for e in eps:
      c = cache[e]
      pills = {"target": np.array([c["u"], c["v"]]),
               "other": np.array(c["wrong"])}
      # 두 알약의 u 가 너무 붙어 있으면 pan 으로 구분할 수 없다 — 제외
      if abs(pills["target"][0] - pills["other"][0]) < 0.06:
          continue
      # 접근 구간의 여러 시점에서 본다. 한 시점만 보면 표본이 28회뿐이라
      # 신뢰구간이 ±18% 로 벌어져 50% 와 구분되지 않는다.
      for off in offs:
        t = c["grasp"] - off
        if t < c["from"]:
            continue
        sample = ds[t]

        for name, goal in pills.items():
            pred = predict_end(policy, cfg, pre, post, sample, goal.astype(np.float32))
            u_hat = inv(float(pred[0]))
            d_cmd = abs(u_hat - goal[0])
            d_oth = abs(u_hat - pills["other" if name == "target" else "target"][0])
            hit = d_cmd < d_oth
            ok += hit; tot += 1
            margins.append(d_oth - d_cmd)
            rows.append(dict(ep=e, off=off, goal=name, u_hat=round(u_hat, 3),
                             u_cmd=round(float(goal[0]), 3), hit=bool(hit)))

    if tot == 0:
        raise SystemExit("측정할 장면이 없습니다.")
    rate = ok / tot * 100
    m = np.array(margins)

    print(f"\n지정 알약 선택 시험 — {a.run} / {a.ckpt}")
    print("=" * 66)
    print(f"  장면 {len(set(r['ep'] for r in rows))}개 × 목표 2개 = 시행 {tot}회")
    print(f"  (두 알약의 가로 간격이 좁아 pan 으로 구분 불가한 장면은 제외)")
    print()
    se = float(np.sqrt(rate / 100 * (1 - rate / 100) / tot) * 100)
    print(f"  **지정한 알약 쪽으로 간 비율 {rate:.0f}%**   ({ok}/{tot})  95% 구간 ±{1.96*se:.0f}%p")
    print(f"  여유 중앙값 {np.median(m):+.3f}  (양수면 지시한 쪽에 더 가까움)")
    print(f"  50% = 무작위 (둘 중 하나)")
    print()
    if rate >= 85:
        print("  판정: 지정한 알약을 고릅니다. 실기 검증으로 넘어갈 수 있습니다.")
    elif rate >= 65:
        print("  판정: 대체로 고르지만 실패가 섞입니다. 실기에서 오집이 나올 수 있습니다.")
    else:
        print("  판정: 지정한 알약을 고르지 못합니다. 좌표가 결정을 내리지 못하고 있습니다.")

    out = PROJECT / "goal_metric_xy_pick.tsv"
    new = not out.exists()
    with out.open("a") as f:
        if new:
            f.write("run\tckpt\tpick_pct\tmargin\tn\n")
        f.write(f"{a.run}\t{a.ckpt}\t{rate:.1f}\t{np.median(m):+.3f}\t{tot}\n")
    (PROJECT / "probe_xy_pick_rows.json").write_text(json.dumps(rows, indent=1))
    print(f"  기록: {out}")


if __name__ == "__main__":
    main()
