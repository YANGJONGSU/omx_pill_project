#!/usr/bin/env python
"""정책의 파지점 오차를 **위치의 함수로** 적합해 자동 보정식을 만든다.

왜 상수 오프셋으로는 안 되는가
------------------------------
2026-08-06 계측: 좌우 오차가 pan 에 **비례**한다 (상관 +0.78, 기울기 +0.060 도/도).
상수 편향이 아니라 **배율 오차**다. 한 위치에서 맞추면 다른 위치가 어긋난다 —
"위치마다 영점이 다르다"는 관찰이 정확히 이것이다.

  좌우 오차 = a + b·pan          (위치로 61% 설명)
  깊이 오차 = c + d·깊이축        (위치로 23% 설명, 상수 성분이 대부분)

그래서 보정도 위치의 함수여야 한다. 정책이 낸 예측 자세를 보고 그 자리에 맞는
보정량을 계산해서 빼면 된다.

한계: 이 적합은 **학습 데이터 위에서** 잰 것이라 실기의 절대 편향(카메라-작업대
정합 등)은 안 들어 있다. 그건 여전히 상수 오프셋(`--grasp-offset-*`)으로 잡는다.
여기서 잡는 것은 **위치에 따라 달라지는 성분**이다. 둘은 더해서 쓴다.

사용법:
  python ~/omx_pill_project/fit_correction.py --run act_v3_red29 --episodes 0 28
  python ~/omx_pill_project/fit_correction.py --run act_v3_green --episodes 29 58
  → train/<run>/correction.json 저장, run_policy.py 가 자동으로 읽는다
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from validate_policy import (  # noqa: E402
    TRAIN_DIR,
    gripper_close_frame,
    gripper_threshold,
    load_policy,
)

I_PAN, I_LIFT, I_ELBOW = 0, 1, 2
DEG_PER_MM_LAT = 0.207
DEG_PER_MM_DEPTH = 0.31 / 0.65


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="act_v3_red29")
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--repo-id", default="1unasy/pill_v3")
    ap.add_argument("--episodes", nargs=2, type=int, required=True, metavar=("FROM", "TO"),
                    help="이 정책이 학습한 에피소드 범위")
    ap.add_argument("--extra", nargs="*", type=int, default=[], help="추가 에피소드 번호")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id)
    ep = np.asarray(ds.hf_dataset["episode_index"])
    A = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    ck = TRAIN_DIR / args.run / "checkpoints" / args.ckpt / "pretrained_model"
    if not ck.is_dir():
        raise SystemExit(f"체크포인트 없음: {ck}")
    policy, cfg, pre, post = load_policy(ck, ds, args.device)
    H = int(cfg.chunk_size)

    @torch.inference_mode()
    def predict_grasp(gi):
        s = ds[int(gi)]
        obs = {k: (s[k].clone() if torch.is_tensor(s[k]) else s[k]) for k in cfg.input_features}
        obs["task"] = s.get("task", "")
        policy.reset()
        c = post(policy.predict_action_chunk(pre(obs))).float().cpu().numpy()
        return (c[0] if c.ndim == 3 else c)[H - 1]

    eps = list(range(args.episodes[0], args.episodes[1] + 1)) + list(args.extra)
    pan, dep, e_pan, e_dep = [], [], [], []
    for e in eps:
        g = np.where(ep == e)[0]
        if len(g) == 0:
            continue
        gt = A[g]
        thr, low = gripper_threshold(gt)
        c = gripper_close_frame(gt, thr, low)
        if c is None or c - H < 0:
            continue
        p = predict_grasp(g[c - H])
        t = gt[c]
        # **예측값**을 설명변수로 쓴다 — 실기에서는 정답을 모르고 예측만 있기 때문이다
        pan.append(p[I_PAN])
        dep.append(p[I_ELBOW] - p[I_LIFT])
        e_pan.append(p[I_PAN] - t[I_PAN])
        e_dep.append((p[I_ELBOW] - p[I_LIFT]) - (t[I_ELBOW] - t[I_LIFT]))

    pan, dep, e_pan, e_dep = map(np.array, (pan, dep, e_pan, e_dep))
    if len(pan) < 8:
        raise SystemExit(f"표본이 너무 적습니다 ({len(pan)}개)")

    print(f"{args.run} — 에피소드 {len(pan)}개로 적합\n")
    out = {"run": args.run, "ckpt": args.ckpt, "n": int(len(pan))}
    for name, x, err, mmdeg, key in [
        ("좌우(pan)", pan, e_pan, DEG_PER_MM_LAT, "pan"),
        ("깊이축", dep, e_dep, DEG_PER_MM_DEPTH, "depth"),
    ]:
        b1, b0 = np.polyfit(x, err, 1)
        res = err - (b0 + b1 * x)
        r2 = 1 - res.var() / err.var() if err.var() > 0 else 0.0
        print(f"  {name}")
        print(f"    보정 전  평균 {err.mean():+.2f}도  표준편차 {err.std():.2f}  "
              f"(±{err.std()/mmdeg:.1f}mm)")
        print(f"    적합     오차 = {b0:+.3f} {b1:+.3f}·위치     설명력 {r2*100:.0f}%")
        print(f"    보정 후  표준편차 {res.std():.2f}  (±{res.std()/mmdeg:.1f}mm)\n")
        out[key] = {"b0": float(b0), "b1": float(b1), "r2": float(r2),
                    "std_before": float(err.std()), "std_after": float(res.std())}

    path = TRAIN_DIR / args.run / "correction.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"저장: {path}")
    print("\nrun_policy.py 가 이 파일을 자동으로 읽어 적용합니다 (--no-auto-correct 로 끔).")
    print("주의: 이건 **위치에 따라 달라지는 성분**만 잡는다. 실기의 절대 편향은")
    print("      여전히 --grasp-offset-* 로 잡고, 둘은 더해서 쓴다.")


if __name__ == "__main__":
    main()
