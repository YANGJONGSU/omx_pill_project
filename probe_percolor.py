#!/usr/bin/env python3
"""색별 정책의 자기색 대비를 잰다 (2026-08-10).

색별 정책에는 목표 입력이 없으므로 `probe_policy.py --tests 5`(원-핫을 바꿔보는 방식)를
쓸 수 없다. 대신 **다른 색 에피소드에서 얼마나 못 맞추는가**로 잰다.

    자기색 대비 = (다른 색 에피소드에서의 오차) ÷ (자기 색 에피소드에서의 오차)

    1.00  아무 색이나 같은 궤적을 낸다 = 색을 구분하지 않는다
    3.26  act_v3_green40k 의 실측값 (40K, 96개 데이터)
    5.45  act_v3_red29 의 실측값

파지 시점 근처에서 open-loop 로 청크를 예측해 정답 행동과 비교한다.
색별 정책은 자기 색 데이터만 보고 학습했으므로, 다른 색 장면에서는 알약을 못 찾아
오차가 커져야 정상이다.

    python probe_percolor.py --run act_green_224 --ckpt 040000 --color green
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

from validate_policy import load_policy  # noqa: E402

PROJECT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT / "train"
COLORS = ["red", "green", "yellow"]


def episode_color_map(ds):
    """에피소드 번호 → 색."""
    out = {}
    for i in range(ds.meta.total_episodes):
        task = ds.meta.episodes["tasks"][i]
        task = task.as_py() if hasattr(task, "as_py") else task
        task = task[0] if isinstance(task, (list, tuple)) else task
        for c in COLORS:
            if c in str(task):
                out[i] = c
                break
    return out


def episode_error(policy, cfg, pre, post, ds, ep, n_steps, device, frac=0.7):
    """에피소드의 파지 근처에서 청크를 예측해 정답 행동과의 평균절대오차를 낸다.

    배치 구성은 probe_policy.predict_chunk 와 같은 방식을 쓴다 — 정책의
    input_features 에 있는 키만 골라 넣고 predict_action_chunk 를 호출한다.
    """
    a = int(ds.meta.episodes["dataset_from_index"][ep])
    b = int(ds.meta.episodes["dataset_to_index"][ep])
    idx = max(a, a + int((b - a) * frac) - n_steps)
    sample = ds[idx]

    obs = {}
    for k in cfg.input_features:
        v = sample[k]
        obs[k] = v.clone() if torch.is_tensor(v) else v
    obs["task"] = sample.get("task", "")

    policy.reset()
    with torch.no_grad():
        chunk = post(policy.predict_action_chunk(pre(obs))).float().cpu().numpy()
    pred = chunk[0] if chunk.ndim == 3 else chunk

    gt = np.array([ds[idx + j]["action"].numpy() for j in range(n_steps) if idx + j < b])
    n = min(len(gt), len(pred))
    return float(np.abs(pred[:n] - gt[:n]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--color", required=True, choices=COLORS)
    ap.add_argument("--repo-id", default="1unasy/pill_v3")
    ap.add_argument("--per-color", type=int, default=12, help="색당 표본 에피소드 수")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ds = LeRobotDataset(a.repo_id)
    ckpt = TRAIN_DIR / a.run / "checkpoints" / a.ckpt / "pretrained_model"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트가 없습니다: {ckpt}")
    policy, cfg, pre, post = load_policy(ckpt, ds, a.device)
    n_steps = cfg.n_action_steps

    cmap = episode_color_map(ds)
    to_i = list(ds.meta.episodes["dataset_to_index"])
    fr_i = list(ds.meta.episodes["dataset_from_index"])
    lengths = [int(t) - int(f) for t, f in zip(to_i, fr_i)]

    print(f"\n색별 정책 자기색 대비 — {a.run} / {a.ckpt}  (자기 색: {a.color})")
    print("=" * 66)
    errs = {}
    for c in COLORS:
        # 정지 시연(449프레임)은 빼고 집기만 쓴다
        eps = [e for e, cc in cmap.items() if cc == c and lengths[e] != 449]
        eps = eps[:: max(1, len(eps) // a.per_color)][: a.per_color]
        vals = [episode_error(policy, cfg, pre, post, ds, e, n_steps, a.device) for e in eps]
        errs[c] = float(np.mean(vals))
        mark = " ← 자기 색" if c == a.color else ""
        print(f"  {c:7s} 에피소드 {len(eps):2d}개   평균 오차 {errs[c]:.3f}{mark}")

    others = [errs[c] for c in COLORS if c != a.color]
    ratio = float(np.mean(others)) / errs[a.color]
    print("-" * 66)
    print(f"  **자기색 대비 {ratio:.2f}배**   (다른 색 평균 {np.mean(others):.3f} ÷ 자기 색 {errs[a.color]:.3f})")
    print()
    print("  참고: act_v3_green40k 3.26 / act_v3_red29 5.45 / 원-핫 1.00")
    if ratio >= 3.0:
        print("  판정: 색을 확실히 구분합니다.")
    elif ratio >= 2.0:
        print("  판정: 구분하지만 기존 정책보다 약합니다.")
    else:
        print("  판정: 색 구분이 약합니다. 학습이 덜 됐거나 문제가 있습니다.")

    out = PROJECT / "percolor_metric.json"
    hist = json.loads(out.read_text()) if out.exists() else {}
    hist[f"{a.run}@{a.ckpt}"] = {"color": a.color, "errors": errs, "ratio": ratio}
    out.write_text(json.dumps(hist, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
