#!/usr/bin/env python3
"""좌표 조건화 정책이 목표를 따르는지 잰다 (2026-08-12).

원-핫용 `probe_policy.py --tests 5` 는 목표를 "다른 색 원-핫"으로 바꿔 재지만,
좌표 목표에는 그 방식을 쓸 수 없다. 대신 **같은 장면의 다른 알약 좌표**를 준다.

두 가지를 잰다.

  ① 자기목표 대비 = (틀린 좌표를 줬을 때 오차) ÷ (올바른 좌표를 줬을 때 오차)
       원-핫 실험과 같은 척도라 직접 비교할 수 있다.
       1.02 = 기준선(목표 무시)   1.22 = 원-핫+FiLM 100K   3.0 이상 = 성공

  ② 유도 성공률 — 틀린 좌표를 줬을 때 **실제로 그쪽으로 가는가**
       라벨 188개로 u→shoulder_pan 회귀를 적합해(상관 -0.976) 목표 좌표가 요구하는
       pan 값을 계산하고, 예측이 그 값에 얼마나 가까운지 본다. ①이 "안 틀렸다"를
       본다면 ②는 "제대로 갔다"를 본다.

    python probe_xy.py --run act_xy_224 --ckpt 100000
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

from make_xy_labels import blobs_from, color_mask, frame_bgr  # noqa: E402
from validate_policy import load_policy  # noqa: E402

PROJECT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT / "train"
GOAL = "observation.environment_state"
COLORS = ["red", "green", "yellow"]


def grasp_index(ds, a, b, step=3):
    """그리퍼가 닫히는 프레임의 전역 인덱스와 그때의 정답 행동."""
    act = np.array([ds[i]["action"].numpy() for i in range(a, b, step)])
    g = act[:, 5]
    thr = (g.max() + g.min()) / 2
    idx = np.where(g < thr)[0]
    if len(idx) == 0:
        return None, None
    return a + int(idx[0]) * step, act[int(idx[0])]


def predict_end(policy, cfg, pre, post, sample, goal):
    """관측 하나 + 목표 좌표 → 청크 마지막 행동."""
    obs = {}
    for k in cfg.input_features:
        v = sample[k]
        obs[k] = v.clone() if torch.is_tensor(v) else v
    obs["task"] = sample.get("task", "")
    obs[GOAL] = torch.as_tensor(np.asarray(goal, dtype=np.float32))
    policy.reset()
    with torch.no_grad():
        ch = post(policy.predict_action_chunk(pre(obs))).float().cpu().numpy()
    ch = ch[0] if ch.ndim == 3 else ch
    return ch[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--repo-id", default="1unasy/pill_v3_xy")
    ap.add_argument("--n", type=int, default=40, help="검사할 에피소드 수")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--horizon-back", type=int, default=100,
                    help="파지 몇 프레임 전에서 예측할지 (chunk_size)")
    a = ap.parse_args()

    cache_f = PROJECT / "xy_cache.json"
    if not cache_f.exists():
        raise SystemExit("xy_cache.json 이 없습니다. 먼저 `python make_xy_cache.py` 를 실행하세요.")
    C = json.loads(cache_f.read_text())
    cache = {int(k): v for k, v in C["episodes"].items()}
    pu, pv = C["pan_fit"], C["lift_fit"]

    ds = LeRobotDataset(a.repo_id)
    ckpt = TRAIN_DIR / a.run / "checkpoints" / a.ckpt / "pretrained_model"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트가 없습니다: {ckpt}")
    policy, cfg, pre, post = load_policy(ckpt, ds, a.device)
    print(f"  회귀(캐시): pan = {pu[0]:+.1f}·u {pu[1]:+.1f}   lift = {pv[0]:+.1f}·v {pv[1]:+.1f}")

    eps_all = sorted(k for k, v in cache.items() if v["wrong"])

    # ── 본 측정 ───────────────────────────────────────────────────────────
    eps = eps_all[:: max(1, len(eps_all) // a.n)][: a.n]
    rows = []
    for e in eps:
        c = cache[e]
        t = c["grasp"] - a.horizon_back
        if t < c["from"]:
            continue
        true_act = np.array(c["act"], np.float32)
        tgt = np.array([c["u"], c["v"]], np.float32)
        wrong = np.array(c["wrong"], np.float32)

        sample = ds[t]
        pc = predict_end(policy, cfg, pre, post, sample, tgt)
        pw = predict_end(policy, cfg, pre, post, sample, wrong)

        rows.append(dict(ep=e,
                         ec=float(np.abs(pc - true_act).mean()),
                         ew=float(np.abs(pw - true_act).mean()),
                         d=float(np.abs(pc - pw).mean()),
                         pan_c=float(pc[0]), pan_w=float(pw[0]),
                         want_c=float(np.polyval(pu, tgt[0])),
                         want_w=float(np.polyval(pu, wrong[0])),
                         sep=float(np.hypot(*(wrong - tgt)))))

    if not rows:
        raise SystemExit("측정할 에피소드가 없습니다.")

    ec = np.array([r["ec"] for r in rows]); ew = np.array([r["ew"] for r in rows])
    ratio = float(ew.mean() / ec.mean())
    delta = float(np.mean([r["d"] for r in rows]))

    print(f"\n좌표 조건화 측정 — {a.run} / {a.ckpt}   표본 {len(rows)}개")
    print("=" * 70)
    print(f"  올바른 좌표 오차 {ec.mean():.3f}     틀린 좌표 오차 {ew.mean():.3f}")
    print(f"  **자기목표 대비 {ratio:.2f}**   (목표 변경 시 행동 변화 Δ {delta:.2f})")
    print(f"  참고: 기준선 1.02 / 원-핫+FiLM 100K 1.22 / 색별 정책 3.00~4.77")

    # ② 응답 이득 — 목표를 옮긴 만큼 실제로 따라 움직였는가
    #    이분법(더 가까운가)으로 재면 "덜 갔다"가 전부 실패로 잡혀 0% 가 나온다.
    #    얼마나 따라갔는지를 비율로 재는 편이 훨씬 많은 것을 알려준다.
    need = np.array([r["want_w"] - r["want_c"] for r in rows])
    got = np.array([r["pan_w"] - r["pan_c"] for r in rows])
    gain = float(np.abs(got).mean() / np.abs(need).mean() * 100)
    sign = float((np.sign(got) == np.sign(need)).mean() * 100)
    pc_ = np.array([r["pan_c"] for r in rows]); wc_ = np.array([r["want_c"] for r in rows])
    corr = float(np.corrcoef(pc_, wc_)[0, 1])
    moved = np.array([abs(r["pan_w"] - r["want_w"]) for r in rows])
    stay = np.array([abs(r["pan_w"] - r["want_c"]) for r in rows])
    win = float((moved < stay).mean() * 100)

    print(f"\n  올바른 목표 추종도  pan ↔ 요구각 상관 {corr:+.3f}   (1.0 = 좌표대로 간다)")
    print(f"  **응답 이득 {gain:.0f}%**  — 목표를 {np.abs(need).mean():.1f}도만큼 옮겼을 때 "
          f"실제로 {np.abs(got).mean():.1f}도 움직였습니다")
    print(f"    방향 일치 {sign:.0f}% (50%=무작위)   유도 성공률 {win:.0f}%")

    print()
    if ratio >= 3.0:
        print("  판정: 성공. 좌표 목표를 따릅니다. 단일 정책으로 통합 가능합니다.")
    elif ratio >= 1.5:
        print("  판정: 방향은 맞으나 약합니다. 에폭 부족일 수 있습니다.")
    else:
        print("  판정: 실패. 좌표는 출력과 같은 공간이라 정상 학습에서 이 값이 나오기 어렵습니다.")
        print("        모델이 아니라 피처 배선(environment_state 가 실제로 들어갔는지)을 먼저 보십시오.")

    out = PROJECT / "goal_metric_xy.tsv"
    new = not out.exists()
    with out.open("a") as f:
        if new:
            f.write("run\tckpt\tratio\tdelta\tgain_pct\tsign_pct\tcorr\tguide_pct\tn\n")
        f.write(f"{a.run}\t{a.ckpt}\t{ratio:.3f}\t{delta:.3f}\t{gain:.0f}\t{sign:.0f}\t{corr:+.3f}\t{win:.0f}\t{len(rows)}\n")
    (PROJECT / "probe_xy_rows.json").write_text(json.dumps(rows, indent=1))
    print(f"  기록: {out}")


if __name__ == "__main__":
    main()
