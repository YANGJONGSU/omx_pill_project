#!/usr/bin/env python3
"""SmolVLA 가 **언어로** 목표 색을 구분하는지 잰다.

ACT 원-핫 3색은 자기색 대비 1.00배로 완전히 실패했다 (2026-08-06) — 목표를 무시하고
장면만 외웠다. SmolVLA 는 사전학습된 언어 인코더가 있으니 다를 수 있다.

방법: 같은 화면에 **지시문만 바꿔** 넣고 예측이 얼마나 달라지는지 본다.
  자기 색 지시문 → 오차가 작아야 한다 (정답 궤적과 가까움)
  다른 색 지시문 → 오차가 커야 한다
  비율이 1.0 이면 지시문을 무시한 것이다.

    python probe_vla.py --run smolvla_v1
"""
from __future__ import annotations
import argparse, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import validate_policy as _vp
from validate_policy import TRAIN_DIR, gripper_close_frame, gripper_threshold, load_policy

TASKS = {"red": "pick red pill", "green": "pick green pill", "yellow": "pick yellow pill"}
EPS = {"red": range(0, 29), "green": range(29, 59), "yellow": range(59, 89)}


@torch.inference_mode()
def predict(policy, cfg, pre, post, sample, task):
    obs = {}
    for k in cfg.input_features:
        # 정책이 데이터셋과 다른 카메라 이름을 쓰면(smolvla_base 는 camera1/2)
        # 샘플에는 원래 이름(top/wrist)으로 들어 있다. 역방향으로 찾아 준다.
        src = next((a for a, b in _vp.CAM_RENAME.items() if b == k), k)
        if src not in sample:
            continue          # smolvla_base 의 camera3 — 우리는 카메라가 2대뿐이다
        v = sample[src]
        obs[k] = v.clone() if torch.is_tensor(v) else v
    obs["task"] = task
    policy.reset()
    ch = post(policy.predict_action_chunk(pre(obs))).float().cpu().numpy()
    return (ch[0] if ch.ndim == 3 else ch)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="smolvla_v1")
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--repo-id", default="1unasy/pill_v3")
    ap.add_argument("--per-color", type=int, default=8, help="색당 에피소드 수")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ds = LeRobotDataset(a.repo_id)
    ep_index = np.asarray(ds.hf_dataset["episode_index"])
    acts = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    ck = TRAIN_DIR / a.run / "checkpoints" / a.ckpt / "pretrained_model"
    print(f"정책 로드: {ck}")
    policy, cfg, pre, post = load_policy(ck, ds, a.device)

    err = {c: {t: [] for t in TASKS} for c in TASKS}
    for c, eps in EPS.items():
        sel = list(eps)[:: max(1, len(list(eps)) // a.per_color)][: a.per_color]
        for ep in sel:
            gi = np.where(ep_index == ep)[0]
            if not len(gi):
                continue
            gt = acts[gi]
            thr, low = gripper_threshold(gt)
            close = gripper_close_frame(gt, thr, low)
            if close is None:
                continue
            for off in (-20,):   # 파지 직전 한 시점. 표본은 에피소드 수로 늘린다
                f = close + off
                if not (0 <= f < len(gi)):
                    continue
                s = ds[int(gi[f])]
                for tname, tstr in TASKS.items():
                    p = predict(policy, cfg, pre, post, s, tstr)
                    err[c][tname].append(float(np.abs(p - gt[f]).mean()))
            print(f"\r  진행 {c} ep{ep}", end="", flush=True)
    print()

    print("\n" + "=" * 62)
    print(f"  {a.run} — 지시문을 바꾸면 예측이 달라지는가")
    print("=" * 62)
    print(f"{'에피소드':<10}" + "".join(f"{t+' 지시':>14}" for t in TASKS))
    ratios = []
    for c in TASKS:
        row = f"  {c:<8}"
        own = np.mean(err[c][c]) if err[c][c] else float("nan")
        for t in TASKS:
            v = np.mean(err[c][t]) if err[c][t] else float("nan")
            row += f"{v:>14.3f}"
        print(row)
        others = [np.mean(err[c][t]) for t in TASKS if t != c and err[c][t]]
        if own and others:
            ratios.append(min(others) / own)
    print("-" * 62)
    for c, r in zip(TASKS, ratios):
        print(f"  {c:<8}자기색 대비 {r:.2f}배")
    if ratios:
        m = float(np.mean(ratios))
        print(f"\n  평균 {m:.2f}배")
        print("  참고: ACT 원-핫 3색 1.00배(실패) / ACT 색별 3.26~5.45배(성공)")
        print("  → " + ("지시문을 **무시한다** (1.2배 미만)" if m < 1.2 else
                        "지시문을 읽고 있다" if m < 2.0 else
                        "지시문을 확실히 구분한다"))


if __name__ == "__main__":
    main()
