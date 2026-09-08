#!/usr/bin/env python
"""학습된 체크포인트를 로봇 없이 검증한다 (오프라인 open-loop 재현 오차).

실기 평가 전에 "학습이 실제로 됐는지"를 데이터로 확인하는 용도다. 확인하는 것:

  1. 체크포인트 무결성 — 가중치에 NaN/Inf 없는지, 정규화 통계가 정상인지,
     정책의 input/output 피처가 데이터셋과 일치하는지
  2. Open-loop 액션 오차 — 실제 배포와 같은 방식으로(chunk 단위로 관측을 넣고
     n_action_steps 개를 그대로 실행) 예측한 액션을 정답 액션과 비교
  3. 베이스라인 대비 — "항상 평균 액션 출력", "현재 자세 유지"보다 얼마나 나은지.
     이걸 못 이기면 학습이 안 된 것이다
  4. 그리퍼 타이밍 — 잡는 순간(그리퍼 닫힘)을 몇 프레임 오차로 맞추는지

주의: 30개 에피소드 전부 학습에 쓰였으므로 이건 **일반화 성능이 아니다.**
"학습이 망가지지 않았다"는 하한 확인용이고, 실제 성공률은 실기(eval.sh)로만 나온다.

사용법:
  python ~/omx_pill_project/validate_policy.py                       # last 체크포인트, 5개 에피소드
  python ~/omx_pill_project/validate_policy.py --ckpt 005000 020000  # 체크포인트 비교
  python ~/omx_pill_project/validate_policy.py --episodes 0 1 2 --n-action-steps 20
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

TRAIN_DIR = Path("/home/user/omx_pill_project/train")
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]


CAM_RENAME: dict[str, str] = {}   # 정책이 다른 카메라 이름을 기대할 때의 변환표


def load_policy(ckpt_dir: Path, ds, device: str):
    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    cfg.pretrained_path = str(ckpt_dir)
    cfg.device = device
    # smolvla_base 는 camera1/2/3 을 기대하는데 우리 데이터는 top/wrist 다.
    # 전처리기(rename_map)가 배치를 이미 camera1/2 로 바꿔 주므로, 여기서는
    # **정책 설정 쪽만** 데이터 이름에 맞춰 make_policy 의 검증을 통과시킨 뒤
    # 원래 이름으로 되돌린다. 되돌리지 않으면 정책이 배치에서 이미지를 못 찾는다.
    global CAM_RENAME
    CAM_RENAME = {}
    want = [k for k in getattr(cfg, "input_features", {}) if ".images." in k]
    have = [k for k in ds.meta.features if ".images." in k]
    orig = dict(cfg.input_features) if want else None
    if want and sorted(want) != sorted(have):
        feats = {k: v for k, v in cfg.input_features.items() if k not in want}
        for src, dst in zip(sorted(have), sorted(want)):
            feats[src] = cfg.input_features[dst]
            CAM_RENAME[src] = dst
        cfg.input_features = feats
        print(f"  ↔ 카메라 이름 맞춤: {', '.join(f'{a.split(chr(46))[-1]}→{b.split(chr(46))[-1]}' for a, b in CAM_RENAME.items())}")
    policy = make_policy(cfg, ds_meta=ds.meta)
    if CAM_RENAME and orig is not None:
        cfg.input_features = orig          # 정책은 원래 이름으로 배치를 읽는다
        policy.config.input_features = orig
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(ckpt_dir),
        dataset_stats=ds.meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, cfg, pre, post


def check_integrity(policy, cfg, ds) -> list[str]:
    """가중치·통계·피처 정합성. 문제를 문자열 리스트로 돌려준다 (빈 리스트 = 정상)."""
    problems = []

    n_bad = 0
    n_param = 0
    for name, p in policy.named_parameters():
        n_param += p.numel()
        if not torch.isfinite(p).all():
            n_bad += 1
            problems.append(f"가중치에 NaN/Inf: {name}")
    print(f"  파라미터 {n_param / 1e6:.1f}M, NaN/Inf 있는 텐서 {n_bad}개")

    for key in cfg.input_features:
        if key not in ds.meta.features:
            problems.append(f"정책 입력 {key} 가 데이터셋에 없음")
        else:
            want = tuple(cfg.input_features[key].shape)
            got = tuple(ds.meta.features[key]["shape"])
            # 이미지는 데이터셋이 HWC, 정책이 CHW 로 들고 있어 순서만 다르다 (정상)
            if sorted(want) != sorted(got):
                problems.append(f"{key} shape 불일치: 정책 {want} vs 데이터셋 {got}")
    print(f"  입력 피처 {list(cfg.input_features)} — 데이터셋과 일치 여부 위에 반영")

    for key in ("action", "observation.state"):
        st = ds.meta.stats[key]
        for sk in ("mean", "std"):
            v = np.asarray(st[sk], dtype=np.float64)
            if not np.isfinite(v).all():
                problems.append(f"stats[{key}][{sk}] 에 NaN/Inf")
            if sk == "std" and (v <= 1e-8).any():
                problems.append(
                    f"stats[{key}][std] 에 0에 가까운 값 {np.round(v, 6).tolist()} — "
                    "해당 관절이 전혀 안 움직였다는 뜻"
                )
    print(f"  action std = {np.round(np.asarray(ds.meta.stats['action']['std']), 3).tolist()}")
    return problems


@torch.inference_mode()
def rollout_episode(policy, cfg, pre, post, ds, gidx, gt_actions, n_action_steps, device):
    """chunk 단위 open-loop. 예측 액션 (T,6) 과 정답 (T,6) 을 돌려준다."""
    T = len(gidx)
    pred = np.full((T, gt_actions.shape[1]), np.nan, dtype=np.float32)
    amp = (
        torch.autocast(device_type="cuda")
        if device == "cuda" and getattr(cfg, "use_amp", False)
        else nullcontext()
    )

    policy.reset()
    for t in range(0, T, n_action_steps):
        sample = ds[int(gidx[t])]
        obs = {k: sample[k] for k in cfg.input_features}
        obs["task"] = sample.get("task", "")
        with amp:
            batch = pre(obs)
            chunk = policy.predict_action_chunk(batch)
        chunk = post(chunk).float().cpu().numpy()
        if chunk.ndim == 3:  # (B, chunk_size, action_dim)
            chunk = chunk[0]
        k = min(n_action_steps, chunk.shape[0], T - t)
        pred[t : t + k] = chunk[:k]
    return pred, gt_actions


def gripper_threshold(gt_traj):
    """정답 궤적에서 '닫힘' 판정 임계값과 방향을 구한다.

    그리퍼는 에피소드 대부분을 열린 채로 보내므로 중앙값이 열린 값이다.
    중앙값에서 가장 먼 극값이 닫힘이고, 그 중간을 임계값으로 쓴다.
    (값의 크기·부호로 열림/닫힘을 가정하지 않는다 — 로봇마다 다르다)
    """
    g = gt_traj[:, -1]
    opened = float(np.median(g))
    lo, hi = float(g.min()), float(g.max())
    closed = lo if abs(lo - opened) > abs(hi - opened) else hi
    return (opened + closed) / 2.0, closed < opened


def gripper_close_frame(traj, thr, closed_is_low):
    """그리퍼가 처음 닫힘 임계값을 넘는 프레임. 없으면 None."""
    g = traj[:, -1]
    crossed = np.where(g < thr if closed_is_low else g > thr)[0]
    return int(crossed[0]) if len(crossed) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="act_yellow")
    ap.add_argument("--ckpt", nargs="+", default=["last"])
    ap.add_argument("--repo-id", default="1unasy/pill_v2_onehot")
    ap.add_argument("--episodes", nargs="+", type=int, default=None, help="기본: 0 5 10 15 20")
    ap.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="한 번의 관측으로 실행할 액션 수. 기본은 체크포인트 설정값(=배포와 동일)",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    episodes = args.episodes if args.episodes is not None else [0, 5, 10, 15, 20]

    print(f"데이터셋 로드: {args.repo_id}")
    ds = LeRobotDataset(args.repo_id)
    fps = ds.meta.fps
    print(f"  {ds.meta.total_episodes} 에피소드 / {ds.meta.total_frames} 프레임 / {fps} fps")

    # 정답 액션과 에피소드 경계는 비디오 디코딩 없이 테이블에서 직접 읽는다
    ep_index = np.asarray(ds.hf_dataset["episode_index"])
    all_actions = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    all_states = np.asarray(ds.hf_dataset["observation.state"], dtype=np.float32)
    a_min = all_actions.min(axis=0)
    a_max = all_actions.max(axis=0)
    a_rng = np.maximum(a_max - a_min, 1e-6)
    a_mean = all_actions.mean(axis=0)
    dim = all_actions.shape[1]
    names = JOINT_NAMES[:dim] if dim <= len(JOINT_NAMES) else [f"d{i}" for i in range(dim)]

    for ckpt in args.ckpt:
        ckpt_dir = TRAIN_DIR / args.run / "checkpoints" / ckpt / "pretrained_model"
        print("\n" + "=" * 78)
        print(f"체크포인트: {ckpt_dir}")
        print("=" * 78)
        if not ckpt_dir.is_dir():
            print("  없음 — 건너뜀")
            continue

        policy, cfg, pre, post = load_policy(ckpt_dir, ds, args.device)
        n_steps = args.n_action_steps or cfg.n_action_steps
        print(f"\n[1] 무결성 검사")
        problems = check_integrity(policy, cfg, ds)
        print(
            f"  chunk_size={cfg.chunk_size} n_action_steps={cfg.n_action_steps} "
            f"(이번 검증은 {n_steps} 사용 = {n_steps / fps:.1f}초 open-loop)"
        )
        if problems:
            print("  !! 문제:")
            for p in problems:
                print(f"     - {p}")
        else:
            print("  이상 없음")

        print(f"\n[2] Open-loop 액션 오차 — 에피소드 {episodes}")
        err_model, err_mean, err_hold = [], [], []
        grip_rows = []
        for ep in episodes:
            gidx = np.where(ep_index == ep)[0]
            if len(gidx) == 0:
                print(f"  ep {ep}: 없음")
                continue
            gt = all_actions[gidx]
            st = all_states[gidx]
            pred, gt = rollout_episode(
                policy, cfg, pre, post, ds, gidx, gt, n_steps, args.device
            )
            ok = ~np.isnan(pred[:, 0])
            e_model = np.abs(pred[ok] - gt[ok])
            e_mean = np.abs(a_mean[None, :] - gt[ok])
            # "현재 자세 유지": 각 chunk 시작 시점의 state 를 끝까지 유지
            hold = np.repeat(st[::n_steps], n_steps, axis=0)[: len(gt)]
            e_hold = np.abs(hold[ok] - gt[ok])
            err_model.append(e_model)
            err_mean.append(e_mean)
            err_hold.append(e_hold)

            thr, closed_is_low = gripper_threshold(gt)
            gf_gt = gripper_close_frame(gt, thr, closed_is_low)
            gf_pd = gripper_close_frame(pred[ok], thr, closed_is_low)
            grip_rows.append((ep, len(gt), gf_gt, gf_pd))
            print(
                f"  ep {ep:2d} ({len(gt):4d} 프레임): "
                f"MAE {e_model.mean():.3f} / 평균베이스라인 {e_mean.mean():.3f} "
                f"/ 자세유지 {e_hold.mean():.3f}"
            )

        if not err_model:
            continue
        E = np.concatenate(err_model)
        M = np.concatenate(err_mean)
        H = np.concatenate(err_hold)

        print(f"\n  관절별 MAE (전체 {len(E)} 프레임)")
        print(f"  {'관절':<10}{'MAE':>9}{'범위대비':>10}{'평균BL':>10}{'자세유지BL':>12}{'개선배수':>10}")
        for i, nm in enumerate(names):
            print(
                f"  {nm:<10}{E[:, i].mean():>9.3f}{100 * E[:, i].mean() / a_rng[i]:>9.1f}%"
                f"{M[:, i].mean():>10.3f}{H[:, i].mean():>12.3f}"
                f"{M[:, i].mean() / max(E[:, i].mean(), 1e-9):>10.1f}x"
            )
        print(
            f"  {'전체':<10}{E.mean():>9.3f}{100 * (E.mean(axis=0) / a_rng).mean():>9.1f}%"
            f"{M.mean():>10.3f}{H.mean():>12.3f}{M.mean() / max(E.mean(), 1e-9):>10.1f}x"
        )

        print("\n[3] 판정")
        ratio = M.mean() / max(E.mean(), 1e-9)
        rel = 100 * (E.mean(axis=0) / a_rng).mean()
        if problems:
            print("  X 무결성 문제 있음 — 위 [1] 확인")
        elif ratio < 2:
            print(f"  X 평균 베이스라인의 {ratio:.1f}배 — 학습이 사실상 안 됨")
        elif rel > 15:
            print(f"  △ 액션 범위의 {rel:.1f}% 오차 — 궤적을 대충만 따라간다")
        else:
            print(f"  O 오차 {rel:.1f}% (범위 대비), 베이스라인 대비 {ratio:.1f}배 — 정상 학습")
        print("     ※ 학습 데이터 재현 오차다. 실제 성공률은 eval.sh 로만 확인된다")

        print("\n[4] 그리퍼 닫힘 타이밍 (프레임, 30fps)")
        for ep, T, gf_gt, gf_pd in grip_rows:
            if gf_gt is None or gf_pd is None:
                print(f"  ep {ep:2d}: 정답 {gf_gt} / 예측 {gf_pd}  (임계값 통과 없음)")
            else:
                d = gf_pd - gf_gt
                print(
                    f"  ep {ep:2d}: 정답 {gf_gt:4d} / 예측 {gf_pd:4d}  "
                    f"차이 {d:+4d} 프레임 ({d / fps:+.2f}초)"
                )

        del policy
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
