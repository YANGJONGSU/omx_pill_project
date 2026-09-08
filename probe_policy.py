#!/usr/bin/env python
"""학습된 정책에게 "너 위치를 어디서 보고 있니?" 를 직접 묻는다 (로봇 없이, 재학습 없이).

지금까지의 진단은 전부 **환경**(랜드마크·매트·초점·케이블)과 **데이터**(파지 성공·길이·
배치 준수)를 쟀다. 정책 자체의 입력→출력 민감도는 한 번도 안 쟀고, 그래서 실기가
빗나갈 때마다 남는 선택지가 "다시 찍자 / 다시 학습하자" 뿐이었다. 이 스크립트가 그 빈칸이다.

세 가지를 잰다 (전부 이미 있는 체크포인트 위에서):

  [1] 차폐 — top 을 가릴 때와 wrist 를 가릴 때 중 어느 쪽이 더 망가지는가.
      정책이 위치 정보를 어느 카메라에서 얻는지가 여기서 갈린다.

  [2] top 이동 민감도 — top 화면만 dy 픽셀 밀었을 때 정책이 깊이축(elbow-lift)을
      얼마나 움직이는가. 학습 데이터에서 잰 기울기는 **0.31도/px** 이다
      (run_policy.shift_image 의 환산값, 그룹별 0.355 / 0.277).
      정책 기울기 ÷ 0.31 = **응답률**. 1에 가까우면 정상, 0에 가까우면 top 을 무시하는 것.
      후자면 affine 증강(translate 0.05 = ±32px = ±21mm)이 필요한 정밀도를
      그대로 지워버렸다는 뜻이고, `--top-shift-y` 가 안 들었던 이유도 같다.

  [3] 그룹별 부호 오차 — ep0~11 (옛 화각) 과 ep12~28 (지금 화각) 의 open-loop 오차가
      계통적으로 반대 방향인가. 그렇다면 화각 혼입이 확정된다.
      §8.5 의 "17개만 재학습해서 3시간 만에 확인" 을 재학습 없이 대체한다.

사용법:
  python ~/omx_pill_project/probe_policy.py                       # 셋 다
  python ~/omx_pill_project/probe_policy.py --tests 1 2           # 골라서
  python ~/omx_pill_project/probe_policy.py --run act_v2_yellow20 --repo-id 1unasy/pill_v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.utils.constants import OBS_IMAGES
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from validate_policy import (  # noqa: E402
    TRAIN_DIR,
    gripper_close_frame,
    gripper_threshold,
    load_policy,
    rollout_episode,
)

TOP = "observation.images.top"
WRIST = "observation.images.wrist"

# run_policy.shift_image 의 docstring 에 기록된 학습 데이터 실측 환산.
# 화면 1px ~= 0.31도(elbow-lift) ~= 0.65mm. 이게 [2] 의 기준선이다.
DEG_PER_PX = 0.31
MM_PER_PX = 0.65

# 관절 인덱스 (pan, lift, elbow, wrist_flex, wrist_roll, gripper)
I_PAN, I_LIFT, I_ELBOW = 0, 1, 2
NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]


def depth_axis(a: np.ndarray) -> np.ndarray:
    """깊이(전후) 축. lift+elbow 를 쓰면 서로 상쇄돼 가짜 결론이 난다 (STATUS.md 경고).

    올바른 축은 elbow-lift (전후% 와 상관 +0.87).
    """
    return a[..., I_ELBOW] - a[..., I_LIFT]


def shift_img(img_chw: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """CHW float 텐서를 픽셀 단위로 민다. run_policy.shift_image 와 같은 방식."""
    if dx == 0 and dy == 0:
        return img_chw
    hwc = img_chw.permute(1, 2, 0).cpu().numpy()
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(
        hwc, M, (hwc.shape[1], hwc.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    return torch.from_numpy(out).permute(2, 0, 1)


GOAL = "observation.environment_state"


@torch.inference_mode()

def _horizon_of(cfg):
    """정책의 행동 시퀀스 길이. ACT 는 chunk_size, Diffusion 은 horizon 을 쓴다."""
    return int(getattr(cfg, "chunk_size", None)
               or getattr(cfg, "horizon", None)
               or getattr(cfg, "n_action_steps", 1))


def predict_chunk(policy, cfg, pre, post, sample, *, shift=(0, 0), blackout=None, cam=TOP,
                  goal=None, prev=None):
    """관측 하나 → 액션 청크 (chunk_size, 6). 화면을 밀거나 가린 채로 넣을 수 있다.

    goal 을 주면 목표 원-핫을 그 값으로 바꿔 넣는다 (같은 화면에 목표만 바꿔보기).
    """
    obs = {}
    for k in cfg.input_features:
        v = sample[k]
        obs[k] = v.clone() if torch.is_tensor(v) else v
    obs["task"] = sample.get("task", "")
    if goal is not None and GOAL in obs:
        obs[GOAL] = torch.as_tensor(goal, dtype=torch.float32)

    if shift != (0, 0) and cam in obs:
        obs[cam] = shift_img(obs[cam], shift[0], shift[1])
    if blackout is not None and blackout in obs:
        obs[blackout] = torch.zeros_like(obs[blackout])

    policy.reset()
    batch = pre(obs)

    # Diffusion 은 과거 관측 n_obs_steps 개를 큐에 모아 두고 거기서 배치를 다시 만든다.
    # select_action 이 큐를 채우면서 카메라들을 observation.images 하나로 합치므로,
    # predict_action_chunk 를 직접 부르려면 같은 키를 만들어 넘겨야 큐가 걸러지지 않는다.
    # (관측 하나를 n_obs_steps 번 복제 — 정지 상태와 같다. 목표만 바꿔 비교하는
    #  용도라 두 조건에 같은 처리가 들어가면 비율은 그대로 의미가 있다.)
    if int(getattr(cfg, "n_obs_steps", 1) or 1) > 1 and hasattr(policy, "_queues"):
        # prev 가 있으면 진짜 직전 프레임으로 큐를 채운다 (없으면 현재 관측을 복제).
        if prev is not None:
            pobs = {}
            for k in cfg.input_features:
                v = prev[k]
                pobs[k] = v.clone() if torch.is_tensor(v) else v
            pobs["task"] = prev.get("task", "")
            if goal is not None and GOAL in pobs:
                pobs[GOAL] = torch.as_tensor(goal, dtype=torch.float32)
            policy.select_action(dict(pre(pobs)))
        policy.select_action(dict(batch))
        if getattr(cfg, "image_features", None):
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack(
                [batch[k] for k in cfg.image_features], dim=-4)

    chunk = post(policy.predict_action_chunk(batch)).float().cpu().numpy()
    return chunk[0] if chunk.ndim == 3 else chunk


def approach_frames(all_actions, gidx, offsets=(-60, -30, -10)):
    """파지 직전 프레임들의 전역 인덱스. 접근 중이라 top 위치가 행동을 좌우하는 구간이다."""
    gt = all_actions[gidx]
    thr, low = gripper_threshold(gt)
    close = gripper_close_frame(gt, thr, low)
    if close is None:
        return []
    return [int(gidx[close + o]) for o in offsets if 0 <= close + o < len(gidx)]


# ---------------------------------------------------------------- [1] 차폐

def test_ablation(policy, cfg, pre, post, ds, probes, all_actions):
    print("\n" + "=" * 78)
    print("[1] 차폐 — 정책은 어느 카메라를 보고 있나")
    print("=" * 78)
    print("  각 카메라를 검게 가리고 예측이 정답에서 얼마나 멀어지는지 잰다.")
    print("  많이 망가지는 쪽 = 정책이 실제로 의존하는 쪽.\n")

    rows = {}
    for label, black in (("정상", None), ("top 가림", TOP), ("wrist 가림", WRIST)):
        errs = []
        for gi in probes:
            sample = ds[gi]
            chunk = predict_chunk(policy, cfg, pre, post, sample, blackout=black)
            n = min(len(chunk), len(all_actions) - gi)
            gt = all_actions[gi : gi + n]
            errs.append(np.abs(chunk[:n] - gt).mean())
        rows[label] = float(np.mean(errs))

    base = rows["정상"]
    print(f"  {'조건':<12} {'액션 MAE':>10}   {'정상 대비':>10}")
    for label, v in rows.items():
        ratio = "—" if label == "정상" else f"{v / base:6.2f}배"
        print(f"  {label:<12} {v:10.3f}   {ratio:>10}")

    d_top = rows["top 가림"] / base
    d_wrist = rows["wrist 가림"] / base
    print()
    if d_top < 1.15 and d_wrist < 1.15:
        print("  판정: 두 카메라 다 가려도 거의 안 변한다 → 정책이 영상을 거의 안 보고")
        print("        관절 상태(observation.state)만으로 궤적을 외운 것에 가깝다.")
    elif d_top >= d_wrist * 1.3:
        print("  판정: top 의존. 위치 정보를 위쪽 카메라에서 얻고 있다 → [2] 로 정밀도를 본다.")
    elif d_wrist >= d_top * 1.3:
        print("  판정: **wrist 의존.** 위치를 손목 카메라에서 얻고 있다.")
        print("        wrist 는 계속 죽고 초점이 풀리는 그 카메라다 (STATUS.md).")
        print("        실기 1~2cm 오차와 직결될 수 있다.")
    else:
        print("  판정: 두 카메라를 비슷하게 쓴다.")
    return rows


# ------------------------------------------------- [2] top 이동 민감도

def sweep(policy, cfg, pre, post, ds, probes, shifts, cam, base):
    """cam 화면을 dy 픽셀씩 밀면서 깊이축·pan 이 얼마나 따라오는지."""
    print(f"  {'dy(px)':>7} {'≈mm':>6}   {'Δ깊이(도)':>10} {'기대(도)':>9} {'응답률':>8}   {'Δpan':>7}")
    obs_slope = []
    for dy in shifts:
        d_depth, d_pan = [], []
        for gi in probes:
            ch = predict_chunk(policy, cfg, pre, post, ds[gi], shift=(0, dy), cam=cam)
            d_depth.append(depth_axis(ch).mean() - depth_axis(base[gi]).mean())
            d_pan.append(ch[:, I_PAN].mean() - base[gi][:, I_PAN].mean())
        dd, dp = float(np.mean(d_depth)), float(np.mean(d_pan))
        expect = -DEG_PER_PX * dy  # +dy = 물체가 로봇에 가까워 보임 = 덜 뻗음
        resp = dd / expect if abs(expect) > 1e-6 else float("nan")
        if dy != 0:
            obs_slope.append(dd / dy)
        print(f"  {dy:>7} {dy * MM_PER_PX:>6.1f}   {dd:>10.2f} {expect:>9.2f} {resp:>8.2f}   {dp:>7.2f}")
    slope = float(np.mean(np.abs(obs_slope))) if obs_slope else 0.0
    return slope, slope / DEG_PER_PX


def test_top_sensitivity(policy, cfg, pre, post, ds, probes, shifts):
    print("\n" + "=" * 78)
    print("[2] 이동 민감도 — 화면을 밀면 정책이 따라오나")
    print("=" * 78)
    print(f"  학습 데이터 실측 기울기 = {DEG_PER_PX} 도/px (elbow-lift). 이게 기준선이다.")
    print("  정책이 화면의 절대 위치를 제대로 보면 민 만큼 깊이축이 따라와야 한다.")
    print("  wrist 를 대조군으로 같이 잰다 — 둘 다 0이면 애초에 영상 기하를 안 쓰는 것이고,")
    print("  top 만 0이면 top 의 위치 정보가 지워진 것이다.\n")

    base = {gi: predict_chunk(policy, cfg, pre, post, ds[gi]) for gi in probes}

    print("  ── top ──")
    slope, ratio = sweep(policy, cfg, pre, post, ds, probes, shifts, TOP, base)
    print(f"  정책 기울기 = {slope:.3f} 도/px  →  **응답률 {ratio:.2f}**\n")

    print("  ── wrist (대조군) ──")
    w_slope, w_ratio = sweep(policy, cfg, pre, post, ds, probes, shifts, WRIST, base)
    print(f"  정책 기울기 = {w_slope:.3f} 도/px  →  응답률 {w_ratio:.2f}\n")

    if ratio < 0.3 and w_ratio < 0.3:
        print("  판정: **두 화면 다 밀어도 반응이 없다.** 정책이 영상의 기하(위치)를")
        print("        거의 안 쓰고 관절 상태(observation.state)로 궤적을 재생하고 있다는 뜻이다.")
        print("        차폐가 아프게 나온 [1] 과 합치면: 영상은 '무엇/언제'에만 쓰고")
        print("        '어디'는 안 쓴다. 증강만의 문제가 아니라 더 근본적이다.")
        return ratio
    if ratio < 0.3 <= w_ratio:
        print("  판정: **top 만 반응이 없다.** 절대 위치의 유일한 출처인 top 이 지워졌다.")
        print("        affine 증강(translate 0.05 = ±32px = ±21mm)과 정확히 같은 크기다.")
        print("        조치: --dataset.image_transforms.enable=false 로 재학습 1회.")
        return ratio
    print()
    if ratio < 0.3:
        print("  판정: **정책이 top 의 절대 위치를 거의 무시한다.**")
        print(f"        화면을 2cm(32px) 밀어도 깊이축이 {slope * 32:.1f}도밖에 안 움직인다")
        print(f"        (제대로면 {DEG_PER_PX * 32:.1f}도).")
        print("        → affine 증강(translate 0.05 = ±32px = ±21mm)이 필요한 정밀도를")
        print("          '무시해도 된다'고 가르친 것과 일치한다.")
        print("        → `--top-shift-y` 를 124mm 까지 밀어도 안 들었던 이유도 이것이다.")
        print("        조치: --dataset.image_transforms.enable=false 로 재학습 1회.")
    elif ratio > 0.7:
        print("  판정: 정상. top 위치에 제대로 반응한다 → 증강은 범인이 아니다.")
        print("        1~2cm 오차의 원인은 다른 곳이다 ([1] 결과와 [3] 을 본다).")
    else:
        print("  판정: 애매하다 (부분적으로만 반응). [1]·[3] 과 함께 판단할 것.")
    return ratio


# ------------------------------------------- [3] 그룹별 부호 오차

def test_group_bias(policy, cfg, pre, post, ds, ep_index, all_actions, groups, n_steps, device):
    print("\n" + "=" * 78)
    print("[3] 그룹별 오차 — 옛 화각 vs 지금 화각")
    print("=" * 78)
    print("  두 그룹의 깊이축 오차가 계통적으로 **반대 방향**이면 화각 혼입이 확정된다.")
    print("  (재학습 없이 §8.5 의 결정적 실험을 대신한다)\n")

    out = {}
    for label, eps in groups.items():
        vals = []
        for ep in eps:
            gidx = np.where(ep_index == ep)[0]
            if len(gidx) == 0:
                continue
            gt = all_actions[gidx]
            pred, gt = rollout_episode(policy, cfg, pre, post, ds, gidx, gt, n_steps, device)
            ok = ~np.isnan(pred[:, 0])
            vals.append(float(np.mean(depth_axis(pred[ok]) - depth_axis(gt[ok]))))
        out[label] = np.array(vals)
        m, s = out[label].mean(), out[label].std()
        se = s / max(np.sqrt(len(vals)), 1)
        print(f"  {label:<22} n={len(vals):2d}  깊이 오차 {m:+6.2f}도 "
              f"(±{1.96 * se:.2f}, {m / DEG_PER_PX * MM_PER_PX:+.1f}mm)")

    labels = list(out)
    if len(labels) == 2 and all(len(out[k]) > 1 for k in labels):
        a, b = out[labels[0]], out[labels[1]]
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        ci = 1.96 * se
        print(f"\n  그룹 차이 = {diff:+.2f}도 (95% 구간 {diff - ci:+.2f} ~ {diff + ci:+.2f})")
        print(f"             = {diff / DEG_PER_PX * MM_PER_PX:+.1f}mm")
        if abs(diff) - ci > 0 and np.sign(a.mean()) != np.sign(b.mean()):
            print("\n  판정: **두 그룹이 반대 방향으로 치우쳐 있다 = 화각 혼입 확정.**")
            print("        조치: ep0~11 을 재투영하거나 제외하고 재학습.")
        elif abs(diff) - ci > 0:
            print("\n  판정: 그룹 차이는 유의하지만 부호가 같다 → 혼입만으로는 설명이 안 된다.")
        else:
            print("\n  판정: 0을 포함한다 → 이 표본으로는 화각 혼입을 확정할 수 없다.")
            print("        (§8.3 에서 관절 통계로 확정 못 했던 것과 같은 결론)")
    return out


# --------------------------------- [4] 위치 추종 (알약이 옮겨지면 따라가나)

def test_target_tracking(policy, cfg, pre, post, ds, ep_index, all_actions, all_states, n_eps):
    """에피소드마다 알약이 다른 자리에 있다. 정책의 예측 파지점이 그걸 따라가는가.

    [2] 의 화면 밀기는 알약과 팔이 같이 움직여서, 상대 위치를 보는 정책이면
    정상이어도 0이 나온다. 여기서는 화면을 건드리지 않고 **실제로 알약이 다른 자리에
    있는 29개 에피소드**를 쓴다. 화면 차이는 오직 장면 자체의 차이다.

    파지 3.3초 전(=chunk 가 파지 시점에 닿는 지점)에서 청크를 예측하고,
    청크 끝 액션(=예측 파지 자세)이 실제 파지 자세를 따라가는지 회귀로 잰다.

      기울기 1.0 = 알약이 옮겨진 만큼 정확히 따라간다
      기울기 0.0 = 어디에 있든 늘 같은 자리(평균)로 간다

    관절 상태만으로도 어느 정도 예측되므로(팔이 이미 그쪽으로 가고 있다),
    state 기준선을 같이 계산해 **영상이 추가로 기여하는 몫**을 본다.
    """
    print("\n" + "=" * 78)
    print("[4] 위치 추종 — 알약이 다른 자리에 있으면 정책이 따라가나")
    print("=" * 78)
    print("  화면을 조작하지 않는다. 실제로 알약 위치가 다른 29개 에피소드를 쓴다.\n")

    horizon = int(_horizon_of(cfg))

    def fit(x, y):
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            return 0.0, 0.0
        return float(np.polyfit(x, y, 1)[0]), float(np.corrcoef(x, y)[0, 1])

    # 관측 시점을 여러 개 잡는다. 파지에 가까울수록 팔이 이미 목표를 향해 있어서
    # **state 만으로 정답이 새어나온다** — 그 시점의 결과는 영상 기여를 증명하지 못한다.
    # 에피소드 시작 부근(팔이 홈)에서는 state 가 전부 같으므로, 거기서 예측이
    # 목표를 따라간다면 그건 오직 영상에서 온 것이다.
    # 사람이 목표로 향하기 시작하는 순간과, 그게 관절값에 실리는 순간 사이에는 시차가 있다
    # (액션이 state 를 앞선다). 그 창을 찾아서 거기서 정책을 사람과 비교한다.
    probes = [(f"시작 +{f / 30:.1f}초", "start", f)
              for f in (15, 30, 60, 90, 120, 150, 180, 240, 300, 360, 450)]

    print(f"  {'관측 시점':<12} {'n':>3}   {'state 상관':>10} {'사람':>8} {'정책':>9} {'정책 기울기':>11}"
          f"   판정")
    print(f"  {'':12} {'':3}   {'(누설)':>10} {'(천장)':>8} {'(신호)':>9}")
    rows = []
    for label, ref, off in probes:
        pred_v, true_v, state_v, human_v = [], [], [], []
        for ep in range(n_eps):
            gidx = np.where(ep_index == ep)[0]
            if len(gidx) == 0:
                continue
            gt = all_actions[gidx]
            thr, low = gripper_threshold(gt)
            close = gripper_close_frame(gt, thr, low)
            if close is None:
                continue
            t = off if ref == "start" else close - off
            if not (0 <= t < len(gidx)) or t >= close:
                continue
            chunk = predict_chunk(policy, cfg, pre, post, ds[int(gidx[t])])
            pred_v.append(depth_axis(chunk[min(horizon - 1, len(chunk) - 1)]))
            true_v.append(depth_axis(gt[close]))
            state_v.append(depth_axis(all_states[gidx[t]]))
            # 천장: 사람이 같은 시점에서 horizon 프레임 뒤에 실제로 가 있던 자세.
            # 정책이 이걸 못 넘으면 "너무 이른 시점이라 그렇다" 는 변명이 성립한다.
            human_v.append(depth_axis(all_actions[gidx[min(t + horizon, close)]]))

        tv = np.array(true_v)
        a_p, r_p = fit(tv, np.array(pred_v))
        _, r_s = fit(tv, np.array(state_v))
        _, r_h = fit(tv, np.array(human_v))
        leak = abs(r_s) > 0.6
        if leak:
            note = "state 누설 — 무효"
        elif abs(r_h) < 0.4:
            note = "사람도 아직 안 향함 — 판정 보류"
        elif abs(r_p) > 0.6:
            note = "영상으로 위치를 읽는다"
        else:
            note = f"**사람 {r_h:.2f} vs 정책 {r_p:.2f} — 못 읽는다**"
        print(f"  {label:<12} {len(tv):>3}   {r_s:>10.2f} {r_h:>8.2f} {r_p:>9.2f} {a_p:>11.2f}   {note}")
        rows.append((label, r_s, r_p, a_p, r_h))

    # 판정 창 = 사람은 이미 목표를 향했는데(r_h 높음) 관절값은 아직 안 흘리는(r_s 낮음) 시점.
    # 그 창에서만 "영상에서 위치를 읽었는가" 가 깨끗하게 갈린다.
    usable = [r for r in rows if abs(r[4]) > 0.6 and abs(r[1]) < 0.6]
    print()
    if not usable:
        print("  판정: 사람 상관과 state 누설이 겹쳐서 깨끗한 판정 창이 없다.")
        print("        이 태스크는 목표 정보가 액션에 실리는 즉시 관절에도 실린다는 뜻.")
        return rows
    best = max(usable, key=lambda r: abs(r[4]) - abs(r[1]))
    label, r_s, r_p, a_p, r_h = best
    print(f"  판정 창: {label} — 사람은 목표를 향했고(상관 {r_h:.2f}) "
          f"관절값은 아직 안 흘린다(상관 {r_s:.2f})")
    gap = abs(r_h) - abs(r_p)
    if abs(r_p) < 0.4:
        print(f"  판정: **영상에서 알약 위치를 못 읽는다** — 사람 {r_h:.2f} vs 정책 {r_p:.2f}.")
        print("        같은 화면을 보고 사람은 목표로 향했는데 정책은 안 향한다.")
        print("        실기에서 첫 접근 방향이 어긋난 채로 굳어 1~2cm 빗나간다.")
    elif gap > 0.25:
        print(f"  판정: 부분적으로만 읽는다 — 사람 {r_h:.2f} vs 정책 {r_p:.2f} (기울기 {a_p:.2f}).")
        print("        평균 자세 쪽으로 끌린 만큼이 계통 오차로 남는다.")
    else:
        print(f"  판정: 사람만큼 읽는다 (사람 {r_h:.2f} vs 정책 {r_p:.2f}).")
        print("        인지는 정상 → 1~2cm 오차는 계통 편향 쪽이다 ([3] 을 본다).")
    return rows


# ------------------------------------------- [5] 목표 조건화 (원-핫이 먹히나)

def test_goal_conditioning(policy, cfg, pre, post, ds, ep_index, all_actions, n_eps,
                           episodes=None):
    """같은 화면에 **목표만 바꿔** 넣었을 때 정책이 다른 데로 가는가.

    3색 원-핫 학습에서 이게 안 되면 나머지는 볼 필요가 없다 — "빨강"이라고 해도
    아무 알약이나 집는 정책이 된다. ACT 는 태스크 문자열을 안 읽으므로 목표는
    observation.environment_state 하나로만 들어간다 (make_onehot.py 가 붙인 것).

    알약 검출 없이 판정하는 방법:
      각 에피소드의 **진짜 목표색**을 알고 있으니, 그 색 원-핫을 넣었을 때의 예측이
      실제 파지 자세에 가깝고, **틀린 색**을 넣으면 멀어져야 한다.
        올바른 목표 오차 << 틀린 목표 오차   →  조건화 성공
        둘이 비슷하다                        →  목표를 무시하고 있다
    """
    print("\n" + "=" * 78)
    print("[5] 목표 조건화 — 원-핫을 바꾸면 정책이 다른 알약으로 가나")
    print("=" * 78)

    if GOAL not in cfg.input_features:
        print("  이 정책에는 목표 원-핫 입력이 없습니다 (단일색 정책) — 건너뜁니다.")
        return None

    names = list(cfg.input_features[GOAL].shape and
                 ds.meta.features[GOAL].get("names") or [])
    dim = int(cfg.input_features[GOAL].shape[0])
    colors = [n.replace("goal_", "") for n in names] or [f"d{i}" for i in range(dim)]
    print(f"  원-핫 차원: {colors}\n")

    horizon = int(_horizon_of(cfg))
    eps = episodes if episodes is not None else list(range(n_eps))
    per_color = {c: {"correct": [], "wrong": [], "delta": []} for c in colors}
    used = 0
    for e in eps:
        g = np.where(ep_index == e)[0]
        if len(g) == 0:
            continue
        gt = all_actions[g]
        thr, low = gripper_threshold(gt)
        c = gripper_close_frame(gt, thr, low)
        if c is None or c - horizon < 0:
            continue
        t = int(g[c - horizon])
        true_pose = gt[c]
        # 이 에피소드의 진짜 목표색 = 데이터에 들어 있는 원-핫
        truth = np.asarray(ds[t][GOAL], dtype=np.float32)
        ti = int(np.argmax(truth))
        preds = {}
        for i, col in enumerate(colors):
            oh = np.zeros(dim, dtype=np.float32)
            oh[i] = 1.0
            ch = predict_chunk(policy, cfg, pre, post, ds[t], goal=oh,
                               prev=ds[t - 1] if t > 0 else None)
            preds[i] = ch[min(horizon - 1, len(ch) - 1)]
        err = {i: float(np.abs(p - true_pose).mean()) for i, p in preds.items()}
        d = [float(np.abs(preds[i] - preds[j]).mean())
             for i in range(dim) for j in range(i + 1, dim)]
        col = colors[ti]
        per_color[col]["correct"].append(err[ti])
        per_color[col]["wrong"] += [err[i] for i in range(dim) if i != ti]
        per_color[col]["delta"] += d
        used += 1

    if used == 0:
        print("  쓸 수 있는 에피소드가 없습니다.")
        return None

    print(f"  {'색':<8} {'n':>3} {'올바른 목표':>11} {'틀린 목표':>10} {'비율':>7} {'목표 바꿀때 Δ':>13}")
    ratios = []
    for col in colors:
        d = per_color[col]
        if not d["correct"]:
            print(f"  {col:<8} {'0':>3}   (해당 에피소드 없음)")
            continue
        a = float(np.mean(d["correct"]))
        b = float(np.mean(d["wrong"]))
        r = b / a if a > 1e-9 else float("nan")
        ratios.append(r)
        print(f"  {col:<8} {len(d['correct']):>3} {a:>11.2f} {b:>10.2f} {r:>7.2f} "
              f"{np.mean(d['delta']):>13.2f}")

    delta = float(np.mean([v for c in colors for v in per_color[c]["delta"]]))
    ratio = float(np.mean(ratios)) if ratios else float("nan")
    print(f"\n  에피소드 {used}개 사용")
    print(f"  **평균 비율 {ratio:.2f}**   (틀린 목표 오차 ÷ 올바른 목표 오차)")
    print(f"  목표를 바꿀 때 액션 변화 {delta:.2f} (액션 단위, 0이면 목표를 아예 안 본다)")
    print()
    if delta < 0.5:
        print("  판정: **목표를 무시한다.** 원-핫을 바꿔도 출력이 거의 그대로다.")
        print("        조건화가 학습되지 않았다 — 색을 지정해도 아무 알약이나 집는다.")
    elif ratio > 1.5:
        print("  판정: **조건화 성공.** 올바른 목표를 줄 때 실제 파지 자세에 훨씬 가깝다.")
    elif ratio > 1.15:
        print("  판정: 부분적으로 조건화됐다. 방향은 맞지만 구분이 약하다.")
    else:
        print("  판정: 목표에 반응은 하지만 **올바른 목표가 더 낫지 않다.**")
        print("        원-핫을 보긴 하는데 색↔위치 대응을 못 배운 상태다.")
    return {"ratio": ratio, "delta": delta, "n": used}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="act_v3_red29")
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--repo-id", default="1unasy/pill_v3")
    ap.add_argument("--tests", nargs="+", default=["1", "2", "3", "4", "5"])
    ap.add_argument("--episodes", nargs="+", type=int, default=None,
                    help="[1][2] 에 쓸 에피소드. 기본 0 4 8 14 20 24 28")
    ap.add_argument("--shifts", nargs="+", type=int, default=[-32, -16, 16, 32],
                    help="[2] 에서 top 을 밀 dy 픽셀")
    ap.add_argument("--group-split", type=int, default=12,
                    help="[3] 그룹 경계 (기본 12 = ep0~11 / ep12~)")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--goal-episodes", nargs="+", type=int, default=None,
                    help="[5] 에 쓸 에피소드. 기본은 전체 (학습 중이면 몇 개만 주면 빠르다)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"데이터셋 로드: {args.repo_id}")
    ds = LeRobotDataset(args.repo_id)
    ep_index = np.asarray(ds.hf_dataset["episode_index"])
    all_actions = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    all_states = np.asarray(ds.hf_dataset["observation.state"], dtype=np.float32)
    n_eps = int(ep_index.max()) + 1
    print(f"  {n_eps} 에피소드 / {len(all_actions)} 프레임")

    ckpt_dir = TRAIN_DIR / args.run / "checkpoints" / args.ckpt / "pretrained_model"
    if not ckpt_dir.is_dir():
        raise SystemExit(f"체크포인트 없음: {ckpt_dir}")
    print(f"정책 로드: {ckpt_dir}")
    policy, cfg, pre, post = load_policy(ckpt_dir, ds, args.device)
    n_steps = args.n_action_steps or cfg.n_action_steps

    eps = args.episodes if args.episodes is not None else [
        e for e in (0, 4, 8, 14, 20, 24, 28) if e < n_eps
    ]
    probes = []
    for ep in eps:
        gidx = np.where(ep_index == ep)[0]
        if len(gidx):
            probes += approach_frames(all_actions, gidx)
    print(f"측정 프레임: {len(probes)}개 (에피소드 {eps} 의 파지 2.0/1.0/0.33초 전)")

    if "1" in args.tests:
        test_ablation(policy, cfg, pre, post, ds, probes, all_actions)
    if "2" in args.tests:
        test_top_sensitivity(policy, cfg, pre, post, ds, probes, [0] + list(args.shifts))
    if "3" in args.tests:
        groups = {
            f"ep0~{args.group_split - 1} (옛 화각)": list(range(0, args.group_split)),
            f"ep{args.group_split}~{n_eps - 1} (지금 화각)": list(range(args.group_split, n_eps)),
        }
        test_group_bias(policy, cfg, pre, post, ds, ep_index, all_actions,
                        groups, n_steps, args.device)
    if "4" in args.tests:
        test_target_tracking(policy, cfg, pre, post, ds, ep_index,
                             all_actions, all_states, n_eps)
    if "5" in args.tests:
        test_goal_conditioning(policy, cfg, pre, post, ds, ep_index,
                               all_actions, n_eps, args.goal_episodes)

    print("\n" + "=" * 78)
    print("끝. 세 판정을 같이 읽을 것 — [1] 이 어느 카메라인지, [2] 가 그 정밀도,")
    print("[3] 이 데이터 혼입이다. 재학습은 이 결과가 원인을 지목한 뒤에 한 번만.")
    print("=" * 78)


if __name__ == "__main__":
    main()
