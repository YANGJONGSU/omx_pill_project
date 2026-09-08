#!/usr/bin/env python
"""수집한 데이터셋을 전수 검사한다 (학습 전에 반드시 한 번).

v1·v2 에서 학습을 다 돌린 뒤에야 데이터 문제를 발견해 사이클을 통째로 버린 적이
여러 번 있다. 여기서 잡는 것들:

  파지 성공   — 그리퍼 채널이 열림→닫힘으로 전이했는가 (실패 시연이 섞였는지)
  회복 시연   — 닫힘이 2회 이상 = 한 번 놓치고 다시 잡은 에피소드. 목표 색당 10개
  길이        — 너무 짧은(사고) / 너무 긴(헤맨) 에피소드
  위치 분포   — 파지 시점 pan 이 몇 자리에 몰렸는가. 파일럿 30개가 3자리에만 몰려
                모델이 위치를 찾는 대신 그 자리를 외운 전례가 있다
  화각 일관성 — 에피소드 시작 프레임끼리 비교. 수집 중간에 카메라·작업대가 밀리면
                두 화각이 섞여 학습된다 (v3 ep11↔12 에서 2.3cm 이동이 실제로 있었다)
  프레임 중복 — 카메라가 멈춘 구간. 자동복구 중에는 마지막 프레임이 반복 기록된다

사용법:
  python ~/omx_pill_project/check_dataset.py                 # 전체
  python ~/omx_pill_project/check_dataset.py --fast          # 영상 검사 생략 (빠름)
  python ~/omx_pill_project/check_dataset.py --range 29 58   # 초록만
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from validate_policy import gripper_close_frame, gripper_threshold  # noqa: E402

I_PAN, I_LIFT, I_ELBOW, I_GRIP = 0, 1, 2, 5


def close_events(grip: np.ndarray, thr: float, closed_is_low: bool) -> int:
    """열림→닫힘 전이 횟수. 2 이상이면 놓쳤다 다시 잡은 것 = 회복 시연."""
    closed = grip < thr if closed_is_low else grip > thr
    return int(np.sum(closed[1:] & ~closed[:-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="1unasy/pill_v3")
    ap.add_argument("--range", nargs=2, type=int, default=None, metavar=("FROM", "TO"))
    ap.add_argument("--fast", action="store_true", help="영상 검사 생략")
    ap.add_argument("--red-until", type=int, default=28, help="이 번호까지가 빨강")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id)
    ep_index = np.asarray(ds.hf_dataset["episode_index"])
    A = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    fps = ds.meta.fps
    n_eps = int(ep_index.max()) + 1
    lo, hi = (args.range if args.range else (0, n_eps - 1))
    eps = [e for e in range(lo, hi + 1)]
    print(f"{args.repo_id}: 전체 {n_eps}개 / {len(A)} 프레임 — ep{lo}~{hi} 검사\n")

    rows = []
    for e in eps:
        g = np.where(ep_index == e)[0]
        if len(g) == 0:
            continue
        gt = A[g]
        thr, low = gripper_threshold(gt)
        c = gripper_close_frame(gt, thr, low)
        rows.append({
            "ep": e,
            "color": "빨강" if e <= args.red_until else "초록",
            "len": len(g) / fps,
            "grasp": c is not None,
            "closes": close_events(gt[:, I_GRIP], thr, low),
            "pan": gt[c, I_PAN] if c is not None else np.nan,
            "depth": (gt[c, I_ELBOW] - gt[c, I_LIFT]) if c is not None else np.nan,
        })

    def show(label, sel):
        if not sel:
            return
        L = np.array([r["len"] for r in sel])
        ok = sum(r["grasp"] for r in sel)
        rec = [r["ep"] for r in sel if r["closes"] >= 2]
        pans = np.array([r["pan"] for r in sel if np.isfinite(r["pan"])])
        print(f"── {label} ({len(sel)}개) " + "─" * 40)
        print(f"  파지 성공     {ok}/{len(sel)}" +
              ("" if ok == len(sel) else f"   실패: {[r['ep'] for r in sel if not r['grasp']]}"))
        print(f"  회복 시연     {len(rec)}개 (목표 10)" + (f"  ep{rec}" if rec else ""))
        print(f"  길이          {L.min():.1f}~{L.max():.1f}초 (평균 {L.mean():.1f})")
        short = [r["ep"] for r in sel if r["len"] < 15]
        if short:
            print(f"    !! 15초 미만 {short} — 사고로 끊긴 것일 수 있다")
        if len(pans) > 2:
            # 5도 구간을 몇 칸이나 덮는가. 파일럿 실패작이 5/12 였고 v3 빨강은 8/12 였다
            bins = np.floor((pans - pans.min()) / 5).astype(int)
            print(f"  파지 pan      {pans.min():+.1f}~{pans.max():+.1f}, "
                  f"표준편차 {pans.std():.1f}, 5도 구간 {len(set(bins))}칸")
        print()

    show("빨강", [r for r in rows if r["color"] == "빨강"])
    show("초록", [r for r in rows if r["color"] == "초록"])

    if not args.fast:
        print("── 화각 일관성 " + "─" * 44)
        print("  에피소드 시작 프레임을 서로 비교한다 (수집 중 카메라·작업대가 밀렸는지)")
        import cv2

        key = "observation.images.top"
        refs = {}
        for e in eps:
            g = np.where(ep_index == e)[0]
            if len(g) <= 30:
                continue
            img = ds[int(g[30])][key]
            refs[e] = cv2.cvtColor((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                                   cv2.COLOR_RGB2GRAY)
        if len(refs) >= 2:
            base_e = min(refs)
            base = refs[base_e]
            # 알약·팔이 안 지나가는 고정 구조물 패치로 비교 (align 계열과 같은 영역)
            y0, y1, x0, x1 = 400, 478, 190, 400
            tpl = base[y0:y1, x0:x1]
            drift = {}
            for e, im in refs.items():
                pad = 30
                m = cv2.matchTemplate(im[y0 - pad:y1 + pad, x0 - pad:x1 + pad], tpl,
                                      cv2.TM_CCOEFF_NORMED)
                _, sc, _, loc = cv2.minMaxLoc(m)
                drift[e] = (loc[0] - pad, loc[1] - pad, sc)
            d = np.array([[v[0], v[1]] for v in drift.values()])
            mag = np.hypot(d[:, 0], d[:, 1])
            print(f"  기준 ep{base_e} 대비 최대 이동 {mag.max():.1f}px ({mag.max() * 0.65:.1f}mm)")
            bad = {e: v for e, v in drift.items() if np.hypot(v[0], v[1]) > 8}
            if bad:
                print(f"  !! 8px 넘게 밀린 에피소드 {len(bad)}개:")
                for e in sorted(bad)[:12]:
                    x, y, s = bad[e]
                    print(f"       ep{e:<3} dx {x:+3.0f} dy {y:+3.0f} ({np.hypot(x, y) * 0.65:.1f}mm)")
                print("     → 이 에피소드들은 다른 화각이다. 섞어 학습하면 깊이 매핑이 중간값으로 치우친다")
            else:
                print("  전부 8px 이내 — 수집 내내 화각이 고정됐다")
        print()

    print("── 요약 " + "─" * 51)
    red = [r for r in rows if r["color"] == "빨강"]
    grn = [r for r in rows if r["color"] == "초록"]
    print(f"  빨강 {len(red)}개 / 초록 {len(grn)}개")
    rec_all = sum(1 for r in rows if r["closes"] >= 2)
    print(f"  회복 시연 합계 {rec_all}개")
    print(f"  노랑 0개 — 3색 학습하려면 노랑 수집이 남아 있다")


if __name__ == "__main__":
    main()
