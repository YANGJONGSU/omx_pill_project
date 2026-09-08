#!/usr/bin/env python
"""학습 영상에서 화각·선명도·밝기 기준을 뽑는다 (PLAN_v3 §4, WORKLOG §11.4).

**녹화 전에 저장한 기준을 쓰면 안 된다.** v2 에서 `preflight_ref/top.png` 가 실제 학습
화각과 6.4px 달랐고, 그걸 기준으로 평가 1차를 돌려 성적을 통째로 버렸다. 기준은 반드시
**학습에 실제로 들어간 영상**에서 뽑는다.

v3 에는 이 기준이 아예 없었다. 그래서 ep11↔12 사이에 카메라-작업대가 2.3cm 움직인 것을
녹화 중에 아무도 못 잡았고, 하루 뒤 사후 분석에서야 나왔다.

만드는 것 (`preflight_ref/v3/`):
  top_train.png / wrist_train.png  — 에피소드 시작 프레임들의 **중앙값 영상**
                                     (팔·알약은 매번 다르니 중앙값을 내면 고정 배경만 남는다)
  meta.json — 카메라별 밝기 평균, **선명도(라플라시안 분산) 범위**, 뽑은 에피소드 수

선명도 범위가 핵심이다. 그게 있어야 나중에 "렌즈 초점이 풀렸다"를 숫자로 잡는다 —
v2 때 wrist 가 학습 범위(1287~5369)의 15분의 1인 184 로 흐려진 채 평가 2회를 돌렸다.

사용법:
  python ~/omx_pill_project/make_ref.py                       # pill_v3 → preflight_ref/v3/
  python ~/omx_pill_project/make_ref.py --repo-id 1unasy/pill_v2 --out v2b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT = Path("/home/user/omx_pill_project")
CAMS = {"top": "observation.images.top", "wrist": "observation.images.wrist"}


def sharpness(bgr: np.ndarray) -> float:
    """라플라시안 분산. focus_wrist.py 와 같은 척도라 숫자를 그대로 비교할 수 있다."""
    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="1unasy/pill_v3")
    ap.add_argument("--out", default="v3", help="preflight_ref/ 아래 폴더 이름")
    ap.add_argument("--at", type=float, default=1.0,
                    help="에피소드 시작 몇 초 지점을 쓸지 (기본 1.0 — 첫 프레임은 워밍업 JPEG 손상이 있다)")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id)
    ep_index = np.asarray(ds.hf_dataset["episode_index"])
    n_eps = int(ep_index.max()) + 1
    fps = ds.meta.fps
    off = int(args.at * fps)
    print(f"{args.repo_id}: {n_eps} 에피소드 / {fps} fps — 시작 +{args.at}초 프레임 사용")

    out_dir = PROJECT / "preflight_ref" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {}
    for name, key in CAMS.items():
        frames, sharps = [], []
        for ep in range(n_eps):
            gidx = np.where(ep_index == ep)[0]
            if len(gidx) <= off:
                continue
            img = ds[int(gidx[off])][key]  # CHW float [0,1]
            bgr = cv2.cvtColor(
                (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            frames.append(bgr)
            sharps.append(sharpness(bgr))

        stack = np.stack(frames)
        # 중앙값: 팔·알약처럼 매 에피소드 달라지는 것은 지워지고 고정 배경만 남는다.
        # 평균을 쓰면 움직이는 것이 흐릿하게 번져 템플릿 매칭 점수가 떨어진다.
        ref = np.median(stack, axis=0).astype(np.uint8)
        path = out_dir / f"{name}_train.png"
        cv2.imwrite(str(path), ref)

        s = np.array(sharps)
        meta[name] = {
            "mean": float(stack.reshape(len(stack), -1).mean()),
            "sharpness_min": float(s.min()),
            "sharpness_med": float(np.median(s)),
            "sharpness_max": float(s.max()),
            "n_episodes": len(frames),
        }
        print(f"  {name}: {len(frames)}장 → {path.name}")
        print(f"        밝기 평균 {meta[name]['mean']:.1f}  "
              f"선명도 {s.min():.0f} ~ {s.max():.0f} (중앙값 {np.median(s):.0f})")

    meta["_source"] = {"repo_id": args.repo_id, "at_sec": args.at}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"\n저장: {out_dir}")
    print("이제 실기 전에 `python check_live.py` 로 지금 카메라와 대조할 수 있다.")


if __name__ == "__main__":
    main()
