#!/usr/bin/env python
"""목표를 **원-핫 대신 좌표**로 붙인 데이터셋을 만든다 (2026-08-12).

배경
----
원-핫 목표 조건화는 네 가지 구성(ACT 토큰 / ACT+FiLM / CVAE 제거 / Diffusion)에서
전부 기준선(1.02)에 머물렀다. 원인은 주입 강도가 아니다 — Diffusion 은 목표를 전
계층에 넣고도 0.97 이었다(Δ 는 오히려 ACT 보다 컸다). 즉 신호는 도달하는데
**그것이 무엇을 가리키는지** 배우지 못한다.

목표를 기호가 아니라 좌표로 주면 그 문제가 사라진다. 목표가 출력과 같은 공간에
있으므로 무시하는 순간 손실이 오른다 — 지름길이 존재하지 않는다.

    원-핫   [1,0,0]        목표 값 3종류 (에피소드를 224개 찍어도 3종류)
    좌표    [0.62, 0.31]   목표 값 188종류 — 랜덤 배치가 여기서 신호가 된다

라벨은 `make_xy_labels.py` 가 탑뷰 첫 프레임에서 HSV 로 검출한 목표 알약의 위치다.

주의
----
- 라벨이 없는 에피소드(검출 실패 4개 + 정지 시연 32개)에는 [0,0] 이 들어간다.
  **학습 시 `--dataset.episodes` 로 반드시 제외할 것.** 목록은 xy_episodes.json 에 쓴다.
- 정지 시연을 뺀 이유: 목표가 트레이에 없으면 좌표 자체가 존재하지 않는다.
  그 상황은 학습이 아니라 검출기가 처리한다 (덩어리 0개 → 정책을 호출하지 않는다).

    python make_xy_dataset.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
FEATURE = "observation.environment_state"      # 이 이름이어야 FeatureType.ENV 로 잡힌다
NAMES = ["goal_u", "goal_v"]                   # 추론 시 obs["goal_u"], obs["goal_v"] 로 주입
HF_HOME = Path.home() / ".cache/huggingface/lerobot"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="1unasy/pill_v3")
    ap.add_argument("--out-repo-id", default="1unasy/pill_v3_xy")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.dataset_tools import add_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    lab = json.loads((PROJECT / "xy_labels.json").read_text())["labels"]
    goal = {int(k): np.array([v["u"], v["v"]], dtype=np.float32) for k, v in lab.items()}
    zero = np.zeros(2, dtype=np.float32)
    print(f"라벨 {len(goal)}개 로드")

    ds = LeRobotDataset(args.repo_id)
    if FEATURE in ds.meta.features:
        raise SystemExit(f"입력에 이미 {FEATURE} 가 있습니다 — 원본을 지정하세요.")

    out_root = HF_HOME / args.out_repo_id
    if out_root.exists():
        if not args.force:
            raise SystemExit(f"이미 있습니다: {out_root}  (--force 로 덮어쓰기)")
        shutil.rmtree(out_root)

    missing = sorted(set(range(ds.meta.total_episodes)) - set(goal))
    print(f"라벨 없는 에피소드 {len(missing)}개 → [0,0] 으로 채우고 학습에서 제외합니다")

    def value_fn(_row, ep_idx, _frame_in_ep):
        return goal.get(int(ep_idx), zero)

    print(f"\n좌표 목표를 붙입니다 → {args.out_repo_id}")
    new_ds = add_features(
        ds,
        {FEATURE: (value_fn, {"dtype": "float32", "shape": [2], "names": NAMES})},
        output_dir=out_root,
        repo_id=args.out_repo_id,
    )

    eps = sorted(goal)
    (PROJECT / "xy_episodes.json").write_text(json.dumps(
        {"episodes": eps, "n": len(eps), "excluded": missing,
         "feature": FEATURE, "names": NAMES}, indent=1))
    (out_root / "xy_spec.json").write_text(json.dumps(
        {"feature": FEATURE, "names": NAMES, "source_repo_id": args.repo_id,
         "note": "목표 = 탑뷰 픽셀 좌표 / 640, / 480 로 정규화"}, indent=2))

    # ── 검증: 실제로 읽어서 라벨과 일치하는지 ──────────────────────────────
    print("\n검증")
    from lerobot.datasets.utils import dataset_to_policy_features
    ft = dataset_to_policy_features(new_ds.meta.features)[FEATURE]
    print(f"  피처 타입 {ft.type}  ← ENV 여야 목표로 들어간다")

    fr = list(new_ds.meta.episodes["dataset_from_index"])
    to = list(new_ds.meta.episodes["dataset_to_index"])
    bad = 0
    for e in eps[:: max(1, len(eps) // 40)][:40]:
        for i in (int(fr[e]), (int(fr[e]) + int(to[e])) // 2, int(to[e]) - 1):
            v = np.asarray(new_ds[i][FEATURE], dtype=np.float32)
            if not np.allclose(v, goal[e], atol=1e-5):
                bad += 1
                if bad <= 3:
                    print(f"  불일치 ep{e} frame{i}: {v} != {goal[e]}")
    print(f"  표본 확인 완료 — 불일치 {bad}개")
    if bad:
        raise SystemExit("불일치가 있습니다. 학습 전에 확인하세요.")

    print(f"\n완료: {args.out_repo_id}   학습 대상 {len(eps)} 에피소드")


if __name__ == "__main__":
    main()
