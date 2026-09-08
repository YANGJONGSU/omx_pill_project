#!/usr/bin/env python
"""녹화한 데이터셋에 목표 원-핫을 `observation.environment_state` 로 붙인다.

왜 필요한가
-----------
ACT 는 태스크 문자열("pick yellow pill")을 **읽지 않는다** — `modeling_act.py` 에
tokenizer 도 language encoder 도 없다. 문자열만 다르고 화면이 같은 세 색을 그냥
섞어서 학습시키면, 모델은 목표를 모른 채 "아무 알약이나 집기"를 배운다.

ACT 가 실제로 받는 목표 입력 통로는 `observation.environment_state` 하나뿐이다
(`modeling_act.py:344-357` 의 `encoder_env_state_input_proj`). 그런데 `lerobot-record`
는 이 피처를 만들지 못한다. 그래서 녹화가 끝난 뒤 여기서 `task_index` 를 보고
원-핫 벡터를 계산해 새 피처로 붙인다.

확인해 둔 사실 (lerobot v0.4.4 소스 기준)
-----------------------------------------
- 키 이름이 정확히 `observation.environment_state` 여야 `FeatureType.ENV` 로 잡힌다
  (`datasets/utils.py:726-727`). 다른 이름은 STATE 로 분류돼 관절값과 섞인다.
- ENV 는 ACT 의 `normalization_mapping` 에 없고, 없는 타입은 IDENTITY 로 처리된다
  (`processor/normalize_processor.py:305`). 즉 **정규화 없이 0/1 그대로** 들어간다.
  `add_features` 가 새 피처의 통계를 만들어주지 않지만(`dataset_tools.py:1082-1090`)
  그래서 문제가 되지 않는다.
- `add_features` 는 원본을 고치지 않고 **새 데이터셋으로 복사**한다. 원본은 그대로 남는다.

사용법
------
  python ~/omx_pill_project/make_onehot.py                    # pill_v2 -> pill_v2_onehot
  python ~/omx_pill_project/make_onehot.py --check            # 붙이지 않고 원본만 점검
  python ~/omx_pill_project/make_onehot.py --force            # 기존 출력 데이터셋을 지우고 다시

학습·평가는 **출력 데이터셋**(`pill_v2_onehot`)을 쓴다. 색 순서는 아래 COLORS 이고
`onehot_spec.json` 으로도 남기니, 추론 쪽에서는 그 파일을 읽어 맞출 것.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

# 원-핫 차원 순서. **한 번 정하면 바꾸지 말 것** — 바꾸면 이미 학습한 정책이
# 다른 색을 집는다. task_index 는 녹화 순서에 따라 달라지므로 쓰지 않는다.
COLORS = ["yellow", "red", "green"]

FEATURE = "observation.environment_state"
DEFAULT_REPO = "1unasy/pill_v2"
HF_HOME = Path.home() / ".cache/huggingface/lerobot"

# 각 차원의 이름. 추론 때 이 이름이 그대로 쓰인다 — `build_dataset_frame` 은
# 피처의 names 를 키로 관측 dict 에서 값을 꺼내므로(`datasets/utils.py:690`),
# run_policy.py 가 obs["goal_yellow"]=1.0 식으로 넣어주면 원-핫이 조립된다.
NAMES = [f"goal_{c}" for c in COLORS]


def color_of_task(task: str) -> str:
    """태스크 문자열에서 색 하나를 뽑는다. 애매하면 예외를 던진다."""
    hits = [c for c in COLORS if c in task.lower()]
    if len(hits) != 1:
        raise SystemExit(
            f"태스크 문자열에서 색을 특정할 수 없습니다: {task!r} (찾은 색: {hits})\n"
            f"녹화 시 --dataset.single_task 를 'pick <색> pill' 형식으로 맞춰야 합니다."
        )
    return hits[0]


def build_task_map(meta) -> dict[int, str]:
    """task_index -> 색. meta.tasks 는 태스크 문자열이 인덱스인 DataFrame이다."""
    return {int(row.task_index): color_of_task(task) for task, row in meta.tasks.iterrows()}


def report(ds, task_map: dict[int, str]) -> None:
    """어떤 색이 몇 에피소드/몇 프레임인지 — 붙이기 전에 눈으로 확인할 것."""
    import pandas as pd

    files = sorted((ds.root / "data").glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "task_index"]) for f in files])

    print(f"\n데이터셋 {ds.repo_id} — {ds.meta.total_episodes} 에피소드 / {len(df)} 프레임")
    print(f"{'색':<8} {'차원':>4} {'에피소드':>8} {'프레임':>8}")
    print("-" * 32)
    counted = 0
    for i, color in enumerate(COLORS):
        idxs = [ti for ti, c in task_map.items() if c == color]
        sub = df[df.task_index.isin(idxs)]
        n_ep = sub.episode_index.nunique()
        counted += n_ep
        print(f"{color:<8} {i:>4} {n_ep:>8} {len(sub):>8}")
    if counted < ds.meta.total_episodes:
        print(f"\n주의: 색이 붙지 않은 에피소드가 {ds.meta.total_episodes - counted}개 있습니다")

    missing = [c for c in COLORS if c not in task_map.values()]
    if missing:
        print(f"\n주의: 아직 안 찍은 색이 있습니다 — {', '.join(missing)}")
        print("      그 차원은 학습 내내 0이라, 나중에 그 색을 지정해도 모델이 반응하지 않습니다.")
        print("      3색을 다 찍은 뒤에 학습할 것.")


def main() -> None:
    ap = argparse.ArgumentParser(description="목표 원-핫을 environment_state 로 붙이기")
    ap.add_argument("--repo-id", default=DEFAULT_REPO, help="입력 데이터셋")
    ap.add_argument("--out-repo-id", default=None, help="출력 데이터셋 (기본: 입력 + _onehot)")
    ap.add_argument("--check", action="store_true", help="붙이지 않고 원본 점검만")
    ap.add_argument("--force", action="store_true", help="기존 출력 데이터셋을 지우고 다시 만든다")
    args = ap.parse_args()

    from lerobot.datasets.dataset_tools import add_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_repo = args.out_repo_id or f"{args.repo_id}_onehot"
    out_root = HF_HOME / out_repo

    print(f"입력 데이터셋 로드: {args.repo_id}")
    ds = LeRobotDataset(args.repo_id)

    if FEATURE in ds.meta.features:
        raise SystemExit(
            f"입력 데이터셋에 이미 {FEATURE} 가 있습니다. 원본이 아니라 이미 후처리된 "
            f"데이터셋을 가리키고 있는 것 같습니다 (--repo-id 확인)."
        )

    task_map = build_task_map(ds.meta)
    print("\ntask_index -> 색 -> 원-핫 차원")
    for ti in sorted(task_map):
        c = task_map[ti]
        vec = [1 if c == x else 0 for x in COLORS]
        print(f"  {ti} -> {c:<7} -> {vec}")

    report(ds, task_map)

    if args.check:
        print("\n--check 모드라 여기서 멈춥니다 (아무것도 쓰지 않았습니다).")
        return

    if out_root.exists():
        if not args.force:
            raise SystemExit(
                f"\n출력 데이터셋이 이미 있습니다: {out_root}\n"
                f"덮어쓰려면 --force 를 붙이세요 (그 폴더를 통째로 지웁니다)."
            )
        print(f"\n--force: 기존 출력 데이터셋을 지웁니다 -> {out_root}")
        shutil.rmtree(out_root)

    onehot = {
        ti: np.array([1.0 if task_map[ti] == c else 0.0 for c in COLORS], dtype=np.float32)
        for ti in task_map
    }

    def value_fn(row, _ep_idx, _frame_in_ep):
        return onehot[int(row["task_index"])]

    print(f"\n원-핫을 붙여 새 데이터셋을 만듭니다 -> {out_repo}")
    print("  (영상은 복사만 하므로 재인코딩 없이 몇 분이면 끝납니다)")
    new_ds = add_features(
        ds,
        {FEATURE: (value_fn, {"dtype": "float32", "shape": [len(COLORS)], "names": NAMES})},
        output_dir=out_root,
        repo_id=out_repo,
    )

    spec = {
        "colors": COLORS,
        "names": NAMES,
        "feature": FEATURE,
        "source_repo_id": args.repo_id,
        "task_index_to_color": {str(k): v for k, v in task_map.items()},
    }
    (out_root / "onehot_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    # 검증 — 몇 프레임을 실제로 읽어서 원-핫이 맞게 들어갔는지 본다
    print("\n검증")
    feat = new_ds.meta.features[FEATURE]
    print(f"  피처 등록: {FEATURE} {feat['dtype']} {feat['shape']}")

    from lerobot.datasets.utils import dataset_to_policy_features

    ft = dataset_to_policy_features(new_ds.meta.features)[FEATURE]
    print(f"  정책 피처 타입: {ft.type}  <- ENV 여야 ACT 가 목표로 받는다")

    bad = 0
    step = max(1, len(new_ds) // 200)
    for i in range(0, len(new_ds), step):
        item = new_ds[i]
        vec = np.asarray(item[FEATURE], dtype=np.float32)
        want = onehot[int(item["task_index"])]
        if vec.shape != (len(COLORS),) or not np.array_equal(vec, want):
            bad += 1
            if bad <= 3:
                print(f"  불일치 frame {i}: {vec} != {want}")
    n_checked = len(range(0, len(new_ds), step))
    print(f"  표본 {n_checked} 프레임 중 불일치 {bad}개")

    print(f"\n완료. 학습·평가는 이제 이 데이터셋을 씁니다: {out_repo}")
    print(f"  색 순서: {COLORS}  ({out_root / 'onehot_spec.json'})")
    if bad:
        raise SystemExit("불일치가 있습니다 — 학습 전에 원인을 확인하세요.")


if __name__ == "__main__":
    main()
