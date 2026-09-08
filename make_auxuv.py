#!/usr/bin/env python
"""SmolVLA 보조 손실용 목표 좌표 데이터셋을 만든다 (2026-08-19).

왜 필요한가
-----------
smolvla_v3(사전학습, VLM 동결)를 실기에서 재보니 지시문을 바꿔도 같은 곳으로
갔다. 그런데 언어 임베딩은 색을 구분했고(red↔green 코사인 0.35~0.48), 시각
인코더도 알약 색을 100% 구분했다. 재료는 다 있는데 "language → 어느 알약"
연결이 안 만들어진 것이다 — 그 연결은 액션 전문가의 교차 어텐션 하나에만
걸려 있었다(train_expert_only=true).

이 스크립트는 **정답을 하나 더 준다**: "그 알약이 화면 어디 있어?" 를 훈련
중에만 추가로 묻는다. 답을 안 보면 위치를 못 맞히므로 언어를 볼 수밖에
없다. 추론(실행)에는 전혀 관여하지 않는다 — lerobot 쪽 패치
(configuration_smolvla.py, modeling_smolvla.py) 가 aux_position_loss_weight=0
이면 이 컬럼을 아예 읽지 않는다.

라벨 출처
---------
xy_labels.json 을 그대로 쓴다 — 좌표 조건화(act_xy_224)용으로 이미 만들어
검증된 것이다(집기 시연 192개 중 188개, 97.9%). 정지 시연 32개와 검출 실패
4개는 **(-1, -1)** 로 채운다(행동 손실은 그대로 배우고 위치 손실만 안 걸린다).
u, v 는 0~1 로 정규화돼 있어 -1 은 절대 정답과 안 겹치는 안전한 표식이다.

별도의 valid 스칼라 컬럼을 안 쓰는 이유: lerobot 0.4.4 의 add_features 가
크기 1 배열을 자동으로 파이썬 스칼라로 접는데(dataset_tools.py:998), 그
스칼라를 다시 arrow 스키마에 쓰는 경로에 버그가 있어 shape=[1] 도 shape=[]
도 둘 다 실패한다(검증됨, 2026-08-19). sentinel 값이면 이 경로를 아예 안
탄다 — 검증 안 된 라이브러리 경로에 기대지 않는다.

  python make_auxuv.py                    # 1unasy/pill_v3_auxuv 생성
  python make_auxuv.py --force            # 있으면 지우고 새로
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
HF_HOME = Path.home() / ".cache/huggingface/lerobot"

FEATURE_UV = "aux_target_uv"
INVALID = -1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="1unasy/pill_v3")
    ap.add_argument("--out-repo-id", default=None)
    ap.add_argument("--labels", default=str(PROJECT / "xy_labels.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.dataset_tools import add_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_repo = args.out_repo_id or f"{args.repo_id}_auxuv"
    out_root = HF_HOME / out_repo

    lab = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    labels = {int(k): v for k, v in lab["labels"].items()}
    print(f"좌표 라벨 {len(labels)}개 로드 ({args.labels})")

    ds = LeRobotDataset(args.repo_id)
    fr = list(ds.meta.episodes["dataset_from_index"])
    to = list(ds.meta.episodes["dataset_to_index"])
    n_ep = ds.meta.total_episodes

    n_valid = sum(1 for e in range(n_ep) if e in labels)
    n_stop = sum(1 for e in range(n_ep) if (int(to[e]) - int(fr[e])) == 449)
    print(f"전체 에피소드 {n_ep}개 — 좌표 있음 {n_valid}개 / 정지 시연 {n_stop}개 / "
          f"나머지(검출 실패) {n_ep - n_valid - n_stop}개")

    if out_root.exists():
        if not args.force:
            raise SystemExit(f"\n출력 데이터셋이 이미 있습니다: {out_root}\n"
                              f"덮어쓰려면 --force 를 붙이세요.")
        print(f"--force: 기존 출력을 지웁니다 -> {out_root}")
        shutil.rmtree(out_root)

    def uv_fn(_row, ep_idx, _frame_in_ep):
        d = labels.get(int(ep_idx))
        if d is None:
            return np.array([INVALID, INVALID], dtype=np.float32)
        return np.array([d["u"], d["v"]], dtype=np.float32)

    print(f"\n좌표 정답을 붙여 새 데이터셋을 만듭니다 -> {out_repo}")
    print("  (영상은 복사만 하므로 재인코딩 없이 몇 분이면 끝납니다)")
    new_ds = add_features(
        ds,
        {FEATURE_UV: (uv_fn, {"dtype": "float32", "shape": [2], "names": ["u", "v"]})},
        output_dir=out_root,
        repo_id=out_repo,
    )

    spec = {
        "source_repo_id": args.repo_id,
        "labels_file": str(Path(args.labels).resolve()),
        "n_valid": n_valid, "n_stop": n_stop, "n_total": n_ep,
        "feature_uv": FEATURE_UV, "invalid_sentinel": INVALID,
    }
    (out_root / "auxuv_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    print("\n검증")
    bad = 0
    step = max(1, len(new_ds) // 200)
    checked = 0
    n_true_valid = 0
    for i in range(0, len(new_ds), step):
        item = new_ds[i]
        uv = np.asarray(item[FEATURE_UV], dtype=np.float32)
        checked += 1
        n_true_valid += bool(uv[0] >= 0)
        if uv.shape != (2,):
            bad += 1
            if bad <= 3:
                print(f"  형태 이상 frame {i}: {uv.shape}")
    print(f"  표본 {checked} 프레임 중 형태 이상 {bad}개, 유효 좌표 프레임 {n_true_valid}개")
    print(f"\n완료 -> {out_repo}")
    print(f"  훈련에서: --dataset.repo_id={out_repo} --policy.aux_position_loss_weight=0.3")


if __name__ == "__main__":
    main()
