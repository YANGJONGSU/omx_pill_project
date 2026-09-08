#!/usr/bin/env python3
"""좌표를 지시문 안에 넣은 데이터셋을 만든다 (2026-08-20).

v3~v8 이 전부 실패했다. 언어로 색을 알려주면 모델이 화면에서 그 색을 **찾아야**
하는데, 224개로는 그 능력이 안 붙는다.

ACT 에서 실제로 성공한 방식은 색이 아니라 좌표였다:
    act_xy_224   3.675배,  상관계수 +0.981,  부호일치 100%,  실기 3색 조제 성공
    film224      1.25배    (원-핫으로는 거의 못 따라감)
어려운 부분(색 찾기)을 검출기가 대신하고, 정책은 "거기로 가서 집기" 만 배운다.

같은 논리를 SmolVLA 에 적용한다. 색 이름 대신 좌표를 문장에 쓴다:
    "pick red pill"              →  "pick the pill at 0.40 0.33"

**v5 와 다르다.** v5 는 좌표를 *맞히게* 시켰고(보조 손실), 실패했다(0.89배).
여기서는 좌표를 *입력으로 준다* — ACT xy 와 같은 구조다.

좌표는 pill_v3_auxuv 에 이미 들어 있다 (에피소드별 목표 알약의 top 화면 uv).

주의: 이러면 색 판단은 SmolVLA 가 아니라 검출기가 한다. 발표에서
"SmolVLA 가 언어로 색을 구분한다" 가 아니라 "SmolVLA + 검출기로 조제한다" 로
정확히 말해야 한다.

    python make_uvtext.py [--force]
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / ".cache/huggingface/lerobot/1unasy"
SRC, DST = ROOT / "pill_v3_auxuv", ROOT / "pill_v3_uvtext"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if not SRC.exists():
        print(f"원본이 없습니다: {SRC}")
        return 1
    if DST.exists():
        if not a.force:
            print(f"이미 있습니다: {DST}\n--force 를 붙이세요.")
            return 1
        shutil.rmtree(DST)

    sys.path.insert(0, str(Path(__file__).parent))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("1unasy/pill_v3_auxuv")
    n_ep = ds.meta.total_episodes

    # 에피소드별 목표 좌표.
    # 검출기가 실패한 프레임은 (-1,-1) 로 들어 있다 — 첫 프레임만 보면 16% 가
    # "pick the pill at -1.00 -1.00" 이라는 쓰레기 지시문이 된다.
    # 유효 프레임의 **중앙값**을 쓴다 (팔이 알약을 가리는 후반부에 흔들리므로
    # 평균보다 중앙값이 안정적이다).
    uv_col = np.asarray(ds.hf_dataset["aux_target_uv"], dtype=np.float32)
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    ep_uv, no_valid = {}, []
    for e in range(n_ep):
        v = uv_col[ep_col == e]
        ok = v[(v[:, 0] >= 0) & (v[:, 1] >= 0)]
        if len(ok) == 0:
            no_valid.append(e)
            continue
        ep_uv[e] = np.median(ok, axis=0)
    if no_valid:
        print(f"  ⚠ 유효 좌표가 하나도 없는 에피소드 {len(no_valid)}개 — 제외합니다")
    n_ep_used = len(ep_uv)
    print(f"  좌표 확보 {n_ep_used}/{n_ep} 에피소드")

    # 색도 함께 읽어 검증에 쓴다 (지시문에는 색을 **안** 쓴다).
    def ep_task(e: int) -> str:
        t = ds.meta.episodes["tasks"][e]
        t = t.as_py() if hasattr(t, "as_py") else t
        return str(t[0] if isinstance(t, (list, tuple)) else t)

    new_task = {}
    for e in sorted(ep_uv):
        u, v = ep_uv[e]
        new_task[e] = f"pick the pill at {u:.2f} {v:.2f}"

    print(f"  에피소드 {n_ep}개 → 좌표 지시문 생성")
    for e in (0, 50, 100, 150):
        if e < n_ep:
            print(f"    ep{e:3d}  {ep_task(e):26s} → {new_task[e]}")

    # 좌표가 실제로 색을 가르는지 검증 — 같은 좌표에 다른 색이 있으면 안 된다.
    by_color = {}
    for e in sorted(ep_uv):
        c = next((k for k in ("red", "green", "yellow") if k in ep_task(e)), "?")
        by_color.setdefault(c, []).append(ep_uv[e])
    print("\n  색별 좌표 분포 (겹치면 좌표만으로 색을 못 가른다)")
    for c, arr in sorted(by_color.items()):
        arr = np.stack(arr)
        print(f"    {c:8s} u {arr[:,0].min():.2f}~{arr[:,0].max():.2f}  "
              f"v {arr[:,1].min():.2f}~{arr[:,1].max():.2f}  ({len(arr)}개)")
    print("    → 겹쳐도 된다. 좌표는 '어느 색' 이 아니라 '어디' 를 말한다.")

    print(f"\n데이터셋을 복사합니다 -> 1unasy/pill_v3_uvtext")
    shutil.copytree(SRC, DST)

    # tasks.jsonl 재작성
    tasks = sorted(set(new_task.values()) | {ep_task(e) for e in range(n_ep) if e not in new_task})
    t2i = {t: i for i, t in enumerate(tasks)}
    with open(DST / "meta/tasks.jsonl", "w", encoding="utf-8") as f:
        for t, i in t2i.items():
            f.write(json.dumps({"task_index": i, "task": t}, ensure_ascii=False) + "\n")

    # data parquet 의 task_index 재작성
    n_pq = 0
    for pq in sorted((DST / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        df["task_index"] = df["episode_index"].map(
            lambda e: t2i.get(new_task.get(int(e), ep_task(int(e))), 0)).astype("int64")
        df.to_parquet(pq, index=False)
        n_pq += 1
    print(f"  data parquet {n_pq}개의 task_index 재작성")

    # meta/episodes 의 tasks 재작성
    n_ep_pq = 0
    for pq in sorted((DST / "meta/episodes").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        if "tasks" in df.columns:
            df["tasks"] = df["episode_index"].map(
                lambda e: [new_task.get(int(e), ep_task(int(e)))])
            df.to_parquet(pq, index=False)
            n_ep_pq += 1
    print(f"  meta/episodes {n_ep_pq}개의 tasks 재작성")

    print(f"\n완료 -> 1unasy/pill_v3_uvtext  (지시문 {len(tasks)}종류)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
