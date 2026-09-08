#!/usr/bin/env python3
"""색 원-핫을 **관측 상태에** 넣은 데이터셋을 만든다 (2026-08-21).

목표: **검출기 없이 SmolVLA 가 스스로 색을 찾게 한다.**

v10 은 성공했지만 색 판별을 HSV 검출기가 한다. 정책은 "거기로 가서 집기" 만
한다. 그건 우리가 원한 것이 아니다.

두 가지가 이미 증명돼 있다:

    film224 (ACT)   원-핫 + FiLM        **실기 3색 연속 성공**
                    검출기 없이 정책이 화면에서 색을 찾는다
                    오프라인 자기색 대비는 1.25배뿐인데 실기는 됐다

    smolvla_v10     좌표를 observation.state 로   **실기 성공**
                    언어(굳은 VLM)를 우회하고 학습되는 층으로 목표를 넣으면 된다

v11 은 둘을 합친다 — **원-핫을 observation.state 로** 넣는다.
좌표가 아니라 색 자체를 주므로, 화면에서 그 색을 찾는 일은 정책이 한다.

    v10   상태 = [관절6] + [u,v]×13        검출기가 색을 찾음
    v11   상태 = [관절6] + [원-핫3]×8+2    정책이 색을 찾음  ← 이번 목표

왜 언어가 아니라 상태인가 (v6 에서 실측):
    언어는 사전학습 VLM 을 통과한다. 신호를 79배(0.017→1.335%)로 키워도
    1.09배였다 — 무시하도록 이미 굳었다. 상태는 train_state_proj=True 라
    그래디언트가 직접 신호를 키운다.

원-핫은 pill_v3_onehot_224 의 environment_state 를 **그대로** 가져온다 —
film224 가 실기에 성공한 바로 그 값이다(goal_yellow, goal_red, goal_green).

    python make_ohstate.py [--force]
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / ".cache/huggingface/lerobot/1unasy"
SRC, DST = ROOT / "pill_v3_onehot_224", ROOT / "pill_v3_ohstate"
N_JOINT = 6
N_COLOR = 3
N_REPEAT = 8      # 6 + 3×8 = 30 (max_state_dim 32 안, 2칸은 남긴다)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--repeat", type=int, default=N_REPEAT)
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

    ds = LeRobotDataset("1unasy/pill_v3_onehot_224")
    oh = np.asarray(ds.hf_dataset["observation.environment_state"], dtype=np.float32)
    color_names = list(ds.meta.features["observation.environment_state"]["names"])
    print(f"  원-핫 {len(oh)}프레임, 색 순서 {color_names}")
    print(f"  분포 {dict(zip(color_names, oh.sum(0).astype(int).tolist()))}")

    print(f"\n데이터셋을 복사합니다 -> 1unasy/pill_v3_ohstate")
    shutil.copytree(SRC, DST)

    new_dim = N_JOINT + N_COLOR * a.repeat
    row = 0
    for pq in sorted((DST / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        n = len(df)
        st = np.stack(df["observation.state"].to_numpy())[:, :N_JOINT]
        v = oh[row:row + n]
        assert len(v) == n, f"프레임 수가 안 맞습니다 {len(v)} != {n}"
        wide = np.concatenate([st, np.tile(v, (1, a.repeat))], axis=1).astype(np.float32)
        df["observation.state"] = list(wide)
        df.to_parquet(pq, index=False)
        row += n
    print(f"  observation.state  {N_JOINT} → {new_dim} 차원 ({row}프레임)")

    info = json.load(open(DST / "meta/info.json", encoding="utf-8"))
    ft = info["features"]["observation.state"]
    ft["shape"] = [new_dim]
    base = list(ft.get("names") or [])[:N_JOINT]
    # 이름은 run_policy 가 실기에서 채울 자리다. goal_yellow0, goal_red0, ...
    ft["names"] = base + [f"{c}{i}" for i in range(a.repeat) for c in color_names]
    json.dump(info, open(DST / "meta/info.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  info.json 갱신 — 색 칸 이름 예: {ft['names'][6:9]}")

    st_path = DST / "meta/stats.json"
    if st_path.exists():
        stats = json.load(open(st_path, encoding="utf-8"))
        allst = np.concatenate([
            np.stack(pd.read_parquet(pq)["observation.state"].to_numpy())
            for pq in sorted((DST / "data").rglob("*.parquet"))])
        s = stats["observation.state"]
        s["mean"] = allst.mean(0).tolist()
        s["std"] = (allst.std(0) + 1e-8).tolist()
        s["min"] = allst.min(0).tolist()
        s["max"] = allst.max(0).tolist()
        json.dump(stats, open(st_path, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  stats.json 재계산 — 색 칸 std {allst[:,6:9].std(0).round(3).tolist()}")

    print(f"\n완료 -> 1unasy/pill_v3_ohstate  (state {new_dim}차원)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
