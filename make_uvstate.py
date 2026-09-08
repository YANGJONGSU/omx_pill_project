#!/usr/bin/env python3
"""목표 좌표를 **관측 상태에** 넣은 데이터셋을 만든다 (2026-08-20).

v3~v8 이 전부 실패했다. 전부 목표를 **언어로** 줬다.

    v3 1.10 / v4 1.09 / v5 0.89 / v6 1.09 / v7 0.96 / v8 2.02(동작 망가짐)

실기에서 성공한 유일한 것은 좌표를 **숫자로** 받는 ACT 였다.

    act_xy_224   3.675배, 상관 +0.981, 자기색오차 0.833, 실기 3색 연속 성공
    observation.environment_state = (goal_u, goal_v)   ← 언어를 안 거친다

SmolVLA 는 environment_state 를 읽지 않는다(OBS_STATE 와 이미지만 본다).
그래서 좌표를 observation.state 에 접어 넣는다. 상태는 32차원까지 채워지고
로봇 관절은 6개뿐이라 26칸이 비어 있다.

**왜 이번엔 다른가 — 계측으로 확인한 것**

    언어로 색 바꾸기            prefix 변화 0.079%   ← v3~v8 이 쓴 경로
    좌표를 상태 2칸에            prefix 변화 0.004%
    좌표를 26칸에 반복           prefix 변화 0.013%  (같은 척도에서 3.5배)

초기값만 보면 상태 경로가 더 약해 보인다. 그런데 결정적 차이가 있다:

    train_state_proj = True     좌표가 지나는 층이 **학습된다**

언어는 사전학습된 VLM 을 통과한다 — 모델이 무시하도록 이미 굳어 있고,
v6 에서 신호를 79배로 키워도(0.017%→1.335%) 1.09배에 그쳤다. 반면 state_proj
는 그래디언트가 직접 모양을 잡는 선형층이라 학습이 신호를 키운다.
그리고 act_xy_224 가 **같은 224개 데이터로** 3.675배를 냈다 — 이 매핑이
학습 가능하다는 증거다.

26칸 반복은 공짜로 그래디언트 경로를 13배 늘린다. 정규화가 척도를 맞추므로
값 자체를 키우지 않는다 (키우면 정규화가 도로 줄인다).

지시문은 원래대로 "pick red pill" 을 쓴다 — 실기에서 색 이름으로 부를 수
있어야 하고, 언어는 어차피 무시되므로 해가 없다.

    python make_uvstate.py [--force]
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / ".cache/huggingface/lerobot/1unasy"
SRC, DST = ROOT / "pill_v3_xy", ROOT / "pill_v3_uvstate"
N_REPEAT = 13          # 26칸 = 2좌표 × 13
N_JOINT = 6


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

    ds = LeRobotDataset("1unasy/pill_v3_xy")

    # **되는 데이터를 그대로 쓴다.** act_xy_224 가 3.675배를 낸 좌표가
    # pill_v3_xy 의 observation.environment_state 다. 검출 실패 18,945프레임
    # (7.7%)은 거기서 (0,0) 으로 표시돼 있고, 그 표기 그대로 학습해서 성공했다.
    # 중앙값으로 메우는 등 손대지 않는다 — 되는 것과 달라지면 그 차이가
    # 실패 원인이 됐을 때 가려낼 수 없다.
    filled = np.asarray(ds.hf_dataset["observation.environment_state"], dtype=np.float32)
    z = (filled[:, 0] == 0) & (filled[:, 1] == 0)
    print(f"  좌표 {len(filled)}프레임 (검출실패 {int(z.sum())}프레임은 (0,0) 그대로)")
    print(f"  좌표 범위  u {filled[:,0].min():.2f}~{filled[:,0].max():.2f}"
          f"   v {filled[:,1].min():.2f}~{filled[:,1].max():.2f}")

    print(f"\n데이터셋을 복사합니다 -> 1unasy/pill_v3_uvstate")
    shutil.copytree(SRC, DST)

    # ── observation.state 를 6 → 6+2×repeat 으로 늘린다 ────────────────
    new_dim = N_JOINT + 2 * a.repeat
    row = 0
    for pq in sorted((DST / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        n = len(df)
        st = np.stack(df["observation.state"].to_numpy())[:, :N_JOINT]
        uv = filled[row:row + n]
        assert len(uv) == n, f"프레임 수가 안 맞습니다 {len(uv)} != {n}"
        wide = np.concatenate([st, np.tile(uv, (1, a.repeat))], axis=1).astype(np.float32)
        df["observation.state"] = list(wide)
        df.to_parquet(pq, index=False)
        row += n
    print(f"  observation.state  {N_JOINT} → {new_dim} 차원 ({row}프레임)")

    # info.json 의 shape / names 갱신
    info = json.load(open(DST / "meta/info.json", encoding="utf-8"))
    ft = info["features"]["observation.state"]
    ft["shape"] = [new_dim]
    base_names = list(ft.get("names") or [])[:N_JOINT]
    ft["names"] = base_names + [f"goal_{c}{i}" for i in range(a.repeat) for c in ("u", "v")]
    json.dump(info, open(DST / "meta/info.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  info.json 갱신")

    # stats 재계산 — 정규화가 좌표 척도를 관절과 맞춘다. 옛 6차원 통계를
    # 그대로 두면 차원이 안 맞아 학습이 죽는다.
    st_path = DST / "meta/stats.json"
    if st_path.exists():
        stats = json.load(open(st_path, encoding="utf-8"))
        allst = []
        for pq in sorted((DST / "data").rglob("*.parquet")):
            allst.append(np.stack(pd.read_parquet(pq)["observation.state"].to_numpy()))
        allst = np.concatenate(allst)
        s = stats["observation.state"]
        s["mean"] = allst.mean(0).tolist()
        s["std"] = (allst.std(0) + 1e-8).tolist()
        s["min"] = allst.min(0).tolist()
        s["max"] = allst.max(0).tolist()
        json.dump(stats, open(st_path, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  stats.json 재계산 — 좌표 std {allst[:,6:8].std(0).round(3).tolist()}")

    print(f"\n완료 -> 1unasy/pill_v3_uvstate  (state {new_dim}차원)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
