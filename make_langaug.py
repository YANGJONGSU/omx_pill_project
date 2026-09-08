#!/usr/bin/env python
"""지시문을 다양화한 데이터셋을 만든다 — 촬영 없이 (2026-08-19).

왜 필요한가
-----------
v3·v4·v5 가 전부 실패했다(지시문 반응 1.10/1.09/0.89배, 기준선 1.02).
prefix 안을 직접 재보니 지시문을 red→green 으로 바꿔도 **0.017%** 밖에 안
변한다. 정책이 언어를 무시해도 손실이 안 오르니 실제로 무시한다.

이건 우리만의 문제가 아니라 VLA 에서 잘 알려진 현상이다.

  "Information Collapse" — 지시문이 이미지만으로 예측 가능하면, 모델은
  언어가 추가 정보를 안 준다고 학습하고 최적화 과정에서 버린다.
  원인은 modality imbalance: 시각·행동은 다양한데 언어는 템플릿 몇 개뿐이다.
  (labelstud.io/blog/why-your-vla-model-ignores-language-and-the-cheapest-fix-in-robotics)

  LIBERO-PRO: 90% 성공률이 지시문을 흔들면 무너지고, 의미 없는 문장에도
  출력이 그대로였다. 우리 관찰(엉뚱한 문장이 색 변경보다 예측을 더 바꿈)과 같다.

보고된 해법과 효과 (전부 새 촬영 없이 지시문만 다시 씀):
  LangGap   같은 궤적으로 0% → 90%
  CAST      +27%p
  FineVLA   색 속성 +18점. 지시문 9.3 → 96.8 단어

핵심 설계
---------
① 세 색이 **같은 문장 틀**을 공유한다.
   빨강만 길게 쓰면 문장 길이로 색을 맞혀버려 또 우회한다. 틀을 공유하고
   색 단어 하나만 다르게 해야 그걸 볼 수밖에 없다.

② 원래 지시문을 절반 남긴다 (FineVLA 실측: FG:Raw = 1:1 이 최적, 역U자).
   전부 긴 문장으로 바꾸면 실기에서 "pick red pill" 을 못 알아듣는다.
   학습과 실행의 문장 형식이 어긋나면 소용없다.

③ 반사실 지시문을 넣는다 — "pick red, not green or yellow".
   다른 색을 집으면 명시적으로 틀린 것이 되므로 색을 무시할 수 없다.

④ (선택) 위치 표현. --with-pos 를 주면 xy_labels.json 의 좌표로
   "on the left side" 같은 말을 붙인다. 좌표가 있는 188개에만 적용하고
   나머지는 위치 없는 표현을 쓴다 — 틀린 위치를 가르치지 않기 위해서다.

  python make_langaug.py                 # 1unasy/pill_v3_langaug 생성
  python make_langaug.py --with-pos      # 위치 표현까지
  python make_langaug.py --force         # 있으면 지우고 새로
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent
HF_HOME = Path.home() / ".cache/huggingface/lerobot"
COLORS = ["red", "green", "yellow"]

# 세 색이 공유하는 틀. {c} 자리에만 색이 들어간다.
TEMPLATES = [
    "pick {c} pill",                                   # 원래 형식 (실기에서 쓰는 것)
    "grab the {c} capsule from the tray",
    "take the {c} one and put it in the bottle",
    "pick up the {c} pill",
    "move the {c} capsule to the bottle",
]
# 반사실 — 무엇이 아닌지도 말한다
COUNTERFACTUAL = "pick the {c} pill, not the {o1} or the {o2}"
# 위치 표현 (--with-pos 일 때만, 좌표가 있는 에피소드에만)
POSITIONAL = "pick the {c} pill {pos}"


def pos_phrase(u: float, v: float) -> str:
    lr = "on the left" if u < 0.4 else ("on the right" if u > 0.6 else "in the middle")
    fb = "near the top" if v < 0.4 else ("near the bottom" if v > 0.6 else "")
    return f"{lr} {fb}".strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="1unasy/pill_v3")
    ap.add_argument("--out-repo-id", default=None)
    ap.add_argument("--labels", default=str(PROJECT / "xy_labels.json"))
    ap.add_argument("--with-pos", action="store_true", help="좌표 기반 위치 표현 추가")
    ap.add_argument("--raw-ratio", type=float, default=0.5,
                    help="원래 지시문을 남길 비율 (FineVLA 실측 최적 1:1 = 0.5)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_repo = a.out_repo_id or f"{a.repo_id}_langaug"
    src_root = HF_HOME / a.repo_id
    out_root = HF_HOME / out_repo

    if out_root.exists():
        if not a.force:
            raise SystemExit(f"\n이미 있습니다: {out_root}\n--force 를 붙이세요.")
        print(f"--force: 기존 출력을 지웁니다 -> {out_root}")
        shutil.rmtree(out_root)

    ds = LeRobotDataset(a.repo_id)
    n_ep = ds.meta.total_episodes

    def color_of(e: int) -> str | None:
        t = ds.meta.episodes["tasks"][e]
        t = t.as_py() if hasattr(t, "as_py") else t
        t = t[0] if isinstance(t, (list, tuple)) else t
        return next((c for c in COLORS if c in str(t)), None)

    labels = {}
    if a.with_pos:
        lab = json.loads(Path(a.labels).read_text(encoding="utf-8"))
        labels = {int(k): v for k, v in lab["labels"].items()}
        print(f"좌표 라벨 {len(labels)}개 로드 (위치 표현용)")

    rng = random.Random(a.seed)

    # ── 에피소드마다 지시문 하나 고르기 ────────────────────────────────
    ep_task: dict[int, str] = {}
    n_raw = 0
    for e in range(n_ep):
        c = color_of(e)
        if c is None:
            continue
        if rng.random() < a.raw_ratio:
            ep_task[e] = TEMPLATES[0].format(c=c)      # 원래 형식 그대로
            n_raw += 1
            continue
        others = [x for x in COLORS if x != c]
        pool = [t.format(c=c) for t in TEMPLATES[1:]]
        pool.append(COUNTERFACTUAL.format(c=c, o1=others[0], o2=others[1]))
        if a.with_pos and e in labels:
            d = labels[e]
            pool.append(POSITIONAL.format(c=c, pos=pos_phrase(d["u"], d["v"])))
        ep_task[e] = rng.choice(pool)

    tasks = sorted(set(ep_task.values()))
    print(f"\n에피소드 {len(ep_task)}개 → 지시문 {len(tasks)}종류")
    print(f"  원래 형식 유지 {n_raw}개 ({n_raw / len(ep_task) * 100:.0f}%), "
          f"다양화 {len(ep_task) - n_raw}개")
    print("\n  만들어진 지시문:")
    for t in tasks:
        k = sum(1 for v in ep_task.values() if v == t)
        print(f"    {k:3d}개  {t}")

    # 색마다 같은 틀을 쓰는지 검증 — 문장 길이로 색을 맞힐 수 있으면 실패다
    print("\n  검증 — 색별 지시문 길이 (비슷해야 색 단어를 봐야만 한다)")
    for c in COLORS:
        ls = [len(v.split()) for e, v in ep_task.items() if color_of(e) == c]
        print(f"    {c:7s} 평균 {sum(ls) / len(ls):.2f} 단어  (에피소드 {len(ls)}개)")

    # ── 데이터셋 복사 후 task 재작성 ───────────────────────────────────
    print(f"\n데이터셋을 복사합니다 -> {out_repo}")
    shutil.copytree(src_root, out_root)

    task_to_idx = {t: i for i, t in enumerate(tasks)}
    pd.DataFrame({"task_index": list(task_to_idx.values())},
                 index=pd.Index(list(task_to_idx.keys()), name="task")
                 ).to_parquet(out_root / "meta" / "tasks.parquet")

    info = json.loads((out_root / "meta" / "info.json").read_text(encoding="utf-8"))
    info["total_tasks"] = len(tasks)
    (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

    ep_to_idx = {e: task_to_idx[t] for e, t in ep_task.items()}
    n_files = 0
    for pq in sorted((out_root / "data").glob("*/*.parquet")):
        df = pd.read_parquet(pq)
        df["task_index"] = df["episode_index"].map(ep_to_idx).fillna(0).astype(df["task_index"].dtype)
        df.to_parquet(pq, index=False)
        n_files += 1
    print(f"  data parquet {n_files}개의 task_index 재작성")

    # 에피소드 메타의 tasks 컬럼도 맞춘다
    for pq in sorted((out_root / "meta" / "episodes").glob("*/*.parquet")):
        df = pd.read_parquet(pq)
        if "tasks" in df.columns:
            df["tasks"] = df["episode_index"].map(lambda e: [ep_task.get(int(e), "pick red pill")])
            df.to_parquet(pq, index=False)
    print("  meta/episodes 의 tasks 재작성")

    print("\n검증")
    new = LeRobotDataset(out_repo)
    print(f"  총 에피소드 {new.meta.total_episodes}개, 지시문 {len(new.meta.tasks)}종류")
    for e in (0, 50, 100, 150):
        t = new.meta.episodes["tasks"][e]
        t = t.as_py() if hasattr(t, "as_py") else t
        print(f"    ep{e:3d}  {t}")
    print(f"\n완료 -> {out_repo}")


if __name__ == "__main__":
    main()
