#!/usr/bin/env python3
"""3색 연속 조제 평가용 배치표를 만든다 (2026-08-24).

`plan_positions.py` 는 촬영용이라 **목표 하나**를 뽑는다. 여기서는 정책 비교가
목적이므로 **알약 3색이 항상 트레이에 있고, 처방 순서만 다른** 배치를 만든다.

**왜 seed 를 고정하는가**

정책을 공정하게 비교하려면 두 조건이 동시에 필요하다.

    무작위여야 한다     특정 정책에 유리한 자리에 놓는 편향을 없앤다
    동일해야 한다       배치가 매번 다르면 정책 차이인지 운 차이인지 못 가른다

그래서 **seed 고정 난수**로 표를 한 번 뽑고, **모든 정책이 같은 표를 쓴다.**
표를 다시 뽑고 싶으면 같은 seed 로 이 스크립트를 돌리면 된다.

    python plan_eval3.py 10 --seed 8241
    python plan_eval3.py 10 --seed 8241 --overlay    # top 화면에 겹쳐 확인

배치는 5×4 격자에서 **서로 다른 칸**을 뽑는다 — 같은 칸에 두 알약이 겹치면
검출기가 하나로 보고, 그건 정책 성능과 무관한 실패가 된다.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
COLORS = ["red", "green", "yellow"]
KOR = {"red": "빨강", "green": "초록", "yellow": "노랑"}
COLS, ROWS = 5, 4                      # 격자
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("count", type=int, nargs="?", default=10, help="시행 횟수")
    ap.add_argument("--seed", type=int, default=None, help="같은 표를 다시 뽑으려면 지정")
    ap.add_argument("--out", default=str(PROJECT / "eval3_placements.md"))
    a = ap.parse_args()

    seed = a.seed if a.seed is not None else random.randint(1000, 9999)
    rng = random.Random(seed)

    area = json.loads((PROJECT / "area.json").read_text())

    rows = []
    for i in range(a.count):
        # 서로 다른 칸 3개. 같은 칸이면 알약이 겹쳐 검출기가 하나로 본다.
        cells = rng.sample(range(COLS * ROWS), 3)
        order = COLORS[:]
        rng.shuffle(order)                       # 처방 순서도 무작위
        placed = {}
        for c, cell in zip(COLORS, cells):
            cx, cy = cell % COLS, cell // COLS
            placed[c] = {
                "칸": f"{chr(ord('A') + cy)}{cx + 1}",
                "좌우": round((cx + 0.5) / COLS * 100),
                "전후": round((cy + 0.5) / ROWS * 100),
                "각도": rng.choice(ANGLES),
            }
        rows.append({"i": i, "order": order, "p": placed})

    L = []
    L.append(f"# 3색 연속 조제 평가 배치표 — {a.count}회\n")
    L.append(f"- seed `{seed}` (같은 표를 다시 뽑으려면 `--seed {seed}`)")
    L.append(f"- 배치 영역 `{area}` · 격자 {COLS}×{ROWS} ({COLS*ROWS}칸)")
    L.append("- 좌우/전후 = 영역을 0~100%로 본 위치, 각도 = 색깔 있는 끝이 향하는 방향")
    L.append("- **알약 3색이 항상 트레이에 있습니다.** 약통은 비운 상태로 시작합니다.")
    L.append("- **모든 정책이 이 표를 그대로 씁니다.** 배치를 바꾸면 비교가 성립하지 않습니다.\n")

    L.append("## 판정 표기\n")
    L.append("| 표기 | 뜻 |")
    L.append("|---|---|")
    L.append("| `O` | 1차 시도에 성공 (집어서 약통에 넣음) |")
    L.append("| `O2` | 두 번째 시도에 성공 |")
    L.append("| `x-a` | 위치를 못 찾음 — 엉뚱한 데로 가거나 안 움직임 |")
    L.append("| `x-b` | 갔는데 빗나감 — 파지 실패 |")
    L.append("| `x-c` | 집었는데 놓기 실패 — 약통 밖에 떨어뜨림 |")
    L.append("| `x-d` | **다른 색을 집음** — 목표 조건화 실패 |\n")

    L.append("## 배치와 판정\n")
    L.append("| # | 처방 순서 | 빨강 | 초록 | 노랑 | 1번째 | 2번째 | 3번째 | 메모 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        p = r["p"]
        seq = " → ".join(KOR[c] for c in r["order"])
        cell = lambda c: f"{p[c]['칸']} {p[c]['각도']}°"
        L.append(f"| {r['i']} | {seq} | {cell('red')} | {cell('green')} | {cell('yellow')} | | | | |")

    L.append("\n## 정책별 집계\n")
    L.append("| 정책 | 1차 성공 | 재시도 포함 | 다른 색 집음 | 비고 |")
    L.append("|---|---|---|---|---|")
    for pol, desc in (("act_*_224", "색별 모델 3개"),
                      ("act_film_224", "원-핫 + FiLM"),
                      ("act_xy_224", "화면 좌표"),
                      ("smolvla_v10", "좌표를 관측 상태로")):
        L.append(f"| `{pol}` | /{a.count*3} | /{a.count*3} | | {desc} |")

    L.append("\n## 실행 명령\n")
    L.append("```bash")
    L.append("export SECS=90 REC=1 SHOW=1 HOLD=0")
    L.append("POLICY=film224   bash dispense_onehot.sh red green yellow   # 처방 순서는 표대로")
    L.append("POLICY=xy        bash dispense_onehot.sh red green yellow")
    L.append("POLICY=smolvla10 bash dispense_onehot.sh red green yellow")
    L.append("```")
    L.append("\n색별 정책 3개는 `bash dispense.sh red green yellow` 로 돌립니다.")
    L.append("\n> 연속 조제에서 두 번째 색으로 못 넘어가면 `HOME_RETURN=1` 을 붙이십시오 —")
    L.append("> 담은 직후 팔이 약통 위에 있어 학습 분포 밖에서 출발하는 문제입니다.")

    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    print(f"  seed {seed} · {a.count}회 · 시행당 3색 = 총 {a.count*3} 판정")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
