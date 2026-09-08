#!/usr/bin/env python
"""목표를 **이미지 안에 그려 넣은** 데이터셋을 만든다 (2026-08-13).

왜 이 방식인가
--------------
지금까지 네 번 모두 목표를 화면 밖 숫자로 주고 모델이 알아서 잇기를 기대했다.

    원-핫 토큰        1.00      목표를 무시
    원-핫 + FiLM      1.22      장면 암기분
    Diffusion         0.97      Δ 는 큰데 방향이 무작위
    좌표 (act_xy)     선택률 51%  알약으로 가지만 어느 것인지는 화면이 정함

마지막 실험이 원인을 특정했다. 예측은 80% 가 알약 근처에 떨어지는데, 목표를 바꿔도
18% 만 따라간다. **모델은 알약을 찾을 줄 알지만 어느 것인지 고르지 못한다.**
그리고 못 고르는 이유는 학습 데이터의 52%(구 98개)가 장면만으로 정답이 결정되어
목표를 참조할 이유가 없기 때문이다.

표식은 그 연결 과정 자체를 없앤다. 목표가 화면 **안에** 있으므로 이을 것이 없다.
결정적으로, 구 98개에서도 표식은 정답 위치에 찍힌다. 즉 **"표식으로 간다"는 규칙
하나가 188개 전부를 설명한다.** 장면 188개를 외우는 것보다 쉽다. 지름길이 사라지는
게 아니라 정답이 지름길보다 쉬워진다.

구현
----
탑뷰 프레임의 목표 좌표에 원을 그린다. 손목 카메라는 건드리지 않는다.
색은 알약 색과 겹치지 않게 **자홍(magenta)** 을 쓴다 — HSV 검출기가 쓰는
빨강/노랑/초록 범위 밖이다.

    python make_marker_dataset.py            # 전체
    python make_marker_dataset.py --dump 6   # 확인용 이미지도 저장
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent
HF_HOME = Path.home() / ".cache/huggingface/lerobot"
MARKER_BGR = (255, 0, 255)      # 자홍 — 알약 색(빨강 354, 노랑 50, 초록 134)과 안 겹친다
RADIUS, THICK = 13, 3


def draw_marker(bgr, u, v):
    """정규화 좌표 (u, v) 에 표식을 그린다. 원본을 바꾸지 않는다."""
    h, w = bgr.shape[:2]
    x, y = int(round(u * w)), int(round(v * h))
    out = bgr.copy()
    cv2.circle(out, (x, y), RADIUS, MARKER_BGR, THICK, cv2.LINE_AA)
    cv2.drawMarker(out, (x, y), MARKER_BGR, cv2.MARKER_CROSS, RADIUS - 4, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="1unasy/pill_v3")
    ap.add_argument("--out-repo-id", default="1unasy/pill_v3_marker")
    ap.add_argument("--dump", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    lab = json.loads((PROJECT / "xy_labels.json").read_text())["labels"]
    goal = {int(k): (v["u"], v["v"]) for k, v in lab.items()}
    print(f"라벨 {len(goal)}개")

    src = LeRobotDataset(a.repo_id)
    out_root = HF_HOME / a.out_repo_id
    if out_root.exists():
        if not a.force:
            raise SystemExit(f"이미 있습니다: {out_root}  (--force 로 덮어쓰기)")
        shutil.rmtree(out_root)

    # 원본을 통째로 복사한 뒤 top 영상만 다시 인코딩한다.
    print(f"원본 복사 → {out_root}")
    shutil.copytree(src.root, out_root)

    vids = sorted((out_root / "videos").rglob("*.mp4"))
    top = [p for p in vids if "top" in str(p)]
    print(f"top 영상 {len(top)}개 / 전체 {len(vids)}개")

    fr = list(src.meta.episodes["dataset_from_index"])
    to = list(src.meta.episodes["dataset_to_index"])
    # 영상 파일 하나에 여러 에피소드가 들어 있다. 프레임 인덱스로 어느 에피소드인지 찾는다.
    def ep_of_global(gi):
        for e, (f, t) in enumerate(zip(fr, to)):
            if f <= gi < t:
                return e
        return None

    dump_dir = Path("/home/user/.claude/jobs/fd2b9c3d/tmp/marker")
    if a.dump:
        dump_dir.mkdir(parents=True, exist_ok=True)
    dumped = 0
    skipped_frames = 0

    for n, vp in enumerate(top, 1):
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 이 파일이 담당하는 전역 프레임 구간을 파일명에서 알 수 없으므로,
        # 파일 안의 프레임 수와 에피소드 경계를 맞춰 순서대로 소비한다.
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tmp = vp.with_suffix(".tmp.mp4")
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        # 이 파일에 속한 에피소드들 (파일명 chunk/file 로 매핑되지 않으므로 길이로 추정)
        # LeRobot 은 파일별로 에피소드를 순서대로 담는다.
        eps_here = []
        acc = 0
        start_ep = getattr(main, "_next_ep", 0)
        e = start_ep
        while e < len(fr) and acc < n_frames:
            L = int(to[e]) - int(fr[e])
            if acc + L > n_frames:
                break
            eps_here.append(e); acc += L; e += 1
        main._next_ep = e

        seq = []
        for ee in eps_here:
            seq += [ee] * (int(to[ee]) - int(fr[ee]))

        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ep = seq[i] if i < len(seq) else None
            if ep is not None and ep in goal:
                u, v = goal[ep]
                frame = draw_marker(frame, u, v)
                if a.dump and dumped < a.dump and i % 400 == 0:
                    cv2.imwrite(str(dump_dir / f"ep{ep}_f{i}.png"), frame); dumped += 1
            else:
                skipped_frames += 1
            vw.write(frame)
            i += 1
        cap.release(); vw.release()
        tmp.replace(vp)
        if n % 5 == 0 or n == len(top):
            print(f"  {n}/{len(top)}  ({vp.name})")

    print(f"\n표식 없는 프레임 {skipped_frames}개 (라벨 없는 에피소드 — 학습에서 제외됨)")
    (out_root / "marker_spec.json").write_text(json.dumps(
        {"source_repo_id": a.repo_id, "marker_bgr": MARKER_BGR, "radius": RADIUS,
         "note": "목표 알약 위치에 자홍 원+십자. 손목 카메라는 미변경."}, indent=2))
    if a.dump:
        print(f"확인 이미지: {dump_dir}")
    print(f"완료: {a.out_repo_id}")


if __name__ == "__main__":
    main()
