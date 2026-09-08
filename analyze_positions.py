#!/usr/bin/env python
"""데이터셋의 '집는 순간' 자세로 위치 다양성을 측정한다.

알약 위치는 데이터에 라벨로 없다. 대신 그리퍼가 닫히는 프레임의 팔 관절값이
알약이 있던 위치의 대리 지표다 — 같은 자리를 잡으면 같은 자세가 나온다.

이 측정으로 파일럿 30개가 3개 자리에만 몰려 있었다는 것을 찾아냈다.
자세한 배경은 v1_pilot/STATUS.md 참고.

사용법:
  python ~/omx_pill_project/analyze_positions.py              # 전체 요약
  python ~/omx_pill_project/analyze_positions.py --split 30   # 파일럿/새 데이터 경계 지정
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pyarrow.parquet as pq

DATA = "/home/user/.cache/huggingface/lerobot/1unasy/pill_v2/data/chunk-000/*.parquet"

# 파일럿에서 나온 3개 덩어리 (pan 범위). 기록용 상수.
PILOT_CLUSTERS = [(-2.66, 1.05), (5.79, 8.52), (12.87, 14.97)]


def grasp_table(pattern: str = DATA):
    """에피소드별 (집는 순간 관절값, 길이, 그리퍼 전환 횟수)."""
    poses, lengths, transitions = {}, {}, {}
    for f in sorted(glob.glob(pattern)):
        t = pq.read_table(f)
        ei = np.array(t.column("episode_index").to_pylist())
        st = np.array(
            [np.asarray(x) for x in t.column("observation.state").to_pylist()], dtype=np.float32
        )
        for ep in sorted(set(ei.tolist())):
            s = st[ei == ep]
            grip = s[:, 5]
            closed = grip < (grip.max() + grip.min()) / 2
            idx = int(np.argmax(closed)) if closed.any() else int(np.argmin(grip))
            lengths[ep] = len(s)
            transitions[ep] = int(np.sum(np.diff(closed.astype(int)) != 0))
            if idx > 30:  # 프레임 0 근처면 검출 실패 (홈 자세를 집는 순간으로 오인)
                poses[ep] = s[idx, :3]
    return poses, lengths, transitions


def cluster_split(pan: np.ndarray, gap: float = 2.0):
    """정렬한 값에서 gap 이상 벌어진 곳을 잘라 덩어리로 나눈다."""
    p = np.sort(pan)
    return np.split(p, np.flatnonzero(np.diff(p) > gap) + 1)


def summarize(name: str, grp: dict) -> None:
    arr = np.stack(list(grp.values()))
    pan = arr[:, 0]
    gs = cluster_split(pan)
    print(f"[{name}]  n={len(grp)}")
    print(
        f"  pan {pan.min():+.2f} ~ {pan.max():+.2f} "
        f"(폭 {np.ptp(pan):.1f} ≈ {np.ptp(pan) * 1.8:.0f}°), std {pan.std():.2f}"
    )
    print(f"  덩어리 {len(gs)}개, 최대 이웃 간격 {np.diff(np.sort(pan)).max():.2f}")
    print(
        f"  lift {arr[:, 1].min():+.1f}~{arr[:, 1].max():+.1f}  "
        f"elbow {arr[:, 2].min():+.1f}~{arr[:, 2].max():+.1f}"
    )
    print()


def main():
    ap = argparse.ArgumentParser(description="위치 다양성 측정")
    ap.add_argument("--split", type=int, default=30, help="파일럿/새 데이터 경계 에피소드 번호")
    args = ap.parse_args()

    poses, lengths, transitions = grasp_table()
    pilot = {k: v for k, v in poses.items() if k < args.split}
    new = {k: v for k, v in poses.items() if k >= args.split}
    print(f"집는 순간 검출: 파일럿 {len(pilot)}, 새 데이터 {len(new)}\n")

    summarize(f"파일럿 ep0~{args.split - 1}", pilot)
    if new:
        summarize(f"새 데이터 ep{args.split}~", new)
        summarize("전체", poses)

        # 파일럿 내부 최근접 거리 = "같은 자리에 또 놓았을 때" 나오는 거리.
        # 새 에피소드가 이보다 멀면 진짜 새 위치다.
        P = np.stack(list(pilot.values()))
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        thr = d.min(axis=1).max()
        far = [ep for ep, v in new.items() if np.linalg.norm(P - v, axis=-1).min() > thr]
        print(f"파일럿 내부 최근접 거리 최대 = {thr:.2f} (이보다 멀면 '새 위치')")
        print(f"새 데이터 중 파일럿과 안 겹치는 것: {len(far)}/{len(new)}")
        print(f"  겹친 것: {sorted(set(new) - set(far))}")

    bad_len = {e: n for e, n in lengths.items() if not (600 <= n <= 1100)}
    bad_tr = {e: t for e, t in transitions.items() if t != 2}
    print(f"\n[품질] 길이 이상(20~36초 밖): {bad_len if bad_len else '없음'}")
    print(f"[품질] 그리퍼 전환 2회 아님: {bad_tr if bad_tr else '없음'}")
    print("  전환 4회 이상 = 잡다가 놓치고 다시 잡음, 1회 = 사이클 미완성 의심")


if __name__ == "__main__":
    main()
