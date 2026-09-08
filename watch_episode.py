#!/usr/bin/env python
"""저장된 에피소드 영상을 찾아 재생한다.

lerobot은 에피소드를 개별 파일로 저장하지 않고 여러 개를 한 mp4에 이어붙인다.
어느 파일의 몇 초 지점인지는 meta/episodes 에 들어 있어서, 그걸 읽어 바로 그
지점부터 틀어준다.

사용법:
  python ~/omx_pill_project/watch_episode.py            # 목록만 보기
  python ~/omx_pill_project/watch_episode.py 55         # ep55 top 영상 재생
  python ~/omx_pill_project/watch_episode.py 55 --cam wrist
  python ~/omx_pill_project/watch_episode.py 2 --eval    # 평가 데이터셋
  python ~/omx_pill_project/watch_episode.py 55 --print  # 명령만 출력
"""

from __future__ import annotations

import argparse
import glob
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/home/user/.cache/huggingface/lerobot/1unasy")


def load(repo: str):
    d = ROOT / repo
    rows = []
    for f in sorted(glob.glob(str(d / "meta/episodes/*/*.parquet"))):
        t = pq.read_table(f).to_pylist()
        rows.extend(t)
    return d, {r["episode_index"]: r for r in rows}


def video_path(d: Path, cam: str, row) -> tuple[Path, float, float]:
    key = f"observation.images.{cam}"
    ci = row[f"videos/{key}/chunk_index"]
    fi = row[f"videos/{key}/file_index"]
    p = d / f"videos/{key}/chunk-{ci:03d}/file-{fi:03d}.mp4"
    return p, float(row[f"videos/{key}/from_timestamp"]), float(row[f"videos/{key}/to_timestamp"])


def main():
    ap = argparse.ArgumentParser(description="저장된 에피소드 영상 보기")
    ap.add_argument("episode", type=int, nargs="?", help="에피소드 번호 (없으면 목록만)")
    ap.add_argument("--cam", default="top", choices=["top", "wrist"])
    ap.add_argument("--eval", action="store_true", help="평가 데이터셋(eval_pill_v2)")
    ap.add_argument("--print", dest="only_print", action="store_true", help="명령만 출력")
    args = ap.parse_args()

    repo = "eval_pill_v2" if args.eval else "pill_v2"
    d, eps = load(repo)
    if not eps:
        raise SystemExit(f"메타를 못 읽었습니다: {d}")

    if args.episode is None:
        print(f"[{repo}]  에피소드 {len(eps)}개\n")
        print(f"{'ep':>4}  {'영상 파일':<14} {'시작':>8} {'끝':>8} {'길이':>7}")
        for ep in sorted(eps):
            p, a, b = video_path(d, args.cam, eps[ep])
            print(f"{ep:>4}  {p.name:<14} {a:>7.1f}s {b:>7.1f}s {b - a:>6.1f}s")
        print(f"\n영상 폴더: {d / 'videos'}")
        print("특정 에피소드: python watch_episode.py <번호> [--cam wrist] [--eval]")
        return

    if args.episode not in eps:
        raise SystemExit(f"ep{args.episode} 없음. 있는 범위: {min(eps)}~{max(eps)}")

    p, a, b = video_path(d, args.cam, eps[args.episode])
    if not p.exists():
        raise SystemExit(f"영상 파일이 없습니다: {p}")

    cmd = ["ffplay", "-autoexit", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}",
           "-window_title", f"{repo} ep{args.episode} [{args.cam}]", str(p)]
    print(f"ep{args.episode} [{args.cam}]  {p.name}  {a:.1f}s ~ {b:.1f}s ({b - a:.1f}초)")
    if args.only_print:
        print(" ".join(cmd))
        return
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
