#!/usr/bin/env python3
"""데이터셋 중간의 에피소드를 지우고 번호를 다시 매긴다 (2026-08-10).

lerobot v0.4.4 에는 에피소드 삭제 기능이 없다. 맨 뒤라면 파일만 치우면 되지만
중간이면 뒤따르는 에피소드의 번호와 전역 프레임 인덱스를 전부 다시 매겨야 한다.

    python remove_episodes.py 190 199            # 미리보기 (아무것도 안 고침)
    python remove_episodes.py 190 199 --apply    # 실행

**주의 — 한 파일에 에피소드가 여러 개 들어 있다.** 원본 96개는 최대 20개까지
한 parquet 에 묶여 있고, 오늘 찍은 것은 대개 1개씩이다. 그래서 "에피소드마다
파일을 통째로 다시 쓰는" 방식은 틀린다 (같은 파일을 여러 번 덮어써서 데이터가
망가진다 — 2026-08-10 에 실제로 그렇게 깨뜨렸다). 반드시 **파일 단위로** 처리한다.

전역 index 는 `start[에피소드] + frame_index` 로 계산한다. frame_index 가 에피소드
안에서 0..길이-1 이므로 파일 안에 에피소드가 몇 개든 정확하다.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = os.path.expanduser("~/.cache/huggingface/lerobot/1unasy/pill_v3")
TRASH = os.path.expanduser("~/omx_pill_project/.trash_removed")


def load_meta():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True)):
        for r in pq.read_table(f).to_pylist():
            r["_meta_file"] = f
            rows.append(r)
    rows.sort(key=lambda r: r["episode_index"])
    return rows


def vid_path(r, cam):
    k = f"videos/observation.images.{cam}/"
    return f"{ROOT}/videos/observation.images.{cam}/chunk-{r[k + 'chunk_index']:03d}/file-{r[k + 'file_index']:03d}.mp4"


def data_path(r):
    return f"{ROOT}/data/chunk-{r['data/chunk_index']:03d}/file-{r['data/file_index']:03d}.parquet"


def to_trash(p, tag):
    d = os.path.join(TRASH, tag, os.path.relpath(p, ROOT))
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.move(p, d)
    print(f"   옮김 {os.path.relpath(p, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    drop = set(a.episodes)

    rows = load_meta()
    info = json.load(open(f"{ROOT}/meta/info.json"))
    print(f"현재 {len(rows)}개 에피소드, {info['total_frames']} 프레임")
    if len(rows) != info["total_episodes"]:
        print("★ meta 와 info.json 개수 불일치 — 중단")
        return 1

    keep = [r for r in rows if r["episode_index"] not in drop]
    gone = [r for r in rows if r["episode_index"] in drop]
    if len(gone) != len(drop):
        print(f"★ 못 찾은 에피소드 {drop - {r['episode_index'] for r in gone}}")
        return 1

    print("\n지울 것:")
    for r in gone:
        print(f"   ep {r['episode_index']}  {r['length']:5d}프레임  {r['tasks'][0]}")

    # 지울 에피소드가 쓰는 파일이 남는 에피소드와 공유되는지 확인한다
    keep_data = {data_path(r) for r in keep}
    keep_vid = {vid_path(r, c) for r in keep for c in ("top", "wrist")}
    free_data = [data_path(r) for r in gone if data_path(r) not in keep_data]
    free_vid = [vid_path(r, c) for r in gone for c in ("top", "wrist") if vid_path(r, c) not in keep_vid]
    shared = [data_path(r) for r in gone if data_path(r) in keep_data]
    print(f"\n통째로 치울 파일: data {len(free_data)}개, 영상 {len(free_vid)}개")
    if shared:
        print(f"남는 에피소드와 공유되는 data 파일 {len(shared)}개 — 행만 걸러낸다")

    newidx = {r["episode_index"]: i for i, r in enumerate(keep)}
    start, cur = {}, 0
    for r in keep:
        start[r["episode_index"]] = cur
        cur += r["length"]
    print(f"\n남는 것 {len(keep)}개, {cur} 프레임")
    if not a.apply:
        print("\n(미리보기입니다. 실행하려면 --apply)")
        return 0

    os.makedirs(TRASH, exist_ok=True)
    shutil.copytree(f"{ROOT}/meta", os.path.join(TRASH, "meta_backup"), dirs_exist_ok=True)

    for p in free_data + free_vid:
        if os.path.exists(p):
            to_trash(p, "files")

    # ── data parquet 을 **파일 단위로** 다시 쓴다 ──────────────────────
    n_files = 0
    for f in sorted(glob.glob(f"{ROOT}/data/chunk-*/*.parquet")):
        t = pq.read_table(f)
        eps = t.column("episode_index").to_pylist()
        if not (set(eps) & set(newidx)):
            continue
        mask = pa.array([e in newidx for e in eps])
        t = t.filter(mask)
        eps = t.column("episode_index").to_pylist()
        fis = t.column("frame_index").to_pylist()
        cols = {n: t.column(n) for n in t.column_names}
        cols["episode_index"] = pa.array([newidx[e] for e in eps], type=t.schema.field("episode_index").type)
        cols["index"] = pa.array([start[e] + fi for e, fi in zip(eps, fis)], type=t.schema.field("index").type)
        pq.write_table(pa.table(cols, schema=t.schema), f)
        n_files += 1
    print(f"\ndata parquet {n_files}개 다시 씀")

    # ── meta/episodes 도 파일 단위로 ───────────────────────────────────
    n_meta = 0
    for f in sorted(glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True)):
        tbl = pq.read_table(f)
        rs = [r for r in tbl.to_pylist() if r["episode_index"] in newidx]
        if not rs:
            to_trash(f, "files")
            continue
        for r in rs:
            old = r["episode_index"]
            r["episode_index"] = newidx[old]
            r["dataset_from_index"] = start[old]
            r["dataset_to_index"] = start[old] + r["length"]
        pq.write_table(pa.Table.from_pylist(rs, schema=tbl.schema), f)
        n_meta += 1
    print(f"meta parquet {n_meta}개 다시 씀")

    info["total_episodes"] = len(keep)
    info["total_frames"] = cur
    info["splits"] = {"train": f"0:{len(keep)}"}
    json.dump(info, open(f"{ROOT}/meta/info.json", "w"), indent=4)
    print(f"\n완료 — {len(keep)}개 에피소드, {cur} 프레임")
    print(f"휴지통: {TRASH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
