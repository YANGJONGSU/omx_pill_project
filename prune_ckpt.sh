#!/usr/bin/env bash
# 학습 중 오래된 체크포인트를 지운다 (2026-08-21).
#
# v10 을 100,000 스텝까지 돌리는데 save_freq 가 2,500 이라 40개(52GB)가 쌓인다.
# 디스크는 21GB 뿐이고, 저장 중에 차면 학습이 죽는다 (8/19 에 v6 가 그렇게 죽었다).
#
# 남기는 것: 10,000 배수 + 마지막 2개.
# 10,000 배수만 있으면 학습량별 비교표를 만드는 데 충분하다.
set -uo pipefail
D=/home/user/omx_pill_project/train/smolvla_v10/checkpoints
while pgrep -f "lerobot-train" > /dev/null; do
  last2=$(ls "$D" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -2)
  for c in $(ls "$D" 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
    echo "$last2" | grep -qx "$c" && continue
    [ $((10#$c % 10000)) -eq 0 ] && continue
    rm -rf "$D/$c" && echo "$(date +%H:%M)  정리 $c"
  done
  sleep 300
done
