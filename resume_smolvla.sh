#!/usr/bin/env bash
# SmolVLA 를 마지막 온전한 체크포인트에서 이어서 130K 까지 (2026-08-15).
#
# 08-15 17:29 에 디스크가 차서 100K 저장 중 죽었다. 마지막 온전한 상태는 90K 다.
# 학습률 스케줄과 옵티마이저 상태가 training_state 에 저장되어 있어 이어붙여도
# 이음매가 없다.
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate
OUT=train/smolvla_v2
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

FREE=$(df -m /home/user | tail -1 | awk '{print $4}')
say "디스크 여유 ${FREE}MB"
[ "$FREE" -lt 8000 ] && { say "!! 8GB 미만 — 체크포인트 4개도 못 씁니다. 중단."; exit 1; }

say "═══ 90K → 130K 이어서 학습 ═══"
lerobot-train --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true --steps=130000 >> "${OUT}.log" 2>&1
RC=$?
say "종료 (코드 $RC) — 체크포인트 $(ls $OUT/checkpoints | grep -v last | wc -l)개"
df -h /home/user | tail -1 | awk '{print "  여유 "$4}'
