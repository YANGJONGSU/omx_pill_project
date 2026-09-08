#!/usr/bin/env bash
# v10 을 20000 까지 이어 돌리고, 끝나면 체크포인트마다 좌표 추종을 잰다.
# (2026-08-20)
#
# 2500스텝 결과: 자유좌표 상관 +0.252
#   기준 act_xy_224 @150K = +0.484 (같은 시험, 실기 3색 연속 성공)
#   1/60 스텝에서 절반 — 방향이 맞다.
#
# 스텝별로 재야 "오르는 중인가 멈췄는가" 를 가릴 수 있다. act_xy_224 도
# 학습 장면 기준으로 1.055(10K) → 3.675(150K) 로 천천히 올랐다.
set -uo pipefail
cd /home/user/omx_pill_project

while pgrep -f "lerobot-train" > /dev/null; do sleep 60; done
sleep 10

echo "$(date +%H:%M:%S)  === v10 2500 → 20000 이어서 학습 ==="
bash train_smolvla_v10.sh 20000

while pgrep -f "lerobot-train" > /dev/null; do sleep 60; done
sleep 15

source /home/user/venv/il/bin/activate
export HF_HUB_OFFLINE=1
echo
echo "$(date +%H:%M:%S)  === 체크포인트별 좌표 추종 ==="
for c in $(ls train/smolvla_v10/checkpoints | grep -E '^[0-9]+$' | sort -n); do
  echo
  echo "  --- $c 스텝 ---"
  python probe_uvstate.py --run smolvla_v10 --ckpt "$c" 2>&1 \
    | grep -E "상관|판정|기준" | sed 's/^/  /'
done

echo
echo "$(date +%H:%M:%S)  전부 완료. 추세: goal_metric_uvstate.tsv"
