#!/usr/bin/env bash
# v10 이 끝나면 체크포인트별로 재고, 이어서 v9 를 돌린다 (2026-08-21).
#
# v10 은 2500~10000 에서 상관이 +0.302/+0.312/+0.305 로 평평했다. 20000 까지
# 확인해 "더 돌려도 안 오른다" 를 확정한 뒤 v9 로 넘어간다.
#
# v9 는 반사실 가중치를 1.0 → 0.1 로 낮춘 판이다. v8 은 지시문 구분(2.02배)에
# 성공했지만 자기색 오차가 2.111 로 되는 정책(0.833)의 2.5배라 실기에서 못 썼다.
set -uo pipefail
cd /home/user/omx_pill_project

wait_gpu() {
  while pgrep -f "lerobot-train" > /dev/null; do sleep 60; done
  sleep 15
}

wait_gpu
source /home/user/venv/il/bin/activate

export HF_HUB_OFFLINE=1
echo "$(date +%H:%M:%S)  === v10 완료 — 체크포인트별 좌표 추종 ==="
echo "  기준 act_xy_224 @150K = 상관 +0.484, 기울기 +0.29 (실기 3색 연속 성공)"
for c in $(ls train/smolvla_v10/checkpoints | grep -E '^[0-9]+$' | sort -n); do
  printf "  %s  " "$c"
  python probe_uvstate.py --run smolvla_v10 --ckpt "$c" --scenes 8 --grid 3 2>&1 \
    | grep -oE "상관 [+-][0-9.]+   기울기 [+-][0-9.]+" | tail -1
done
unset HF_HUB_OFFLINE

echo
echo "$(date +%H:%M:%S)  === v9 시작 (반사실 가중치 0.1) ==="
for bs in 12 8 6 4; do
  echo "$(date +%H:%M:%S)  배치 $bs 시도"
  if BS=$bs bash train_smolvla_v9.sh 10000; then
    echo "$(date +%H:%M:%S)  배치 $bs 로 완료"; break
  fi
  if [ -d train/smolvla_v9/checkpoints ]; then
    echo "$(date +%H:%M:%S)  중간에 멈춤 — 체크포인트가 있으니 재시도하지 않는다"; break
  fi
  echo "$(date +%H:%M:%S)  배치 $bs 실패 — 낮춰서 다시"
  rm -rf train/smolvla_v9
done

wait_gpu
export HF_HUB_OFFLINE=1
echo
echo "$(date +%H:%M:%S)  === v9 판정 ==="
python probe_vla.py --run smolvla_v9 --repo-id 1unasy/pill_v3 2>&1 | tail -16
echo
echo "  성공 기준: 구분 1.2배 이상 **이면서** 자기색 오차 1.2 이하"
echo "  v8 은 구분 2.02배였지만 오차 2.111 이라 실기에서 못 썼다."
echo "$(date +%H:%M:%S)  전부 완료."
