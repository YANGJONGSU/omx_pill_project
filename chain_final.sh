#!/usr/bin/env bash
# v7 완주 → 판정 → v6 완주 → 판정 → v8(반사실) → 판정 (2026-08-19).
#
# 세 판을 순서대로 돌려 어느 개입이 효과였는지 가린다.
#   v6   언어 신호 크기 ×86 + FiLM              신호 0.017% → 1.335%
#   v7   위 + 지시문 3→18종류                   엔트로피 1.58 → 3.76 bits
#   v8   위 + 반사실(틀린 지시문을 밀어냄)        회피가 구조적으로 불가능
#
# v8 이 마지막 수단이다 (촬영은 더 못 한다는 전제). 이것도 실패하면
# "224개로는 SmolVLA 언어 조건화가 안 된다" 가 측정된 결론이 된다.
#
# GPU 가 하나뿐이라 순차로만 돈다. 판정도 GPU 를 쓰므로 학습 종료 후에만 실행.
set -uo pipefail
cd /home/user/omx_pill_project

wait_gpu() {
  while pgrep -f "lerobot-train" > /dev/null; do sleep 60; done
  sleep 15   # GPU 메모리가 실제로 풀릴 시간
}

judge() {   # $1 = 정책 이름
  export HF_HUB_OFFLINE=1
  echo
  echo "$(date +%H:%M:%S)  === $1 판정 ==="
  python probe_vla.py --run "$1" --repo-id 1unasy/pill_v3 2>&1 | tail -16
  unset HF_HUB_OFFLINE
}

source /home/user/venv/il/bin/activate

wait_gpu
judge smolvla_v7

echo
echo "$(date +%H:%M:%S)  === v6 재개 (체크포인트 5000 → 20000) ==="
bash train_smolvla_v6.sh 20000
wait_gpu
judge smolvla_v6

echo
echo "$(date +%H:%M:%S)  === v8 시작 (반사실 학습) ==="
# forward 를 두 번 돌아 활성값이 2배다. 배치를 낮춰가며 시도한다 —
# OOM 으로 그냥 죽으면 밤새 아무것도 안 돌아간다.
for bs in 12 8 6 4; do
  echo "$(date +%H:%M:%S)  v8 배치 $bs 시도"
  if BS=$bs bash train_smolvla_v8.sh 10000; then
    echo "$(date +%H:%M:%S)  v8 배치 $bs 로 완료"
    break
  fi
  if [ -d train/smolvla_v8/checkpoints ]; then
    echo "$(date +%H:%M:%S)  중간에 멈춤 — 체크포인트가 있으니 재시도하지 않는다"
    break
  fi
  echo "$(date +%H:%M:%S)  배치 $bs 실패 — 낮춰서 다시"
  rm -rf train/smolvla_v8
done
wait_gpu
judge smolvla_v8

echo
echo "$(date +%H:%M:%S)  전부 완료. 실기:"
echo '  export POLICY=smolvla8 SECS=90 REC=1 SHOW=1 EXTRA="--n-action-steps 10 --hold-home 0"'
echo '  bash dispense_onehot.sh yellow'
