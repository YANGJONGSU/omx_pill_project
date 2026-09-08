#!/usr/bin/env bash
# v4(VLM 동결 해제) 학습이 끝나면 자동으로 v5(보조 손실) 를 잇는다 (2026-08-19).
#
# GPU 가 하나뿐이라 동시에 못 돌린다. v4 프로세스가 사라질 때까지 기다렸다가,
# 짧은 스모크 테스트(3스텝)로 v5 코드가 실제로 도는지 먼저 확인한 뒤 본 학습을
# 시작한다 — 밤새 무인으로 돌리는 것이라 코드에 문제가 있으면 GPU 시간을
# 낭비하지 않고 바로 멈추는 게 낫다.
#
#   nohup bash chain_v5_after_v4.sh > chain_v5.log 2>&1 &
set -uo pipefail
cd /home/user/omx_pill_project

echo "$(date '+%H:%M:%S')  v4 종료를 기다립니다…"
while pgrep -f "lerobot-train" > /dev/null; do
  sleep 60
done
sleep 10   # GPU 메모리가 실제로 풀릴 시간을 준다
echo "$(date '+%H:%M:%S')  v4 종료 확인 — v5 스모크 테스트 시작"

source /home/user/venv/il/bin/activate
unset HF_HUB_OFFLINE

if [ ! -d "$HOME/.cache/huggingface/lerobot/1unasy/pill_v3_auxuv" ]; then
  echo "  1unasy/pill_v3_auxuv 가 없습니다 — 만듭니다"
  python make_auxuv.py --force 2>&1 | tail -5
fi

rm -rf /tmp/v5smoke
if timeout 900 lerobot-train \
    --dataset.repo_id=1unasy/pill_v3_auxuv \
    --dataset.image_transforms.enable=true \
    --policy.path=lerobot/smolvla_base \
    --policy.aux_position_loss_weight=0.3 \
    --policy.device=cuda --policy.use_amp=true \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --batch_size=24 --steps=3 --save_freq=1000000 \
    --output_dir=/tmp/v5smoke --policy.push_to_hub=false \
    > /tmp/v5smoke.log 2>&1; then
  rm -rf /tmp/v5smoke
  echo "$(date '+%H:%M:%S')  스모크 테스트 통과 (배치 24) — 본 학습 시작"
  bash train_smolvla_v5.sh 20000
else
  echo "$(date '+%H:%M:%S')  스모크 테스트 실패 (배치 24) — 배치 8로 재시도"
  rm -rf /tmp/v5smoke
  if timeout 900 lerobot-train \
      --dataset.repo_id=1unasy/pill_v3_auxuv \
      --dataset.image_transforms.enable=true \
      --policy.path=lerobot/smolvla_base \
      --policy.aux_position_loss_weight=0.3 \
      --policy.device=cuda --policy.use_amp=true \
      --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
      --batch_size=8 --steps=3 --save_freq=1000000 \
      --output_dir=/tmp/v5smoke --policy.push_to_hub=false \
      > /tmp/v5smoke.log 2>&1; then
    rm -rf /tmp/v5smoke
    echo "$(date '+%H:%M:%S')  스모크 테스트 통과 (배치 8) — 본 학습 시작"
    BS=8 bash train_smolvla_v5.sh 20000
  else
    echo "$(date '+%H:%M:%S')  스모크 테스트 실패 — v5 를 시작하지 않는다."
    echo "  로그: /tmp/v5smoke.log (이 로그는 스크립트 종료 후 사라지지 않게 복사해 둔다)"
    cp /tmp/v5smoke.log train/smolvla_v5_smoke_fail.log 2>/dev/null
    rm -rf /tmp/v5smoke
    exit 1
  fi
fi

echo "$(date '+%H:%M:%S')  v5 학습 종료"
