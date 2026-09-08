#!/usr/bin/env bash
# SmolVLA 재학습 v5 — 목표 알약 위치를 보조 손실로 직접 가르친다 (2026-08-19).
#
# 왜 다시 하는가:
#   v4(VLM 동결 해제, 진행 중)는 "언어→행동" 연결을 전체 파라미터로 배우게
#   맡긴 것이다. v5 는 다르게 접근한다 — "그 알약이 화면 어디 있어?" 를
#   훈련 중에 직접 물어본다. 답을 안 보면 못 맞히므로 언어를 볼 수밖에 없다.
#
#   좌표는 xy_labels.json (act_xy_224 용으로 이미 검증된 것, 97.9%) 을 그대로
#   재사용한다. make_auxuv.py 가 pill_v3 에 aux_target_uv 컬럼을 붙여
#   1unasy/pill_v3_auxuv 를 만들어 뒀다 — 영상 재인코딩 없이 컬럼만 추가.
#
#   추론에는 전혀 관여하지 않는다. aux_position_loss_weight=0 인 기존 정책은
#   이 패치의 영향을 받지 않는다(검증됨: v3 체크포인트 정상 로드).
#
#   실제 lerobot 코드 3곳을 고쳤다 (전부 .bak_auxuv 백업 있음):
#     configuration_smolvla.py   aux_position_loss_weight 필드 추가
#     modeling_smolvla.py        aux_pos_head 추가 + 보조 손실 계산
#     processor/converters.py    aux_ 로 시작하는 배치 키가 파이프라인
#                                 진입 시점에 버려지던 것을 통과시킴
#                                 (_extract_complementary_data 허용목록 확장)
#
#   단위 시험(2026-08-19, CPU)으로 확인:
#     일부 유효 라벨   aux_position_loss 계산됨, aux_pos_head 로 그래디언트 흐름
#     전부 무효 라벨   aux_position_loss 자체가 안 생김 (정지 시연에서 안전)
#     총 loss = 행동손실 + weight × aux손실   수치 일치 확인
#
#   bash train_smolvla_v5.sh          # 20K 까지
#   bash train_smolvla_v5.sh 60000    # 이어서 60K 까지 (v4 와 표본 수 맞춤)
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3_auxuv}
OUT=train/smolvla_v5
BS=${BS:-24}      # v3 와 같은 동결 구조(100M 학습). aux_pos_head 는 파라미터
                  # 1,922개뿐이라 무시할 만하다 — v3 의 실측 한계(24)를 그대로 쓴다.
WEIGHT=${WEIGHT:-0.3}
STEPS=${1:-20000}
CKPT=$OUT/checkpoints/last/pretrained_model/train_config.json

unset HF_HUB_OFFLINE

if [ ! -d "$(python3 -c "import os; print(os.path.expanduser('~/.cache/huggingface/lerobot/$REPO'))")" ]; then
  echo "  $REPO 가 없습니다 — 먼저 만듭니다"
  python make_auxuv.py --force
fi

echo "════════════════════════════════════════════════════"
if [ -f "$CKPT" ]; then
  DONE=$(ls $OUT/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  echo "  이어서 학습 — 현재 ${DONE:-?} 스텝 → $STEPS 스텝"
  echo "════════════════════════════════════════════════════"
  lerobot-train --config_path="$CKPT" --resume=true --steps=$STEPS
else
  echo "  새로 학습 — smolvla_base 에서 시작, VLM 동결(전문가만), 보조손실 가중치 $WEIGHT"
  echo "  $STEPS 스텝, 배치 $BS"
  echo "════════════════════════════════════════════════════"
  # train_expert_only 는 건드리지 않는다(v3 와 같은 동결 조건) — 보조 손실이
  # 연결을 만드는 역할을 대신하는지 보는 것이 이 실험의 목적이라, VLM 해동과
  # 동시에 하면 어느 쪽 덕인지 구분이 안 된다.
  lerobot-train \
    --dataset.repo_id=$REPO \
    --dataset.image_transforms.enable=true \
    --policy.path=lerobot/smolvla_base \
    --policy.aux_position_loss_weight=$WEIGHT \
    --policy.device=cuda --policy.use_amp=true \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --batch_size=$BS --steps=$STEPS --save_freq=5000 \
    --output_dir=$OUT --policy.push_to_hub=false
fi

echo
echo "  판정:  python probe_vla.py --run smolvla_v5 --repo-id 1unasy/pill_v3"
echo "  이어서: bash train_smolvla_v5.sh 60000"
