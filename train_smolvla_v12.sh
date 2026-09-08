#!/usr/bin/env bash
# SmolVLA v12 — 언어가 화면의 **위치**를 지목한다 (2026-08-21).
#
# 언어로 목표를 준 아홉 판이 전부 실패했다 (v3~v9, v11).
#   v3 1.10 / v4 1.09 / v5 0.89 / v6 1.09 / v7 0.96 / v8 2.02(실기 실패) / v9 0.96
#   기준선 1.02
#
# 실측으로 원인을 좁혔다:
#   ① 언어 신호가 prefix 의 0.017%
#      v6 에서 86배 증폭 + FiLM 으로 1.335% (79배) 로 키웠다 → 여전히 1.09배
#   ② **색만으로는 행동이 안 갈린다** (구간별 최대 0.204)
#      알약 위치가 매 에피소드 바뀌므로 "빨강" 자체에는 방향 정보가 없다.
#      필요한 것은 시각적 지목 — 화면에서 빨간 것을 찾아 거기로 가는 것.
#
# v6 의 FiLM 은 채널별 변조라 **모든 시각 토큰에 같은 값**을 곱한다. "빨강"
# 이라는 정보가 화면 전체에 균일하게 퍼지니 "어디" 를 못 가리킨다. 크기만
# 키운 것이 소용없었던 이유다.
#
# v12 는 언어를 질의로 삼아 시각 토큰마다 점수를 내고 그 위치를 키운다.
#     score = <img_token, W·lang>/√D ,  img *= 1 + α·(2σ(score) − 1)
# W 를 0 으로 초기화하므로 시작은 정확히 항등이다(사전학습 동작 보존).
#
# 학습 전 CPU 계측:
#   지시문을 바꿨을 때 prefix 변화   24.8%   (v6 는 1.335%)
#   토큰별 변화의 흩어짐             0.879   (0 이면 모든 위치가 똑같이 변함)
# **흩어짐이 핵심이다.** 이제 언어가 화면의 한 곳을 고를 수 있다.
#
# 반사실은 0.3 으로 둔다. v8(1.0)은 구분 2.02배였지만 오차 2.111 로 실기 불가,
# v9(0.1)는 오차 0.735 였지만 구분 0.96배였다. 공간 지목이 구분을 대신 맡아
# 주면 낮은 가중치로도 될 것으로 본다.
#
#   bash train_smolvla_v12.sh 2500     # 먼저 짧게 — 방향 확인
#   bash train_smolvla_v12.sh 20000    # 이어서
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3_langaug}
OUT=train/smolvla_v12
BS=${BS:-12}
STEPS=${1:-2500}
CKPT=$OUT/checkpoints/last/pretrained_model/train_config.json
unset HF_HUB_OFFLINE

echo "════════════════════════════════════════════════════"
if [ -f "$CKPT" ]; then
  DONE=$(ls $OUT/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  echo "  이어서 학습 — 현재 ${DONE:-?} → $STEPS 스텝"
  echo "════════════════════════════════════════════════════"
  lerobot-train --config_path="$CKPT" --resume=true --steps=$STEPS
else
  echo "  새로 학습 — 언어가 화면의 위치를 지목 (α=1.0)"
  echo "  + 언어 크기 ×86 + FiLM + 반사실 0.3"
  echo "  $STEPS 스텝, 배치 $BS"
  echo "════════════════════════════════════════════════════"
  lerobot-train \
    --dataset.repo_id=$REPO \
    --dataset.image_transforms.enable=true \
    --policy.path=lerobot/smolvla_base \
    --policy.language_spatial_alpha=1.0 \
    --policy.language_film=true \
    --policy.language_emb_scale=86.0 \
    --policy.counterfactual_weight=0.3 \
    --policy.counterfactual_margin=0.3 \
    --policy.device=cuda --policy.use_amp=true \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --batch_size=$BS --steps=$STEPS --save_freq=2500 \
    --output_dir=$OUT --policy.push_to_hub=false
fi

echo
echo "  판정:  python probe_vla.py --run smolvla_v12 --repo-id 1unasy/pill_v3"
