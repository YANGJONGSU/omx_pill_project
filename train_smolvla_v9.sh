#!/usr/bin/env bash
# SmolVLA v9 — 반사실을 약하게 (2026-08-21).
#
# v8 은 반사실 가중치 1.0 으로 **지시문 구분에는 성공**했다:
#   지시문 반응 2.02배 (기준선 1.02, v3~v7 은 0.89~1.10)
# 그런데 실기에서 못 썼다:
#   자기색 오차 2.111   (실기 성공한 act_xy_224 는 0.833 — 2.5배 부정확)
#   그리퍼 0회, elbow 가 한 스텝에 18도씩 튐, 팔이 화면 밖으로
#
# 왜 그랬나 — 반사실 손실은 "틀린 지시문일 때 손실이 높아야 한다" 만 요구한다.
# 모델이 이걸 만족시키는 길이 둘이다:
#   ① 언어를 제대로 이해해서 다른 색으로 간다      ← 원한 것
#   ② 언어에 과민반응해서 출력을 마구 흔든다        ← 실제로 배운 것
# ②가 더 쉽다. 지표는 오르고 예측은 불안정해진다.
#
# 그래서 가중치를 1.0 → 0.1 로 낮춘다. 행동 정확도를 주로 하고 반사실은
# 거드는 정도로 맞춘다. margin 도 0.5 → 0.15 로 낮춘다 — 되는 정책(2.53배)도
# 자기색 오차가 0.833 이다. **정확도를 지키면서** 구분한 것이다.
#
#   성공 기준: 지시문 반응 1.2배 이상 **이면서** 자기색 오차 1.2 이하
#   (둘 중 하나만으로는 실기에서 못 쓴다 — v8 이 그것을 보여줬다)
#
#   bash train_smolvla_v9.sh          # 10K
#   bash train_smolvla_v9.sh 20000    # 이어서
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3_langaug}
OUT=train/smolvla_v9
BS=${BS:-12}      # forward 2회라 활성값이 2배
SCALE=${SCALE:-86.0}
CFW=${CFW:-0.1}   # v8 은 1.0 이었다
MARGIN=${MARGIN:-0.15}
STEPS=${1:-10000}
CKPT=$OUT/checkpoints/last/pretrained_model/train_config.json

unset HF_HUB_OFFLINE

echo "════════════════════════════════════════════════════"
if [ -f "$CKPT" ]; then
  DONE=$(ls $OUT/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  echo "  이어서 학습 — 현재 ${DONE:-?} 스텝 → $STEPS 스텝"
  echo "════════════════════════════════════════════════════"
  lerobot-train --config_path="$CKPT" --resume=true --steps=$STEPS
else
  echo "  새로 학습 — 반사실 약하게 (가중치 $CFW, margin $MARGIN)"
  echo "  v8 은 가중치 1.0 / margin 0.5 → 구분 2.02배, 오차 2.111 (실기 불가)"
  echo "  $STEPS 스텝, 배치 $BS"
  echo
  echo "  성공 기준: 구분 1.2배 이상 **이면서** 오차 1.2 이하"
  echo "════════════════════════════════════════════════════"
  lerobot-train \
    --dataset.repo_id=$REPO \
    --dataset.image_transforms.enable=true \
    --policy.path=lerobot/smolvla_base \
    --policy.counterfactual_weight=$CFW \
    --policy.counterfactual_margin=$MARGIN \
    --policy.language_emb_scale=$SCALE \
    --policy.language_film=true \
    --policy.device=cuda --policy.use_amp=true \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --batch_size=$BS --steps=$STEPS --save_freq=2500 \
    --output_dir=$OUT --policy.push_to_hub=false
fi

echo
echo "  판정:  python probe_vla.py --run smolvla_v9 --repo-id 1unasy/pill_v3"
