#!/usr/bin/env bash
# SmolVLA v7 — 지시문 다양화 (2026-08-19).
#
# v3~v5 실패 후 조사해 보니 이건 VLA 에서 잘 알려진 현상이었다.
#
#   "Information Collapse" — 지시문이 이미지만으로 예측 가능하면, 모델은
#   언어가 추가 정보를 안 준다고 학습하고 최적화 과정에서 버린다.
#   원인은 modality imbalance: 시각·행동은 다양한데 언어는 템플릿 몇 개뿐이다.
#
#   LIBERO-PRO: 90% 성공률이 지시문을 흔들면 무너지고, 의미 없는 문장에도
#   출력이 그대로였다 — 우리 관찰과 정확히 같다.
#
# 보고된 해법과 효과 (전부 새 촬영 없이 지시문만 다시 씀):
#   LangGap  0% → 90%   /   CAST  +27%p   /   FineVLA  색 속성 +18
#
# make_langaug.py 로 만든 1unasy/pill_v3_langaug 를 쓴다.
#   지시문      3종류 → 18종류
#   엔트로피    1.584 → 3.757 bits  (언어가 나르는 정보량 2.4배)
#   색별 길이   5.27~5.68 단어로 비슷 — 문장 길이로 색을 맞힐 수 없다
#   원래 형식   절반 유지 (FineVLA 실측: FG:Raw=1:1 최적, 역U자 곡선)
#               전부 바꾸면 실기의 "pick red pill" 을 못 알아듣는다
#   반사실      "pick the red pill, not the green or the yellow"  21개
#
# **좌표는 안 쓴다.** 순수하게 언어만으로 색을 배우는지 본다.
# 이것도 실패하면 그때 좌표를 마지막 수단으로 쓴다 — 그 순서라야
# "언어만으로는 안 돼서 위치를 줬다" 는 이야기가 성립한다.
#
#   bash train_smolvla_v7.sh          # 20K
#   bash train_smolvla_v7.sh 60000    # 이어서
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3_langaug}
OUT=train/smolvla_v7
BS=${BS:-24}
SCALE=${SCALE:-86.0}
STEPS=${1:-20000}
CKPT=$OUT/checkpoints/last/pretrained_model/train_config.json

unset HF_HUB_OFFLINE

echo "════════════════════════════════════════════════════"
if [ -f "$CKPT" ]; then
  DONE=$(ls $OUT/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  echo "  이어서 학습 — 현재 ${DONE:-?} 스텝 → $STEPS 스텝"
  echo "════════════════════════════════════════════════════"
  lerobot-train --config_path="$CKPT" --resume=true --steps=$STEPS
else
  echo "  새로 학습 — 지시문 18종류 + 언어 크기 보정 ×$SCALE + 언어 FiLM"
  echo "  $STEPS 스텝, 배치 $BS"
  echo
  echo "  판정 기준: 지시문 반응 1.2배 이상 (기준선 1.02, v3~v5 는 0.89~1.10)"
  echo "════════════════════════════════════════════════════"
  lerobot-train \
    --dataset.repo_id=$REPO \
    --dataset.image_transforms.enable=true \
    --policy.path=lerobot/smolvla_base \
    --policy.language_emb_scale=$SCALE \
    --policy.language_film=true \
    --policy.device=cuda --policy.use_amp=true \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --batch_size=$BS --steps=$STEPS --save_freq=5000 \
    --output_dir=$OUT --policy.push_to_hub=false
fi

echo
echo "  판정:  python probe_vla.py --run smolvla_v7 --repo-id 1unasy/pill_v3"
echo "  이어서: bash train_smolvla_v7.sh 60000"
