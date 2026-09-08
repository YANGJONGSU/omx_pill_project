#!/usr/bin/env bash
# SmolVLA v6 — 언어 신호가 묻히는 근본 원인을 고친다 (2026-08-19).
#
# v3·v4·v5 는 전부 실패했다 (지시문 반응 1.10 / 1.09 / 0.89배, 기준선 1.02).
# 사전학습을 제대로 불러오고, VLM 을 해동하고, 위치를 직접 가르쳐도 안 됐다.
# 그래서 "왜 안 되는가" 를 prefix 안에서 직접 쟀다.
#
#   이미지 임베딩(증폭 후)   토큰 64개,  값 크기 123.5
#   언어 임베딩(증폭 후)     토큰 48개,  값 크기   1.43     ← 86배 작다
#
# 토큰 수는 43% 로 충분한데 **값 크기가 86배 작아** 어텐션에서 존재감이 없다.
# 지시문을 red→green 으로 바꿔도 prefix 는 0.017% 밖에 안 변한다. 정책이
# 언어를 무시해도 손실이 안 오르니 실제로 무시한다 — 그것이 세 판의 실패였다.
#
# 두 가지를 함께 고친다.
#   ① language_emb_scale=86   언어를 이미지와 같은 자릿수로 올린다.
#                             구조를 바꾸는 게 아니라 원래 있어야 할 균형이다.
#   ② language_film=true      언어에서 γ,β 를 뽑아 시각 특징의 모든 채널을
#                             곱셈 변조한다: f' = (1+γ)·f + β.
#                             토큰은 어텐션이 무시할 수 있지만 곱셈은 못 한다.
#                             ACT 에서 같은 방식(act_film_224)이 원-핫으로
#                             실기 3색 조제에 성공했다.
#
# 실측한 신호 세기 (지시문을 바꿨을 때 prefix 변화율):
#   지금(v3~v5)        0.017%
#   ① 만              1.022%   (60배)
#   ② 만              0.456%   (27배)
#   ①+②              1.335%   (79배)   ← 이 설정
#
#   bash train_smolvla_v6.sh          # 20K
#   bash train_smolvla_v6.sh 60000    # 이어서 60K
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3}
OUT=train/smolvla_v6
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
  echo "  새로 학습 — 언어 크기 보정 ×$SCALE + 언어 FiLM"
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
echo "  판정:  python probe_vla.py --run smolvla_v6 --repo-id $REPO"
echo "  이어서: bash train_smolvla_v6.sh 60000"
