#!/usr/bin/env bash
# SmolVLA v8 — 반사실 학습 (2026-08-19).
#
# v3~v7 이 다 실패하면 쓰는 마지막 학습 수단이다 (촬영은 더 못 한다는 전제).
#
# 지금까지의 모든 판이 공유한 약점:
#   "빨강 지시 + 빨강 궤적" 만 가르친다 → 지시문을 무시해도 손실이 안 오른다.
#   실제로 무시한다 (v3 1.10 / v4 1.09 / v5 0.89배, 기준선 1.02).
#
# 반사실은 틀린 짝을 명시적으로 밀어낸다:
#   "pick red pill"   + 빨강 궤적  →  가까워지게  (기존)
#   "pick green pill" + 빨강 궤적  →  멀어지게    (추가)
# 지시문을 무시하면 두 손실이 같아지고 벌점이 붙는다. 회피가 구조적으로 불가능하다.
#
# 조사에서 이 방향이 가장 효과가 컸다 — LangGap 은 같은 궤적으로 0% → 90%.
#
# 구현
#   틀린 지시문은 배치 안에서 언어 토큰을 한 칸 굴려 만든다. 토크나이저가
#   필요 없고, 굴려도 같은 문장인 표본은 자동 제외된다. 같은 장면×3목표가
#   78% 이므로 재료는 충분하다.
#   손실은 비율 힌지: relu(margin - (L_wrong - L_correct) / L_correct)
#   비율이라 학습이 진행돼 손실이 줄어도 척도가 유지된다.
#
# v6/v7 의 언어 신호 보정(×86 + FiLM)도 함께 켠다 — 신호가 안 닿으면
# 반사실도 밀어낼 대상이 없다.
#
# **학습 시간이 약 2배다** (forward 를 두 번 돈다). 그래서 10K 로 시작한다 —
# 신호가 훨씬 세므로 더 빨리 갈릴 것으로 본다.
#
#   bash train_smolvla_v8.sh          # 10K
#   bash train_smolvla_v8.sh 20000    # 이어서
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3_langaug}
OUT=train/smolvla_v8
BS=${BS:-12}      # forward 2회라 활성값이 2배 — v7(24)의 절반에서 시작
SCALE=${SCALE:-86.0}
CFW=${CFW:-1.0}
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
  echo "  새로 학습 — 반사실(가중치 $CFW) + 언어 보정 ×$SCALE + FiLM"
  echo "  $STEPS 스텝, 배치 $BS"
  echo
  echo "  학습 중 볼 것: cf_gap 이 0 에서 0.5 쪽으로 올라가야 한다"
  echo "  (틀린 지시문일 때 손실이 얼마나 더 높은가의 비율)"
  echo "════════════════════════════════════════════════════"
  lerobot-train \
    --dataset.repo_id=$REPO \
    --dataset.image_transforms.enable=true \
    --policy.path=lerobot/smolvla_base \
    --policy.counterfactual_weight=$CFW \
    --policy.counterfactual_margin=0.5 \
    --policy.language_emb_scale=$SCALE \
    --policy.language_film=true \
    --policy.device=cuda --policy.use_amp=true \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --batch_size=$BS --steps=$STEPS --save_freq=2500 \
    --output_dir=$OUT --policy.push_to_hub=false
fi

echo
echo "  판정:  python probe_vla.py --run smolvla_v8 --repo-id 1unasy/pill_v3"
echo "  이어서: bash train_smolvla_v8.sh 20000"
