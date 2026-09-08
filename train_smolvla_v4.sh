#!/usr/bin/env bash
# SmolVLA 재학습 v4 — VLM 동결을 푼다 (2026-08-19).
#
# 왜 다시 하는가:
#   v3(공식 권장, VLM 동결)를 실기에서 재보니 지시문을 바꿔도 같은 데로 갔다.
#   그런데 부품은 멀쩡했다.
#     언어 임베딩    red/green/yellow 가 뚜렷이 구분됨 (코사인 0.35~0.48)
#     시각 인코더    알약 색을 최근접 이웃 100% 로 구분함
#     엉뚱한 문장    실제 색을 바꾸는 것보다 예측을 더 크게 흔들었다
#       ("xyzzy qwerty" 차이 1.37 > red↔green 차이 0.30)
#   즉 언어→행동 연결이 안 만들어졌다. 그 연결은 액션 전문가의 교차
#   어텐션에서만 배우는데(train_expert_only=true, VLM 동결), 224개로는
#   부족했다. `film224`(ACT, 5천만 파라미터)가 224개로 색 구분에 성공한
#   전례가 있으므로, 연결을 배울 파라미터 용량을 늘려 본다.
#
#   bash train_smolvla_v4.sh          # 20K 까지 (판정 지점)
#   bash train_smolvla_v4.sh 60000    # 이어서 60K 까지
#
# 이어서 돌릴 때는 --resume 과 --config_path 가 함께 필요하다 (v0.4.4 요구사항).
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3}
OUT=train/smolvla_v4
BS=${BS:-8}      # 316M 학습(VLM 포함)은 100M 보다 무겁다. 24 는 v3(100M) 기준값이라
                  # 여기선 보수적으로 8 에서 시작 — 첫 스텝에서 OOM 이 없으면 그대로 간다.
STEPS=${1:-20000}
CKPT=$OUT/checkpoints/last/pretrained_model/train_config.json

# smolvla_base 를 받아야 하므로 오프라인 모드를 쓰지 않는다
unset HF_HUB_OFFLINE

echo "════════════════════════════════════════════════════"
if [ -f "$CKPT" ]; then
  DONE=$(ls $OUT/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  echo "  이어서 학습 — 현재 ${DONE:-?} 스텝 → $STEPS 스텝"
  echo "════════════════════════════════════════════════════"
  lerobot-train --config_path="$CKPT" --resume=true --steps=$STEPS
else
  echo "  새로 학습 — smolvla_base 에서 시작, VLM 동결 해제, $STEPS 스텝, 배치 $BS"
  echo
  echo "  20K 판정 기준:"
  echo "    지시 반응 3도 이상  → 계속 간다 (필요량 20도의 15%)"
  echo "    3도 미만            → 중단. 224개로는 이 구조도 안 된다고 결론"
  echo "════════════════════════════════════════════════════"
  # v3 와 같고 --policy.train_expert_only=false 만 다르다.
  # --rename_map, --policy.use_amp 필요 이유는 train_smolvla_v3.sh 참고.
  lerobot-train \
    --dataset.repo_id=$REPO \
    --dataset.image_transforms.enable=true \
    --policy.path=lerobot/smolvla_base \
    --policy.train_expert_only=false \
    --policy.device=cuda --policy.use_amp=true \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --batch_size=$BS --steps=$STEPS --save_freq=5000 \
    --output_dir=$OUT --policy.push_to_hub=false
fi

echo
echo "  판정:  python probe_vla.py --run smolvla_v4 --repo-id $REPO"
echo "  이어서: bash train_smolvla_v4.sh 60000"
