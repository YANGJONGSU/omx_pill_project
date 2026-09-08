#!/usr/bin/env bash
# SmolVLA 재학습 — 사전학습 가중치를 제대로 불러온다 (2026-08-18).
#
# 왜 다시 하는가:
#   smolvla_v1·v2 는 `--policy.type=smolvla` 로만 시작해 load_vlm_weights 기본값
#   False 로 학습됐다. smolvlm_with_expert.py:76 의 else 분기로 가서 **구조만
#   가져오고 가중치는 무작위**였다. SmolVLM2-500M 을 백지에서 학습시킨 셈이라
#   "red" 와 "green" 이 다른 단어라는 것조차 모르는 상태였다.
#   지시를 바꿔도 0.3~1.0도(필요량 20도의 2~5%)만 움직인 것이 이것으로 설명된다.
#
# 이번에는 `--policy.path=lerobot/smolvla_base` 로 시작한다. 시각-언어 이해에
# 더해 **로봇 조작 사전학습까지** 들어 있는 정책이다.
#
#   bash train_smolvla_v3.sh          # 20K 까지 (판정 지점, 약 6.2시간)
#   bash train_smolvla_v3.sh 60000    # 이어서 60K 까지
#
# 이어서 돌릴 때는 --resume 과 --config_path 가 함께 필요하다 (v0.4.4 요구사항).
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3}
OUT=train/smolvla_v3
BS=${BS:-24}     # 8GB VRAM 실측 한계. 32·48 은 OOM
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
  echo "  새로 학습 — smolvla_base 에서 시작, $STEPS 스텝, 배치 $BS"
  echo
  echo "  20K 판정 기준 (약 6.2시간):"
  echo "    지시 반응 3도 이상  → 계속 간다 (필요량 20도의 15%)"
  echo "    3도 미만            → 중단. 데이터의 한계로 결론"
  echo "════════════════════════════════════════════════════"
  # **공식 권장 설정을 따른다.** (huggingface.co/docs/lerobot/en/smolvla)
  #   lerobot-train --policy.path=lerobot/smolvla_base --batch_size=64 --steps=20000
  # 우리가 더하는 것은 아래 두 가지뿐이고, 둘 다 필요해서 넣는다.
  #
  # ① --rename_map (필수)
  #    smolvla_base 는 camera1/2/3 을 기대하는데 우리 데이터는 top/wrist 다.
  #    factory.py:472 는 input_features 가 **비어 있을 때만** 데이터셋 특징으로
  #    채우는데 smolvla_base 는 이미 차 있어 덮어쓰이지 않는다.
  #    없이 돌리면 ValueError: Feature mismatch 로 즉시 실패한다.
  #    카메라가 2개뿐이어도 camera3 없이 통과한다 (실측).
  #
  # ② --policy.use_amp=true
  #    공식 문서는 --policy.dtype=bfloat16 을 쓰지만 **v0.4.4 에는 그 인자가 없다**
  #    (gradient_checkpointing 도 없다 — 최신 판 문서다). use_amp 가 대응된다.
  #
  # train_expert_only 는 **건드리지 않는다.** smolvla_base 기본값 true 이고
  # 공식 권장도 그대로 둔다. VLM 은 얼리고 액션 전문가만 학습하는 구조인데,
  # 언어→행동 연결은 전문가의 교차 어텐션에서 배우므로 VLM 을 풀 필요가 없다.
  # 오히려 224개로 316M 을 학습하면 사전학습 언어 이해를 덮어쓸 위험이 있다.
  # (2026-08-18: 한때 false 로 잡았다가 이 판단으로 되돌렸다.)
  #
  # 배치는 실측 한계 24 (8GB VRAM). 공식 권장 64 는 OOM, 32 도 OOM.
  # 1.11 s/step → 20K 에 약 6.2시간.
  lerobot-train \
    --dataset.repo_id=$REPO \
    --dataset.image_transforms.enable=true \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda --policy.use_amp=true \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --batch_size=$BS --steps=$STEPS --save_freq=5000 \
    --output_dir=$OUT --policy.push_to_hub=false
fi

echo
echo "  판정:  python probe_vla.py --run smolvla_v3 --repo-id $REPO"
echo "  이어서: bash train_smolvla_v3.sh 60000"
