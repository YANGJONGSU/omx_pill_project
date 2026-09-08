#!/usr/bin/env bash
# SmolVLA 재학습 — 설정 최적화 (2026-08-14).
#
# 이전 학습(smolvla_v1)의 문제:
#   ① 20K 스텝 = 1.39 에폭.  ACT 색별 정책은 9 에폭에서 최적이었다. 압도적으로 부족.
#   ② 학습률 스케줄이 decay_steps=30000 인데 20K 에서 끊었다 — 감쇠가 끝나기 전에
#      멈춰, 마지막까지 큰 학습률로 흔들리다 끝났다.
#   ③ train_expert_only=True — 행동 전문가만 학습하고 VLM 은 통째로 얼렸다.
#      언어("pick red pill")로 목표를 구분해야 하는 과제인데, 언어를 읽는 쪽이
#      과제에 맞게 조정될 기회가 없었다.
#
# 이번에 바꾸는 것:
#   - 스텝 20K → 130K (약 9 에폭). ACT 색별 정책이 최적이었던 학습량과 같게 맞춘다.
#     스케줄도 130K 에 맞춰 끝까지 감쇠시킨다.
#   - train_expert_only=false — VLM 쪽도 함께 학습해 언어 조건화를 살린다
#   - **freeze_vision_encoder=false** — 시각 타워까지 학습한다.
#
# 시각 인코더를 푸는 이유:
#   실기 확인 결과 원-핫 단순 토큰(act_v3_3color)만 실패하고, 목표를 넓게 주입한
#   FiLM · 좌표 · 표식은 전부 지시한 색으로 갔다. 즉 목표 신호가 **시각 특징에
#   닿느냐**가 갈랐다. SmolVLA 는 지금까지 두 번 다 시각 타워를 얼린 채 학습해,
#   색이라는 시각 속성이 과제에 맞게 조정될 기회가 없었다. 남은 지렛대는 이것뿐이다.
#
# 배치는 6 으로 시작하고 OOM 이면 4 → 시각 동결 순으로 물러선다.
#
#   nohup bash train_smolvla2.sh > train_smolvla2.log 2>&1 &
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

OUT=train/smolvla_v2
STEPS=${STEPS:-130000}
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

[ -d "$OUT" ] && { say "$OUT 이미 있음 — 중단"; exit 1; }

run_train() {
  local BS=$1 FREEZE=${2:-false}
  say "═══ SmolVLA 학습 — 배치 $BS, ${STEPS} 스텝, 시각동결=$FREEZE ═══"
  lerobot-train \
    --dataset.repo_id=1unasy/pill_v3 \
    --dataset.image_transforms.enable=true \
    --policy.type=smolvla \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=$FREEZE \
    --policy.scheduler_decay_steps=$STEPS \
    --policy.device=cuda --policy.use_amp=true \
    --batch_size=$BS --steps=$STEPS --save_freq=10000 --num_workers=8 \
    --output_dir="$OUT" --policy.push_to_hub=false \
    > "${OUT}.log" 2>&1
  return $?
}

oom() { grep -qi "out of memory" "${OUT}.log"; }

run_train 6 false; RC=$?
if [ $RC -ne 0 ] && oom; then
  say "!! VRAM 부족 — 배치 4 로 재시도"
  rm -rf "$OUT"; run_train 4 false; RC=$?
fi
if [ $RC -ne 0 ] && oom; then
  say "!! 여전히 부족 — 시각 인코더를 동결하고 배치 8 로 재시도"
  rm -rf "$OUT"; run_train 8 true; RC=$?
fi

say "학습 종료 (코드 $RC) — 체크포인트 $(ls $OUT/checkpoints 2>/dev/null | grep -v last | wc -l)개"
say "실기 확인:  RUN=smolvla_v2 TASK='pick red pill' bash run.sh last --repo-id 1unasy/pill_v3 \\"
say "              --show --local-keys --relax-on-exit --temporal-ensemble --no-freeze-on-grasp"
