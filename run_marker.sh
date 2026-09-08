#!/usr/bin/env bash
# 표식 조건화 학습 + 조기 판정 (2026-08-13).
#
# 앞선 다섯 번은 목표를 화면 밖 값으로 주고 모델이 잇기를 기대했다. 전부 실패했다.
# 이번엔 목표를 화면 안에 그려 넣는다 — 이을 것이 없다.
#
# **조기 판정**: 지난번엔 150K 까지 돌리고서야 평평한 걸 알았다. 이번엔
#   10K · 20K · 30K 에서 **떼어 둔 장면**으로 선택률을 재고,
#   30K 에서도 55% 미만이면 즉시 중단한다. 약 3시간이면 결론이 난다.
#
#   nohup bash run_marker.sh > run_marker.log 2>&1 &
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

RUN=act_marker2
OUT=train/$RUN
STEPS=120000
KILL_AT=30000
KILL_BELOW=55
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

EPS=$(python3 -c "
import json; d=json.load(open('marker_episodes.json'))
print('[' + ','.join(map(str, d['train'])) + ']')")
N=$(python3 -c "import json; print(len(json.load(open('marker_episodes.json'))['train']))")

[ -d "$OUT" ] && { say "$OUT 이미 있음 — 중단"; exit 1; }
say "═══ 표식 조건화 학습 — 학습 $N 에피소드, 떼어 둔 15개는 측정 전용 ═══"

lerobot-train \
  --dataset.repo_id=1unasy/pill_v3_xy \
  --dataset.episodes="$EPS" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.02,0.02]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.7,1.3]}},"affine":{"weight":0.0,"type":"RandomAffine","kwargs":{"degrees":[-5.0,5.0],"translate":[0.05,0.05]}}}' \
  --policy.type=act \
  --policy.marker_conditioning=true \
  --policy.film_conditioning=true \
  --policy.device=cuda --policy.use_amp=true \
  --batch_size=8 --steps=$STEPS --save_freq=10000 --num_workers=16 \
  --output_dir="$OUT" --policy.push_to_hub=false \
  > "${OUT}.log" 2>&1 &
TRAIN_PID=$!
say "학습 시작 (PID $TRAIN_PID)"

# ── 조기 판정 감시 ────────────────────────────────────────────────────────
CK_DIR=$OUT/checkpoints
DONE=""
while kill -0 $TRAIN_PID 2>/dev/null; do
  for CK in $(ls $CK_DIR 2>/dev/null | grep -E '^[0-9]+$' | sort); do
    case " $DONE " in *" $CK "*) continue;; esac
    [ -d "$CK_DIR/$CK/pretrained_model" ] || continue
    sleep 20
    LOG="probe_marker_${CK}.log"
    say "$CK 측정 (떼어 둔 장면)"
    timeout 1800 python -u probe_xy_pick.py --run $RUN --ckpt $CK --holdout > "$LOG" 2>&1
    RC=$?
    RATE=$(grep -oE "간 비율 [0-9]+%" "$LOG" | grep -oE "[0-9]+")
    if [ -z "$RATE" ]; then
      say "  !! 측정 실패 (코드 $RC). 마지막 5줄:"; tail -5 "$LOG" | sed 's/^/      /'
    else
      say "  ★ 선택률 ${RATE}%   $(grep -oE '여유 중앙값 [-+][0-9.]+' "$LOG")"
      if [ "$CK" -ge "$KILL_AT" ] && [ "$RATE" -lt "$KILL_BELOW" ]; then
        say "  ══ 조기 중단 — ${CK} 에서 ${RATE}% (기준 ${KILL_BELOW}%). 더 돌려도 안 오릅니다."
        kill $TRAIN_PID 2>/dev/null
        sleep 10
        say "중단 완료. GPU 를 비웠습니다."
        exit 2
      fi
    fi
    DONE="$DONE $CK"
  done
  sleep 60
done

say "학습 종료 — 최종 곡선:"
sed 's/^/    /' goal_metric_xy_pick.tsv
