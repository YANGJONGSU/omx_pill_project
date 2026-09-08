#!/usr/bin/env bash
# Diffusion 체크포인트별 목표 조건화 측정 (2026-08-12).
#
# 학습 중 자동 측정(watch_goal_metric.sh)이 probe_policy.py 의 chunk_size 에러를
# 2>/dev/null 로 버리고 조용히 넘어가는 바람에 10개 체크포인트가 전부 비었다.
# 그래서 여기서는 **출력을 통째로 파일에 남기고, 값을 못 뽑으면 그 사실을 기록한다.**
#
#   nohup bash measure_diffusion_curve.sh > measure_diffusion_curve.log 2>&1 &
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

RUN=diffusion_224
REPO=1unasy/pill_v3_onehot_224
OUT=goal_metric_diffusion.tsv
FRAMES=30691          # 에폭 환산용 (전체 프레임 ÷ 배치 6 ... 실제로는 스텝당 샘플수 기준)

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 진행 중인 100K 재측정이 끝나기를 기다린다 (GPU 8GB — 동시에 돌리면 OOM)
while pgrep -f "probe_policy.py --run $RUN" > /dev/null 2>&1; do sleep 30; done
say "선행 측정 종료 확인"

for CK in 040000 080000; do
  LOG="probe_${RUN}_${CK}.log"
  say "═══ $CK 측정 시작 (로그: $LOG) ═══"
  timeout 3600 python -u probe_policy.py --run $RUN --ckpt $CK \
    --repo-id $REPO --tests 5 > "$LOG" 2>&1
  RC=$?

  RATIO=$(grep -oE "평균 비율 [0-9.]+" "$LOG" | grep -oE "[0-9.]+" | tail -1)
  DELTA=$(grep -oE "액션 변화 [0-9.]+" "$LOG" | grep -oE "[0-9.]+" | tail -1)

  if [ -z "$RATIO" ]; then
    # 조용히 넘어가지 않는다. 실패도 표에 남긴다.
    say "!! $CK 측정 실패 (종료코드 $RC). 마지막 5줄:"
    tail -5 "$LOG" | sed 's/^/    /'
    printf "%s\tFAIL\tFAIL\tFAIL\tFAIL(rc=%s)\t\n" "$CK" "$RC" >> $OUT
    continue
  fi

  # 색별 비율은 결과표 5번째 칸에 있다 (색 n 올바른 틀린 비율 Δ)
  pc() { awk -v c="$1" '$1==c {print $5}' "$LOG" | tail -1; }
  say "$CK → 비율 $RATIO / Δ $DELTA"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$CK" "$(pc yellow)" "$(pc red)" "$(pc green)" \
         "$RATIO" "$DELTA" >> $OUT
done

say "끝. 결과:"
cat $OUT | sed 's/^/    /'
