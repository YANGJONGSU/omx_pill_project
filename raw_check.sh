#!/usr/bin/env bash
# 보정을 **하나도** 넣지 않고 세 정책을 그대로 돌린다.
#
# 왜 (2026-08-07, 사용자 질문): "보정 안 해두고 오늘 아침에 학습된 3개 그대로
# 위치 따라가는지 궁금하다. 그동안 보정 건드려서 달라진 거 아닌가."
# 타당한 의심이다. 오늘 얹은 것들(위치별 자동보정, 파지점 오프셋, 구역별 값,
# 좌표계 해석)이 전부 정책 위에 덧씌운 것이라, 정책 자체의 실력을 가린다.
#
# 끄는 것:
#   --no-auto-correct            correction.json (위치별 직선 보정)
#   --grasp-offset-* 0           손으로 맞춘 오프셋 (저장값은 **안 지운다**, 이번만 0)
#   --zones 안 씀                구역별 값
#   --radial-offset 안 씀        좌표계 해석 (오프셋이 0이라 어차피 무관)
#
# 남기는 것 (정책 구동에 필수):
#   --temporal-ensemble          학습과 같은 추론 방식
#   --no-freeze-on-grasp         켜면 교착에 빠진다 (2026-08-05)
#
#   bash ~/omx_pill_project/raw_check.sh              # 빨강 노랑 초록
#   bash ~/omx_pill_project/raw_check.sh green        # 한 색만
#   SECS=90 bash ~/omx_pill_project/raw_check.sh
set -u
cd /home/user/omx_pill_project

SECS=${SECS:-120}
declare -A RUN=( [red]=act_v3_red40k [green]=act_v3_green40k [yellow]=act_v3_yellow40k )
PILLS=("$@"); [ ${#PILLS[@]} -eq 0 ] && PILLS=(red yellow green)

echo "════════════════════════════════════════════════════════"
echo "  순수 정책 확인 — 보정 전부 끔"
echo "  ${PILLS[*]}   (색당 ${SECS}초)"
echo "════════════════════════════════════════════════════════"
echo "  자동보정 끔 / 파지점 오프셋 0 / 구역 안 씀"
echo "  * 저장해둔 보정값은 그대로 남습니다 (이번 실행만 0)"
echo
for c in "${PILLS[@]}"; do
  [ -z "${RUN[$c]:-}" ] && { echo "모르는 색: $c"; exit 1; }
  echo "   $c  ← ${RUN[$c]}"
done
echo
read -p "  트레이에 알약을 놓고 엔터 > " _

# run_policy 는 오프셋이 바뀌면 자동 저장한다. 여기서는 0 으로 시작하므로 그대로 두면
# **맞춰둔 값이 0 으로 덮어써진다.** 백업했다가 끝나면 되돌린다.
BK=$(mktemp /tmp/grasp_offset_bk.XXXX.json)
cp grasp_offset.json "$BK"
restore() {
  cp "$BK" grasp_offset.json
  echo
  echo "  ↩ 맞춰둔 보정값 복원됨 (빨강 노랑 초록 그대로)"
  rm -f "$BK"
}
trap restore EXIT INT TERM

i=0
for c in "${PILLS[@]}"; do
  i=$((i+1))
  echo
  echo "────────────────────────────────────────────────────────"
  echo " [$i/${#PILLS[@]}] $c   순수 정책   ($(date '+%H:%M:%S'))"
  echo "────────────────────────────────────────────────────────"
  RUN="${RUN[$c]}" TASK="pick $c pill" \
  timeout -s INT "$SECS" bash run.sh last \
    --repo-id 1unasy/pill_v3_onehot --show --local-keys --relax-on-exit \
    --temporal-ensemble --no-freeze-on-grasp \
    --no-auto-correct \
    --grasp-offset-depth 0 --grasp-offset-lateral 0 --grasp-offset-z 0 \
    --trace \
    2>&1 | grep -vE "QFontDatabase|Note that Qt|Corrupt JPEG|^INFO|WARN:"
  echo " [$i/${#PILLS[@]}] $c 종료"
  sleep 3
done

echo
echo "════════════════════════════════════════════════════════"
echo "  끝  $(date '+%H:%M:%S')"
echo "  잘 따라갔으면 → 학습은 제대로 됐고, 보정이 문제였다"
echo "  못 따라갔으면 → 보정이 그동안 실력을 메우고 있었다"
echo "════════════════════════════════════════════════════════"
