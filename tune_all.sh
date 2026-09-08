#!/usr/bin/env bash
# 세 색을 보정 0 으로 돌리며 **파지 순간 사진**을 모은다.
# 끝나면 analyze_grasp.py 가 그 사진에서 보정값을 계산해 넣는다.
#
# 화살표로 감을 잡는 대신 사진을 재서 값을 낸다 (2026-08-07).
#   bash tune_all.sh              # 빨강 노랑 초록
#   SECS=100 bash tune_all.sh red
set -u
cd /home/user/omx_pill_project
SECS=${SECS:-120}
declare -A RUN=( [red]=act_v3_red40k [green]=act_v3_green40k [yellow]=act_v3_yellow40k )
PILLS=("$@"); [ ${#PILLS[@]} -eq 0 ] && PILLS=(red yellow green)

echo "════════════════════════════════════════════════════════"
echo "  파지 순간 촬영 — 보정 0 으로 돌린다"
echo "  ${PILLS[*]}   (색당 ${SECS}초)"
echo "════════════════════════════════════════════════════════"
echo "  집을 때마다 사진이 남습니다. 여러 번 집을수록 정확해집니다."
echo "  집었으면 알약을 다시 놓아주세요 (자리를 조금씩 바꿔가며)."
echo
BK=$(mktemp /tmp/go_bk.XXXX.json); cp grasp_offset.json "$BK"
trap 'cp "$BK" grasp_offset.json; rm -f "$BK"; echo; echo "  ↩ 보정값 원복"' EXIT INT TERM
read -p "  트레이에 알약을 놓고 엔터 > " _

for c in "${PILLS[@]}"; do
  mkdir -p "grasp_shots/$c"
  echo
  echo "──────────────────────────────────────────"
  echo " $c   촬영 중   ($(date '+%H:%M:%S'))"
  echo "──────────────────────────────────────────"
  RUN="${RUN[$c]}" TASK="pick $c pill" \
  timeout -s INT "$SECS" bash run.sh last \
    --repo-id 1unasy/pill_v3_onehot --show --local-keys --relax-on-exit \
    --temporal-ensemble --no-freeze-on-grasp \
    --no-auto-correct --grasp-offset-depth 0 --grasp-offset-lateral 0 \
    --dump-grasp "grasp_shots/$c" --trace \
    2>&1 | grep -vE "QFontDatabase|Note that Qt|Corrupt JPEG|^INFO|WARN:"
  n=$(ls grasp_shots/$c/*_wrist.png 2>/dev/null | wc -l)
  echo " $c 종료 — 사진 ${n}장"
  sleep 3
done
echo
echo "════════════════════════════════════════════════════════"
for c in "${PILLS[@]}"; do
  echo "  $c: $(ls grasp_shots/$c/*_wrist.png 2>/dev/null | wc -l)장"
done
echo "  다음:  python analyze_grasp.py --color <색> --apply"
echo "════════════════════════════════════════════════════════"
