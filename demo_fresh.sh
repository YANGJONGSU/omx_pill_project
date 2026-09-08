#!/usr/bin/env bash
# 색마다 **카메라를 완전히 다시 잡고** 하나씩 돌린다.
#
# 왜 (2026-08-07 실측): 단독 실행은 다 성공했고 연속 실행은 다 실패했다.
#   노랑 단독 읽기실패 22회 성공 / 초록 단독 11회 성공 / 빨강 단독 77회·미달0.2% 성공
#   3색 연속 419회 실패 / 3색 연속 154회·미달8.2% 실패
# wrist 케이블이 불량이라 오래 돌릴수록 무너진다. 색 사이에 카메라를 쉬게 해서
# 각 색이 "단독 실행"과 같은 조건에서 시작하게 한다.
#
#   bash demo_fresh.sh red yellow green
#   REST=20 bash demo_fresh.sh          # 색 사이 대기(초)
set -u
cd /home/user/omx_pill_project
SECS=${SECS:-150}; REST=${REST:-15}
declare -A RUN=( [red]=act_v3_red29 [green]=act_v3_green40k [yellow]=act_v3_yellow40k )
declare -A EXTRA=( [red]="--radial-offset" [yellow]="--radial-offset" [green]="" )
PILLS=("$@"); [ ${#PILLS[@]} -eq 0 ] && PILLS=(red yellow green)
W=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0

health() {   # 최근 2분 wrist 끊김
  journalctl -k --since "2 min ago" --no-pager 2>/dev/null | grep -c "usb 1-3.3: USB disconnect"
}

echo "════════════════════════════════════════════════════"
echo "  색마다 카메라를 다시 잡고 하나씩   ${PILLS[*]}"
echo "════════════════════════════════════════════════════"
read -p "  트레이에 알약을 놓고 엔터 > " _
WIN=${WIN:-1} bash record_demo.sh "$(IFS=-; echo "${PILLS[*]}")_fresh"
sleep 2

i=0
for c in "${PILLS[@]}"; do
  i=$((i+1))
  mkdir -p "grasp_shots/$c"
  echo
  echo "──────────────────────────────────────────"
  echo " [$i/${#PILLS[@]}] $c   ${RUN[$c]}   ($(date '+%H:%M:%S'))"
  [ ! -e "$W" ] && { echo "  ⚠ wrist 없음 — 케이블 확인 후 엔터"; read _; }
  echo "  최근 2분 끊김 $(health)회"
  echo "──────────────────────────────────────────"
  RUN="${RUN[$c]}" TASK="pick $c pill" \
  timeout -s INT "$SECS" bash run.sh last \
    --repo-id 1unasy/pill_v3_onehot --show --local-keys --relax-on-exit \
    --temporal-ensemble --no-freeze-on-grasp ${EXTRA[$c]} --offset-step 1 \
    --dump-grasp "grasp_shots/$c" --trace \
    2>&1 | grep -vE "QFontDatabase|Note that Qt|Corrupt JPEG|^INFO|WARN:"
  echo " [$i/${#PILLS[@]}] $c 종료 — 사진 $(ls grasp_shots/$c/*_wrist.png 2>/dev/null|wc -l)장"
  # 카메라를 쉬게 한다. 다음 색이 새로 잡으면서 상태가 회복된다.
  if [ $i -lt ${#PILLS[@]} ]; then
    echo "  … 카메라 회복 대기 ${REST}초"
    sleep "$REST"
  fi
done
sleep 2; bash record_demo.sh stop
echo
echo "════════════════════════════════════════════════════"
echo "  끝  $(date '+%H:%M:%S')"
echo "  영점 계산:  python analyze_grasp.py --color <색>"
echo "════════════════════════════════════════════════════"
