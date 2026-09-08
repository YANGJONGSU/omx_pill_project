#!/usr/bin/env bash
# 연속 조제 시연 — 색을 순서대로 하나씩 집는다.
#
# **검증된 `run.sh`(run_policy.py) 를 색마다 한 번씩 부른다.** 2026-08-06 에 내가 새로 짠
# pick 루프(prescribe.py)는 각 색이 따로는 되는데 이어서 돌리면 팔이 안 움직였다.
# 그래서 새 루프를 쓰지 않고, 되는 것을 그대로 순서대로 호출한다.
#
# 각 색마다 제한 시간을 주고, 시간이 되면 SIGINT 를 보낸다 — run_policy.py 는 그걸 받으면
# **홈으로 복귀한 뒤 정상 종료**한다(강제 종료가 아니다).
#
#   bash ~/omx_pill_project/demo_chain.sh red green
#   bash ~/omx_pill_project/demo_chain.sh red green yellow
#   SECS=90 bash ~/omx_pill_project/demo_chain.sh red green    # 색당 90초
set -u
cd /home/user/omx_pill_project

SECS=${SECS:-70}          # 색 하나에 줄 시간
# 빨강은 red29 를 쓴다 (2026-08-07). red40k 는 회복 시연 7개(ep89~95)가 들어 있어
# 집고 나서도 계속 재시도한다 — 실기 촬영으로 확인했다: 알약을 물고 있으면서
# 87초 동안 그리퍼를 7번 다시 닫았다. red29 는 회복 시연이 없어 담고 멈춘다.
# 색 구분 지표도 red29 가 5.45배로 red40k(3.85배)보다 높다.
declare -A RUN=( [red]=act_v3_red29 [green]=act_v3_green40k [yellow]=act_v3_yellow40k )
# 색마다 **영점을 맞춘 조건 그대로** 돌려야 한다 (2026-08-07).
# 오프셋 해석이 두 가지다:
#   --radial-offset 있음 = 팔의 반경 방향 (옛 동작)
#   없음                 = 책상 좌표계 (기본) — 좌우로 갈수록 둘이 벌어진다
# 빨강·노랑은 반경 기준으로, 초록은 책상 기준으로 맞췄다. 섞으면 그만큼 어긋난다.
declare -A EXTRA=( [red]="--radial-offset" [yellow]="--radial-offset" [green]="" )
declare -A REPO=( [red]=1unasy/pill_v3_onehot [green]=1unasy/pill_v3_onehot [yellow]=1unasy/pill_v3_onehot )

PILLS=("$@")
if [ ${#PILLS[@]} -eq 0 ]; then
  # 정형외과 처방 — 색 조합을 번호로 고른다
  declare -a MENU_NAME=( "단순 타박상" "급성 요통" "근육 염좌" "만성 관절염" "위염 동반 통증" )
  declare -a MENU_PILL=( "red" "red green" "red yellow" "red yellow green" "green yellow" )
  declare -A DRUG=( [red]=소염진통제 [yellow]=근이완제 [green]=위장보호제 )
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║            약국 자동 조제 — 정형외과                 ║"
  echo "╠══════════════════════════════════════════════════════╣"
  for i in "${!MENU_NAME[@]}"; do
    d=""
    for c in ${MENU_PILL[$i]}; do d="$d ${DRUG[$c]}($c)"; done
    printf "║ %d. %-12s%-32s║\n" "$((i+1))" "${MENU_NAME[$i]}" "$d"
  done
  echo "╚══════════════════════════════════════════════════════╝"
  read -p "  병명 번호 > " SEL
  idx=$((SEL-1))
  [ -z "${MENU_PILL[$idx]:-}" ] && { echo "없는 번호입니다."; exit 1; }
  echo "  → ${MENU_NAME[$idx]}"
  read -ra PILLS <<< "${MENU_PILL[$idx]}"
fi

echo "════════════════════════════════════════════"
echo " 연속 조제:  ${PILLS[*]}"
echo " 색당 제한 시간 ${SECS}초 (지나면 홈 복귀 후 다음 색)"
echo "════════════════════════════════════════════"
for c in "${PILLS[@]}"; do
  [ -z "${RUN[$c]:-}" ] && { echo "모르는 색: $c"; exit 1; }
  echo "   $c  ← ${RUN[$c]}"
done
echo
read -p "트레이에 알약을 놓고 엔터를 누르세요 > " _

i=0
for c in "${PILLS[@]}"; do
  i=$((i+1))
  echo
  echo "──────────────────────────────────────────"
  mkdir -p "grasp_shots/$c"
  echo " [$i/${#PILLS[@]}] $c   ($(date '+%H:%M:%S'))"
  echo "──────────────────────────────────────────"
  # --start-paused 를 빼서 홈 복귀 후 바로 시작한다.
  # timeout -s INT 로 끝내면 run_policy 가 홈 복귀 후 정상 종료한다.
  RUN="${RUN[$c]}" TASK="pick $c pill" \
  timeout -s INT "$SECS" bash run.sh last \
    --repo-id "${REPO[$c]}" --show --local-keys --relax-on-exit \
    --temporal-ensemble --no-freeze-on-grasp ${EXTRA[$c]} --offset-step 1 \
    --dump-grasp "grasp_shots/$c" --trace \
    2>&1 | grep -vE "QFontDatabase|Note that Qt|Corrupt JPEG|^INFO|WARN:"
  echo " [$i/${#PILLS[@]}] $c 종료"
  sleep 3
done

echo
echo "════════════════════════════════════════════"
echo " 연속 조제 끝  $(date '+%H:%M:%S')"
echo "════════════════════════════════════════════"
