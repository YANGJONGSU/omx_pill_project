#!/usr/bin/env bash
# 약국 자동 조제 — 병명을 번호로 고르면 그 처방대로 연속 조제한다.
#
# 실행은 두 방식 중 하나로 한다:
#   prescribe (기본) — 정책 세 개를 미리 올려두고 색마다 갈아끼운다.
#                      **카메라를 껐다 켜지 않고**, 시간 제한 대신 그리퍼 신호로 완료를 감지한다.
#   chain           — 색마다 run.sh(run_policy.py)를 새로 부른다. 카메라가 색마다 껐다 켜지고
#                      시간 제한으로 끊는다. prescribe 가 안 움직일 때의 대안.
#
#   bash ~/omx_pill_project/menu.sh
#   ENGINE=chain bash ~/omx_pill_project/menu.sh
#   TIMEOUT=200 bash ~/omx_pill_project/menu.sh
set -u
cd /home/user/omx_pill_project
ENGINE=${ENGINE:-prescribe}
TIMEOUT=${TIMEOUT:-150}

# 정형외과 처방. 소염진통제에는 위장보호제를 같이 내는 실제 처방을 반영했다.
declare -a NAME=( "단순 타박상" "급성 요통" "근육 염좌" "만성 관절염" "위염 동반 통증" )
declare -a PILL=( "red" "red green" "red yellow" "red yellow green" "green yellow" )
declare -A DRUG=( [red]="소염진통제" [yellow]="근이완제" [green]="위장보호제" )

while true; do
  echo
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║             약국 자동 조제 — 정형외과                    ║"
  echo "╠══════════════════════════════════════════════════════════╣"
  for i in "${!NAME[@]}"; do
    d=""
    for c in ${PILL[$i]}; do d="$d ${DRUG[$c]}"; done
    printf "║ %d. %-16s%-32s║\n" "$((i+1))" "${NAME[$i]}" "$d"
  done
  echo "║ 0. 종료                                                  ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  read -p "  병명 번호 > " SEL
  [ "${SEL:-0}" = "0" ] && break
  idx=$((SEL-1))
  if [ -z "${PILL[$idx]:-}" ]; then
    echo "  없는 번호입니다."
    continue
  fi
  echo
  echo "  ■ ${NAME[$idx]}  —  ${PILL[$idx]}"
  read -p "  트레이에 알약을 놓고 엔터 > " _
  if [ "$ENGINE" = "chain" ]; then
    SECS=$TIMEOUT bash demo_chain.sh ${PILL[$idx]} < /dev/null
  else
    source /home/user/venv/il/bin/activate
    PYTHONUNBUFFERED=1 python prescribe.py --pills ${PILL[$idx]} \
      --relax-on-exit --show --timeout "$TIMEOUT"
  fi
  # 같은 처방을 바로 다시 할 수 있게 한다 (실험할 때 번호를 매번 안 고르도록)
  while true; do
    echo
    echo "  ── ${NAME[$idx]} 끝 ──   r 다시 / m 메뉴 / q 종료"
    read -p "  > " AGAIN
    case "${AGAIN:-m}" in
      r|R)
        read -p "  트레이를 다시 채우고 엔터 > " _
        if [ "$ENGINE" = "chain" ]; then
          SECS=$TIMEOUT bash demo_chain.sh ${PILL[$idx]} < /dev/null
        else
          source /home/user/venv/il/bin/activate
          PYTHONUNBUFFERED=1 python prescribe.py --pills ${PILL[$idx]} --relax-on-exit --show --timeout "$TIMEOUT"
        fi
        ;;
      q|Q) echo "종료합니다."; exit 0 ;;
      *) break ;;
    esac
  done
done
echo "종료합니다."
