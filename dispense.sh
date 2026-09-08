#!/usr/bin/env bash
# 처방 조합대로 연속 조제한다 — 신규 색별 정책(224 재학습) (2026-08-15).
#
# demo_fresh.sh 의 구조를 그대로 쓴다. 핵심은 **색 사이에 카메라를 완전히 다시 잡는
# 것**이다 (2026-08-07 실측: 단독 실행은 다 성공, 연속 실행은 다 실패했다.
# wrist 케이블이 불량이라 오래 돌릴수록 무너진다). 색마다 프로세스를 새로 띄우고
# 사이에 쉬게 해서, 각 색이 "단독 실행"과 같은 조건에서 시작하게 한다.
#
# 정책은 색마다 다르고 체크포인트도 다르다 — 마지막이 아니라 측정해서 가장
# 좋았던 지점이다 (초록 87502 / 노랑 60000 / 빨강 90000).
#
#   bash dispense.sh red green          빨강 → 초록 순서로
#   bash dispense.sh OS-02              처방 코드로 (web/prescriptions.json)
#   bash dispense.sh                    처방 목록 보기
#   OLD=1 bash dispense.sh red green    기존 정책으로 비교
#   SECS=120 REST=20 bash dispense.sh red green
set -u
cd /home/user/omx_pill_project

SECS=${SECS:-150}      # 색 하나에 주는 **최대** 시간. 보통은 그 전에 끝난다 —
                       # 알약을 약통에 놓는 순간 run_policy 가 "✅ 담기 완료" 를 찍고,
                       # 아래에서 그걸 보면 바로 다음 색으로 넘어간다. 시간은 안전망이다.
                       # (정지 조건이 아직 없어 그냥 두면 남은 알약까지 집는다.)
REST=${REST:-5}        # 색 사이 카메라 회복 대기. 0 으로 두면 연속 실행이 무너질 수
                       # 있다 (2026-08-07 실측). 여기가 유일하게 줄일 수 있는 대기다 —
                       # 나머지 지연(정책 로드·카메라 열기·로봇 연결)은 색을 바꿀 때마다
                       # 프로세스를 새로 띄우는 구조에서 오는 고정 비용이다.
HOME_T=${HOME_T:-3.0}  # 홈 복귀 시간. 1.5 로 줄였더니 노랑이 못 집었다 —
                       # 팔이 덜 안정된 상태에서 시작하면 초기 자세가 학습과 어긋난다.
                       # 검증된 기본값(3.0)으로 되돌린다.
SHOW=${SHOW:-1}        # 0 이면 카메라 창을 띄우지 않는다. 창 생성이 빠지면 몇 초 빨라지고
                       # 시연 화면도 깔끔하다. 대신 창에서 키로 보정할 수 없다(Ctrl+C 는 됨).
OLD=${OLD:-0}

declare -A NEW_RUN=( [red]=act_red_224   [green]=act_green_224   [yellow]=act_yellow_224 )
declare -A NEW_CK=(  [red]=090000        [green]=087502          [yellow]=089118 )
declare -A OLD_RUN=( [red]=act_v3_red29  [green]=act_v3_green40k [yellow]=act_v3_yellow40k )
declare -A EXTRA=(   [red]="--radial-offset" [yellow]="--radial-offset" [green]="" )
declare -A KOR=(     [red]=빨강 [green]=초록 [yellow]=노랑 )

W=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
T=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0

# ── 인자 해석: 처방 코드면 조합으로 펼친다 ────────────────────────────────
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  echo "처방 목록 — bash dispense.sh <코드>  또는  bash dispense.sh <색> <색> ..."
  echo
  python3 - <<'PY'
import json
d = json.load(open("web/prescriptions.json"))
약 = d["약품"]
for p in d["처방"]:
    조합 = " + ".join(f"{약[c]['색이름']}({약[c]['이름']})" for c in p["조합"])
    print(f"  {p['코드']}  {p['병명']:<10s}  {조합}")
PY
  echo
  echo "  색 이름으로도 됩니다:  bash dispense.sh red green yellow"
  exit 0
fi

if [[ "${ARGS[0]}" =~ ^[A-Z]{2}-[0-9]{2}$ ]]; then
  CODE="${ARGS[0]}"
  read -r -a PILLS <<< "$(python3 - "$CODE" <<'PY'
import json, sys
d = json.load(open("web/prescriptions.json"))
for p in d["처방"]:
    if p["코드"] == sys.argv[1]:
        print(" ".join(p["조합"])); break
else:
    sys.exit(1)
PY
)" || { echo "모르는 처방 코드: $CODE"; exit 1; }
  BANNER="$CODE — $(python3 -c "
import json,sys
d=json.load(open('web/prescriptions.json'))
print(next(p['병명'] for p in d['처방'] if p['코드']=='$CODE'))")"
else
  PILLS=("${ARGS[@]}")
  BANNER="직접 지정"
fi

for c in "${PILLS[@]}"; do
  case "$c" in red|green|yellow) ;; *) echo "모르는 색: $c"; exit 1;; esac
done

# ── 사전 점검 ─────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════"
echo "  연속 조제  —  $BANNER"
printf "  순서:"; for c in "${PILLS[@]}"; do printf " %s" "${KOR[$c]}"; done; echo
echo "  정책: $([ "$OLD" = 1 ] && echo '기존 3개' || echo '신규 224 재학습')"
echo "════════════════════════════════════════════════════"
[ ! -e "$T" ] && { echo "  ⚠ top 카메라 없음 — 케이블 확인"; exit 1; }
[ ! -e "$W" ] && { echo "  ⚠ wrist 카메라 없음 — 케이블 확인"; exit 1; }
echo "  카메라 2대 정상"
read -p "  트레이에 알약을 놓고 엔터 > " _

i=0
for c in "${PILLS[@]}"; do
  i=$((i+1))
  if [ "$OLD" = 1 ]; then
    R=${OLD_RUN[$c]}; CK=last; REPO=1unasy/pill_v3_onehot; EX="${EXTRA[$c]}"
  else
    R=${NEW_RUN[$c]}; CK=${NEW_CK[$c]}; REPO=1unasy/pill_v3; EX="${EXTRA[$c]}"
  fi
  mkdir -p "grasp_shots/$c"
  echo
  echo "──────────────────────────────────────────"
  echo " [$i/${#PILLS[@]}] ${KOR[$c]}   $R / $CK   ($(date '+%H:%M:%S'))"
  [ ! -e "$W" ] && { echo "  ⚠ wrist 사라짐 — 케이블 확인 후 엔터"; read -r _; }
  echo "──────────────────────────────────────────"

  # 중간 색에서는 토크를 끄지 않는다 — 힘을 빼면 팔이 처지고 다음 색이 처진
  # 자세에서 홈 복귀를 시작한다 (2026-08-18).
  RELAX=""; [ $i -eq ${#PILLS[@]} ] && RELAX="--relax-on-exit"
  LOG="/tmp/dispense_${c}_$$.log"
  RUN="$R" TASK="pick $c pill" \
  timeout -s INT "$SECS" bash run.sh "$CK" \
    --repo-id "$REPO" $([ "$SHOW" = 1 ] && echo "--show --local-keys") $RELAX \
    --temporal-ensemble --no-freeze-on-grasp $EX --offset-step 1 \
    --home-time "$HOME_T" --dump-grasp "grasp_shots/$c" --trace \
    > "$LOG" 2>&1 &
  PID=$!

  # 로그를 보면서 진행 상황을 그대로 보여주고, "담기 완료" 가 나오면 끝낸다.
  tail -n +1 -f "$LOG" --pid=$PID 2>/dev/null \
    | grep -vE "QFontDatabase|Note that Qt|Corrupt JPEG|^INFO|WARN:" &
  TAIL=$!

  DONE=0
  while kill -0 $PID 2>/dev/null; do
    if grep -q "담기 완료" "$LOG" 2>/dev/null; then
      echo "  → 담았습니다. 정리하고 다음으로 넘어갑니다."
      DONE=1
      pkill -INT -f "run_policy.py --run $R" 2>/dev/null
      # 홈 복귀에 시간을 주고, 그래도 안 끝나면 확실히 끝낸다.
      for _ in $(seq 1 12); do kill -0 $PID 2>/dev/null || break; sleep 1; done
      if kill -0 $PID 2>/dev/null; then
        echo "  (정상 종료가 늦어 강제 종료합니다)"
        pkill -f "run_policy.py --run $R" 2>/dev/null; sleep 2
        kill -9 $PID 2>/dev/null
      fi
      break
    fi
    sleep 0.5
  done
  wait $PID 2>/dev/null
  kill $TAIL 2>/dev/null; wait $TAIL 2>/dev/null
  rm -f "$LOG"
  [ "$DONE" = 1 ] || echo "  ⚠ 시간 안에 담지 못했습니다 (${SECS}초)"

  echo " [$i/${#PILLS[@]}] ${KOR[$c]} 종료"
  if [ $i -lt ${#PILLS[@]} ]; then
    echo "  … 카메라 회복 대기 ${REST}초 (연속 실행이 무너지지 않게 하는 핵심입니다)"
    sleep "$REST"
  fi
done

echo
echo "════════════════════════════════════════════════════"
echo "  조제 완료  $(date '+%H:%M:%S')"
echo "  영점 확인:  python analyze_grasp.py --color <색>"
echo "════════════════════════════════════════════════════"
