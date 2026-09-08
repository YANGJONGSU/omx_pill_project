#!/usr/bin/env bash
# 처방 조합대로 연속 조제 — **원-핫 단일 정책** (2026-08-16).
#
# dispense.sh(색별 정책 3개)와 같은 처방표를 쓰되, **정책 파일이 하나**다.
# 색이 바뀌어도 다른 모델을 불러오지 않고 목표 원-핫만 바꾼다.
#
#   색별 정책 3개   모델 3개를 따로 관리, 색마다 다른 모델 로드
#   원-핫 정책 1개  모델 하나로 세 색 — 색 추가 시 데이터만 늘리면 된다
#
# 색 전환은 한 프로세스 안에서 이뤄진다. 카메라·모델을 다시 열지 않으므로
# 색별 정책 방식보다 전환이 빠르다.
#
#   bash dispense_onehot.sh OS-04           처방 코드
#   bash dispense_onehot.sh red yellow green 색 직접
#   bash dispense_onehot.sh                  처방 목록
#   POLICY=novae224 bash dispense_onehot.sh OS-02   다른 원-핫 정책으로
#   MODE=fresh bash dispense_onehot.sh OS-04        색마다 재시작 (대안)
#   REC=1 bash dispense_onehot.sh red yellow green  정책이 본 화면을 mp4 로 남긴다
set -uo pipefail
cd /home/user/omx_pill_project

SECS=${SECS:-150}      # 색 하나에 주는 최대 시간
SHOW=${SHOW:-1}
REST=${REST:-5}        # 색 사이 대기
# 색 전환 방식:
#   seq(기본)  한 프로세스 안에서 목표만 바꾼다. 모델을 한 번만 로드한다.
#              전환 시 목표 원-핫 교체 + 정책 큐 초기화만 한다.
#              홈 복귀·카메라 재연결을 넣어 봤으나 오히려 성공 개수가 줄었다.
#   fresh      색마다 프로세스를 새로 띄운다. seq 가 안 될 때의 대안.
HOLD=${HOLD:-1}        # 목표 색이 없을 때 홈 유지 (안전장치).
                       # **정책 자체의 능력을 볼 때는 HOLD=0 으로 끄십시오** —
                       # 이건 학습이 못 하는 일을 코드로 막는 것이지 정책이 배운 게 아닙니다.
MODE=${MODE:-seq}
SEQ_HOME=${HOME_RETURN:-0}   # 색 전환 시 강제 홈 복귀 (기본 끔).
                             # 시연은 "약통에 넣고 스스로 홈으로 돌아가기" 까지 찍혀 있다.
                             # 강제로 되돌리면 그 학습된 복귀 동작을 끊는다. 정책에 맡기고,
                             # 정말 멈췄을 때만 --stall-secs 가 개입한다.
                             # 변수명에 HOME 을 쓰면 셸의 홈 디렉토리를 덮어써
                             # 캐시 경로가 깨진다 (2026-08-16에 겪음).
POLICY=${POLICY:-film224}

declare -A RUN=( [token96]=act_v3_3color [tokenaug]=act_v3_onehot_aug
                 [film96]=act_v3_film    [film224]=act_film_224  [novae224]=act_novae_224
                 [xy]=act_xy_224         [marker]=act_marker2
                 [smolvla3]=smolvla_v3
                 [smolvla4]=smolvla_v4  [smolvla5]=smolvla_v5
                 [smolvla6]=smolvla_v6  [smolvla7]=smolvla_v7  [smolvla8]=smolvla_v8
                 [smolvla9]=smolvla_v9  [smolvla10]=smolvla_v10  [smolvla12]=smolvla_v12 )
declare -A CK=(  [token96]=080000 [tokenaug]=128000
                 [film96]=128000  [film224]=100000 [novae224]=080000
                 [xy]=150000      [marker]=030000  [smolvla3]=020000
                 [smolvla4]=060000 [smolvla5]=020000
                 [smolvla6]=020000 [smolvla7]=020000 [smolvla8]=020000
                 [smolvla9]=010000 [smolvla10]=080000 [smolvla12]=020000 )
declare -A REPO=([token96]=1unasy/pill_v3_onehot [tokenaug]=1unasy/pill_v3_onehot
                 [film96]=1unasy/pill_v3_onehot  [film224]=1unasy/pill_v3_onehot_224
                 [novae224]=1unasy/pill_v3_onehot_224
                 [xy]=1unasy/pill_v3_xy [marker]=1unasy/pill_v3_xy
                 [smolvla3]=1unasy/pill_v3
                 [smolvla4]=1unasy/pill_v3  [smolvla5]=1unasy/pill_v3_auxuv
                 [smolvla6]=1unasy/pill_v3  [smolvla7]=1unasy/pill_v3_langaug
                 [smolvla8]=1unasy/pill_v3_langaug
                 [smolvla9]=1unasy/pill_v3_langaug  [smolvla10]=1unasy/pill_v3_uvstate  [smolvla12]=1unasy/pill_v3_langaug )
declare -A KOR=( [red]=빨강 [green]=초록 [yellow]=노랑 )

[ -n "${RUN[$POLICY]:-}" ] || { echo "모르는 정책: $POLICY  (가능: ${!RUN[*]})"; exit 1; }

# 체크포인트를 실행할 때 바꿔 쓸 수 있게 한다 (2026-08-21).
#   CKPT=010000 POLICY=smolvla12 bash dispense_onehot.sh yellow
# 오프라인 지표가 실기를 예측하지 못하므로(film224 1.25 성공 / v8 2.02 실패)
# 여러 체크포인트를 실기로 직접 비교해야 한다.
if [ -n "${CKPT:-}" ]; then
  # 폴더 이름은 6자리다(090000). 사용자가 90000 이나 88000 을 넣어도 찾아준다 —
  # 6자리로 채워 보고, 없으면 가장 가까운 것을 쓴다.
  _dir="train/${RUN[$POLICY]}/checkpoints"
  _c=$(printf "%06d" "$((10#$CKPT))" 2>/dev/null || echo "$CKPT")
  if [ ! -d "$_dir/$_c" ]; then
    _near=$(ls "$_dir" 2>/dev/null | grep -E '^[0-9]+$' \
            | awk -v t="$((10#$CKPT))" '{d=$1-t; if(d<0)d=-d; print d, $1}' \
            | sort -n | head -1 | cut -d" " -f2)
    if [ -n "$_near" ]; then
      echo "  ⚠ $CKPT 가 없어 가장 가까운 $_near 을(를) 씁니다"
      echo "     있는 것: $(ls "$_dir" | grep -E '^[0-9]+$' | tr '\n' ' ')"
      _c="$_near"
    fi
  fi
  CK[$POLICY]="$_c"
  echo "  체크포인트 덮어씀: $_c"
fi

W=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
T=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0

# ── 인자 해석 ─────────────────────────────────────────────────────────────
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  echo "처방 목록 — bash dispense_onehot.sh <코드>  또는  <색> <색> ..."
  echo
  python3 - <<'PY'
import json
d = json.load(open("web/prescriptions.json")); 약 = d["약품"]
for p in d["처방"]:
    조합 = " + ".join(f"{약[c]['색이름']}({약[c]['이름']})" for c in p["조합"])
    print(f"  {p['코드']}  {p['병명']:<10s}  {조합}")
PY
  echo
  echo "  정책 선택:  POLICY={film224|novae224|film96|tokenaug|token96|xy|marker}"
  echo "              (기본 film224. xy·marker 는 좌표 목표 — 화면에서 HSV 로 검출)"
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
import json
d=json.load(open('web/prescriptions.json'))
print(next(p['병명'] for p in d['처방'] if p['코드']=='$CODE'))")"
else
  PILLS=("${ARGS[@]}"); BANNER="직접 지정"
fi

for c in "${PILLS[@]}"; do
  case "$c" in red|green|yellow) ;; *) echo "모르는 색: $c"; exit 1;; esac
done

# ── 실행 ──────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════"
echo "  연속 조제 (원-핫 단일 정책)  —  $BANNER"
printf "  순서:"; for c in "${PILLS[@]}"; do printf " %s" "${KOR[$c]}"; done; echo
echo "  정책: ${RUN[$POLICY]}/${CK[$POLICY]}   전환방식: $MODE"
echo "════════════════════════════════════════════════════"
[ ! -e "$T" ] && { echo "  ⚠ top 카메라 없음"; exit 1; }
[ ! -e "$W" ] && { echo "  ⚠ wrist 카메라 없음"; exit 1; }
echo "  카메라 2대 정상"
read -p "  트레이에 알약을 놓고 엔터 > " _

FILTER="QFontDatabase|Note that Qt|Corrupt JPEG|^INFO|WARN:"

if [ "$MODE" = "seq" ]; then
  SEQ=$(IFS=,; echo "${PILLS[*]}")
  RUN="${RUN[$POLICY]}" TASK="pick ${PILLS[0]} pill" \
  timeout -s INT $((SECS * ${#PILLS[@]})) bash run.sh "${CK[$POLICY]}" \
    --repo-id "${REPO[$POLICY]}" $([ "$SHOW" = 1 ] && echo "--show --local-keys") \
    --relax-on-exit $([ "${POLICY#smolvla}" = "$POLICY" ] && [ "$POLICY" != diffusion ] && echo "--temporal-ensemble") --no-freeze-on-grasp --offset-step 1 \
    --sequence "$SEQ" $([ "$SEQ_HOME" = 1 ] && echo "--seq-home") \
      $([ "$HOLD" = 0 ] && echo "--hold-home 0") \
    --dump-grasp "grasp_shots/$POLICY" --trace ${EXTRA:-} \
    ${REC:+--record-video "report/run_${POLICY}_$(date +%H%M%S).mp4"} \
    2>&1 | grep -vE "$FILTER"
else
  i=0
  for c in "${PILLS[@]}"; do
    i=$((i+1))
    echo
    echo "──────────────────────────────────────────"
    echo " [$i/${#PILLS[@]}] ${KOR[$c]}   ($(date '+%H:%M:%S'))"
    echo "──────────────────────────────────────────"
    LOG="/tmp/oh_${c}_$$.log"
    # 중간 색에서는 토크를 끄지 않는다. --relax-on-exit 로 힘을 빼면 팔이 중력으로
    # 처지고, 다음 색이 **처진 자세에서** 홈 복귀를 시작한다. 단독 실행이 되는데
    # 연속만 안 되던 차이가 이것이다 (2026-08-18). 마지막 색에서만 푼다.
    RELAX=""; [ $i -eq ${#PILLS[@]} ] && RELAX="--relax-on-exit"
    RUN="${RUN[$POLICY]}" TASK="pick $c pill" \
    timeout -s INT "$SECS" bash run.sh "${CK[$POLICY]}" \
      --repo-id "${REPO[$POLICY]}" $([ "$SHOW" = 1 ] && echo "--show --local-keys") \
      $RELAX --temporal-ensemble --no-freeze-on-grasp --offset-step 1 \
      --trace > "$LOG" 2>&1 &
    PID=$!
    tail -n +1 -f "$LOG" --pid=$PID 2>/dev/null | grep -vE "$FILTER" &
    TAIL=$!
    DONE=0
    while kill -0 $PID 2>/dev/null; do
      if grep -q "담기 완료" "$LOG" 2>/dev/null; then
        echo "  → 담았습니다. 다음으로 넘어갑니다."
        DONE=1
        pkill -INT -f "run_policy.py --run ${RUN[$POLICY]}" 2>/dev/null
        # 홈 복귀(3초) + 장치 정리에 시간을 넉넉히 준다. kill -9 는 로봇·카메라
        # 연결을 정리하지 않아 다음 색이 이상한 상태에서 시작한다.
        for _ in $(seq 1 25); do kill -0 $PID 2>/dev/null || break; sleep 1; done
        if kill -0 $PID 2>/dev/null; then
          echo "  ⚠ 정상 종료가 늦습니다 — 한 번 더 기다립니다"
          pkill -INT -f "run_policy.py --run ${RUN[$POLICY]}" 2>/dev/null
          for _ in $(seq 1 15); do kill -0 $PID 2>/dev/null || break; sleep 1; done
          kill -0 $PID 2>/dev/null && kill -9 $PID 2>/dev/null
        fi
        break
      fi
      sleep 0.5
    done
    wait $PID 2>/dev/null; kill $TAIL 2>/dev/null; wait $TAIL 2>/dev/null; rm -f "$LOG"
    [ "$DONE" = 1 ] || echo "  ⚠ 시간 안에 담지 못했습니다 (${SECS}초)"
    [ $i -lt ${#PILLS[@]} ] && { echo "  … ${REST}초 대기"; sleep "$REST"; }
  done
fi

echo
echo "════════════════════════════════════════════════════"
echo "  종료  $(date '+%H:%M:%S')"
echo "════════════════════════════════════════════════════"
