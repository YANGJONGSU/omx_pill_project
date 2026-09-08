#!/usr/bin/env bash
# 시연 영상 녹화 — 화면(카메라 창 + 터미널)을 mp4 로 남긴다.
#
# 발표 자료용. 실행하면 녹화가 시작되고, Ctrl+C 나 stop 으로 끝낸다.
# 로봇 실행과 **따로** 돌린다 — 먼저 이걸 켜고, 다른 창에서 prescribe/pharmacy 를 돌린다.
#
#   bash ~/omx_pill_project/record_demo.sh              # 전체 화면
#   bash ~/omx_pill_project/record_demo.sh 처방_초록노랑   # 이름 지정
#   bash ~/omx_pill_project/record_demo.sh stop         # 녹화 중지
#
# 결과: report/demo_<이름>_<시각>.mp4
set -u
OUT=/home/user/omx_pill_project/report
PIDF=/tmp/omx_record.pid
mkdir -p "$OUT"

if [ "${1:-}" = "stop" ]; then
  if [ -f "$PIDF" ]; then
    kill -INT "$(cat "$PIDF")" 2>/dev/null
    sleep 2
    rm -f "$PIDF"
    echo "녹화 중지. 파일:"
    ls -lt "$OUT"/demo_*.mp4 2>/dev/null | head -3
  else
    echo "녹화 중이 아닙니다."
  fi
  exit 0
fi

if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "이미 녹화 중입니다 (PID $(cat "$PIDF")). 먼저 'stop' 하세요."
  exit 1
fi

NAME=${1:-demo}
F="$OUT/demo_${NAME}_$(date +%m%d_%H%M).mp4"

# 화면 크기를 짝수로 맞춘다 (libx264 는 홀수 해상도를 거부한다)
DIM=$(DISPLAY=:1 xdpyinfo 2>/dev/null | awk '/dimensions/{print $2}')
W=${DIM%x*}; H=${DIM#*x}
W=$((W - W % 2)); H=$((H - H % 2))

echo "녹화 시작: $F  (${W}x${H})"
echo "  끝내려면:  bash ~/omx_pill_project/record_demo.sh stop"

# WIN=1 이면 카메라 창만 잡는다. 전체 화면을 찍으면 터미널 오류 메시지가 화면 절반을
# 덮어 발표 영상으로 못 쓴다 (2026-08-07). 창 위치는 xdotool 로 찾는다.
GEO=""
if [ "${WIN:-0}" = "1" ]; then
  WID=$(DISPLAY=:1 xdotool search --name "policy view" 2>/dev/null | head -1)
  if [ -n "$WID" ]; then
    eval "$(DISPLAY=:1 xdotool getwindowgeometry --shell "$WID" 2>/dev/null)"
    WW=$((WIDTH - WIDTH % 2)); WH=$((HEIGHT - HEIGHT % 2))
    GEO="-video_size ${WW}x${WH}"; OFF="+${X},${Y}"
    echo "  카메라 창만 녹화: ${WW}x${WH} @ ${X},${Y}"
  else
    # xdotool 이 없다. 2026-08-07 녹화본에서 잰 창 위치를 쓴다 (3424x1926 화면 기준,
    # 카메라 두 장이 나란히 있는 영역만). 창을 옮겼으면 CAMGEO 로 덮어쓴다.
    GEO="-video_size ${CAMW:-2552}x${CAMH:-956}"; OFF="+${CAMX:-136},${CAMY:-216}"
    echo "  카메라 영역만 녹화: ${CAMW:-2552}x${CAMH:-956} @ ${CAMX:-136},${CAMY:-216}"
    echo "  (창을 옮겼으면 CAMX/CAMY/CAMW/CAMH 로 조정하세요)"
  fi
fi
DISPLAY=:1 nohup ffmpeg -y -f x11grab -framerate 20 ${GEO:--video_size "${W}x${H}"} -i ":1.0${OFF:-}" \
  -c:v libx264 -preset veryfast -crf 26 -pix_fmt yuv420p "$F" \
  > /tmp/omx_record.log 2>&1 &
echo $! > "$PIDF"
sleep 3
if kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "  녹화 중 (PID $(cat "$PIDF"))"
else
  echo "  !! 시작 실패 — /tmp/omx_record.log 확인"
  tail -5 /tmp/omx_record.log
  rm -f "$PIDF"
fi
