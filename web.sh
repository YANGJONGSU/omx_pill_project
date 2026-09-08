#!/usr/bin/env bash
# 조제 웹 화면을 켠다. venv 활성화를 알아서 한다.
#
#   bash web.sh              내 컴퓨터에서만 (실제 로봇)
#   bash web.sh 공유         같은 와이파이 사람들에게 공개
#   bash web.sh 시뮬         로봇 없이 화면만 (남이 눌러도 안전)
#   bash web.sh 공유 시뮬    화면만 + 공개
#
# REC=1 을 앞에 붙이면 정책이 본 화면을 mp4 로 남긴다.
set -uo pipefail
cd /home/user/omx_pill_project/web
source /home/user/venv/il/bin/activate
export HF_HUB_OFFLINE=1

HOST=127.0.0.1
REAL="--real"
for a in "$@"; do
  case "$a" in
    공유|share) HOST=0.0.0.0 ;;
    시뮬|sim)   REAL="" ;;
  esac
done

PORT=${PORT:-8000}
echo "════════════════════════════════════════════"
if [ -n "$REAL" ]; then echo "  실제 로봇 모드 — 조제 버튼이 팔을 움직입니다"
else                    echo "  시뮬레이션 — 로봇은 움직이지 않습니다"; fi
if [ "$HOST" = 0.0.0.0 ]; then
  IP=$(hostname -I | awk '{print $1}')
  echo
  echo "  다른 사람은 이 주소로 접속합니다:"
  echo "    http://$IP:$PORT"
  echo
  echo "  안 열리면 방화벽을 한 번만 여십시오:"
  echo "    sudo ufw allow $PORT/tcp"
else
  echo "  이 컴퓨터에서만: http://127.0.0.1:$PORT"
fi
echo "════════════════════════════════════════════"
echo

python app.py $REAL --host "$HOST" --port "$PORT"
