#!/usr/bin/env bash
# 같은 장면 × 3목표 촬영 — 목표 조건화를 가르치는 데이터를 찍는다 (2026-08-09).
#
# 왜 이렇게 찍나:
#   지금 데이터는 각 배치가 **딱 한 번, 한 목표로만** 나온다. 그래서 장면만 외워도
#   학습 손실이 전부 맞고, 목표(원-핫/언어)를 읽을 이유가 없다.
#   실측 결과 원-핫 1.00배 · SmolVLA 1.02배 — 목표를 완전히 무시했다.
#   같은 배치에 세 목표를 붙이면 장면만으로는 답이 셋이라, 목표를 읽어야만 손실이 준다.
#
# 정지 시연도 같이 찍는다:
#   목표 색을 **약통에 넣어둔 채로** 그 색을 시키고 팔을 10초간 가만히 둔다.
#   이것이 로봇이 실제로 마주치는 장면이다 — 빨강을 담고 나면 빨강은 약통에 있고
#   트레이엔 초록·노랑만 남는다. 그때 멈춰야 하는데 지금은 회복 동작을 반복한다.
#
#   bash record_sets.sh          # 세트 1 부터
#   bash record_sets.sh 7        # 세트 7 부터 (이어서)
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3}
EP=${EP:-120}          # 에피소드 상한(초) — → 로 넘긴다
RESET=${RESET:-90}
STOP_EP=${STOP_EP:-15} # 정지 시연 길이(초)
START=${1:-1}
PROG=.record_sets_progress

TOP=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0
WRIST=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
CAMS="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }"

# declare -A 가 없으면 bash 가 한글 첨자를 산술식으로 읽어 터진다 (색이 뒤섞인다).
declare -A EN=( [빨강]=red [초록]=green [노랑]=yellow )

kill_preview() {   # 미리보기가 켜져 있으면 카메라를 뺏겨 녹화가 통째로 날아간다
  if pgrep -f preview_grid.py > /dev/null 2>&1; then
    echo "  ※ 미리보기가 켜져 있어 끕니다 (카메라 충돌 방지)"
    pkill -f preview_grid.py; sleep 2
  fi
}

rec1() {   # $1 = 영문색, $2 = 안내문
  kill_preview
  echo
  echo "  ▶ 녹화: pick $1 pill    $2"
  echo "    (→ 로 종료·저장 / ← 로 버리고 다시 / ESC 중단)"
  lerobot-record \
    --resume=true \
    --robot.type=omx_follower --robot.port=/dev/omx_follower --robot.id=omx_follower_arm \
    --robot.cameras="$CAMS" \
    --teleop.type=omx_leader --teleop.port=/dev/omx_leader --teleop.id=omx_leader_arm \
    --dataset.repo_id="$REPO" \
    --dataset.single_task="pick $1 pill" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="$3" \
    --dataset.reset_time_s="$RESET" \
    --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
}

TOTAL=$(python3 -c "import json;print(len(json.load(open('layouts.json'))))")
echo "════════════════════════════════════════════════════════════"
echo "  같은 장면 × 3목표 촬영    세트 $START ~ $TOTAL"
echo "  데이터셋: $REPO  (기존에 이어붙임)"
echo "════════════════════════════════════════════════════════════"
echo "  격자      A~E 왼→오 / 1~4 가까움→멂"
echo "  화살표    **색깔 끝**이 향하는 방향 (↑ = 로봇에서 먼 쪽)"
echo "  ※ 정확한 모양은 BATCH_SHEET.pdf 그림을 보세요"
echo
echo "  ★ 세트 안에서 되돌릴 때 **칸과 방향이 모두 같아야** 합니다."
echo "     달라지면 '같은 장면' 이 아니게 되어 이 촬영의 의미가 없어집니다."
echo

for ((S=START; S<=TOTAL; S++)); do
  eval "$(python3 - "$S" <<'PY'
import json,sys
L=json.load(open('layouts.json')); x=L[int(sys.argv[1])-1]
ARROW={0:'→',45:'↗',90:'↑',135:'↖',180:'←',225:'↙',270:'↓',315:'↘'}
print(f"R={x['빨강']}; G={x['초록']}; Y={x['노랑']}")
print(f"RA='{ARROW[x['빨강방향']]}'; GA='{ARROW[x['초록방향']]}'; YA='{ARROW[x['노랑방향']]}'")
print(f"ORDER='{' '.join(x['촬영순서'])}'")
print(f"STOP={1 if x['정지시연'] else 0}")
PY
)"
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  세트 $S / $TOTAL"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  printf "    빨강 %-3s %s      초록 %-3s %s      노랑 %-3s %s\n" "$R" "$RA" "$G" "$GA" "$Y" "$YA"
  echo "    촬영 순서: $ORDER"
  [ "$STOP" = "1" ] && echo "    ※ 이 세트 끝에 정지 시연 있음"
  echo
  read -p "  세 알약을 표대로 놓고 엔터 (q=중단) > " a; [ "$a" = "q" ] && break

  i=0
  for KO in $ORDER; do
    i=$((i+1))
    case "$KO" in 빨강) CELL=$R; ANG=$RA;; 초록) CELL=$G; ANG=$GA;; 노랑) CELL=$Y; ANG=$YA;; esac
    echo
    echo "  ── [$i/3] $KO   $CELL   색깔끝 $ANG ──"
    rec1 "${EN[$KO]}" "$KO 를 집어 약통에 넣으세요" "$EP"
    echo "$S $i" > $PROG
    if [ $i -lt 3 ]; then
      echo
      read -p "  ✅ $KO 를 $CELL 칸에 색깔끝 $ANG 방향으로 되돌리고 엔터 > " _
    fi
  done

  if [ "$STOP" = "1" ]; then
    # 정지 시연: 목표 색을 **약통에 둔 채로** 그 색을 시킨다 = 담은 직후의 장면
    eval "$(python3 - "$S" <<'PY'
import json,sys
L=json.load(open('layouts.json')); x=L[int(sys.argv[1])-1]
print(f"STOPKO={x['정지색']}; STOPTYPE={x['정지유형']}")
print(f"STOPADD={x.get('정지추가색','')}")
left=[c for c in ('빨강','초록','노랑') if c not in (x['정지색'], x.get('정지추가색',''))]
print(f"STOPLEFT={left[0] if x.get('정지추가색') else ''}; STOPLEFTCELL={x[left[0]] if x.get('정지추가색') else ''}")
PY
)"
    case "$STOPKO" in 빨강) CELL=$R;; 초록) CELL=$G;; 노랑) CELL=$Y;; esac
    case "$STOPKO" in 빨강) STOPEN=red;; 초록) STOPEN=green;; 노랑) STOPEN=yellow;; esac
    echo
    echo "  ── 정지 시연 : $STOPKO  ($STOPTYPE) ──"
    if [ "$STOPTYPE" = "빈트레이" ]; then
      echo "     알약 **세 개 모두** 약통에 넣으세요. 트레이를 완전히 비웁니다."
      echo "     (3색 처방이 끝난 직후의 장면입니다)"
    elif [ "$STOPTYPE" = "두개담김" ]; then
      echo "     $STOPKO 와 $STOPADD **두 개**를 약통에 넣으세요."
      echo "     트레이에는 **$STOPLEFT 하나만** 남습니다 ($STOPLEFTCELL 칸, 각도 그대로)."
      echo "     (2색 처방이 끝난 직후의 장면입니다)"
    else
      echo "     $STOPKO 만 약통에 넣으세요. **나머지 두 색은 트레이에 그대로 둡니다**"
      echo "     (칸도 각도도 건드리지 마세요 — 담은 직후의 장면이어야 합니다)"
    fi
    echo "     → 과제는 'pick $STOPEN pill' 인데 트레이에 없는 상태입니다."
    echo
    echo "     ★ 리더암을 **홈 자세**에 두고 시작하세요."
    echo "       (집기 에피소드가 끝난 자리 = 홈. 담고 돌아온 그 자세입니다)"
    echo "       홈이 아닌 자세에서 찍으면 엉뚱한 자세에서 멈추는 걸 배웁니다."
    echo
    echo "     → 그 자세에서 **전혀 움직이지 말고** ${STOP_EP}초를 흘려보내세요."
    echo
    read -p "  준비되면 엔터 > " _
    rec1 "$STOPEN" "움직이지 마세요 — ${STOP_EP}초" "$STOP_EP"
    echo
    read -p "  ✅ 약통의 알약을 원래 칸·각도로 되돌리고 엔터 > " _
  fi
  echo "  세트 $S 완료."
done

echo
echo "════════════════════════════════════════════════════════════"
echo "  촬영 종료. 다음:"
echo "    python make_onehot.py --check     # 색별 개수 확인"
echo "    python make_onehot.py             # 원-핫판 갱신"
echo "════════════════════════════════════════════════════════════"
