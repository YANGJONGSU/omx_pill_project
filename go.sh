#!/usr/bin/env bash
# 신규 색별 정책(224 에피소드 재학습)을 색 이름만으로 켠다 (2026-08-14).
#
# 저장하지 않는다. lerobot-record 가 아니라 run.sh → run_policy.py 를 부르므로
# "켜면 그냥 돈다". 멈추는 건 키를 누를 때뿐이다.
#
#   bash go.sh red        빨강
#   bash go.sh green      초록
#   bash go.sh yellow     노랑
#   bash go.sh red old    기존 정책(act_v3_red29)으로 비교 실행
#
# 실행 중 키 (카메라 창에 포커스):
#   스페이스   홈 복귀 후 정지
#   s / p      재개 / 그 자리 정지
#   ESC 또는 q 홈 복귀 후 종료
set -uo pipefail
cd /home/user/omx_pill_project

C=${1:-}
MODE=${2:-new}

case "$C" in
  red|green|yellow) ;;
  *) echo "사용법: bash go.sh {red|green|yellow} [old]"; exit 1;;
esac

# 체크포인트는 색마다 다르다. 마지막이 아니라 **측정해서 가장 좋았던 지점**이다.
#   초록 87502 (최종)  노랑 89118 (최종)  빨강 90000 (9에폭 직전 정점)
#
# 노랑은 오프라인 지표가 60000 을 최고(4.61)로 지목했으나 **실기에서 못 집었다.**
# 최종 체크포인트 89118 (오프라인 3.81) 로 바꾸니 정상 동작했다 — 2026-08-15 실기 확인.
# 오프라인 지표가 실기와 어긋난 또 하나의 사례다.
declare -A NEW_RUN=( [red]=act_red_224     [green]=act_green_224   [yellow]=act_yellow_224 )
declare -A NEW_CK=(  [red]=090000          [green]=087502          [yellow]=089118 )
declare -A OLD_RUN=( [red]=act_v3_red29    [green]=act_v3_green40k [yellow]=act_v3_yellow40k )

# 빨강·노랑은 반경 기준으로 보정값을 잡아 왔다. 초록은 책상 기준(기본값).
declare -A EXTRA=(   [red]="--radial-offset" [yellow]="--radial-offset" [green]="" )

if [ "$MODE" = "old" ]; then
  R=${OLD_RUN[$C]}; CK=last; REPO=1unasy/pill_v3_onehot
else
  R=${NEW_RUN[$C]}; CK=${NEW_CK[$C]}; REPO=1unasy/pill_v3
fi

echo "═══ $C  —  $R / $CK  (${MODE}) ═══"
RUN="$R" TASK="pick $C pill" exec bash run.sh "$CK" \
  --repo-id "$REPO" --show --local-keys --relax-on-exit \
  --temporal-ensemble --no-freeze-on-grasp ${EXTRA[$C]} --offset-step 1
