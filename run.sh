#!/usr/bin/env bash
# 학습된 정책을 실기에서 계속 돌린다 (저장 없음, 시간 제한 없음, 홈 복귀 버튼 있음).
#
# eval.sh 는 lerobot-record 라서 반드시 에피소드 단위로 끊고 데이터셋으로 저장한다.
# 이 스크립트는 그게 아니라 "켜면 그냥 돈다". 멈추는 건 내가 키를 누를 때뿐이다.
#
# 사용법:
#   bash ~/omx_pill_project/run.sh                 # last 체크포인트로 계속 실행
#   bash ~/omx_pill_project/run.sh 010000          # 다른 체크포인트
#   bash ~/omx_pill_project/run.sh last --n-action-steps 20   # open-loop 짧게
#   bash ~/omx_pill_project/run.sh --set-home      # 홈 자세를 손으로 가르치기
#   bash ~/omx_pill_project/run.sh last --task "pick red pill"   # 목표 색 지정
#
# 목표 조건화(원-핫) 정책은 --task 의 색으로 목표가 정해진다. 뒤에 붙인 --task 가
# 이긴다(argparse 는 나중 값을 쓴다). eval.sh 는 원-핫 정책에 쓸 수 없으니 평가도 여기서 한다.
#
# 실행 중 키:
#   스페이스   홈(안정 자세)으로 복귀 후 정지
#   s / p      재개 / 그 자리 정지
#   ESC 또는 q 홈 복귀 후 종료
set -euo pipefail

RUN=${RUN:-act_v2}
TASK=${TASK:-"pick yellow pill"}

# 첫 인자가 - 로 시작하면 체크포인트가 아니라 옵션이다
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  CKPT=$1; shift
else
  CKPT=last
fi

source /home/user/venv/il/bin/activate

# 데이터셋은 전부 로컬 캐시에 있다(~/.cache/huggingface/lerobot). 그런데
# LeRobotDataset 이 매번 허브에 refs 를 조회해서, 토큰이 만료되거나 네트워크가
# 끊기면 "Repository Not Found / Invalid username or password" 로 죽는다.
# 오프라인 모드로 두면 캐시만 본다 — 실기 실행에 인터넷이 필요할 이유가 없다.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

exec python /home/user/omx_pill_project/run_policy.py \
  --run "$RUN" --ckpt "$CKPT" --task "$TASK" "$@"
