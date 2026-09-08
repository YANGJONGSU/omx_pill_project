#!/usr/bin/env bash
# 데이터셋과 정책을 허깅페이스에 올린다 (2026-08-23).
#
# 로컬 폴더 이름은 `1unasy/...` 인데 이건 **이전 노트북 계정**이다. 캐시 경로일
# 뿐이라 코드가 참조하는 이름은 그대로 두고, **업로드할 때만** 현재 로그인
# 계정으로 올린다. 이러면 실기·학습이 하나도 안 깨진다.
#
#   huggingface-cli login    # 먼저 Write 토큰으로 로그인
#   bash hf_upload.sh                 # 무엇을 올릴지 목록만 보여준다
#   bash hf_upload.sh pill_v3         # 데이터셋 하나
#   bash hf_upload.sh --policies      # 실기 성공 정책 3개
#   bash hf_upload.sh --all           # 전부 (오래 걸린다)
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

CACHE=$HOME/.cache/huggingface/lerobot/1unasy
TRAIN=/home/user/omx_pill_project/train

WHO=$(python3 -c "
from huggingface_hub import whoami
try: print(whoami()['name'])
except Exception: print('')
" 2>/dev/null)

if [ -z "$WHO" ]; then
  echo "  ⚠ 허깅페이스에 로그인되어 있지 않습니다."
  echo "     huggingface-cli login   (Write 권한 토큰이 필요합니다)"
  exit 1
fi
if [ "$WHO" = "1unasy" ]; then
  echo "  ⚠ 아직 이전 노트북 계정(1unasy)으로 로그인되어 있습니다."
  echo "     본인 계정 토큰으로 다시 로그인하십시오:  huggingface-cli login"
  exit 1
fi
echo "  올릴 계정: $WHO"
echo

# 올릴 후보 — 실기에서 확인된 것만 고른다. 실패한 판까지 올리면 받는 사람이
# 무엇을 써야 할지 알 수 없다.
DATASETS=(pill_v3 pill_v3_xy pill_v3_onehot_224 pill_v3_uvstate)
declare -A POLICIES=(
  [act_xy_224]=150000        # 좌표 조건화, 실기 3색 연속 성공
  [act_film_224]=100000      # 원-핫 + FiLM, 검출기 없이 실기 성공
  [smolvla_v10]=080000       # SmolVLA 로 실기에 성공한 첫 판
)

if [ $# -eq 0 ]; then
  echo "  === 데이터셋 ==="
  for d in "${DATASETS[@]}"; do
    [ -d "$CACHE/$d" ] && printf "    %-22s %s\n" "$d" "$(du -sh $CACHE/$d | cut -f1)"
  done
  echo "  === 정책 ==="
  for p in "${!POLICIES[@]}"; do
    m=$TRAIN/$p/checkpoints/${POLICIES[$p]}/pretrained_model
    [ -d "$m" ] && printf "    %-22s %s\n" "$p/${POLICIES[$p]}" "$(du -sh $m | cut -f1)"
  done
  echo
  echo "  bash hf_upload.sh pill_v3        데이터셋 하나"
  echo "  bash hf_upload.sh --policies     정책 3개"
  echo "  bash hf_upload.sh --all          전부"
  exit 0
fi

up_dataset() {
  local name=$1 src=$CACHE/$1
  [ -d "$src" ] || { echo "  없음: $name"; return 1; }
  echo "  ▶ 데이터셋 $name  ($(du -sh $src | cut -f1))"
  python3 - "$WHO" "$name" "$src" <<'PY'
import sys
from huggingface_hub import HfApi
who, name, src = sys.argv[1:4]
api = HfApi()
repo = f"{who}/{name}"
api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
api.upload_folder(folder_path=src, repo_id=repo, repo_type="dataset",
                  commit_message="OMX 알약 조제 모방학습 데이터")
print(f"    -> https://huggingface.co/datasets/{repo}")
PY
}

up_policy() {
  local run=$1 ck=$2 src=$TRAIN/$1/checkpoints/$2/pretrained_model
  [ -d "$src" ] || { echo "  없음: $run/$ck"; return 1; }
  echo "  ▶ 정책 $run/$ck  ($(du -sh $src | cut -f1))"
  python3 - "$WHO" "$run" "$src" <<'PY'
import sys
from huggingface_hub import HfApi
who, name, src = sys.argv[1:4]
api = HfApi()
repo = f"{who}/{name}"
api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
api.upload_folder(folder_path=src, repo_id=repo, repo_type="model",
                  commit_message="OMX 알약 조제 정책 (실기 검증됨)")
print(f"    -> https://huggingface.co/{repo}")
PY
}

case "${1:-}" in
  --policies)
    for p in "${!POLICIES[@]}"; do up_policy "$p" "${POLICIES[$p]}"; done ;;
  --all)
    for d in "${DATASETS[@]}"; do up_dataset "$d"; done
    for p in "${!POLICIES[@]}"; do up_policy "$p" "${POLICIES[$p]}"; done ;;
  *)
    up_dataset "$1" ;;
esac

echo
echo "  완료. 로컬 경로(1unasy/...)는 그대로이므로 실기·학습에 영향이 없습니다."
