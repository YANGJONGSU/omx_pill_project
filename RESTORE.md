# 다른 컴퓨터에서 이어서 작업하기

노트북 반납 전 백업한 내용과 복구 방법입니다. 이 파일만 보면 전부 되살릴 수 있습니다.

## 1. 어디에 무엇이 있나

| 자산 | 위치 |
|---|---|
| 코드·문서·실험 기록 | GitHub `YANGJONGSU/omx_pill_project` (비공개) |
| 학습된 정책 | HuggingFace `kimy0420` |
| 데이터셋 | HuggingFace `kimy0420` |
| 시연·실기 영상 | HuggingFace `kimy0420/omx_videos` |
| 지원 서류·발표 자료 | 저장소 안 `문서백업/` |

## 2. 코드 받기

```bash
git clone https://github.com/YANGJONGSU/omx_pill_project.git
cd omx_pill_project
```

비공개 저장소라 GitHub 로그인이 필요합니다. 토큰을 새로 발급받아야 하면
github.com → Settings → Developer settings → Personal access tokens에서 `repo` 권한으로 만듭니다.

저장소 구성:

```
omx_pill_project/
├── run_policy.py          정책 실행 (실기 제어의 핵심)
├── dispense_onehot.sh     처방 순서대로 연속 조제
├── make_*.py              데이터셋 변환 (원-핫, 좌표, 상태주입)
├── probe_*.py             체크포인트 직접 계측 도구
├── plan_positions.py      배치표 생성
├── area.json              트레이 배치 영역
├── home_pose.json         홈 자세
├── grasp_offset.json      색별 파지 보정값
├── SETTINGS.json          정책별 설정과 오프셋
├── EXPERIMENT_LOG_*.md    실험 전체 기록
├── WORKLOG_*.md           날짜별 작업 기록
└── 문서백업/
    ├── 01_사용기술.md          기술 명세 (아키텍처, 학습 설정)
    ├── 02_트러블슈팅.md        A~E 분류 트러블슈팅 전체
    ├── BATCH_SHEET.pdf        동일 장면×3목표 촬영표
    ├── 위키/                   발표용 차트 이미지 13개 포함
    ├── 대화기록/               작업 세션 기록 21개
    ├── 발표사진/               발표용 이미지
    ├── 슬라이드캡처/           개발 여정 슬라이드 캡처
    ├── 발표자료/               발표 관련 문서·이미지
    └── 지원서류/
        ├── 이력서.html
        ├── 포트폴리오.html
        └── 발표_예상질문.html
```

`지원서류/` 안 HTML은 브라우저로 바로 열립니다.

## 3. 정책과 데이터 받기

전부 `kimy0420` 계정에 있습니다. 로그인:

```bash
pip install huggingface_hub
huggingface-cli login
```

**정책 (실기 성공한 것들)**

| 이름 | 방식 | 결과 |
|---|---|---|
| `kimy0420/act_film_224` | 원-핫 + FiLM | 3색 연속 성공 |
| `kimy0420/act_xy_224` | 좌표 조건화 | 3색 연속 성공 |
| `kimy0420/smolvla_v10` | 좌표를 상태에 주입 | SmolVLA 첫 성공 |
| `kimy0420/act_red_224` 외 2종 | 색별 정책 | 동작 확인 |

```bash
huggingface-cli download kimy0420/act_xy_224 --local-dir ./train/act_xy_224
```

**데이터셋**

| 이름 | 용도 |
|---|---|
| `kimy0420/pill_v3` | 원본 224 에피소드 |
| `kimy0420/pill_v3_xy` | 좌표 조건화용 |
| `kimy0420/pill_v3_onehot_224` | 원-핫 조건화용 |
| `kimy0420/pill_v3_uvstate` | SmolVLA 상태주입용 |

LeRobot이 찾는 위치에 두려면:

```bash
huggingface-cli download kimy0420/pill_v3 --repo-type dataset \
  --local-dir ~/.cache/huggingface/lerobot/kimy0420/pill_v3
```

**영상** — `kimy0420/omx_videos`

| 파일 | 내용 |
|---|---|
| `발표_xy_좌표조건화.mp4` | 좌표 정책 3색 연속 조제 (91초) |
| `발표_원핫_film224.mp4` | FiLM 정책 3색 연속 조제 |
| `demo_원핫_목표무시.mp4` | 원-핫이 지시를 무시하는 장면 |
| `run_smolvla10_성공.mp4` | SmolVLA v10 성공 |
| `run_smolvla8_떨림.mp4` | v8 떨림·헤맴 실패 사례 |

## 4. 환경 다시 만들기

```bash
python3 -m venv ~/venv/il
source ~/venv/il/bin/activate

# LeRobot v0.4.4 고정 — 상위 버전은 호환 안 됨
git clone https://github.com/huggingface/lerobot.git ~/il_ws/src/lerobot
cd ~/il_ws/src/lerobot && git checkout v0.4.4
pip install -e .
pip install "transformers<5" num2words   # SmolVLA용
```

**중요 — LeRobot 로컬 패치 3개**

이 프로젝트는 LeRobot을 직접 수정해서 씁니다. 재설치하면 날아갑니다.

| 파일 | 수정 내용 |
|---|---|
| `lerobot_record.py` | 저장 순서 변경 (에피소드 종료 → 저장 → 리셋 안내) |
| `camera_opencv.py` | 카메라가 죽어도 제자리 자동 복구 |
| `control_utils.py` | ESC를 2초 안에 두 번 눌러야 종료 |
| `modeling_act.py` | FiLM 조건화 추가 |
| `configuration_act.py` | `film_conditioning` 옵션 추가 |

패치 내용은 `문서백업/01_사용기술.md`의 부록에 코드까지 적혀 있습니다.

## 5. 하드웨어 세팅

| 구성 | 사양 |
|---|---|
| 로봇 | OpenManipulator-X 팔로워 + 리더암, OpenRB-150 컨트롤러 |
| 카메라 | top(오버헤드) Jieli USB Composite, wrist Innomaker U20CAM-720P |
| 연산 | RTX 4060 Laptop 8GB — 배치 8, Mixed Precision 필수 |

카메라 장치 번호는 재부팅마다 바뀌므로 `/dev/v4l/by-id/` 고정 경로를 씁니다.
`dispense_onehot.sh` 안에 경로가 하드코딩돼 있으니 카메라를 바꾸면 그 줄을 고쳐야 합니다.

## 6. 실기 실행

```bash
cd ~/omx_pill_project
POLICY=xy bash dispense_onehot.sh red yellow green
```

`POLICY`에 `xy`, `film224`, `smolvla10` 중 하나를 넣습니다.

주의: 웹 대시보드와 **동시에 못 씁니다.** 시리얼 포트와 카메라는 한 프로그램만 점유할 수 있습니다.

## 7. 백업 안 한 것

| 항목 | 이유 |
|---|---|
| 실패한 실험 체크포인트 130GB+ | 다시 쓸 일이 없음 |
| 실행 녹화 영상 60여 개 | 용량이 크고 핵심 5개만 남김 |

필요해지면 `EXPERIMENT_LOG_20260813.md`에 어떤 설정으로 학습했는지 다 적혀 있어 재현 가능합니다.

## 8. 현재 상태 요약

3색 연속 조제에 성공한 정책이 네 가지입니다. 색별 정책 3개, 원-핫+FiLM, 좌표 조건화, SmolVLA v10.

측정이 남은 것: 같은 배치를 고정해 정책별 성공률을 정량 비교하는 실험.
설계는 `eval3_placements.md`에 되어 있고 실행만 하면 됩니다.
