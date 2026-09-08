# 전체 자료

발표에 직접 쓰는 것은 `../01_핵심/` 입니다. 여기는 **근거를 되짚을 때** 보는 곳입니다.
질문이 들어왔을 때 원본을 열어 보이거나, 보고서를 쓸 때 인용하는 용도입니다.

## 문서 (19개)

### 실험 기록 — 날짜순

| 파일 | 내용 |
|---|---|
| `WORKLOG_20260804_05.md` | 환경 세팅, 첫 데이터 수집 |
| `WORKLOG_20260805.md` | **위치 암기 발견** — 파지 분포 실측, 배치 랜덤화 |
| `WORKLOG_20260806.md` | 색별 정책 학습, 체크포인트 선정 문제 |
| `WORKLOG_20260807.md` | **목표 조건화 실패 다섯 건** — SmolVLA 포함 |
| `EXPERIMENT_LOG_20260813.md` | 목표 조건화 실험 결론, 기각한 가설 정리 |
| `REALROBOT_20260814.md` | **실기 검증** — 오프라인과 어긋난 사례들, SmolVLA 원인 |
| `PERCOLOR_RESULT.md` | 색별 정책 224개 재학습 결과 |

### 평가

| 파일 | 내용 |
|---|---|
| `eval_results.md` · `eval_round2.md` · `eval_round3.md` | 실기 평가 회차별 |

### 정리 문서

| 파일 | 내용 |
|---|---|
| `03_발전과정.md` | **문제가 풀려 간 순서로 본 전체 과정** (핵심 자료와 같은 것) |
| `01_사용기술.md` | 시스템 구성·아키텍처 |
| `02_트러블슈팅.md` | 현상 → 원인 → 조치 → 검증 |
| `TROUBLESHOOTING.md` | 하드웨어·환경 문제 모음 |
| `RELATED_WORK.md` | 참고한 연구 |
| `PLAN_v3.md` · `ONEHOT_CODE.md` · `STATUS.md` | 설계·구현 메모 |

## 그림 (8개)

| 파일 | 쓰임 |
|---|---|
| `fig_goal_methods.png` | **목표 조건화 6가지 비교** — 발표 중심 그림 |
| `fig_checkpoints.png` | 체크포인트별 지표 — 정점이 마지막이 아니다 |
| `fig_offline_vs_real.png` | 오프라인 지표 vs 실기 결과 |
| `fig_onehot_ignores_goal.png` | 목표를 바꿔도 예측이 겹친다 |
| `camera_setup.png` | 카메라 2대 배치 |
| `area_v3.png` | 알약 배치 영역 (`area.json`) |
| `top_ALIGN.png` | 카메라 정렬 검증 |
| `bin_2pills.png` | 약통에 담긴 상태 |

## 지표 (10개, tsv)

`fig_*.png` 의 원본 수치입니다. 표로 인용하거나 다시 그릴 때 씁니다.

| 파일 | 내용 |
|---|---|
| `goal_metric_curve.tsv` | 원-핫 + FiLM 체크포인트별 자기색 대비 |
| `goal_metric_curve_vae.tsv` | CVAE 포함 판 |
| `goal_metric_novae.tsv` | CVAE 제거 판 |
| `goal_metric_diffusion.tsv` | Diffusion Policy |
| `goal_metric_xy.tsv` · `_xy_pick.tsv` · `_xy_free.tsv` | 좌표 조건화 (선택률 포함) |
| `percolor_red.tsv` · `_green.tsv` · `_yellow.tsv` | 색별 정책 체크포인트별 |

## 영상 (2개)

| 파일 | 내용 |
|---|---|
| `발표_xy_좌표조건화.mp4` | 91초, 3색 연속. 설명 띠 포함 |
| `발표_원핫_film224.mp4` | 166초, 검출기 없이 3색 연속 |

원본 녹화 21개는 `report/run_*.mp4` 에 있습니다(총 15GB). 위 둘은 축소·라벨을
붙인 판이라 이것만 보관합니다.

## 기록 (10개, csv)

실기 조제의 관절 궤적입니다. **담기 성공 3개 이상**인 것만 추렸습니다.

| 파일 | 성공 | 길이 |
|---|---|---|
| `trace_132233.csv` | **11개** | 51분 |
| `trace_152859.csv` | 8개 | 14분 |
| `trace_152713.csv` | 7개 | 12분 |
| `trace_152239.csv` | 6개 | 19분 |
| 나머지 6개 | 3개씩 | 2~9분 |

성공 판정 기준(224개 실측): 그리퍼 최솟값 ≥ 49.85 **그리고** 놓는 순간
`shoulder_pan` 이 −28 ~ −21(약통 위치).

## 여기에 없는 것

| | 위치 | 이유 |
|---|---|---|
| 파지 순간 사진 316장 | `grasp_shots/` (188MB) | 보정값 계산용. 발표에는 안 씀 |
| 원본 녹화 21개 | `report/*.mp4` (15GB) | 너무 큼. 축소판을 영상 폴더에 둠 |
| 학습 체크포인트 23개 | `train/` | 수십 GB. 필요하면 그 자리에서 실행 |
| 기록 134개 | `report/trace_*.csv` | 실패·중단분. 성공분만 추림 |
