# OMX 알약 자동 조제 — 목표 지정 모방학습

트레이에 섞인 알약 3종 중 **지정된 색만 골라 담는** 로봇팔입니다.
ROBOTIS OpenManipulator-X + LeRobot(ACT / SmolVLA)로 만들었고,
파이널 프로젝트 「밍키케어」(병원 안내·조제 로봇 시스템)의 **약국 조제 파트**입니다.

정해진 자리의 물건을 집는 pick&place가 아니라, **매번 트레이를 보고 어느 알약인지
판단해서 집는** 것이 목표였습니다. 조제는 잘못된 알약이 하나만 섞여도 환자에게
그대로 가기 때문입니다.

---

## 결과

서로 다른 방식으로 **3색 연속 조제에 성공한 정책 세 개**를 확보했습니다.

| 정책 | 목표 전달 방식 | 실기 |
|---|---|---|
| `act_xy_224` | 목표 알약의 화면 좌표 (u, v) | 3색 연속 성공 · 91초 |
| `act_film_224` | 원-핫 + FiLM (시각 특징 512채널 변조) | 3색 연속 성공 · 166초 |
| `smolvla_v10` | 좌표를 `observation.state`에 주입 | 색 구분 + 정확히 파지 |

시연 데이터 **224 에피소드 / 245,529 프레임**을 직접 촬영했습니다.

**이 프로젝트의 결론은 하나입니다 — 목표를 "어떤 경로로" 넣느냐가 모델 크기나
학습량보다 중요했습니다.** 원-핫 토큰은 인코더 입력 603조각 중 1개(0.17%)라
무시해도 손실이 오르지 않았습니다.

---

## 먼저 볼 것

| | |
|---|---|
| **[`발표자료/01_핵심/README.md`](발표자료/01_핵심/README.md)** | **여기부터 보십시오.** 문제 → 진단 → 해결이 순서대로, 기각한 가설과 근거까지 |
| [`발표자료/03_준비과정/`](발표자료/03_준비과정/) | 데이터를 어떻게 설계했나 — 난수 배치표, 촬영 계획 |
| [`wiki/`](wiki/) | 사용 기술 · 트러블슈팅 · 발전 과정 |

## 핵심 실험 기록

| 파일 | 내용 |
|---|---|
| [`ONEHOT_CODE.md`](ONEHOT_CODE.md) | 원-핫이 무시된 이유, FiLM 도입과 그 한계(학습 장면 1.33 / 미학습 1.03) |
| [`PERCOLOR_RESULT.md`](PERCOLOR_RESULT.md) | 색별 정책 재학습 전후 자기색 대비·오차 |
| [`EXPERIMENT_LOG_20260813.md`](EXPERIMENT_LOG_20260813.md) | SmolVLA 언어 조건화 v1~v9 실패 기록 |
| [`REALROBOT_20260814.md`](REALROBOT_20260814.md) | 실기 검증. 사전학습 가중치 미로드가 원인이었던 것 |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | 오프라인 지표와 실기가 어긋난 사례들 |
| [`WORKLOG_20260804_05.md`](WORKLOG_20260804_05.md) ~ `20260807` | 일자별 작업 기록 |

## 진단 도구 — 재학습 대신 계측

실패할 때마다 8시간짜리 재학습을 돌리는 대신, **체크포인트를 고정한 채 입력을
망가뜨려 반응을 재는** 방식으로 바꿨습니다.

| 파일 | 하는 일 |
|---|---|
| [`probe_policy.py`](probe_policy.py) | 카메라 차폐 / 목표 교체 / 그룹 편향 / 위치 추종 5종 검사 |
| [`preflight.py`](preflight.py) | 실기 전 환경·카메라·정책 사전 점검 |
| [`align_top.py`](align_top.py) | 위상 상관으로 카메라 화각 이탈 검출 |
| [`validate_policy.py`](validate_policy.py) | 체크포인트 오프라인 검증 |

> 진단 도구의 코드 구현에는 Claude Code를 사용했습니다. 검사 항목과 판정 기준은
> 직접 설계했습니다.

## 데이터 설계

| 파일 | 하는 일 |
|---|---|
| [`plan_positions.py`](plan_positions.py) | 난수 배치표 생성 (5×4 격자 20칸 × 8방향, 색별 시드) |
| [`make_xy_labels.py`](make_xy_labels.py) | HSV 검출로 목표 좌표 라벨 생성 (재촬영 없이) |
| [`make_onehot.py`](make_onehot.py) / [`make_ohstate.py`](make_ohstate.py) | 원-핫 조건화 데이터셋 |
| [`make_uvstate.py`](make_uvstate.py) | 좌표를 상태 입력에 주입 (SmolVLA v10) |
| [`check_dataset.py`](check_dataset.py) | 데이터셋 무결성 검사 |

## 실행

| 파일 | 하는 일 |
|---|---|
| [`run_policy.py`](run_policy.py) | 정책 추론 + 로봇 구동 |
| [`prescribe.py`](prescribe.py) | 처방 순서대로 연속 조제 |
| [`pharmacy.py`](pharmacy.py) | 약국 단계 연동 |
| [`web/`](web/) | 조제 화면. 로봇 없이 `python app.py`로 동작 |
| [`patches/`](patches/) | LeRobot v0.4.4 로컬 패치 (ACT에 FiLM 조건화 추가) |

## 환경 재현

[`RESTORE.md`](RESTORE.md) · [`SETUP_NEW_LAPTOP.md`](SETUP_NEW_LAPTOP.md) ·
[`RECORDING.md`](RECORDING.md)

```
Ubuntu 24.04 · LeRobot v0.4.4 (로컬 패치) · PyTorch
ROBOTIS OpenManipulator-X · Dynamixel SDK
카메라 2대 (오버헤드 + 손목) · RTX 4060 Laptop 8GB (batch 8 + AMP)
```

---

## 관련 자료

- **실기 동작 영상** — https://youtu.be/ukF5k4bYa9o (밍키케어 팀 시연 중 OMX 조제 파트)
- **약국 조제 화면 (배포 운영 중)** — https://mingkycarepro.site/pharmacy
- **학습한 정책과 데이터셋** — https://huggingface.co/kimy0420
- **자율주행 로봇 프로젝트** — https://github.com/YANGJONGSU/jongky_magic

---

양종수 · kimy042089@gmail.com
