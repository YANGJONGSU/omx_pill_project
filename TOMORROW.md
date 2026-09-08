# 이어서 할 일 (2026-08-18 새벽 기준)

## 지금 상태 — 시연 가능

| | 명령 |
|---|---|
| 좌표 조건화 (3색 연속 성공) | `POLICY=xy bash dispense_onehot.sh red yellow green` |
| 원-핫 + FiLM (3색 연속 성공) | `POLICY=film224 bash dispense_onehot.sh red yellow green` |
| 색별 정책 3개 | `bash dispense.sh red yellow green` |
| 웹 (실제 로봇) | `cd web && python app.py --real` → http://127.0.0.1:8000 |
| 정책 비교 목록 | `bash try_failed.sh` |

`REC=1` 을 앞에 붙이면 정책이 본 화면을 mp4 로 남깁니다.

## 바로 확인할 것 하나

정지 안전장치를 **켠 채로** 돌려 확인이 안 됐습니다.

```bash
# 트레이에 빨강·노랑, 초록은 약통에 둔 상태에서
POLICY=film224 SECS=40 bash dispense_onehot.sh green
```

기대: `⏸ 트레이에 green 이(가) 없습니다 — 홈 자세를 유지합니다` 가 뜨고 팔이 멈춤.

02:02 실행은 `HOLD=0` (안전장치 끔)이라 5초 후 표류해 빨강을 집었습니다.
그것은 **정책의 실제 한계**이고, 켜면 막힙니다.

두 결과를 나란히 보여주면 발표 자료가 됩니다.

```
HOLD=0   정책만    5초 후 표류 → 다른 색을 집는다   (한계)
HOLD=1   안전장치  홈을 유지한다                    (해결)
```

## 남은 일

- [ ] 위키 push (커밋은 되어 있음)
      `cd ~/Desktop/mingky_care_pro.wiki && git push`
- [ ] 실기 성공률 측정 — 정책별 10회씩, 발표용 정량 근거
- [ ] 시연 영상 촬영
- [ ] 파지 보정값 정리 — `grasp_shots/` 에 사진이 쌓여 있음,
      `python analyze_grasp.py --color red` 로 mm 단위 계산

## 문서

| 파일 | 내용 |
|---|---|
| `wiki/03_발전과정.md` | **문제가 풀려 간 순서로 전체 과정** (발표용) |
| `wiki/01_사용기술.md` | 시스템 구성·아키텍처·실험 결과 |
| `wiki/02_트러블슈팅.md` | 현상→원인→조치→검증, A~E 계열 |
| `REALROBOT_20260814.md` | 실기 검증 기록 |
| `EXPERIMENT_LOG_20260813.md` | 목표 조건화 실험 결론 |

## 오늘 고친 것 (2026-08-17~18)

1. 성공 판정 — 그리퍼 최솟값 + 약통 위치 (빈손·재시도를 성공으로 오인하던 것)
2. 실패 판정의 `continue` 가 `robot.send_action` 을 건너뛰어 팔이 멈춤
3. 시간 앙상블이 Diffusion·SmolVLA 를 마비 (ACT 전용 옵션)
4. 좌표 정책이 로봇 팔을 목표로 잡음 → `lift <= -35` 일 때만 갱신
5. 정지 감지가 알약 접근 중을 끊음 → 0.5도 / 20초 (정상 집기 오인 0/3950)
6. 목표 색이 없으면 홈 유지 → 정상 집기 0/48 막음, 정지 실패 0/32
7. 트레이 개수 검출 — 카메라 워밍업 + 검증된 HSV + 5장 최빈값
8. `HF_HUB_OFFLINE=1` — 로컬 캐시만 보게 (인증 오류 방지)
