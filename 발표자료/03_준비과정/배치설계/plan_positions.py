#!/usr/bin/env python
"""에피소드별 알약 배치표를 만든다 (3색 섞어 배치, 위치·각도 랜덤화용).

왜 필요한가
-----------
파일럿 30개는 목표 알약이 `shoulder_pan` 기준 3개 덩어리(9/9/10개, 각 폭 2~4단위)에만
몰렸고 덩어리 사이 4.7단위(≈8.5°)에는 데이터가 한 프레임도 없었다. 모델이 배운 것은
"위치를 찾기"가 아니라 "외운 3개 접근 자세 중 고르기"였다. 손으로 "대충 흩어놓기"는
트레이의 익숙한 칸으로 돌아간다. 그래서 자리를 미리 뽑아놓고 그대로 놓는다.

v2 에서 달라진 것
-----------------
- **3색을 매번 다 올린다.** 목표 하나만 집고 나머지 둘은 그대로 둔다 (CLAUDE.md 1번).
  방해 알약이 없으면 "색을 보고 고르기"를 배울 수가 없다.
- 목표색이 균등하게 섞이도록 뽑는다 (`--color` 로 한 색만 뽑을 수도 있다).
- 각도를 0~360 으로 준다 — **색깔 있는 끝이 향하는 방향**이다. 4구간(90°씩) 층화라
  자연히 유색 끝이 좌우 절반씩 나온다. 0~180 만 쓰면 캡슐 방향이 한쪽으로 쏠린다.
- 배치 영역은 하드코딩하지 않는다. 화각이 바뀌면 옛 좌표가 통째로 무효가 되므로,
  카메라를 고정한 뒤 `--set-area` 로 한 번 잡아 `area.json` 에 저장해서 쓴다.

사용법
------
  python ~/omx_pill_project/plan_positions.py --set-area   # 처음 한 번 (카메라 고정 후)
  python ~/omx_pill_project/plan_positions.py 20 --color yellow
  python ~/omx_pill_project/plan_positions.py 60           # 3색 균등하게 섞어서
  python ~/omx_pill_project/plan_positions.py 20 --overlay # top 화면에 점 찍어 이미지로

`--overlay` 는 실제 top 화면 위에 번호 붙은 점을 찍어준다. 숫자표(좌우 63%, 전후 7%)
만으로는 책상 앞에서 어디인지 알 수 없다. 에피소드별로 한 칸씩 떼어낸 시트
(`placements_sheet.png`)도 같이 나오니 리셋 구간에 그 칸만 보면 된다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PROJECT = Path("/home/user/omx_pill_project")
AREA_FILE = PROJECT / "area.json"
TOP_CAM = "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"

COLORS = ["yellow", "red", "green"]
KR = {"yellow": "노랑", "red": "빨강", "green": "초록"}
# 화면에 그릴 색 (BGR)
BGR = {"yellow": (60, 220, 240), "red": (60, 60, 230), "green": (90, 200, 90)}

# 알약끼리 최소 거리 (배치 영역 짧은 변에 대한 비율). 너무 붙으면 그리퍼가
# 목표를 집을 때 옆 알약을 건드린다.
MIN_DIST = 0.22


# ------------------------------------------------------------------ 배치 영역


def load_area() -> dict:
    if not AREA_FILE.exists():
        raise SystemExit(
            f"배치 영역이 아직 정해지지 않았습니다: {AREA_FILE}\n"
            f"카메라를 최종 위치에 고정한 뒤 먼저 실행하세요:\n"
            f"  python {__file__} --set-area\n"
            f"(옛 화각의 좌표를 그대로 쓰면 알약이 화면 밖이나 팔이 안 닿는 곳에 갑니다)"
        )
    return json.loads(AREA_FILE.read_text())


def save_area(x0: int, y0: int, x1: int, y1: int) -> None:
    area = {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}
    AREA_FILE.write_text(json.dumps(area, indent=2))
    print(f"\n저장했습니다: {AREA_FILE}")
    print(f"  {area}  ({area['x1'] - area['x0']}x{area['y1'] - area['y0']} 픽셀)")
    print("\n이 시점 이후로 카메라를 건드리면 이 좌표가 무효가 됩니다.")


def set_area() -> None:
    """top 화면에서 배치 영역을 드래그로 잡아 area.json 에 저장한다.

    `selectROI` 는 창을 먼저 만들어두지 않으면 Qt 백엔드에서
    "NULL window handler" 로 죽는다 — namedWindow/imshow 를 먼저 부른다.
    그래도 실패하면 좌표를 직접 넣는 --area-coords 를 쓰라고 안내한다.
    """
    import cv2

    img = grab_top()
    print("배치 영역을 드래그로 지정하세요 (알약을 놓을 범위).")
    print("  · 팔이 네 귀퉁이에 다 닿는 범위여야 합니다 — teleop 으로 미리 확인할 것")
    print("  · 목적지 약통은 영역 밖에 두세요")
    print("  ENTER/SPACE 확정, c 취소")

    win = "set area"
    try:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win, img)
        cv2.waitKey(200)  # 창이 실제로 뜰 시간을 준다
        x, y, w, h = cv2.selectROI(win, img, showCrosshair=True)
    except cv2.error as e:
        cv2.destroyAllWindows()
        snap = PROJECT / "report" / "area_pick.png"
        snap.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(snap), grid_overlay(img))
        raise SystemExit(
            f"드래그 창을 띄우지 못했습니다 ({str(e)[:60]}...).\n\n"
            f"좌표를 직접 넣는 방법을 쓰세요. 격자가 그려진 화면을 저장해 뒀습니다:\n"
            f"  xdg-open {snap}\n"
            f"그림에서 범위를 읽고:\n"
            f"  python {__file__} --area-coords x0 y0 x1 y1"
        ) from None
    cv2.destroyAllWindows()
    if w < 10 or h < 10:
        raise SystemExit("영역이 너무 작습니다 — 취소했습니다.")

    save_area(x, y, x + w, y + h)


def grid_overlay(img):
    """좌표를 눈으로 읽을 수 있게 50픽셀 격자와 숫자를 그린다."""
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, 50):
        cv2.line(out, (x, 0), (x, h), (60, 60, 60), 1)
        cv2.putText(out, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    for y in range(0, h, 50):
        cv2.line(out, (0, y), (w, y), (60, 60, 60), 1)
        cv2.putText(out, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return out


def grab_top():
    """top 카메라 한 장. V4L2 + MJPG 필수 (FFMPEG 백엔드는 해상도·포맷 설정이 안 먹는다)."""
    import cv2

    cap = cv2.VideoCapture(TOP_CAM, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("top 카메라를 열 수 없습니다 (녹화가 돌고 있으면 먼저 끄세요)")
    ok = False
    for _ in range(15):  # 자동 노출 안정화
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("프레임 읽기 실패")
    return frame


# ------------------------------------------------------------------ 뽑기


def stratified(n: int, rng: np.random.Generator) -> np.ndarray:
    """n개 칸 격자에 하나씩 배정하고 칸 안에서 흔든다 → 좌우/전후 (0~1).

    순수 난수와 달리 뭉치거나 빈 구역이 생기지 않는다 — 파일럿의 실패가
    "빈 구역"이었으므로 이게 핵심이다.
    """
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    cells = [(r, c) for r in range(rows) for c in range(cols)][:n]
    rng.shuffle(cells)
    out = np.empty((n, 2))
    for i, (r, c) in enumerate(cells):
        out[i, 0] = (c + rng.uniform(0.1, 0.9)) / cols  # 좌우
        out[i, 1] = (r + rng.uniform(0.1, 0.9)) / rows  # 전후
    return out


def stratified_angles(n: int, rng: np.random.Generator) -> np.ndarray:
    """0~360 을 90°씩 4구간으로 나눠 균등 배정. 유색 끝 방향이 절반씩 나온다."""
    bins = np.tile(np.arange(4), int(math.ceil(n / 4)))[:n]
    rng.shuffle(bins)
    return (bins * 90 + rng.uniform(5, 85, size=n)).round().astype(int)


# 약통에 미리 들어있는 알약 개수의 비율. 개수가 늘수록 트레이에 남는 알약이 줄어든다.
# 0개(트레이 3개)를 가장 많이 두는 이유는 아래 stage_counts 주석 참고.
STAGE_WEIGHTS = [0.4, 0.4, 0.2]


def stage_counts(count: int, rng: np.random.Generator) -> np.ndarray:
    """각 에피소드가 "처방의 몇 번째 pick" 상태인지 (=약통에 미리 든 개수 0/1/2).

    왜 필요한가 — 두 가지가 한 번에 걸린다.

    1. **약통 쪽**: 전부 빈 약통으로 시작하면 "약통에 알약이 보이는" 화면이 에피소드
       끝에만 존재한다. 모델은 그걸 "일이 끝났다"는 신호로 배우고, 두 번째 pick 을
       시작조차 안 하게 된다 (CLAUDE.md 7번 causal confusion).
    2. **트레이 쪽**: 항상 3개로 시작하면, 실제 처방의 두 번째 pick(트레이에 2개만
       남은 화면)이 학습 분포 밖이 된다. 첫 알약을 담고 돌아온 순간이 바로 그 상태다.

    그래서 약통에 든 것 = 트레이에서 이미 빠진 것으로 **앞뒤를 맞춘다.**
    배치표 한 줄이 곧 처방 중간 상태 하나가 된다.

    비율을 균등(1/3씩)이 아니라 0개 쪽으로 기울인 이유: 약통이 2개면 트레이에 목표
    하나만 남아서 "아무거나 집기"로도 정답이 된다(원-핫을 안 읽어도 맞는다). 색 구분을
    배우는 신호는 트레이에 3개 있을 때 가장 강하므로 그쪽을 많이 둔다.
    """
    counts = rng.choice([0, 1, 2], size=count, p=STAGE_WEIGHTS)
    # 뽑기 운으로 한쪽이 비지 않게, 개수가 충분하면 각 단계를 최소 1개는 보장한다
    if count >= 6:
        for stage in (0, 1, 2):
            if not (counts == stage).any():
                counts[rng.integers(count)] = stage
    return counts


def draw_distractors(target_xy: np.ndarray, rng: np.random.Generator, aspect: float, n: int = 2):
    """방해 알약 n개의 자리. 목표와도, 서로도 너무 붙지 않게 뽑는다."""
    placed = [target_xy]
    for _ in range(n):
        for _try in range(200):
            p = rng.uniform(0.05, 0.95, size=2)
            # 화면 비율을 반영해 실제 거리로 잰다 (영역이 가로로 길면 좌우 여유가 크다)
            if all(
                math.hypot((p[0] - q[0]) * aspect, p[1] - q[1]) >= MIN_DIST for q in placed
            ):
                break
        placed.append(p)
    return placed[1:]


def build(count: int, color: str | None, seed: int, area: dict):
    rng = np.random.default_rng(seed)

    if color:
        targets = [color] * count
    else:
        # 목표색 균등 — 남는 개수는 앞 색부터 하나씩
        targets = [COLORS[i % len(COLORS)] for i in range(count)]
        rng.shuffle(targets)

    pos = stratified(count, rng)
    ang = stratified_angles(count, rng)
    stages = stage_counts(count, rng)
    aspect = (area["x1"] - area["x0"]) / (area["y1"] - area["y0"])

    rows = []
    for i in range(count):
        # 약통에 든 것 = 트레이에서 이미 빠진 것. 목표색은 아직 트레이에 있어야 하므로
        # 나머지 두 색 중에서 뺀다 (그래서 약통에 목표색은 안 들어간다 —
        # 실제 처방에서도 이미 담은 색을 또 집을 일이 없다).
        others = [c for c in COLORS if c != targets[i]]
        rng.shuffle(others)
        n_bin = int(stages[i])
        in_bin, on_tray = others[:n_bin], others[n_bin:]

        xys = draw_distractors(pos[i], rng, aspect, n=len(on_tray))
        # 방해 알약도 각도를 준다. 손으로 대충 놓으면 늘 비슷하게 눕고, 그러면
        # "각도가 특이한 놈이 목표"라는 지름길이 생긴다. 목표만큼 정확할 필요는
        # 없으니(±20° 오차 무방) 4구간에서 하나씩 뽑아준다.
        d_angs = stratified_angles(len(on_tray), rng) if on_tray else []
        rows.append(
            {
                "target": targets[i],
                "xy": pos[i],
                "ang": int(ang[i]),
                "distractors": [
                    (c, xy, int(a)) for (c, xy), a in zip(zip(on_tray, xys, strict=True), d_angs, strict=True)
                ],
                "bin": in_bin,
            }
        )
    return rows


# ------------------------------------------------------------------ 그리기


def _arrow(img, x, y, deg, color, scale=1.0):
    """색깔 있는 끝이 향하는 방향으로 화살표. 화면 좌표는 y가 아래로 증가한다."""
    import cv2

    t = math.radians(deg)
    dx, dy = 13 * scale * math.cos(t), -13 * scale * math.sin(t)
    p0 = (int(x - dx), int(y - dy))
    p1 = (int(x + dx), int(y + dy))
    cv2.arrowedLine(img, p0, p1, (255, 255, 255), int(4 * scale), tipLength=0.45)
    cv2.arrowedLine(img, p0, p1, color, int(2 * scale), tipLength=0.45)


def draw_overlay(rows, start, out_path, seed, area):
    """실제 top 화면 위에 번호 붙은 목표 점을 찍는다 (목표만, 방해는 시트에서 본다)."""
    import cv2

    img = grab_top()
    a = area
    overlay = img.copy()
    cv2.rectangle(overlay, (a["x0"], a["y0"]), (a["x1"], a["y1"]), (0, 200, 255), -1)
    img = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
    cv2.rectangle(img, (a["x0"], a["y0"]), (a["x1"], a["y1"]), (0, 200, 255), 1)

    for i, r in enumerate(rows):
        x = int(a["x0"] + r["xy"][0] * (a["x1"] - a["x0"]))
        y = int(a["y0"] + r["xy"][1] * (a["y1"] - a["y0"]))
        _arrow(img, x, y, r["ang"], BGR[r["target"]])
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        label = str(start + i)
        cv2.putText(img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
        cv2.putText(img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    txt = f"ep {start}~{start + len(rows) - 1}  seed={seed}"
    cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(str(out_path), img)
    print(f"\n오버레이 저장: {out_path}")
    print("  화살표 = 목표 알약과 색깔 끝 방향, 노란 테두리 = 배치 영역")


def draw_sheet(rows, start, out_path, area, cols=5, scale=1.6):
    """에피소드별로 한 칸씩 떼어낸 시트. 3개를 다 그린다 (목표는 빨간 테두리).

    한 장에 겹쳐 찍으면 번호를 못 읽는다. 리셋 구간에 다음 에피소드 칸만 보면 된다.
    """
    import cv2

    base = grab_top()
    a = area
    crop = base[a["y0"] : a["y1"], a["x0"] : a["x1"]]
    ch, cw = crop.shape[:2]
    cell = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_LINEAR)
    cell = cv2.addWeighted(cell, 0.55, np.full_like(cell, 255), 0.45, 0)  # 점이 잘 보이게
    h, w = cell.shape[:2]
    pad, band = 4, 38  # 라벨 두 줄 (목표 / 트레이·약통 구성)

    n = len(rows)
    nrows = int(math.ceil(n / cols))
    sheet = np.full((nrows * (h + band + pad) + pad, cols * (w + pad) + pad, 3), 235, np.uint8)

    for i, r in enumerate(rows):
        gr, gc = divmod(i, cols)
        x0 = pad + gc * (w + pad)
        y0 = pad + gr * (h + band + pad)
        panel = cell.copy()

        # 방해 알약 먼저, 목표를 위에. 방해도 화살표(각도)를 그린다 — 늘 비슷하게
        # 눕혀 놓으면 "각도가 특이한 놈이 목표"라는 지름길이 생긴다
        for color, xy, dang in r["distractors"]:
            px, py = int(xy[0] * (w - 1)), int(xy[1] * (h - 1))
            _arrow(panel, px, py, dang, BGR[color], scale=1.1)
            cv2.circle(panel, (px, py), 7, BGR[color], -1)
            cv2.circle(panel, (px, py), 7, (60, 60, 60), 1)
            cv2.putText(panel, KR[color][0], (px - 5, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1, cv2.LINE_AA)

        px, py = int(r["xy"][0] * (w - 1)), int(r["xy"][1] * (h - 1))
        _arrow(panel, px, py, r["ang"], BGR[r["target"]], scale=1.3)
        cv2.circle(panel, (px, py), 5, (0, 0, 255), 2)
        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (150, 150, 150), 1)

        sheet[y0 + band : y0 + band + h, x0 : x0 + w] = panel

        # 1줄: 몇 번 에피소드에 목표가 무엇인지 / 2줄: 트레이와 약통에 뭐가 있는지
        tray = "+".join([KR[r["target"]][0]] + [KR[c][0] for c, _, _ in r["distractors"]])
        binlabel = "".join(KR[c][0] for c in r["bin"]) if r["bin"] else "없음"
        line1 = f"ep {start + i}   {KR[r['target']]} {r['ang']}deg"
        line2 = f"트레이 {len(r['distractors']) + 1}개({tray})  통 {binlabel}"
        for k, text in enumerate((line1, line2)):
            cv2.putText(sheet, text, (x0 + 2, y0 + 14 + k * 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), sheet)
    print(f"에피소드별 시트 저장: {out_path}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("  화살표 = 목표(빨간 원), 채운 원 = 방해 알약")


# ------------------------------------------------------------------ main


def save_table(rows, start, path: Path, seed: int, area: dict) -> None:
    """배치표를 마크다운으로 저장한다 — 터미널은 스크롤로 사라지므로 이게 원본이다.

    녹화하면서 체크박스를 채워 나가면 어디까지 했는지도 남는다.
    """
    from collections import Counter

    cfg = Counter((r["target"], tuple(sorted(c for c, _, _ in r["distractors"]))) for r in rows)

    L = [
        f"# 배치표 — ep {start}~{start + len(rows) - 1} ({len(rows)}개)",
        "",
        f"- seed `{seed}` (같은 표를 다시 뽑으려면 `--seed {seed}`)",
        f"- 배치 영역 `{area}`",
        "- 좌우/전후 = 영역을 0~100%로 본 위치",
        "- 각도 = **색깔 있는 끝이 향하는 방향** (0=오른쪽, 90=위, 반시계)",
        "- 약통에 든 색은 트레이에 없다 — 한 줄이 곧 처방 중간 상태 하나다",
        "",
        "## 구성별 개수",
        "",
        "| 목표 | 트레이에 함께 | 개수 |",
        "|---|---|---|",
    ]
    for (t, d), n in sorted(cfg.items()):
        dd = " + ".join(KR[c] for c in d) if d else "(목표 하나만)"
        L.append(f"| {KR[t]} | {dd} | {n} |")

    L += [
        "",
        "## 에피소드별",
        "",
        "| ✓ | ep | 목표 | 좌우% | 전후% | 각도 | 트레이에 함께 | 약통에 미리 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows):
        d = ", ".join(f"{KR[c]} {xy[0]*100:.0f}/{xy[1]*100:.0f} {a}°" for c, xy, a in r["distractors"])
        b = " ".join(KR[c] for c in r["bin"]) if r["bin"] else "(빈 약통)"
        L.append(
            f"| [ ] | {start + i} | {KR[r['target']]} | {r['xy'][0] * 100:.0f} | "
            f"{r['xy'][1] * 100:.0f} | {r['ang']}° | {d or '(목표 하나만)'} | {b} |"
        )

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n배치표 저장: {path}")


def main():
    ap = argparse.ArgumentParser(description="알약 배치표 생성 (3색 섞어 배치)")
    ap.add_argument("count", type=int, nargs="?", default=20, help="찍을 에피소드 수")
    ap.add_argument("--color", choices=COLORS, default=None,
                    help="목표색 하나로 몰아 찍을 때. 생략하면 3색 균등")
    ap.add_argument("--seed", type=int, default=None, help="같은 표를 다시 뽑으려면 지정")
    ap.add_argument("--start", type=int, default=0, help="에피소드 번호 시작값 (이어 찍기면 기존 개수)")
    ap.add_argument("--set-area", action="store_true", help="배치 영역을 드래그로 다시 지정")
    ap.add_argument(
        "--area-coords",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="드래그 대신 좌표로 배치 영역 지정 (창이 안 뜰 때)",
    )
    ap.add_argument("--overlay", action="store_true", help="top 화면에 점을 찍어 이미지로 저장")
    ap.add_argument("--out", default=str(PROJECT / "placements.png"), help="오버레이 저장 경로")
    args = ap.parse_args()

    if args.area_coords:
        x0, y0, x1, y1 = args.area_coords
        if x1 <= x0 or y1 <= y0:
            raise SystemExit("좌표 순서가 잘못됐습니다: x0 y0 x1 y1 (왼쪽위 → 오른쪽아래)")
        save_area(x0, y0, x1, y1)
        return

    if args.set_area:
        set_area()
        return

    area = load_area()
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 10_000)
    rows = build(args.count, args.color, seed, area)

    print(f"배치표 — {args.count}개 (seed={seed}, 같은 표가 필요하면 --seed {seed})")
    print(f"배치 영역: {area}  ({AREA_FILE})")
    print("좌우/전후 = 영역을 0~100%로 본 위치, 각도 = **색깔 있는 끝이 향하는 방향**")
    print("  (0=오른쪽, 90=위, 반시계. 화면 기준)")
    print()
    print(
        f"{'ep':>4} {'목표':>5} {'좌우%':>6} {'전후%':>6} {'각도':>5}   "
        f"{'트레이에 같이 놓을 것 (색 좌우/전후)':<30} {'약통에 미리':<14} {'체크':>4}"
    )
    print("-" * 94)
    for i, r in enumerate(rows):
        d = "  ".join(f"{KR[c]} {xy[0]*100:.0f}/{xy[1]*100:.0f} {a}°" for c, xy, a in r["distractors"])
        d = d or "(목표 하나만)"
        b = " ".join(KR[c] for c in r["bin"]) if r["bin"] else "(빈 약통)"
        print(
            f"{args.start + i:>4} {KR[r['target']]:>5} {r['xy'][0] * 100:>6.0f} "
            f"{r['xy'][1] * 100:>6.0f} {r['ang']:>4}°   {d:<28} {b:<14} [  ]"
        )

    # 확인용 — 이 표 자체가 뭉쳐 있으면 뽑기가 잘못된 것이다
    print()
    print("확인 — 목표 알약의 퍼짐 (파일럿은 pan 덩어리 3개였다):")
    xy = np.array([r["xy"] for r in rows])
    for name, col in (("좌우", xy[:, 0] * 100), ("전후", xy[:, 1] * 100)):
        gaps = np.diff(np.sort(col))
        print(f"  {name}: 범위 {col.min():.0f}~{col.max():.0f}%, 최대 이웃 간격 {gaps.max():.1f}%p")
    angs = np.array([r["ang"] for r in rows])
    print(f"  각도 4구간 개수: {[int(((angs // 90) == b).sum()) for b in range(4)]}")
    cnt = {c: sum(1 for r in rows if r["target"] == c) for c in COLORS}
    print(f"  목표색: {', '.join(f'{KR[c]} {n}개' for c, n in cnt.items())}")
    nb = [sum(1 for r in rows if len(r["bin"]) == k) for k in range(3)]
    print(
        f"  처방 단계: 첫 번째 pick(트레이 3개) {nb[0]}개 / "
        f"두 번째(트레이 2개) {nb[1]}개 / 세 번째(트레이 1개) {nb[2]}개"
    )
    print()
    print("놓을 때: '트레이' 칸의 알약만 올리고, '약통에 미리' 칸은 약통 안에 넣어둔 채 시작한다.")
    print("  (약통에 든 것 = 이미 담은 것. 그래서 그 색은 트레이에 없다 — 실제 처방 중간 상태다)")
    print("목적지 약통 자체는 흔들지 않는다. 목표 하나만 집어 약통에 담으면 끝.")

    save_table(rows, args.start, Path(str(args.out).replace(".png", ".md")), seed, area)

    if args.overlay:
        draw_overlay(rows, args.start, args.out, seed, area)
        draw_sheet(rows, args.start, str(args.out).replace(".png", "_sheet.png"), area)


if __name__ == "__main__":
    main()
