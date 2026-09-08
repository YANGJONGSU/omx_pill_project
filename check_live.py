#!/usr/bin/env python
"""지금 카메라가 학습 때와 같은 화면을 주고 있는지 대조한다 (로봇 없이, 카메라만).

오프라인 계측(`probe_policy.py`)에서 나온 결과: **정책은 멀쩡하다.** 깊이 오차가
데이터 위에서 0.3~0.5도(약 1mm)인데 실기에서는 1~2cm 빗나간다. 10~20배 차이는
정책이 아니라 **실기에서 정책에게 들어가는 그림**에서 생겼을 가능성이 크다.

v2 때 실제로 그랬다 — wrist 초점이 학습 범위의 15분의 1로 흐려진 채 평가를 2회 돌렸고,
화각이 6.4px 어긋난 채로 평가 1차를 통째로 버렸다. 둘 다 사전에 이 검사를 했으면 걸렸다.

세 가지를 학습 분포와 대조한다 (기준은 `make_ref.py` 가 학습 영상에서 뽑은 것):
  화각   — 고정 구조물 템플릿 매칭으로 dx, dy 픽셀
  선명도 — 라플라시안 분산이 학습 범위 안인지 (렌즈 초점)
  밝기   — 평균 밝기가 기준 ±5 안쪽인지 (노출·조명)

사용법:
  python ~/omx_pill_project/check_live.py             # 한 번 재고 판정
  python ~/omx_pill_project/check_live.py --watch     # 계속 보면서 맞추기 (q 로 종료)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path("/home/user/omx_pill_project")
DEV = {
    "top": "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0",
    "wrist": "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0",
}
# 알약·팔이 지나가지 않는 고정 구조물 (y0, y1, x0, x1) — align_top.py 와 같은 패치
PATCHES = {
    "base": (400, 478, 190, 400),
    "tray_left": (150, 350, 100, 180),
    "tray_bottom": (340, 400, 150, 450),
}
SHIFT_WARN, SHIFT_FAIL = 5.0, 12.0   # preflight.py:48 과 같은 값
MM_PER_PX = 0.65


def open_cam(path: str):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap if cap.isOpened() else None


def grab(cap, n=10):
    """첫 프레임은 워밍업 JPEG 손상이 있으므로 몇 장 버리고 받는다 (STATUS.md)."""
    frame = None
    for _ in range(n):
        ok, f = cap.read()
        if ok:
            frame = f
    return frame


def sharpness(bgr):
    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def shift_vs_ref(g_ref, g_now):
    """고정 구조물 패치별 템플릿 매칭 → 중앙값 dx, dy 와 최저 일치도.

    구역별로 재는 이유: 중앙값 하나만 보면 카메라는 멀쩡한데 매트만 밀린 경우를 놓친다.
    v2 에서 실제로 놓쳤다 (PLAN_v3 §2).
    """
    res = {}
    for name, (y0, y1, x0, x1) in PATCHES.items():
        tpl = g_ref[y0:y1, x0:x1]
        pad = 30
        sy0, sy1 = max(0, y0 - pad), min(g_now.shape[0], y1 + pad)
        sx0, sx1 = max(0, x0 - pad), min(g_now.shape[1], x1 + pad)
        m = cv2.matchTemplate(g_now[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(m)
        res[name] = (loc[0] + sx0 - x0, loc[1] + sy0 - y0, score)
    dx = float(np.median([v[0] for v in res.values()]))
    dy = float(np.median([v[1] for v in res.values()]))
    return dx, dy, min(v[2] for v in res.values()), res


def report(ref_dir: Path, frames: dict, meta: dict, verbose=True):
    verdict = []
    for name, now in frames.items():
        if now is None:
            print(f"\n[{name}] 카메라를 못 엽니다 — USB 확인 필요")
            verdict.append((name, "FAIL", "카메라 없음"))
            continue
        m = meta[name]
        sh = sharpness(now)
        br = float(now.mean())
        print(f"\n[{name}]")

        ref_path = ref_dir / f"{name}_train.png"
        if ref_path.exists() and name == "top":
            g_ref = cv2.cvtColor(cv2.imread(str(ref_path)), cv2.COLOR_BGR2GRAY)
            g_now = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY)
            dx, dy, score, per = shift_vs_ref(g_ref, g_now)
            mag = float(np.hypot(dx, dy))
            tag = "OK" if mag < SHIFT_WARN else ("WARN" if mag < SHIFT_FAIL else "FAIL")
            print(f"  화각   dx {dx:+.1f}  dy {dy:+.1f} px  "
                  f"({mag * MM_PER_PX:.1f}mm)  일치도 {score:.2f}   [{tag}]")
            if verbose:
                for k, (a, b, s) in per.items():
                    print(f"           {k:<12} dx {a:+3.0f} dy {b:+3.0f} (일치도 {s:.2f})")
            verdict.append((name, tag, f"화각 {mag:.1f}px"))

        # 기준은 **압축된 학습 영상**에서, 지금 값은 **생 MJPG** 프레임에서 나온다.
        # H.264 가 디테일을 깎으므로 학습 쪽이 체계적으로 낮게 읽힌다 — 즉 지금 값이
        # 범위보다 높은 것은 정상이고, 문제는 **아래로 벗어나는 것**(초점 풀림)뿐이다.
        # v2 의 실제 사고는 1287 범위에 184 였다 (7배 아래).
        lo, hi = m["sharpness_min"], m["sharpness_max"]
        pct = sh / m["sharpness_med"]
        if sh < lo:
            tag, note = "FAIL", "초점 풀림"
        elif sh > hi:
            tag, note = "OK", "범위 위 — 압축 차이로 정상"
        else:
            tag, note = "OK", ""
        print(f"  선명도 {sh:7.0f}   학습 {lo:.0f}~{hi:.0f} "
              f"(중앙값 {m['sharpness_med']:.0f}, 지금은 {pct:.2f}배)   [{tag}] {note}")
        verdict.append((name, tag, f"선명도 {sh:.0f} ({note})" if note else f"선명도 {sh:.0f}"))

        d = br - m["mean"]
        tag = "OK" if abs(d) <= 5 else ("WARN" if abs(d) <= 12 else "FAIL")
        print(f"  밝기   {br:7.1f}   학습 {m['mean']:.1f} ({d:+.1f})   [{tag}]")
        verdict.append((name, tag, f"밝기 {d:+.1f}"))
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="v3", help="preflight_ref/ 아래 폴더")
    ap.add_argument("--watch", action="store_true", help="계속 보면서 맞춘다")
    args = ap.parse_args()

    ref_dir = PROJECT / "preflight_ref" / args.ref
    meta_path = ref_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"기준이 없습니다: {meta_path}\n먼저 `python make_ref.py` 를 돌리세요.")
    meta = json.loads(meta_path.read_text())
    print(f"기준: {ref_dir}  (출처 {meta.get('_source', {}).get('repo_id', '?')})")

    caps = {n: open_cam(p) for n, p in DEV.items()}
    try:
        if args.watch:
            print("q 로 종료. 초점 링·카메라를 만지면서 OK 가 되게 맞추세요.")
            while True:
                frames = {n: (grab(c, 2) if c else None) for n, c in caps.items()}
                print("\033[2J\033[H", end="")
                report(ref_dir, frames, meta, verbose=False)
                if cv2.waitKey(200) & 0xFF == ord("q"):
                    break
        else:
            frames = {n: (grab(c) if c else None) for n, c in caps.items()}
            v = report(ref_dir, frames, meta)
            bad = [x for x in v if x[1] == "FAIL"]
            warn = [x for x in v if x[1] == "WARN"]
            print("\n" + "=" * 60)
            if bad:
                print("FAIL — 이 상태로 평가하면 그 판은 무효다:")
                for n, _, msg in bad:
                    print(f"  {n}: {msg}")
                print("\n선명도 FAIL 이면 `python focus_wrist.py`, 화각 FAIL 이면 "
                      "`python align_top.py` 로 맞춘다.")
            elif warn:
                print("WARN — 돌려도 되지만 기록해둘 것:")
                for n, _, msg in warn:
                    print(f"  {n}: {msg}")
            else:
                print("전부 OK — 지금 카메라는 학습 때와 같은 화면을 주고 있다.")
                print("실기 오차의 원인은 카메라가 아니다.")
            print("=" * 60)
    finally:
        for c in caps.values():
            if c:
                c.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
