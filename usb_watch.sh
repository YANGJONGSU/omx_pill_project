#!/usr/bin/env bash
# USB 를 꽂고 뽑을 때 커널이 뭘 보는지 실시간으로 보여준다.
# 케이블/카메라 고장을 "컴퓨터 문제"와 가르는 데 쓴다.
echo "════════════════════════════════════════════════════════════"
echo "  USB 실시간 감시 — 꽂거나 뽑으면 여기에 바로 뜹니다"
echo "════════════════════════════════════════════════════════════"
echo
echo "  ① 먼저 마우스를 뽑았다 꽂아보세요 (감시가 되는지 확인용)"
echo "     → 줄이 뜨면 감시는 정상입니다"
echo
echo "  ② 그다음 wrist 카메라를 꽂아보세요"
echo "     → 아무 줄도 안 뜨면 케이블이나 카메라 고장입니다"
echo
echo "────────────────────────────────────────────────────────────"
stdbuf -oL journalctl -kf -n 0 2>/dev/null | stdbuf -oL grep -iE "usb [0-9]|uvcvideo|error -" |
while IFS= read -r l; do
  t=$(echo "$l" | awk '{print $3}')
  m=$(echo "$l" | sed 's/.*kernel: //')
  case "$m" in
    *disconnect*)     echo "  [$t] 🔌 뽑힘   $m" ;;
    *"New USB device found"*) echo "  [$t] ✅ 인식   $m" ;;
    *Product:*)       echo "  [$t]    └─ $m" ;;
    *Manufacturer:*)  echo "  [$t]    └─ $m" ;;
    *error*|*"can't"*|*fail*) echo "  [$t] ❌ 오류   $m" ;;
    *new*speed*)      echo "  [$t] ⚡ 연결됨 $m" ;;
  esac
done
