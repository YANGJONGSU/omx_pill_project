#!/usr/bin/env python3
"""위키 마크다운을 올리기 전에 브라우저로 미리 본다 (의존성 없음).

GitHub 위키에 push 하기 전에 표·링크·코드블록이 제대로 나오는지 확인하는 용도.
`[[페이지명]]` 위키 링크는 로컬 HTML 끼리 이어지게 바꾼다.

    python wiki_preview.py ~/Desktop/mingky_care_pro.wiki
"""
import html
import re
import sys
import webbrowser
from pathlib import Path

MERMAID_JS = """
<script src="mermaid.min.js"></script>
<script>
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  mermaid.initialize({startOnLoad: true, theme: dark ? "dark" : "default",
                      themeVariables: {fontFamily: "Pretendard, Noto Sans KR, sans-serif"}});
</script>
"""

CSS = """
:root{--bg:#fff;--ink:#1f2328;--muted:#59636e;--line:#d1d9e0;--code:#f6f8fa;--link:#0969da;--quote:#59636e}
@media(prefers-color-scheme:dark){:root{--bg:#0d1117;--ink:#e6edf3;--muted:#9198a1;--line:#3d444d;--code:#151b23;--link:#4493f8;--quote:#9198a1}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,"Pretendard","Noto Sans KR","Segoe UI",sans-serif}
.bar{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 24px;display:flex;gap:16px;flex-wrap:wrap;align-items:center;z-index:5}
.bar b{font-size:13px;color:var(--muted);font-weight:600}
.bar a{font-size:13px;color:var(--link);text-decoration:none}
.bar a:hover{text-decoration:underline}
main{max-width:1012px;margin:0 auto;padding:32px 24px 120px}
h1,h2,h3,h4{line-height:1.25;margin:24px 0 16px;font-weight:600}
h1{font-size:2em;padding-bottom:.3em;border-bottom:1px solid var(--line)}
h2{font-size:1.5em;padding-bottom:.3em;border-bottom:1px solid var(--line)}
h3{font-size:1.25em} h4{font-size:1em}
p,ul,ol{margin:0 0 16px} li{margin:.25em 0}
a{color:var(--link);text-decoration:none} a:hover{text-decoration:underline}
code{background:var(--code);padding:.2em .4em;border-radius:6px;font-size:85%;
     font-family:ui-monospace,"D2Coding","Noto Sans Mono CJK KR",monospace}
pre{background:var(--code);padding:16px;border-radius:6px;overflow-x:auto;margin:0 0 16px}
pre code{background:none;padding:0;font-size:85%;line-height:1.45}
table{border-collapse:collapse;margin:0 0 16px;display:block;width:max-content;max-width:100%;overflow:auto}
th,td{border:1px solid var(--line);padding:6px 13px}
th{background:var(--code);font-weight:600}
tr:nth-child(2n) td{background:var(--code)}
blockquote{margin:0 0 16px;padding:0 1em;color:var(--quote);border-left:.25em solid var(--line)}
hr{height:.25em;background:var(--line);border:0;margin:24px 0}
.missing{color:#cf222e;border-bottom:1px dashed #cf222e}\npre.mermaid{background:none;padding:8px 0;text-align:center;overflow-x:auto}
"""


def inline(s: str, pages: set) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)

    def wiki(m):
        name = m.group(1)
        cls = "" if name in pages else ' class="missing"'
        return f'<a href="{name}.html"{cls}>{name}</a>'

    return re.sub(r"\[\[([^\]]+)\]\]", wiki, s)


def convert(md: str, pages: set) -> str:
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]

        if ln.startswith("```"):                      # 코드블록
            lang = ln[3:].strip().lower()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i])); i += 1
            body = "\n".join(buf)
            if lang == "mermaid":                     # GitHub 처럼 다이어그램으로 그린다
                out.append('<pre class="mermaid">' + body + "</pre>")
            else:
                out.append("<pre><code>" + body + "</code></pre>")
            i += 1; continue

        if re.match(r"^\|.*\|\s*$", ln) and i + 1 < len(lines) \
           and re.match(r"^\|[\s:|-]+\|\s*$", lines[i + 1]):        # 표
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
            head = cells(ln); i += 2
            rows = []
            while i < len(lines) and re.match(r"^\|.*\|\s*$", lines[i]):
                rows.append(cells(lines[i])); i += 1
            t = ["<table><thead><tr>"] + [f"<th>{inline(c, pages)}</th>" for c in head] + ["</tr></thead><tbody>"]
            for r in rows:
                t += ["<tr>"] + [f"<td>{inline(c, pages)}</td>" for c in r] + ["</tr>"]
            out.append("".join(t + ["</tbody></table>"])); continue

        if re.match(r"^#{1,6}\s", ln):                # 제목
            n = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{n}>{inline(ln[n:].strip(), pages)}</h{n}>"); i += 1; continue

        if ln.startswith(">"):                        # 인용
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip()); i += 1
            out.append("<blockquote>" + convert("\n".join(buf), pages) + "</blockquote>"); continue

        if re.match(r"^\s*[-*]\s", ln) or re.match(r"^\s*\d+\.\s", ln):   # 목록
            ordered = bool(re.match(r"^\s*\d+\.\s", ln))
            items = []
            while i < len(lines) and (re.match(r"^\s*[-*]\s", lines[i]) or re.match(r"^\s*\d+\.\s", lines[i])):
                indent = len(lines[i]) - len(lines[i].lstrip())
                txt = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", lines[i])
                items.append((indent, txt)); i += 1
            tag = "ol" if ordered else "ul"
            buf, depth = [f"<{tag}>"], 0
            for ind, txt in items:
                d = ind // 2
                while d > depth: buf.append(f"<{tag}>"); depth += 1
                while d < depth: buf.append(f"</{tag}>"); depth -= 1
                buf.append(f"<li>{inline(txt, pages)}</li>")
            buf += [f"</{tag}>"] * (depth + 1)
            out.append("".join(buf)); continue

        if re.match(r"^-{3,}\s*$", ln) or re.match(r"^={3,}\s*$", ln):
            out.append("<hr>"); i += 1; continue

        if ln.strip() == "":
            i += 1; continue

        buf = []                                       # 문단
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6}\s|```|\||>|\s*[-*]\s|\s*\d+\.\s|-{3,})", lines[i]):
            buf.append(lines[i]); i += 1
        out.append("<p>" + inline(" ".join(buf), pages) + "</p>")
    return "\n".join(out)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()
    mds = sorted(p for p in root.glob("*.md") if not p.name.startswith("_"))
    pages = {p.stem for p in root.glob("*.md")}
    out_dir = Path("/tmp/claude-1000/-home-user/fd2b9c3d-8350-46d8-b079-9fd69527330d/scratchpad/wikiview")
    out_dir.mkdir(parents=True, exist_ok=True)

    nav = " ".join(f'<a href="{p.stem}.html">{p.stem}</a>' for p in mds)
    for p in root.glob("*.md"):
        body = convert(p.read_text(encoding="utf-8"), pages)
        (out_dir / f"{p.stem}.html").write_text(
            f"<!doctype html><meta charset='utf-8'><title>{p.stem}</title>"
            f"<style>{CSS}</style>"
            f"<div class='bar'><b>미리보기 · push 전</b>{nav}</div><main>{body}</main>"
            + MERMAID_JS,
            encoding="utf-8")
    print(f"  {len(list(root.glob('*.md')))}개 변환 → {out_dir}")
    first = out_dir / "Home.html"
    target = first if first.exists() else out_dir / f"{mds[0].stem}.html"
    print(f"  열기: {target}")
    webbrowser.open(f"file://{target}")


if __name__ == "__main__":
    main()
