"""노트북이 **최근 커밋이 추가한 내용을 지우고 있는지** 본다 — IDE 옛 버퍼 사고 탐지.

두 번 났다: IDE가 파일을 연 채로 커밋이 진행되면, 그 뒤 에디터가 저장할 때 **옛 버퍼 전체**가
디스크를 덮어써 커밋된 셀 수정이 조용히 사라진다. 노트북 diff는 JSON이라 눈으로 못 잡는다.

되돌리기의 서명은 하나다: **최근 커밋이 추가한 줄을 지금 지우고 있다.** 그걸 기계로 본다.
의도한 되돌리기여도 경고는 맞는 정보이므로, 확인하고 진행하면 된다.

사용법:
    python scripts/check_notebook_regression.py          # 커밋 전 확인 (되돌림 있으면 종료코드 1)
    python scripts/check_notebook_regression.py --since 10
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# JSON 구조만 있는 줄은 신호가 아니다 — 실제 내용이 든 줄만 본다.
MIN_CONTENT = 12


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def content_lines(diff: str, sign: str) -> set[str]:
    """diff에서 추가(+)/삭제(-) 줄의 내용만 뽑는다."""
    out = set()
    for line in diff.splitlines():
        if not line.startswith(sign) or line.startswith(sign * 3):
            continue
        body = line[1:].strip().strip(",").strip('"').strip()
        if len(body) >= MIN_CONTENT:
            out.add(body)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="노트북 되돌림 탐지 (IDE 옛 버퍼)")
    parser.add_argument("--since", type=int, default=5, help="최근 몇 커밋을 기준으로 볼지")
    args = parser.parse_args()

    touched = git("diff", "--name-only", "--", "notebooks").split()
    changed = [p for p in touched if p.endswith(".ipynb")]
    if not changed:
        print("노트북에 미커밋 변경 없음 — 검사할 것이 없다")
        return 0

    hits = 0
    for path in changed:
        removing = content_lines(git("diff", "-U0", "--", path), "-")
        if not removing:
            print(f"[ok  ] {path} — 지우는 내용 없음 (추가만)")
            continue
        recent = set()
        for sha in git("log", f"-{args.since}", "--format=%H", "--", path).split():
            recent |= content_lines(git("show", "-U0", sha, "--", path), "+")
        reverted = sorted(removing & recent)
        if not reverted:
            print(f"[ok  ] {path} — 지우는 {len(removing)}줄이 최근 {args.since}커밋과 무관")
            continue
        hits += 1
        print(f"\n[!! ] {path} — 최근 {args.since}커밋이 추가한 {len(reverted)}줄을 지우고 있다")
        for line in reverted[:8]:
            print(f"        − {line[:96]}")
        if len(reverted) > 8:
            print(f"        … 외 {len(reverted) - 8}줄")

    if hits:
        print(
            f"\n{hits}개 노트북이 최근 커밋 내용을 되돌린다. IDE 옛 버퍼가 의심되면:\n"
            "  1) 에디터에서 그 노트북을 닫는다 (저장하지 않는다)\n"
            "  2) git checkout -- <경로> 로 디스크를 HEAD로 되돌린다\n"
            "  3) 다시 열어 편집한다\n"
            "의도한 되돌리기라면 그대로 커밋하고 이유를 커밋 메시지에 적는다."
        )
    return 1 if hits else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
