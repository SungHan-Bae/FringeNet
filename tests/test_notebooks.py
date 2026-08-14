"""Colab 노트북 필수 셀 규약 — IDE 옛 버퍼가 커밋된 fix를 되돌리는 회귀를 잡는다.

두 번 났다: `dbefab4`가 round2를 옛 버퍼로 덮어써 PAT 자동 로드·5초 반납 fix를 날렸고
(round3가 그걸 복사해 물려받았다), PR #12 수정도 같은 방식으로 되돌아갔다. 사람이 "저장 전에
디스크 최신인지 확인"하는 것으로는 두 번 다 못 막았으므로 규약을 **검사 가능한 불변식**으로
바꾼다.

규약 이전에 끝난 라운드는 실행 로그 보존을 위해 고치지 않으므로, 헤더 첫 셀에 `규약 예외`를
적어 명시적으로 제외한다 (조용히 건너뛰지 않는다 — 이유가 아티팩트 옆에 남는다).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.rglob("*.ipynb"))
# 정적 PAT 소스 — 이 중 둘 이상을 시도해야 Run-All이 입력 대기 없이 돈다.
PAT_SOURCES = (r"os\.environ.*GITHUB_PAT|GITHUB_PAT.*os\.environ", r"userdata\.get", r"github_pat\.txt")
# 반납 셀을 식별하는 표지와, 취소 대기 sleep의 상한 [초].
TEARDOWN = re.compile(r"kill_session|terminate|unassign", re.I)
MAX_TEARDOWN_SLEEP = 5
# 대회 데이터 행이 출력에 찍혔는지 — 6자리 이상 소수가 6개 넘게 연달아 나오면 반사율 덤프다.
DATA_DUMP = re.compile(r"0\.\d{6,}(?:,\s*0\.\d{6,}){5,}")


def load(path: Path) -> tuple[dict, str]:
    """(노트북 dict, 전 셀 소스를 이은 문자열)."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    return nb, "\n".join("".join(c.get("source", [])) for c in nb["cells"])


def is_exempt(nb: dict) -> bool:
    """헤더에 `규약 예외`가 적힌 노트북은 규약 이전 라운드다."""
    return "규약 예외" in "".join(nb["cells"][0].get("source", []))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid(path: Path) -> None:
    """JSON으로 파싱되고 nbformat 4이며 셀이 있다."""
    nb, _ = load(path)
    assert nb["nbformat"] == 4, path
    assert nb["cells"], path


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_run_all_is_unattended(path: Path) -> None:
    """Run-All이 입력 대기 없이 돈다 — 정적 PAT 소스를 둘 이상 시도해야 한다.

    `getpass`가 있는 것 자체는 문제가 아니다 (정적 소스가 전부 없을 때의 최후 폴백).
    문제는 그것이 **주 경로**가 되는 것이다.
    """
    nb, src = load(path)
    if is_exempt(nb) or "git push" not in src:
        pytest.skip("규약 예외 라운드이거나 push 셀이 없다")
    found = [p for p in PAT_SOURCES if re.search(p, src)]
    assert len(found) >= 2, f"{path.name}: 정적 PAT 소스가 {len(found)}개뿐 — Run-All이 막힌다"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_drive_mount_and_teardown(path: Path) -> None:
    """Drive는 항상 force_remount, 반납 취소 대기는 5초 이하 (유휴 과금)."""
    nb, src = load(path)
    if is_exempt(nb):
        pytest.skip("규약 예외 라운드")
    if "drive.mount(" in src:
        assert "force_remount=True" in src, f"{path.name}: stale 마운트 위험"
    if TEARDOWN.search(src):
        waits = [int(s) for s in re.findall(r"sleep\((\d+)\)", src)]
        assert waits, f"{path.name}: 반납 셀에 취소 대기가 없다"
        assert max(waits) <= MAX_TEARDOWN_SLEEP, f"{path.name}: sleep {max(waits)}초 (유휴 과금)"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_competition_data_in_outputs(path: Path) -> None:
    """커밋된 출력에 대회 데이터 행이 없다 (재배포 금지 계약)."""
    assert not DATA_DUMP.search(path.read_text(encoding="utf-8")), f"{path.name}: 데이터 행 유출"
