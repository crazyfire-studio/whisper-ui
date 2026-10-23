"""The release job's CHANGELOG extraction (.github/scripts/changelog_section.py)."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / ".github" / "scripts" / "changelog_section.py"

_spec = importlib.util.spec_from_file_location("changelog_section", _SCRIPT)
assert _spec is not None and _spec.loader is not None
changelog_section = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(changelog_section)

_CHANGELOG = """# Changelog

## [Unreleased]

## [2.1.0] - 2026-02-01

Summary line.

### Added

- New thing.

## [2.0.0] - 2026-01-01

### Fixed

- Old bug.

## [1.0.0] - 2025-12-01

[Unreleased]: https://example.invalid/compare/v2.1.0...HEAD
[2.1.0]: https://example.invalid/releases/tag/v2.1.0
"""


def test_a_section_runs_up_to_the_next_heading_without_its_own_heading() -> None:
    assert changelog_section.section(_CHANGELOG, "2.1.0") == "Summary line.\n\n### Added\n\n- New thing.\n"


def test_a_middle_section_does_not_reach_into_the_next_one() -> None:
    assert changelog_section.section(_CHANGELOG, "2.0.0") == "### Fixed\n\n- Old bug.\n"


def test_the_last_section_stops_before_the_link_definitions() -> None:
    changelog = _CHANGELOG.replace("## [1.0.0] - 2025-12-01\n", "## [1.0.0] - 2025-12-01\n\n- First release.\n")
    assert changelog_section.section(changelog, "1.0.0") == "- First release.\n"


def test_a_version_without_a_section_is_an_error() -> None:
    with pytest.raises(LookupError, match=r"no section for 3\.0\.0"):
        changelog_section.section(_CHANGELOG, "3.0.0")


def test_an_empty_section_is_an_error() -> None:
    # The fixture's 1.0.0 heading is followed only by the link definitions.
    with pytest.raises(LookupError, match="empty"):
        changelog_section.section(_CHANGELOG, "1.0.0")


def test_a_version_is_matched_exactly_not_as_a_prefix() -> None:
    with pytest.raises(LookupError):
        changelog_section.section(_CHANGELOG, "2.1")


def test_the_cli_writes_the_section_and_fails_on_a_missing_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(_CHANGELOG, encoding="utf-8")

    assert changelog_section.main(["changelog_section.py", "2.0.0", str(changelog)]) == 0
    assert capsys.readouterr().out == "### Fixed\n\n- Old bug.\n"

    assert changelog_section.main(["changelog_section.py", "9.9.9", str(changelog)]) == 1
    assert "no section for 9.9.9" in capsys.readouterr().err


def test_the_current_version_has_a_changelog_section() -> None:
    # The release job publishes this section when the tag is pushed, so a
    # version bump that forgets the CHANGELOG fails here, in the release PR,
    # rather than at tag time.
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog_section.section(changelog, version).strip()
