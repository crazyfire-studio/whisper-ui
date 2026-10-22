"""The bookkeeping around the copied conformance test that the copy cannot check.

``CONFORMANCE_XFAILS`` still names tests that exist: an entry that
matches nothing is stale, because the test it excused was renamed or
deleted, and leaving it behind reserves a slot in ``xfail_budget`` for
a gap nobody can close. And every xfail reason and every profile waiver
cites a numbered issue, which the copy's own check is too loose to
enforce.

The names are resolved against the imported module rather than
against the items pytest collected. A check that reads the collection
cannot tell a renamed test from a narrowed selection, so it fails on
``pytest tests/unit/test_ytdlp_conformance.py::TestTheExtractorPinHolds``
-- an ordinary thing to type while debugging. Resolving against the
module removes the question instead of answering it.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

import tests.unit.test_ytdlp_conformance as conformance_module
from tests.unit.conftest import CONFORMANCE_XFAILS

CONFORMANCE_PATH = "tests/unit/test_ytdlp_conformance.py"
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "pyproject.toml"

# The copy's own waiver check asks only for a "#" in the reason, which
# "see #TBD" -- and the upstream template's "see #NNN" -- satisfy; a
# placeholder shipped in a waiver that way. That check sits in the
# hashed region, so the stricter one lives here.
_ISSUE_REFERENCE = re.compile(r"#[1-9]\d*\b")


def _cites_an_issue(reason: str) -> bool:
    return _ISSUE_REFERENCE.search(reason) is not None


def _profile() -> dict[str, Any]:
    with MANIFEST_PATH.open("rb") as handle:
        return tomllib.load(handle)["tool"]["ytdlp_conformance"]


def _resolve(nodeid: str) -> object | None:
    """The test a node id names, or ``None`` when it names nothing."""
    path, *attributes = nodeid.split("::")
    if path != CONFORMANCE_PATH:
        return None
    target: object | None = conformance_module
    for attribute in attributes:
        target = getattr(target, attribute, None)
        if target is None:
            return None
    return target


def test_the_registry_is_not_empty() -> None:
    # The other direction. An empty registry would pass every check
    # below while saying nothing.
    assert CONFORMANCE_XFAILS, (
        "no conformance gap is registered; if the backlog really is empty, "
        "xfail_budget should be 0 and this file can go"
    )


def test_every_registered_node_id_names_a_real_test() -> None:
    unresolved = sorted(nodeid for nodeid in CONFORMANCE_XFAILS if _resolve(nodeid) is None)
    assert not unresolved, (
        "these conformance xfail entries name no test in "
        f"{CONFORMANCE_PATH}: {unresolved}. Remove the entry and lower "
        "xfail_budget, or fix the name."
    )


def test_every_reason_cites_an_issue() -> None:
    # A gap without a ticket is an omission with extra steps, which is
    # the same rule the profile's waivers are held to.
    uncited = sorted(nodeid for nodeid, reason in CONFORMANCE_XFAILS.items() if not _cites_an_issue(reason))
    assert not uncited, f"these conformance xfail entries cite no numbered issue: {uncited}"


def test_every_waiver_cites_an_issue() -> None:
    waivers = _profile().get("waivers", {})
    uncited = sorted(aspect for aspect, reason in waivers.items() if not _cites_an_issue(reason))
    assert not uncited, f"these waivers in {MANIFEST_PATH.name} cite no numbered issue: {uncited}"


@pytest.mark.parametrize("reason", ["see #189", "see fdff87554/youtube-downloader#140", "tracked in #7."])
def test_a_numbered_issue_counts_as_a_citation(reason: str) -> None:
    assert _cites_an_issue(reason)


@pytest.mark.parametrize("reason", ["see #TBD", "see #NNN", "see #", "see #0", "see #12a", "no ticket yet"])
def test_a_placeholder_does_not_count_as_a_citation(reason: str) -> None:
    assert not _cites_an_issue(reason)


def test_selecting_one_class_does_not_fail_the_run() -> None:
    # A regression on the real symptom seen in the sibling project: an
    # earlier version checked the registry from
    # pytest_collection_modifyitems, so selecting a class that holds
    # none of the registered tests left them looking stale and the run
    # exited 4 before anything ran. TestTheChallengeSolversMatch holds
    # none of this repository's five entries.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{CONFORMANCE_PATH}::TestTheChallengeSolversMatch",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"selecting a single class failed the run:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
    )
