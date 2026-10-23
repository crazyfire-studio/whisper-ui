"""Print one version's section of CHANGELOG.md, for a GitHub Release body.

Usage: changelog_section.py VERSION [CHANGELOG]

The section is everything between the version's ``## [VERSION]`` heading
and the next ``## [`` heading, without the heading itself, which is how
the existing releases were written. A version with no section, or an empty
one, exits non-zero: a tag pushed without its CHANGELOG entry should fail
the release, not publish blank notes.

Standard library only, because the release job runs it with the runner's
own Python and installs nothing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]", re.MULTILINE)
# The link definitions at the bottom ("[2.21.0]: https://...") follow the
# last section, which has no next heading to stop at.
_LINK_DEFINITION = re.compile(r"^\[[^\]]+\]: ", re.MULTILINE)


def section(changelog: str, version: str) -> str:
    """Return the body of ``version``'s section, ending in one newline.

    Raises ``LookupError`` when the version has no section or only an empty one.
    """
    headings = list(_HEADING.finditer(changelog))
    for index, heading in enumerate(headings):
        if heading.group("version") != version:
            continue
        # Skip the rest of the heading line (" - 2026-10-23").
        start = changelog.find("\n", heading.end()) + 1 or len(changelog)
        if index + 1 < len(headings):
            end = headings[index + 1].start()
        else:
            definition = _LINK_DEFINITION.search(changelog, start)
            end = definition.start() if definition else len(changelog)
        body = changelog[start:end].strip()
        if not body:
            raise LookupError(f"the CHANGELOG section for {version} is empty")
        return body + "\n"
    raise LookupError(f"the CHANGELOG has no section for {version}")


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        sys.stderr.write(__doc__ or "")
        return 2
    path = Path(argv[2]) if len(argv) == 3 else Path("CHANGELOG.md")
    try:
        sys.stdout.write(section(path.read_text(encoding="utf-8"), argv[1]))
    except LookupError as error:
        sys.stderr.write(f"{error}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
