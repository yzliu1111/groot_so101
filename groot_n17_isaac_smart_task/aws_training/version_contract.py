"""Version-number matching shared by the AWS runtime preflight and tests."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


_NUMERIC_RELEASE = re.compile(
    r"^(?P<release>[0-9]+(?:\.[0-9]+)*)(?:\+[0-9A-Za-z][0-9A-Za-z._-]*)?$"
)


def normalized_release(value: object) -> tuple[int, ...]:
    """Return a numeric release, ignoring only a legal local-build suffix."""

    text = str(value).strip()
    match = _NUMERIC_RELEASE.fullmatch(text)
    if match is None:
        raise ValueError(f"not a numeric release version: {text!r}")
    parts = [int(part) for part in match.group("release").split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def version_number_matches(actual: object, expected: object) -> bool:
    """Compare release numbers while rejecting prerelease/postrelease text."""

    try:
        return normalized_release(actual) == normalized_release(expected)
    except ValueError:
        return False


def version_mismatches(
    expected: Mapping[str, object], actual: Mapping[str, object]
) -> dict[str, dict[str, Any]]:
    """Describe missing, malformed, or different runtime version numbers."""

    mismatches: dict[str, dict[str, Any]] = {}
    for name, expected_value in expected.items():
        actual_value = actual.get(name)
        if not version_number_matches(actual_value, expected_value):
            mismatches[name] = {
                "expected": expected_value,
                "actual": actual_value,
            }
    return mismatches
