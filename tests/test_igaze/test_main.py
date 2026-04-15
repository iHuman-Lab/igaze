from __future__ import annotations

from igaze.main import get_hello


def it_prints_hi_to_the_project_author() -> None:
    expected = "Hello, iHuman Labb!"
    actual = get_hello("iHuman Labb")
    assert actual == expected
