import pytest

from src.utils.functions import get_terms


@pytest.mark.parametrize(
    "text, expected", [
        ("", tuple()),
        (
            "Lorem ipsum, dolor sit amet.",
            ("lorem", "ipsum", "dolor", "sit", "amet", )
        ),
        (
            "Ultricies mi eget! Mauris pharetra?",
            ("ultricies", "mi", "eget", "mauris", "pharetra", )
        ),
        (
            "Y neque a suspendisse. E interdum.",
            ("neque", "suspendisse", "interdum", )
        )
    ]
)
def test_get_terms(text, expected):
    assert get_terms(text) == expected

