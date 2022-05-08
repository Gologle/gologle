import re


def get_terms(text: str) -> tuple[str]:
    """Returns the terms of a text. All the terms empty or with length 1 are
    removed.

    Args:
        text: the text that contain the terms

    Returns:
        A tuple with the terms.
    """
    return tuple(
        filter(
            lambda x: len(x) > 1,   # get rid of small terms
            re.split(r"[\s\,\.\!\?\(\)\[\]/\\\{\}'\"]", text)
        )
    )
