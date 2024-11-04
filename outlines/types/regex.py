paragraph = r"(?:[^\r\n]*[\r\n]+)"


def one_or_more(pattern: str):
    return f"({pattern})+"


def zero_or_more(pattern: str):
    return f"({pattern})*"


def optional(pattern: str):
    return f"({pattern})?"


def exactly(number: int, pattern: str):
    raise NotImplementedError


def between(min: int, max: int, pattern: str):
    raise NotImplementedError
