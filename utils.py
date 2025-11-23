from typing import TypeVar

T = TypeVar("T")


def extract_unique(values: list[T]) -> T:
    unique_values = list(set(values))
    assert len(unique_values) == 1
    return unique_values[0]
