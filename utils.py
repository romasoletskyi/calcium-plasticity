from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def extract_unique(values: list[T]) -> T:
    unique_values = list(set(values))
    assert len(unique_values) == 1
    return unique_values[0]


def get_run_path(run_name: str) -> Path:
    return Path(__file__).parent / "runs" / run_name
