from collections.abc import Callable
from datetime import UTC, datetime

import pytest

FIXED_NOW = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def fixed_now() -> datetime:
    return FIXED_NOW


@pytest.fixture
def fixed_clock() -> Callable[[], datetime]:
    return lambda: FIXED_NOW
