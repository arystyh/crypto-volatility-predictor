import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.synthetic import generate_candles  # noqa: E402


@pytest.fixture(scope="session")
def candles():
    return generate_candles("BTCUSDT", days=2, seed=1)
