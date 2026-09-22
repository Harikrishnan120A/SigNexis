import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dsp.signal_generator import generate_signal


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signal_type": "triangle"},
        {"frequency": 0},
        {"amplitude": -1},
        {"duration": 0},
        {"sampling_rate": 0},
        {"noise_amplitude": -1},
    ],
)
def test_invalid_generation_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        generate_signal(**kwargs)


def test_valid_generation_remains_reproducible_with_seeded_rng():
    _, clean_a, noisy_a = generate_signal(rng=np.random.default_rng(7))
    _, clean_b, noisy_b = generate_signal(rng=np.random.default_rng(7))
    assert np.array_equal(clean_a, clean_b)
    assert np.array_equal(noisy_a, noisy_b)