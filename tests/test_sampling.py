"""tests/test_sampling.py – Pytest tests for the sampling module."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from dsp.sampling import analyse_sampling, nyquist_frequency, check_nyquist_condition


class TestNyquistFrequency:
    def test_basic(self):
        assert nyquist_frequency(8000) == pytest.approx(4000.0)

    def test_half_of_sampling_rate(self):
        for fs in [44100, 22050, 8000, 16000]:
            assert nyquist_frequency(fs) == pytest.approx(fs / 2.0)


class TestNyquistCondition:
    def test_properly_sampled(self):
        assert check_nyquist_condition(1000, 8000) is True   # 1 kHz, 8 kHz fs

    def test_exactly_nyquist_fails(self):
        # f_signal == nyquist → NOT properly sampled
        assert check_nyquist_condition(4000, 8000) is False

    def test_undersampled(self):
        assert check_nyquist_condition(5000, 8000) is False  # 5 kHz > 4 kHz nyquist


class TestSamplingAnalysis:
    def test_properly_sampled_example(self):
        """f=1000 Hz, fs=8000 Hz → properly sampled."""
        result = analyse_sampling(1000, 8000)
        assert result.nyquist_freq == pytest.approx(4000.0)
        assert result.is_properly_sampled is True
        assert result.aliasing_detected is False
        assert result.apparent_freq is None

    def test_aliasing_example(self):
        """f=5000 Hz, fs=8000 Hz → aliasing detected."""
        result = analyse_sampling(5000, 8000)
        assert result.nyquist_freq == pytest.approx(4000.0)
        assert result.is_properly_sampled is False
        assert result.aliasing_detected is True
        # apparent frequency should be in [0, 4000]
        assert result.apparent_freq is not None
        assert 0 <= result.apparent_freq <= result.nyquist_freq

    def test_discrete_samples_match_time_axis(self):
        result = analyse_sampling(500, 8000, duration=0.01)
        assert len(result.t_discrete) == len(result.signal_discrete)

    def test_invalid_sampling_rate(self):
        with pytest.raises(ValueError):
            analyse_sampling(1000, 0)

    def test_invalid_signal_freq(self):
        with pytest.raises(ValueError):
            analyse_sampling(-100, 8000)
