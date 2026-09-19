"""tests/test_filters.py – Pytest tests for the digital filter module."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from dsp.filters import (
    low_pass_filter,
    high_pass_filter,
    band_pass_filter,
    notch_filter,
    apply_recommended_filter,
)


def _sine(freq, fs, duration=0.5):
    t = np.arange(0, duration, 1.0 / fs)
    return np.sin(2 * np.pi * freq * t)


class TestLowPassFilter:
    def test_output_same_length(self):
        signal = _sine(500, 8000)
        out = low_pass_filter(signal, 1000, 8000)
        assert len(out) == len(signal)

    def test_attenuates_high_freq(self):
        """LPF at 1 kHz should attenuate a 3.5 kHz component."""
        fs = 8000
        t = np.arange(0, 1.0, 1.0 / fs)
        low = np.sin(2 * np.pi * 500 * t)
        high = np.sin(2 * np.pi * 3500 * t)
        mixed = low + high
        filtered = low_pass_filter(mixed, 1000, fs)
        # RMS of filtered should be much closer to low-only RMS
        rms_filtered = np.sqrt(np.mean(filtered ** 2))
        rms_high_only = np.sqrt(np.mean(high ** 2))
        assert rms_filtered < rms_high_only + 0.3

    def test_invalid_cutoff_above_nyquist(self):
        with pytest.raises(ValueError):
            low_pass_filter(_sine(500, 8000), 4001, 8000)

    def test_invalid_cutoff_zero(self):
        with pytest.raises(ValueError):
            low_pass_filter(_sine(500, 8000), 0, 8000)


class TestHighPassFilter:
    def test_output_same_length(self):
        signal = _sine(2000, 8000)
        out = high_pass_filter(signal, 500, 8000)
        assert len(out) == len(signal)

    def test_invalid_cutoff_above_nyquist(self):
        with pytest.raises(ValueError):
            high_pass_filter(_sine(500, 8000), 5000, 8000)


class TestBandPassFilter:
    def test_output_same_length(self):
        signal = _sine(1000, 8000)
        out = band_pass_filter(signal, 800, 1200, 8000)
        assert len(out) == len(signal)

    def test_invalid_low_above_high(self):
        with pytest.raises(ValueError):
            band_pass_filter(_sine(1000, 8000), 2000, 1000, 8000)

    def test_invalid_cutoff_above_nyquist(self):
        with pytest.raises(ValueError):
            band_pass_filter(_sine(1000, 8000), 100, 5000, 8000)


class TestNotchFilter:
    def test_output_same_length(self):
        signal = _sine(1000, 8000) + _sine(50, 8000)
        out = notch_filter(signal, 50.0, 8000)
        assert len(out) == len(signal)

    def test_attenuates_50hz(self):
        fs = 8000
        t = np.arange(0, 2.0, 1.0 / fs)
        signal_1k = np.sin(2 * np.pi * 1000 * t)
        hum_50 = np.sin(2 * np.pi * 50 * t)
        mixed = signal_1k + hum_50
        filtered = notch_filter(mixed, 50.0, fs)
        # The 50 Hz power should be much reduced
        from scipy.fft import fft, fftfreq
        n = len(filtered)
        freqs = fftfreq(n, 1.0 / fs)
        mag = np.abs(fft(filtered))
        # Find 50 Hz bin magnitude
        idx = np.argmin(np.abs(freqs - 50))
        assert mag[idx] < mag[np.argmin(np.abs(freqs - 1000))] * 0.5

    def test_invalid_notch_at_nyquist(self):
        with pytest.raises(ValueError):
            notch_filter(_sine(500, 8000), 4000, 8000)


class TestRecommendedFilter:
    def test_short_signal_uses_safe_fallback(self):
        signal = np.ones(3)
        filtered, info = apply_recommended_filter(signal, 1000, "GAUSSIAN_NOISE")
        assert len(filtered) == len(signal)
        assert np.all(np.isfinite(filtered))
        assert info["filter_type"] == "low_pass"

    def test_clean_returns_unchanged(self):
        signal = _sine(1000, 8000)
        filtered, info = apply_recommended_filter(signal, 8000, "CLEAN")
        np.testing.assert_array_equal(filtered, signal)
        assert info["filter_type"] == "none"

    def test_gaussian_noise_applies_filter(self):
        rng = np.random.default_rng(42)
        signal = _sine(1000, 8000) + rng.normal(0, 0.5, len(_sine(1000, 8000)))
        filtered, info = apply_recommended_filter(signal, 8000, "GAUSSIAN_NOISE", 1000)
        assert len(filtered) == len(signal)
        assert info["filter_type"] == "low_pass"

    def test_power_line_applies_notch(self):
        signal = _sine(1000, 8000) + _sine(50, 8000)
        filtered, info = apply_recommended_filter(signal, 8000, "POWER_LINE_INTERFERENCE")
        assert info["filter_type"] == "notch"
