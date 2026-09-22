"""tests/test_fft.py – Pytest tests for the FFT analysis module."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from dsp.fft_analysis import compute_fft


class TestFFTBasics:
    def _pure_sine(self, freq, fs=8000, duration=1.0):
        t = np.arange(0, duration, 1.0 / fs)
        return np.sin(2 * np.pi * freq * t), fs

    def test_dominant_frequency_1kHz(self):
        """A 1 kHz sine at fs=8000 Hz → dominant_freq ≈ 1000 Hz."""
        signal, fs = self._pure_sine(1000, 8000)
        result = compute_fft(signal, fs)
        assert result.dominant_freq == pytest.approx(1000.0, abs=10.0)

    def test_dominant_frequency_500Hz(self):
        signal, fs = self._pure_sine(500, 8000)
        result = compute_fft(signal, fs)
        assert result.dominant_freq == pytest.approx(500.0, abs=5.0)

    def test_frequency_axis_positive(self):
        signal, fs = self._pure_sine(1000, 8000)
        result = compute_fft(signal, fs)
        assert np.all(result.freqs >= 0), "All returned frequencies should be non-negative."

    def test_frequency_axis_max(self):
        """Max frequency in axis should not exceed Nyquist."""
        fs = 8000
        signal, _ = self._pure_sine(1000, fs)
        result = compute_fft(signal, fs)
        assert result.freqs[-1] <= fs / 2.0

    def test_magnitude_non_negative(self):
        signal, fs = self._pure_sine(1000, 8000)
        result = compute_fft(signal, fs)
        assert np.all(result.magnitude >= 0)

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError):
            compute_fft(np.array([]), 8000)

    def test_spectral_centroid_positive(self):
        signal, fs = self._pure_sine(1000, 8000)
        result = compute_fft(signal, fs)
        assert result.spectral_centroid >= 0

    def test_band_energies_sum_le_one(self):
        signal, fs = self._pure_sine(1000, 8000)
        result = compute_fft(signal, fs)
        # low + high energy fractions must be <= 1
        assert result.low_freq_energy + result.high_freq_energy <= 1.0 + 1e-9

    def test_odd_length_includes_highest_positive_bin(self):
        signal = np.sin(2 * np.pi * 2 * np.arange(5) / 5)
        result = compute_fft(signal, 5)
        assert result.freqs[-1] == pytest.approx(2.0)

    def test_even_length_includes_nyquist_bin(self):
        signal = (-1.0) ** np.arange(8)
        result = compute_fft(signal, 8)
        assert result.freqs[-1] == pytest.approx(4.0)
        assert result.magnitude[-1] == pytest.approx(1.0)

    def test_dc_and_tone_power_are_weighted_correctly(self):
        fs = 8000
        t = np.arange(fs) / fs
        result = compute_fft(2.0 + np.sin(2 * np.pi * 1000 * t), fs)
        assert result.spectral_centroid == pytest.approx(1000.0 / 9.0, abs=1.0)
