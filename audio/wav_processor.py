"""
audio/wav_processor.py
======================
WAV file ingestion and processing for SigNexis.

Handles:
    - mono and stereo WAV files
    - different sampling rates
    - invalid / empty / too-short files
    - NaN / Inf in audio data
    - very long files (truncation with user warning)

Uses soundfile as primary reader; falls back to scipy.io.wavfile.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

MAX_DURATION_SEC = 30.0     # truncate files longer than this
MIN_DURATION_SEC = 0.1      # reject files shorter than this


@dataclass
class WAVInfo:
    """Metadata and audio data for an uploaded WAV file."""

    filename: str
    sampling_rate: int
    duration: float
    num_samples: int
    num_channels_original: int
    signal: np.ndarray           # mono float64, range normalised to [-1, 1]
    was_truncated: bool
    warnings: list[str]


def load_wav(
    file_obj,
    max_duration: float = MAX_DURATION_SEC,
) -> WAVInfo:
    """Load a WAV file from a file-like object or path.

    Parameters
    ----------
    file_obj    : file-like object (e.g. Streamlit UploadedFile) or str/Path
    max_duration: maximum accepted duration in seconds

    Returns
    -------
    WAVInfo

    Raises
    ------
    ValueError  – for invalid, empty, or too-short audio
    RuntimeError – if neither soundfile nor scipy can read the file
    """
    warnings_list: list[str] = []
    filename = getattr(file_obj, "name", str(file_obj))

    # ── Try soundfile first ───────────────────────────────────────────────────
    audio: Optional[np.ndarray] = None
    sr: Optional[int] = None

    try:
        import soundfile as sf
        audio, sr = sf.read(file_obj, dtype="float64", always_2d=True)
    except Exception as e_sf:
        # Fallback: scipy.io.wavfile (integers only)
        try:
            import io
            from scipy.io import wavfile

            # Rewind if possible
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)

            sr_int, data = wavfile.read(file_obj)
            sr = int(sr_int)
            if data.ndim == 1:
                data = data[:, np.newaxis]
            # Normalise integer PCM to float64 [-1, 1]
            if np.issubdtype(data.dtype, np.integer):
                max_val = np.iinfo(data.dtype).max
                audio = data.astype(np.float64) / float(max_val)
            else:
                audio = data.astype(np.float64)
        except Exception as e_scipy:
            raise RuntimeError(
                f"Could not read WAV file '{filename}'.\n"
                f"  soundfile error: {e_sf}\n"
                f"  scipy error:     {e_scipy}"
            )

    if audio is None or sr is None:
        raise ValueError(f"Failed to load audio from '{filename}'.")

    num_channels_original = audio.shape[1] if audio.ndim == 2 else 1

    # ── Stereo → mono ─────────────────────────────────────────────────────────
    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = np.mean(audio, axis=1)
        warnings_list.append(
            f"Stereo audio converted to mono (averaged {num_channels_original} channels)."
        )
    else:
        audio = audio.flatten()

    # ── Validate ──────────────────────────────────────────────────────────────
    if len(audio) == 0:
        raise ValueError("Audio file is empty (zero samples).")

    duration = len(audio) / float(sr)

    if duration < MIN_DURATION_SEC:
        raise ValueError(
            f"Audio too short ({duration*1000:.1f} ms). "
            f"Minimum required: {MIN_DURATION_SEC*1000:.0f} ms."
        )

    # ── Truncate long files ───────────────────────────────────────────────────
    was_truncated = False
    if duration > max_duration:
        max_samples = int(max_duration * sr)
        audio = audio[:max_samples]
        was_truncated = True
        warnings_list.append(
            f"Audio truncated to {max_duration:.0f} s "
            f"(original: {duration:.1f} s)."
        )
        duration = max_duration

    # ── Handle NaN / Inf ──────────────────────────────────────────────────────
    n_bad = int(np.sum(~np.isfinite(audio)))
    if n_bad > 0:
        audio = np.where(np.isfinite(audio), audio, 0.0)
        warnings_list.append(
            f"{n_bad} non-finite sample(s) (NaN/Inf) replaced with 0."
        )

    # ── Normalise amplitude ───────────────────────────────────────────────────
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return WAVInfo(
        filename=filename,
        sampling_rate=int(sr),
        duration=float(len(audio) / sr),
        num_samples=len(audio),
        num_channels_original=num_channels_original,
        signal=audio,
        was_truncated=was_truncated,
        warnings=warnings_list,
    )
