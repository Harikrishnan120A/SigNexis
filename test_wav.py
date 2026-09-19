"""WAV processor test script."""
import sys
import numpy as np
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from audio.wav_processor import load_wav
import soundfile as sf

# Create a synthetic WAV in memory and test loading it
fs = 16000
duration = 2.0
t = np.arange(0, duration, 1.0 / fs)
audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz tone

buf = io.BytesIO()
sf.write(buf, audio, fs, format="WAV")
buf.seek(0)
buf.name = "test.wav"

info = load_wav(buf)
print(f"WAV test: fs={info.sampling_rate}, duration={info.duration:.2f}s, samples={info.num_samples}")
print(f"Warnings: {info.warnings}")

from ml.predict import predict_signal
pred = predict_signal(info.signal, info.sampling_rate)
print(f"ML on WAV: class={pred.predicted_class}, confidence={pred.confidence*100:.1f}%")
print("WAV processor test PASSED")

# Test stereo conversion
stereo = np.stack([audio, audio * 0.5], axis=1)
buf2 = io.BytesIO()
sf.write(buf2, stereo, fs, format="WAV")
buf2.seek(0)
buf2.name = "stereo_test.wav"
info2 = load_wav(buf2)
print(f"Stereo->Mono: channels_original={info2.num_channels_original}, signal shape={info2.signal.shape}")
assert info2.signal.ndim == 1, "Should be 1D after stereo->mono conversion"
print("Stereo->Mono test PASSED")

print("\nAll WAV tests PASSED!")
