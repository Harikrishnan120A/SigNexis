from dataclasses import dataclass
import numpy as np
MAX_DURATION_SEC=30.0; MIN_DURATION_SEC=.1
@dataclass
class WAVInfo:
    filename: str; sampling_rate: int; duration: float; num_samples: int; num_channels_original: int; signal: np.ndarray; was_truncated: bool; warnings: list[str]
def load_wav(file_obj, max_duration=MAX_DURATION_SEC):
    import soundfile as sf
    filename=getattr(file_obj,"name",str(file_obj)); audio,sr=sf.read(file_obj,dtype="float64",always_2d=True); channels=audio.shape[1]
    if channels>1: audio=audio.mean(axis=1); warnings=[f"Stereo audio converted to mono ({channels} channels)."]
    else: audio=audio.flatten(); warnings=[]
    if len(audio)==0: raise ValueError("Audio file is empty.")
    duration=len(audio)/sr
    if duration<MIN_DURATION_SEC: raise ValueError("Audio is too short.")
    truncated=duration>max_duration
    if truncated: audio=audio[:int(max_duration*sr)]; warnings.append(f"Audio truncated to {max_duration:.0f} s.")
    audio=np.where(np.isfinite(audio),audio,0.0); peak=np.max(np.abs(audio))
    if peak: audio=audio/peak
    return WAVInfo(filename,int(sr),len(audio)/sr,len(audio),channels,audio,truncated,warnings)
