def validate_sampling_rate(fs):
    if fs<=0: return False,f"Sampling rate must be positive (got {fs})."
    if fs<100: return False,f"Sampling rate {fs} Hz is very low."
    if fs>192000: return False,f"Sampling rate {fs} Hz exceeds maximum supported."
    return True,"OK"
def validate_signal_frequency(f_sig,fs):
    if f_sig<=0: return False,f"Signal frequency must be positive (got {f_sig})."
    return True,(f"Aliasing will occur (f_signal={f_sig:.1f} Hz)." if f_sig>=fs/2 else "OK")
def validate_cutoff_frequency(cutoff,fs,label="Cutoff"):
    return (False,f"{label} must be between 0 and Nyquist.") if cutoff<=0 or cutoff>=fs/2 else (True,"OK")
def validate_duration(duration): return (False,"Duration must be positive.") if duration<=0 else ((False,"Duration exceeds 60 s limit.") if duration>60 else (True,"OK"))
