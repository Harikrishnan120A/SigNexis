"""
app.py
======
SigNexis: Intelligent Digital Signal Analysis, Noise Classification & Adaptive Filtering System
Undergraduate ECE Capstone / Mini-Project Interactive Demonstration Application.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import io

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Project modules ────────────────────────────────────────────────────────────
from dsp.signal_generator import generate_signal, SIGNAL_CLASSES
from dsp.sampling import analyse_sampling
from dsp.fft_analysis import compute_fft
from dsp.features import extract_features, FEATURE_NAMES
from dsp.filters import (
    apply_recommended_filter,
    FILTER_RECOMMENDATIONS,
    low_pass_filter,
    high_pass_filter,
    band_pass_filter,
    notch_filter,
)
from dsp.quality import compute_quality, snr_improvement, estimate_snr_rms_ratio
from ml.predict import predict_signal, MODEL_PATH
from audio.wav_processor import load_wav

MODELS_DIR = ROOT / "models"
EVAL_PATH = MODELS_DIR / "evaluation_results.json"

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SigNexis | ECE DSP & ML System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clean Academic CSS Design ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header Container */
.academic-header {
    background: #0f172a;
    border-bottom: 2px solid #1e293b;
    padding: 1.25rem 1.5rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.academic-title {
    color: #f8fafc;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.academic-subtitle {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.25rem;
    font-weight: 400;
}
.academic-badges {
    display: flex;
    gap: 0.5rem;
}
.tech-tag {
    background: #1e293b;
    color: #38bdf8;
    border: 1px solid #334155;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}

/* Category Distinction Headers */
.cat-tag {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    margin-bottom: 0.4rem;
    display: inline-block;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
}
.cat-dsp  { background: #0c4a6e; color: #7dd3fc; border: 1px solid #0284c7; }
.cat-nyq  { background: #451a03; color: #fdba74; border: 1px solid #ea580c; }
.cat-ml   { background: #3b0764; color: #d8b4fe; border: 1px solid #9333ea; }
.cat-filt { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }

/* Metric Cards */
.eng-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
}
.eng-card-title {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}
.eng-card-val {
    color: #f8fafc;
    font-size: 1.35rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.eng-card-unit {
    color: #64748b;
    font-size: 0.8rem;
    font-weight: 500;
    margin-left: 0.2rem;
}

/* Status Banners */
.status-banner-ok {
    background: #022c22;
    border-left: 4px solid #10b981;
    color: #a7f3d0;
    padding: 0.75rem 1rem;
    border-radius: 0 6px 6px 0;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}
.status-banner-warn {
    background: #450a0a;
    border-left: 4px solid #ef4444;
    color: #fecaca;
    padding: 0.75rem 1rem;
    border-radius: 0 6px 6px 0;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}
.status-banner-info {
    background: #082f49;
    border-left: 4px solid #0284c7;
    color: #bae6fd;
    padding: 0.75rem 1rem;
    border-radius: 0 6px 6px 0;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}

/* Section Header */
.section-header {
    color: #38bdf8;
    font-size: 1.05rem;
    font-weight: 600;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.35rem;
    margin: 1rem 0 0.75rem 0;
    letter-spacing: -0.01em;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly Academic Engineering Theme ──────────────────────────────────────────
PLOTLY_THEME = dict(
    plot_bgcolor="#0f172a",
    paper_bgcolor="#0f172a",
    font=dict(color="#cbd5e1", family="Inter", size=11),
    xaxis=dict(
        gridcolor="#1e293b",
        zerolinecolor="#334155",
        tickcolor="#475569",
        linecolor="#334155",
        mirror=True,
    ),
    yaxis=dict(
        gridcolor="#1e293b",
        zerolinecolor="#334155",
        tickcolor="#475569",
        linecolor="#334155",
        mirror=True,
    ),
)

SIG_COLORS = {
    "clean": "#38bdf8",     # Sky blue
    "corrupted": "#f97316", # Orange
    "filtered": "#10b981",  # Emerald
    "ref_analog": "#a78bfa",# Violet
    "samples": "#fbbf24",   # Amber
    "alias": "#f43f5e",     # Rose
}

def plot_waveform(t: np.ndarray, y: np.ndarray, name: str, color: str, title: str,
                  y2: np.ndarray | None = None, name2: str | None = None,
                  color2: str | None = None) -> go.Figure:
    """Generate high-contrast, fully labeled engineering time-domain plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=y, mode="lines", name=name,
        line=dict(color=color, width=1.5)
    ))
    if y2 is not None and name2 is not None:
        fig.add_trace(go.Scatter(
            x=t, y=y2, mode="lines", name=name2,
            line=dict(color=color2 or SIG_COLORS["filtered"], width=1.5, dash="solid")
        ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=title, font=dict(color="#f8fafc", size=13)),
        xaxis_title="Time t [s]",
        yaxis_title="Amplitude x(t) [V]",
        height=270,
        margin=dict(l=55, r=20, t=35, b=40),
        showlegend=True,
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155", font=dict(size=10)),
    )
    return fig

def plot_spectrum(freqs: np.ndarray, mag: np.ndarray, name: str, color: str,
                  title: str, dominant_freq: float | None = None,
                  nyquist_freq: float | None = None) -> go.Figure:
    """Generate high-contrast single-sided frequency magnitude spectrum."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=mag, mode="lines", name=name,
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.12)" if color == SIG_COLORS["clean"] else "rgba(249, 115, 22, 0.12)"
    ))
    if dominant_freq is not None and dominant_freq > 0:
        fig.add_vline(
            x=dominant_freq,
            line=dict(color="#fbbf24", width=1.5, dash="dash"),
            annotation_text=f"f_dom = {dominant_freq:.1f} Hz",
            annotation_position="top right",
            annotation_font=dict(color="#fbbf24", size=10),
        )
    if nyquist_freq is not None and nyquist_freq > 0:
        fig.add_vline(
            x=nyquist_freq,
            line=dict(color="#ef4444", width=1.5, dash="dot"),
            annotation_text=f"Nyquist Limit f_N = {nyquist_freq:.0f} Hz",
            annotation_position="top left",
            annotation_font=dict(color="#ef4444", size=10),
        )
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(text=title, font=dict(color="#f8fafc", size=13)),
        xaxis_title="Frequency f [Hz]",
        yaxis_title="Magnitude |X(f)| [V]",
        height=270,
        margin=dict(l=55, r=20, t=35, b=40),
        showlegend=True,
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155", font=dict(size=10)),
    )
    return fig

# ── Cached Model & Resource Loaders ──────────────────────────────────────────
@st.cache_resource
def get_ml_model():
    """Load serialized RandomForestClassifier (cached across reruns)."""
    import joblib
    if not MODEL_PATH.exists():
        return None, f"Model file missing at: {MODEL_PATH}"
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as exc:
        return None, str(exc)

@st.cache_data
def get_evaluation_results():
    """Load verified evaluation artifact (cached across reruns)."""
    if EVAL_PATH.exists():
        try:
            with open(EVAL_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None

# ── Session State Initialization ─────────────────────────────────────────────
def initialize_session_state():
    defaults = {
        "signal_type": "sine",
        "frequency": 1000.0,
        "amplitude": 1.0,
        "duration": 1.0,
        "sampling_rate": 8000.0,
        "noise_type": "CLEAN",
        "noise_amplitude": 0.3,
        "t": None,
        "clean": None,
        "noisy": None,
        "filtered": None,
        "prediction": None,
        "filter_info": None,
        "quality_before": None,
        "quality_after": None,
        "nav_page": "Dashboard",
        "audio_loaded": False,
        "audio_info": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

initialize_session_state()

# ── Core Action Callbacks ─────────────────────────────────────────────────────
def execute_signal_generation():
    """Generates synthetic signal based on validated parameters."""
    rng = np.random.default_rng(42)
    t, clean, noisy = generate_signal(
        signal_type=st.session_state.signal_type,
        frequency=float(st.session_state.frequency),
        amplitude=float(st.session_state.amplitude),
        duration=float(st.session_state.duration),
        sampling_rate=float(st.session_state.sampling_rate),
        noise_type=st.session_state.noise_type,
        noise_amplitude=float(st.session_state.noise_amplitude),
        rng=rng,
    )
    st.session_state.t = t
    st.session_state.clean = clean
    st.session_state.noisy = noisy
    st.session_state.filtered = None
    st.session_state.prediction = None
    st.session_state.filter_info = None
    st.session_state.quality_before = None
    st.session_state.quality_after = None
    st.session_state.audio_loaded = False

def execute_pipeline_analysis():
    """Runs FFT, feature extraction, ML inference, and adaptive filtering."""
    if st.session_state.noisy is None:
        execute_signal_generation()

    noisy = st.session_state.noisy
    fs = float(st.session_state.sampling_rate)
    f0 = float(st.session_state.frequency)

    # 1. ML Noise Classification
    try:
        pred = predict_signal(noisy, fs, MODEL_PATH)
        st.session_state.prediction = pred
    except Exception:
        st.session_state.prediction = None
        return

    # 2. Adaptive Filtering
    filtered, finfo = apply_recommended_filter(noisy, fs, pred.predicted_class, f0)
    st.session_state.filtered = filtered
    st.session_state.filter_info = finfo

    # 3. Quality Metrics
    clean = st.session_state.clean
    st.session_state.quality_before = compute_quality(noisy, fs, clean)
    st.session_state.quality_after = compute_quality(filtered, fs, clean)

# If no signal generated yet, initialize automatically on startup
if st.session_state.noisy is None:
    execute_signal_generation()
    execute_pipeline_analysis()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROL PANEL
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📡 SigNexis Control Panel")
    st.caption("ECE Capstone / Mini-Project Demonstration")
    st.markdown("---")

    # Navigation menu
    nav_options = [
        "Dashboard",
        "Signal Generator",
        "Sampling & Aliasing",
        "FFT Analysis",
        "ML Classification",
        "Adaptive Filtering",
        "Before vs After",
        "About",
    ]
    selected_page = st.radio(
        "Module Navigation",
        nav_options,
        index=nav_options.index(st.session_state.nav_page) if st.session_state.nav_page in nav_options else 0,
        key="nav_page",
    )

    st.markdown("---")
    st.markdown("#### ⚡ Pipeline Execution")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Generate", use_container_width=True, type="secondary"):
            execute_signal_generation()
            st.rerun()
    with col_btn2:
        if st.button("Analyze", use_container_width=True, type="primary"):
            execute_pipeline_analysis()
            st.rerun()

    st.markdown("---")
    st.markdown("#### 🔍 System Status")
    model, model_err = get_ml_model()
    if model is not None:
        st.markdown('<span style="color:#4ade80;font-weight:600;font-size:0.8rem;">● RF Model Ready (200 Trees)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#ef4444;font-weight:600;font-size:0.8rem;">● Model Missing</span>', unsafe_allow_html=True)

    nyq_limit = st.session_state.sampling_rate / 2.0
    if st.session_state.frequency < nyq_limit:
        st.markdown('<span style="color:#38bdf8;font-weight:600;font-size:0.8rem;">● Nyquist: Satisfied (f < fs/2)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#f97316;font-weight:600;font-size:0.8rem;">● Nyquist: Violated (Aliasing)</span>', unsafe_allow_html=True)

# ── Academic Header Bar ────────────────────────────────────────────────────────
st.markdown("""
<div class="academic-header">
    <div>
        <div class="academic-title">SigNexis</div>
        <div class="academic-subtitle">
            ML-Based Intelligent Digital Signal Analysis, Noise Classification &amp; Adaptive Filtering System
        </div>
    </div>
    <div class="academic-badges">
        <span class="tech-tag">DSP: Scipy Signal</span>
        <span class="tech-tag">ML: Random Forest</span>
        <span class="tech-tag">Sampling: Nyquist Verification</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if selected_page == "Dashboard":
    st.markdown("### 📊 Executive Engineering Dashboard")
    st.caption("Consolidated operational view of measured DSP parameters, Nyquist criteria, ML inference, and adaptive filtering.")

    noisy = st.session_state.noisy
    clean = st.session_state.clean
    t = st.session_state.t
    fs = st.session_state.sampling_rate
    f0 = st.session_state.frequency
    nyq = fs / 2.0
    pred = st.session_state.prediction
    finfo = st.session_state.filter_info
    qb = st.session_state.quality_before
    qa = st.session_state.quality_after

    fft_res = compute_fft(noisy, fs)

    # Row 1: System KPI Metrics (Categorized)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<span class="cat-tag cat-dsp">Measured DSP</span>', unsafe_allow_html=True)
        st.markdown(f"""<div class="eng-card">
            <div class="eng-card-title">Sampling Rate (fs)</div>
            <div class="eng-card-val">{fs:,.0f}<span class="eng-card-unit">Hz</span></div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<span class="cat-tag cat-dsp">Measured DSP</span>', unsafe_allow_html=True)
        st.markdown(f"""<div class="eng-card">
            <div class="eng-card-title">Nyquist Frequency (fN)</div>
            <div class="eng-card-val">{nyq:,.0f}<span class="eng-card-unit">Hz</span></div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown('<span class="cat-tag cat-dsp">Spectral Peak</span>', unsafe_allow_html=True)
        st.markdown(f"""<div class="eng-card">
            <div class="eng-card-title">Dominant Peak (f_dom)</div>
            <div class="eng-card-val">{fft_res.dominant_freq:,.1f}<span class="eng-card-unit">Hz</span></div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown('<span class="cat-tag cat-ml">ML Prediction</span>', unsafe_allow_html=True)
        cls_name = pred.predicted_class if pred else "—"
        st.markdown(f"""<div class="eng-card">
            <div class="eng-card-title">Noise Classification</div>
            <div class="eng-card-val" style="font-size:0.95rem;color:#38bdf8;">{cls_name}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown('<span class="cat-tag cat-filt">Filter Status</span>', unsafe_allow_html=True)
        rec_filt = finfo["filter_type"].upper() if finfo else "—"
        st.markdown(f"""<div class="eng-card">
            <div class="eng-card-title">Recommended Filter</div>
            <div class="eng-card-val" style="font-size:1.1rem;color:#10b981;">{rec_filt}</div>
        </div>""", unsafe_allow_html=True)

    # Row 2: Secondary Operational Status Row
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        if f0 < nyq:
            st.markdown("""<div class="status-banner-ok">
                <b>Sampling Status:</b> Properly Sampled (f < fs/2)<br>
                Signal is free from frequency-domain foldover.
            </div>""", unsafe_allow_html=True)
        else:
            samp_analysis = analyse_sampling(f0, fs)
            app_f = samp_analysis.apparent_freq or 0.0
            st.markdown(f"""<div class="status-banner-warn">
                <b>Sampling Status:</b> ALIASING DETECTED (f ≥ fs/2)<br>
                Apparent aliased tone: <b>{app_f:.1f} Hz</b>
            </div>""", unsafe_allow_html=True)
    with sc2:
        conf_val = f"{pred.confidence*100:.1f}%" if pred else "N/A"
        st.markdown(f"""<div class="status-banner-info">
            <b>ML Confidence:</b> {conf_val}<br>
            Multi-class probability over 6 signal states.
        </div>""", unsafe_allow_html=True)
    with sc3:
        if finfo and finfo.get("cutoff_used"):
            st.markdown(f"""<div class="status-banner-info">
                <b>Filter Cutoff:</b> {finfo['cutoff_used']} Hz<br>
                Optimized based on signal frequency and bandwidth.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="status-banner-info">
                <b>Filter Cutoff:</b> Pass-through<br>
                Clean signal: No filtering required.
            </div>""", unsafe_allow_html=True)
    with sc4:
        if qb and qa and qb.snr_db is not None and qa.snr_db is not None:
            imp = snr_improvement(qb, qa)
            imp_str = f"{imp:+.2f} dB" if imp is not None else "N/A"
            b_class = "status-banner-ok" if (imp and imp > 0) else "status-banner-info"
            st.markdown(f"""<div class="{b_class}">
                <b>SNR Improvement:</b> {imp_str}<br>
                Before: {qb.snr_db:.2f} dB &nbsp;|&nbsp; After: {qa.snr_db:.2f} dB
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="status-banner-info">
                <b>SNR Metric:</b> Reference Clean Signal<br>
                SNR is mathematically infinite (Clean signal).
            </div>""", unsafe_allow_html=True)

    # Waveform & Spectrum Visualizations
    col_w, col_s = st.columns(2)
    with col_w:
        fig_w = plot_waveform(
            t[:min(len(t), 1000)],
            noisy[:min(len(noisy), 1000)],
            "Corrupted Signal x[n]",
            SIG_COLORS["corrupted"],
            "Time-Domain Waveform (First 1,000 Samples)",
            y2=clean[:min(len(clean), 1000)] if clean is not None else None,
            name2="Reference Clean s[n]",
            color2=SIG_COLORS["clean"]
        )
        st.plotly_chart(fig_w, use_container_width=True)
    with col_s:
        fig_s = plot_spectrum(
            fft_res.freqs,
            fft_res.magnitude,
            "Normalized Spectrum |X(f)|",
            SIG_COLORS["clean"],
            "Single-Sided FFT Magnitude Spectrum",
            dominant_freq=fft_res.dominant_freq,
            nyquist_freq=nyq
        )
        st.plotly_chart(fig_s, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_page == "Signal Generator":
    st.markdown("### ⚙️ Digital Signal Synthesis & Noise Injection")
    st.caption("Generate pure analog-like synthetic signals or load uncompressed PCM WAV recordings with controlled noise corruption.")

    source_mode = st.radio("Signal Input Mode", ["Synthetic Signal Generator", "Audio WAV File Upload"], horizontal=True)

    if source_mode == "Synthetic Signal Generator":
        col_ctl, col_disp = st.columns([1, 2])

        with col_ctl:
            st.markdown('<div class="section-header">Generator Configuration</div>', unsafe_allow_html=True)

            sig_type = st.selectbox(
                "Base Waveform Topology",
                ["sine", "multi_sine"],
                format_func=lambda x: "Single Fundamental Sine" if x == "sine" else "Harmonic Multi-Sine (f, 2f, 3f)",
                index=0 if st.session_state.signal_type == "sine" else 1
            )

            freq_val = st.number_input(
                "Signal Frequency f0 [Hz]",
                min_value=10.0,
                max_value=20000.0,
                value=float(st.session_state.frequency),
                step=50.0
            )

            amp_val = st.slider("Signal Peak Amplitude A [V]", 0.1, 5.0, float(st.session_state.amplitude), 0.1)

            fs_val = st.selectbox(
                "Sampling Frequency fs [Hz]",
                [1000.0, 2000.0, 4000.0, 8000.0, 11025.0, 16000.0, 22050.0, 44100.0, 48000.0],
                index=3
            )

            dur_val = st.slider("Record Duration T [s]", 0.1, 2.0, float(st.session_state.duration), 0.1)

            st.markdown('<div class="section-header">Additive Noise / Distortion</div>', unsafe_allow_html=True)
            noise_type = st.selectbox("Noise Model", SIGNAL_CLASSES, index=SIGNAL_CLASSES.index(st.session_state.noise_type))
            noise_amp = st.slider("Noise Intensity Ratio", 0.0, 2.0, float(st.session_state.noise_amplitude), 0.05)

            # Nyquist check on generator
            nyq_calc = fs_val / 2.0
            if freq_val >= nyq_calc:
                st.warning(f"⚠️ Notice: Frequency {freq_val:.0f} Hz exceeds Nyquist limit ({nyq_calc:.0f} Hz). Aliasing will occur upon synthesis.")

            if st.button("Synthesize & Load Signal", type="primary", use_container_width=True):
                st.session_state.signal_type = sig_type
                st.session_state.frequency = freq_val
                st.session_state.amplitude = amp_val
                st.session_state.sampling_rate = fs_val
                st.session_state.duration = dur_val
                st.session_state.noise_type = noise_type
                st.session_state.noise_amplitude = noise_amp
                execute_signal_generation()
                execute_pipeline_analysis()
                st.success("Signal synthesized and pipeline refreshed.")
                st.rerun()

        with col_disp:
            st.markdown('<div class="section-header">Synthesized Waveform Verification</div>', unsafe_allow_html=True)
            t = st.session_state.t
            clean = st.session_state.clean
            noisy = st.session_state.noisy

            if noisy is not None and t is not None:
                # Truncate visual window to prevent browser lag on very high sample counts
                view_samples = min(len(t), 1200)
                fig_gen = plot_waveform(
                    t[:view_samples],
                    noisy[:view_samples],
                    f"Corrupted (+ {st.session_state.noise_type})",
                    SIG_COLORS["corrupted"],
                    f"Discrete Waveform Preview ({view_samples} of {len(t):,} samples displayed)",
                    y2=clean[:view_samples] if clean is not None else None,
                    name2="Clean Reference s[n]",
                    color2=SIG_COLORS["clean"]
                )
                st.plotly_chart(fig_gen, use_container_width=True)

                # Mathematical Specification Card
                st.markdown(f"""<div class="status-banner-info">
                    <b>Signal Model:</b> x[n] = s[n] + η[n]<br>
                    <b>Sampling Interval:</b> Ts = 1/fs = {1.0/st.session_state.sampling_rate*1e6:.2f} µs &nbsp;|&nbsp;
                    <b>Total Points:</b> N = {len(t):,} samples &nbsp;|&nbsp;
                    <b>Peak Amplitude:</b> {np.max(np.abs(noisy)):.3f} V
                </div>""", unsafe_allow_html=True)

    else:
        # Audio WAV File Upload
        st.markdown('<div class="section-header">WAV Audio File Ingestion</div>', unsafe_allow_html=True)
        uploaded_wav = st.file_uploader("Select an uncompressed WAV audio file", type=["wav"], help="Accepts mono or stereo WAV files up to 30 seconds.")

        if uploaded_wav is not None:
            try:
                wav_info = load_wav(uploaded_wav)
                for w in wav_info.warnings:
                    st.warning(f"Audio Parser Notice: {w}")

                st.session_state.t = np.arange(len(wav_info.signal)) / float(wav_info.sampling_rate)
                st.session_state.clean = None  # No ground truth clean reference for arbitrary audio
                st.session_state.noisy = wav_info.signal
                st.session_state.sampling_rate = float(wav_info.sampling_rate)
                st.session_state.frequency = float(compute_fft(wav_info.signal, wav_info.sampling_rate).dominant_freq)
                st.session_state.audio_loaded = True

                execute_pipeline_analysis()

                st.success(f"Successfully loaded '{wav_info.filename}': {wav_info.num_samples:,} samples @ {wav_info.sampling_rate} Hz.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to process WAV file: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: SAMPLING & ALIASING
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_page == "Sampling & Aliasing":
    st.markdown("### 🔬 Nyquist-Shannon Sampling Theorem & Aliasing Verification")
    st.caption("Demonstrates the mathematical boundary where continuous signals become ambiguous due to insufficient discrete sampling.")

    col_s_ctrl, col_s_view = st.columns([1, 2])

    with col_s_ctrl:
        st.markdown('<div class="section-header">Nyquist Test Parameters</div>', unsafe_allow_html=True)
        test_f = st.number_input("Signal Continuous Frequency f0 [Hz]", 10.0, 15000.0, float(st.session_state.frequency), 100.0)
        test_fs = st.number_input("Sampling Frequency fs [Hz]", 100.0, 48000.0, float(st.session_state.sampling_rate), 500.0)
        test_amp = st.slider("Signal Amplitude A [V]", 0.1, 2.0, 1.0, 0.1)
        test_dur = st.slider("Inspection Window [s]", 0.005, 0.05, 0.02, 0.002, help="Short duration ensures individual discrete samples are clearly visible.")

        s_res = analyse_sampling(test_f, test_fs, test_amp, test_dur)

        st.markdown('<div class="section-header">Theoretical Analysis</div>', unsafe_allow_html=True)
        st.markdown(f"""
        * **Continuous Frequency ($f_0$):** `{test_f:,.1f} Hz`
        * **Sampling Rate ($f_s$):** `{test_fs:,.1f} Hz`
        * **Nyquist Limit ($f_N = f_s/2$):** `{s_res.nyquist_freq:,.1f} Hz`
        * **Condition:** `f0 < fN` → **{s_res.is_properly_sampled}**
        """)

        if s_res.is_properly_sampled:
            st.markdown("""<div class="status-banner-ok">
                <b>Condition Satisfied:</b> f0 < fs / 2<br>
                Discrete samples uniquely reconstruct the original continuous waveform.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="status-banner-warn">
                <b>CRITICAL NYQUIST VIOLATION:</b> f0 ≥ fs / 2<br>
                High-frequency tone folds back across Nyquist boundary into baseband.<br>
                <b>Folded Apparent Frequency:</b> <span style="font-size:1.1rem;font-weight:700;">{s_res.apparent_freq:.1f} Hz</span>
            </div>""", unsafe_allow_html=True)

    with col_s_view:
        st.markdown('<div class="section-header">Analog Reference vs Discrete Sample Stems</div>', unsafe_allow_html=True)

        fig_alias = go.Figure()

        # 1. Analog continuous reference waveform (high-density grid)
        fig_alias.add_trace(go.Scatter(
            x=s_res.t_ref, y=s_res.signal_ref,
            mode="lines", name="Original Continuous Signal x(t)",
            line=dict(color=SIG_COLORS["ref_analog"], width=1.2, dash="dot")
        ))

        # 2. Discrete sampling stems/markers
        fig_alias.add_trace(go.Scatter(
            x=s_res.t_discrete, y=s_res.signal_discrete,
            mode="markers+lines", name="Discrete Samples x[n] = x(nTs)",
            marker=dict(color=SIG_COLORS["samples"], size=7, symbol="circle"),
            line=dict(color=SIG_COLORS["samples"], width=1)
        ))

        # 3. If aliasing occurs, plot the apparent folded waveform
        if s_res.apparent_freq is not None:
            t_recon = np.linspace(0, test_dur, 1000)
            apparent_wave = test_amp * np.sin(2 * np.pi * s_res.apparent_freq * t_recon)
            fig_alias.add_trace(go.Scatter(
                x=t_recon, y=apparent_wave,
                mode="lines", name=f"Apparent Alias ({s_res.apparent_freq:.1f} Hz)",
                line=dict(color=SIG_COLORS["alias"], width=2.2, dash="dash")
            ))

        fig_alias.update_layout(
            **PLOTLY_THEME,
            title=dict(text="Time-Domain Sampling Reconstruction & Aliasing Demonstration", font=dict(color="#f8fafc", size=13)),
            xaxis_title="Time t [s]",
            yaxis_title="Amplitude [V]",
            height=380,
            margin=dict(l=55, r=20, t=35, b=40),
            showlegend=True,
            legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155", font=dict(size=10)),
        )
        st.plotly_chart(fig_alias, use_container_width=True)

        st.markdown("""<div class="status-banner-info">
            <b>Mathematical Foundation:</b> According to the Nyquist-Shannon sampling theorem, an analog bandlimited signal
            can be perfectly reconstructed from its discrete samples if and only if the sampling rate strictly exceeds twice the highest
            frequency component ($f_s > 2 f_{\max}$). When sampled below this threshold, frequencies reflect according to:
            $$f_{\\text{alias}} = \\left| f_0 - \\operatorname{round}\\left(\\frac{f_0}{f_s}\\right) f_s \\right|$$
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: FFT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_page == "FFT Analysis":
    st.markdown("### 📈 Fast Fourier Transform (FFT) & Spectral Decomposition")
    st.caption("Decomposition of time-domain signals into discrete frequency components with verified single-sided amplitude scaling.")

    noisy = st.session_state.noisy
    clean = st.session_state.clean
    t = st.session_state.t
    fs = st.session_state.sampling_rate

    if noisy is None:
        st.warning("Please generate a signal first.")
        st.stop()

    fft_noisy = compute_fft(noisy, fs)
    fft_clean = compute_fft(clean, fs) if clean is not None else None

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        fig_sn = plot_spectrum(
            fft_noisy.freqs, fft_noisy.magnitude,
            "Corrupted Signal Spectrum",
            SIG_COLORS["corrupted"],
            "Corrupted Signal Spectrum |X_noisy(f)|",
            dominant_freq=fft_noisy.dominant_freq,
            nyquist_freq=fs/2.0
        )
        st.plotly_chart(fig_sn, use_container_width=True)
    with col_f2:
        if fft_clean is not None:
            fig_sc = plot_spectrum(
                fft_clean.freqs, fft_clean.magnitude,
                "Clean Reference Spectrum",
                SIG_COLORS["clean"],
                "Clean Reference Spectrum |X_clean(f)|",
                dominant_freq=fft_clean.dominant_freq,
                nyquist_freq=fs/2.0
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Uploaded audio mode: No synthetic noiseless reference spectrum available.")

    # Spectral Metrics Table
    st.markdown('<div class="section-header">Quantitative Spectral Metrics</div>', unsafe_allow_html=True)
    metrics_list = [
        {"Parameter": "Dominant Peak Frequency", "Corrupted Signal": f"{fft_noisy.dominant_freq:.2f} Hz", "Clean Reference": f"{fft_clean.dominant_freq:.2f} Hz" if fft_clean else "N/A", "Description": "Frequency bin containing maximum spectral energy"},
        {"Parameter": "Spectral Centroid", "Corrupted Signal": f"{fft_noisy.spectral_centroid:.2f} Hz", "Clean Reference": f"{fft_clean.spectral_centroid:.2f} Hz" if fft_clean else "N/A", "Description": "Amplitude-weighted mean frequency of the spectrum"},
        {"Parameter": "Spectral Bandwidth", "Corrupted Signal": f"{fft_noisy.spectral_bandwidth:.2f} Hz", "Clean Reference": f"{fft_clean.spectral_bandwidth:.2f} Hz" if fft_clean else "N/A", "Description": "Spread of spectral power around the spectral centroid"},
        {"Parameter": "Low-Frequency Energy Fraction", "Corrupted Signal": f"{fft_noisy.low_freq_energy*100:.2f} %", "Clean Reference": f"{fft_clean.low_freq_energy*100:.2f} %" if fft_clean else "N/A", "Description": "Fraction of spectral energy below 250 Hz (< 0.15 fs)"},
        {"Parameter": "High-Frequency Energy Fraction", "Corrupted Signal": f"{fft_noisy.high_freq_energy*100:.2f} %", "Clean Reference": f"{fft_clean.high_freq_energy*100:.2f} %" if fft_clean else "N/A", "Description": "Fraction of spectral energy above 2000 Hz (> 0.35 fs)"},
    ]
    st.dataframe(pd.DataFrame(metrics_list), use_container_width=True, hide_index=True)

    st.markdown(f"""<div class="status-banner-info">
        <b>Discrete Fourier Transform Formula:</b>
        $$X[k] = \\sum_{{n=0}}^{{N-1}} x[n] \\cdot e^{{-j \\frac{{2\\pi}}{{N}} k n}}, \\quad k = 0, 1, \\dots, N-1$$
        <b>Single-Sided Normalization:</b> $A[k] = \\frac{{2}}{{N}} |X[k]|$ for $k > 0$, and $A[0] = \\frac{{1}}{{N}} |X[0]|$ at DC.<br>
        <b>FFT Parameters:</b> Record length $N = {len(noisy):,}$ points &nbsp;|&nbsp; Frequency resolution $\\Delta f = f_s / N = {fs/len(noisy):.2f}\\text{ Hz/bin}$.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: ML CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_page == "ML Classification":
    st.markdown("### 🤖 Machine Learning Signal & Noise Classification")
    st.caption("Random Forest multiclass classification trained on 13 time- and frequency-domain DSP features.")

    pred = st.session_state.prediction
    eval_data = get_evaluation_results()

    if pred is None:
        st.warning("Execute pipeline analysis first.")
        st.stop()

    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        st.markdown('<div class="section-header">Classification Inference</div>', unsafe_allow_html=True)

        st.markdown(f"""<div class="eng-card">
            <div class="eng-card-title">Inferred Signal State</div>
            <div class="eng-card-val" style="color:#38bdf8;font-size:1.5rem;">{pred.predicted_class}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="eng-card">
            <div class="eng-card-title">Prediction Confidence</div>
            <div class="eng-card-val" style="color:#10b981;">{pred.confidence*100:.1f}<span class="eng-card-unit">%</span></div>
        </div>""", unsafe_allow_html=True)

        # Softmax Probability Distribution
        prob_items = sorted(pred.class_probabilities.items(), key=lambda x: -x[1])
        prob_df = pd.DataFrame(
            [{"Noise Class": k, "Posterior Probability": f"{v*100:.2f}%"} for k, v in prob_items]
        )
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    with col_p2:
        st.markdown('<div class="section-header">Class Probability Distribution</div>', unsafe_allow_html=True)
        classes = list(pred.class_probabilities.keys())
        probs = [pred.class_probabilities[c] * 100 for c in classes]
        colors = ["#38bdf8" if c == pred.predicted_class else "#334155" for c in classes]

        fig_pbar = go.Figure(go.Bar(
            x=classes, y=probs, marker_color=colors,
            text=[f"{p:.1f}%" for p in probs], textposition="outside"
        ))
        fig_pbar.update_layout(
            **PLOTLY_THEME,
            title=dict(text="Random Forest Class Posterior Distribution", font=dict(color="#f8fafc", size=13)),
            xaxis_title="Noise / Signal Class",
            yaxis_title="Probability [%]",
            height=280,
            margin=dict(l=50, r=20, t=35, b=60),
            yaxis_range=[0, 115]
        )
        st.plotly_chart(fig_pbar, use_container_width=True)

    # 13 Canonical Features
    st.markdown('<div class="section-header">Extracted 13-Dimensional DSP Feature Vector</div>', unsafe_allow_html=True)
    f_rows = []
    for k, v in pred.features_used.items():
        f_rows.append({"Feature Name": k, "Extracted Value": f"{v:.6f}", "Domain": "Time Domain" if k in ['mean','std','rms','max_amplitude','min_amplitude','peak_to_peak','zero_crossing_rate','signal_power'] else "Frequency Domain"})
    st.dataframe(pd.DataFrame(f_rows), use_container_width=True, hide_index=True)

    # Production Model Evaluation
    if eval_data:
        st.markdown('<div class="section-header">Offline Model Validation Metrics (Test Set, N=480)</div>', unsafe_allow_html=True)
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Empirical Accuracy", f"{eval_data['accuracy']*100:.2f}%")
        ec2.metric("Training Samples", f"{eval_data.get('n_train', 1920):,}")
        ec3.metric("Test Samples", f"{eval_data.get('n_test', 480):,}")

        # Confusion matrix heatmap
        cm = eval_data.get("confusion_matrix")
        classes_cm = eval_data.get("classes", [])
        if cm and classes_cm:
            fig_cm = px.imshow(
                cm,
                x=classes_cm, y=classes_cm,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Class", y="True Class", color="Count"),
                title="Confusion Matrix (480 Independent Test Samples)",
                text_auto=True,
            )
            fig_cm.update_layout(**PLOTLY_THEME, height=360, margin=dict(l=80, r=20, t=40, b=80))
            st.plotly_chart(fig_cm, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: ADAPTIVE FILTERING
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_page == "Adaptive Filtering":
    st.markdown("### 🎛️ Intelligent Adaptive Digital Filtering")
    st.caption("Automatic filter selection driven by machine learning classification, accompanied by a manual digital filter design studio.")

    noisy = st.session_state.noisy
    fs = st.session_state.sampling_rate
    f0 = st.session_state.frequency
    pred = st.session_state.prediction
    finfo = st.session_state.filter_info
    filtered = st.session_state.filtered

    if noisy is None:
        st.warning("Please generate a signal first.")
        st.stop()

    col_rec, col_man = st.columns([1, 1])

    with col_rec:
        st.markdown('<div class="section-header">Automated ML-Driven Recommendation</div>', unsafe_allow_html=True)
        if pred and finfo:
            st.markdown(f"""<div class="status-banner-info">
                <b>Detected Noise State:</b> {pred.predicted_class} (Confidence: {pred.confidence*100:.1f}%)<br>
                <b>Recommended Topology:</b> <span style="font-weight:700;color:#38bdf8;">{finfo['filter_type'].upper()}</span><br>
                <b>Theoretical Justification:</b> {finfo['reason']}<br>
                <b>Applied Cutoff:</b> {finfo.get('cutoff_used', 'N/A')} Hz
            </div>""", unsafe_allow_html=True)

            if st.button("Apply Recommended Filter", use_container_width=True, type="primary"):
                filtered, finfo = apply_recommended_filter(noisy, fs, pred.predicted_class, f0)
                st.session_state.filtered = filtered
                st.session_state.filter_info = finfo
                st.session_state.quality_after = compute_quality(filtered, fs, st.session_state.clean)
                st.success("Recommended filter executed.")
                st.rerun()

    with col_man:
        st.markdown('<div class="section-header">Manual Filter Studio (Examiner Override)</div>', unsafe_allow_html=True)
        nyq = fs / 2.0
        filt_choice = st.selectbox("Filter Topology", ["low_pass", "high_pass", "band_pass", "notch"])

        try:
            if filt_choice == "low_pass":
                co = st.slider("LPF Cutoff Frequency [Hz]", 10.0, nyq * 0.95, min(2000.0, nyq * 0.5))
                ord_v = st.slider("Filter Order", 2, 10, 4)
                if st.button("Execute Manual Low-Pass Filter", use_container_width=True):
                    m_filt = low_pass_filter(noisy, co, fs, ord_v)
                    st.session_state.filtered = m_filt
                    st.session_state.quality_after = compute_quality(m_filt, fs, st.session_state.clean)
                    st.success(f"Butterworth LPF applied (fc = {co:.1f} Hz, order = {ord_v}).")
                    st.rerun()

            elif filt_choice == "high_pass":
                co = st.slider("HPF Cutoff Frequency [Hz]", 10.0, nyq * 0.95, min(200.0, nyq * 0.2))
                ord_v = st.slider("Filter Order", 2, 10, 4)
                if st.button("Execute Manual High-Pass Filter", use_container_width=True):
                    m_filt = high_pass_filter(noisy, co, fs, ord_v)
                    st.session_state.filtered = m_filt
                    st.session_state.quality_after = compute_quality(m_filt, fs, st.session_state.clean)
                    st.success(f"Butterworth HPF applied (fc = {co:.1f} Hz, order = {ord_v}).")
                    st.rerun()

            elif filt_choice == "band_pass":
                c_low = st.slider("Lower Cutoff [Hz]", 10.0, nyq * 0.8, min(200.0, nyq * 0.2))
                c_high = st.slider("Upper Cutoff [Hz]", c_low + 10.0, nyq * 0.95, min(c_low + 1000.0, nyq * 0.7))
                ord_v = st.slider("Filter Order", 2, 8, 4)
                if st.button("Execute Manual Band-Pass Filter", use_container_width=True):
                    m_filt = band_pass_filter(noisy, c_low, c_high, fs, ord_v)
                    st.session_state.filtered = m_filt
                    st.session_state.quality_after = compute_quality(m_filt, fs, st.session_state.clean)
                    st.success(f"Butterworth BPF applied ({c_low:.1f} - {c_high:.1f} Hz).")
                    st.rerun()

            elif filt_choice == "notch":
                notch_f = st.slider("Notch Target Frequency [Hz]", 1.0, min(500.0, nyq * 0.95), 50.0)
                q_fac = st.slider("Quality Factor Q", 5.0, 60.0, 30.0)
                if st.button("Execute Manual Notch Filter", use_container_width=True):
                    m_filt = notch_filter(noisy, notch_f, fs, q_fac)
                    st.session_state.filtered = m_filt
                    st.session_state.quality_after = compute_quality(m_filt, fs, st.session_state.clean)
                    st.success(f"IIR Notch filter applied (f0 = {notch_f:.1f} Hz, Q = {q_fac:.1f}).")
                    st.rerun()
        except ValueError as err:
            st.error(f"Parameter validation error: {err}")

    # Visual Filtered Waveform Comparison
    if filtered is not None:
        t = st.session_state.t
        view_pts = min(len(t), 1000)
        fig_filt_comp = plot_waveform(
            t[:view_pts],
            noisy[:view_pts],
            "Corrupted Input x[n]",
            SIG_COLORS["corrupted"],
            "Digital Filter Effect: Raw Input vs Filtered Output",
            y2=filtered[:view_pts],
            name2="Filtered Output y[n]",
            color2=SIG_COLORS["filtered"]
        )
        st.plotly_chart(fig_filt_comp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: BEFORE VS AFTER
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_page == "Before vs After":
    st.markdown("### ⚖️ Quantitative Before vs After Performance Evaluation")
    st.caption("Side-by-side comparative analysis of time-domain attenuation, spectral purification, and Signal-to-Noise Ratio (SNR) enhancement.")

    noisy = st.session_state.noisy
    filtered = st.session_state.filtered
    clean = st.session_state.clean
    t = st.session_state.t
    fs = st.session_state.sampling_rate
    qb = st.session_state.quality_before
    qa = st.session_state.quality_after

    if noisy is None or filtered is None:
        st.warning("Please synthesize a signal and apply filtering first.")
        st.stop()

    fft_before = compute_fft(noisy, fs)
    fft_after = compute_fft(filtered, fs)

    # Side-by-side Waveforms
    col_w1, col_w2 = st.columns(2)
    view_pts = min(len(t), 1000)
    with col_w1:
        fig_w_bef = plot_waveform(
            t[:view_pts], noisy[:view_pts],
            "Corrupted Signal x[n]", SIG_COLORS["corrupted"],
            "Time Domain: Before Filtering"
        )
        st.plotly_chart(fig_w_bef, use_container_width=True)
    with col_w2:
        fig_w_aft = plot_waveform(
            t[:view_pts], filtered[:view_pts],
            "Filtered Signal y[n]", SIG_COLORS["filtered"],
            "Time Domain: After Filtering"
        )
        st.plotly_chart(fig_w_aft, use_container_width=True)

    # Side-by-side Spectra
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_s_bef = plot_spectrum(
            fft_before.freqs, fft_before.magnitude,
            "Corrupted Spectrum", SIG_COLORS["corrupted"],
            "Frequency Domain: Before Filtering",
            dominant_freq=fft_before.dominant_freq
        )
        st.plotly_chart(fig_s_bef, use_container_width=True)
    with col_s2:
        fig_s_aft = plot_spectrum(
            fft_after.freqs, fft_after.magnitude,
            "Purified Spectrum", SIG_COLORS["filtered"],
            "Frequency Domain: After Filtering",
            dominant_freq=fft_after.dominant_freq
        )
        st.plotly_chart(fig_s_aft, use_container_width=True)

    # Comparative Quantitative Table
    st.markdown('<div class="section-header">Measured Performance & SNR Enhancement</div>', unsafe_allow_html=True)

    if qb and qa:
        q_rows = [
            {"Metric": "RMS Voltage (Vrms)", "Before Filtering": f"{qb.rms:.4f} V", "After Filtering": f"{qa.rms:.4f} V", "Change / Delta": f"{(qa.rms - qb.rms):+.4f} V"},
            {"Metric": "Peak Amplitude (Vpeak)", "Before Filtering": f"{qb.peak_amplitude:.4f} V", "After Filtering": f"{qa.peak_amplitude:.4f} V", "Change / Delta": f"{(qa.peak_amplitude - qb.peak_amplitude):+.4f} V"},
            {"Metric": "Dominant Frequency", "Before Filtering": f"{qb.dominant_freq:.1f} Hz", "After Filtering": f"{qa.dominant_freq:.1f} Hz", "Change / Delta": f"{(qa.dominant_freq - qb.dominant_freq):+.1f} Hz"},
            {"Metric": "Signal Power (W)", "Before Filtering": f"{qb.signal_power:.6f} W", "After Filtering": f"{qa.signal_power:.6f} W", "Change / Delta": f"{(qa.signal_power - qb.signal_power):+.6f} W"},
        ]

        if qb.snr_db is not None and qa.snr_db is not None:
            imp_db = snr_improvement(qb, qa)
            imp_str = f"{imp_db:+.2f} dB" if imp_db is not None else "N/A"
            q_rows.append({"Metric": "True SNR (Clean Reference Ground Truth)", "Before Filtering": f"{qb.snr_db:.2f} dB", "After Filtering": f"{qa.snr_db:.2f} dB", "Change / Delta": imp_str})

        st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)

        if qb.snr_db is not None and qa.snr_db is not None:
            imp = snr_improvement(qb, qa)
            if imp and imp > 0:
                st.markdown(f"""<div class="status-banner-ok">
                    <b>Verified Signal Quality Enhancement:</b> SNR increased by <b>+{imp:.2f} dB</b>.<br>
                    $$\\Delta\\text{{SNR}} = \\text{{SNR}}_{{\\text{{after}}}} - \\text{{SNR}}_{{\\text{{before}}}} = {qa.snr_db:.2f} - ({qb.snr_db:.2f}) = {imp:+.2f}\\text{{ dB}}$$
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="status-banner-info">
                    <b>Signal Quality Note:</b> SNR change: {imp:+.2f} dB (Clean signal preserved).
                </div>""", unsafe_allow_html=True)
        else:
            rms_est = estimate_snr_rms_ratio(noisy, filtered)
            st.markdown(f"""<div class="status-banner-info">
                <b>Reference-Free Note:</b> No noiseless ground truth reference is available for arbitrary external audio.<br>
                Estimated RMS noise attenuation ratio: <b>{rms_est:+.2f} dB</b>.
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif selected_page == "About":
    st.markdown("### ℹ️ Academic Documentation & Viva Preparation")
    st.caption("Undergraduate ECE mini-project technical specifications, theoretical architecture, and oral examination guide.")

    st.markdown("""
    #### 1. Project Abstract & Objectives
    **SigNexis** is a software-engineered, integrated Digital Signal Processing (DSP) and Machine Learning (ML) system.
    It demonstrates real-time signal synthesis, discrete sampling, Nyquist-Shannon verification, spectral analysis via the Fast Fourier Transform (FFT),
    automated noise classification with Random Forest inference, and adaptive digital filter design.

    ---

    #### 2. System Architecture Block Diagram
    ```
    +------------------------+      +--------------------------+
    | Continuous Signal x(t) | ---> | Discrete Sampling x[n]   |
    +------------------------+      | (Nyquist Check: fs > 2f) |
                                    +--------------------------+
                                                 |
                                                 v
    +------------------------+      +--------------------------+
    | Feature Extraction     | <--- | Fast Fourier Transform   |
    | (13 Canonical Metrics) |      | Single-Sided Spectrum    |
    +------------------------+      +--------------------------+
                 |
                 v
    +------------------------+      +--------------------------+
    | Random Forest Model    | ---> | Adaptive Digital Filter  |
    | (6-Class Classifier)   |      | (Butterworth / Notch)    |
    +------------------------+      +--------------------------+
                                                 |
                                                 v
                                    +--------------------------+
                                    | Before vs After Metrics  |
                                    | True Delta SNR (dB)      |
                                    +--------------------------+
    ```

    ---

    #### 3. Theoretical Formulation Reference
    * **Nyquist-Shannon Sampling Theorem:**
      $$f_s > 2 f_{\max}, \\quad f_N = \\frac{f_s}{2}$$
    * **Aliased Foldover Frequency:**
      $$f_{\\text{alias}} = \\left| f_0 - \\operatorname{round}\\left(\\frac{f_0}{f_s}\\right) f_s \\right|$$
    * **Discrete Fourier Transform (DFT):**
      $$X[k] = \\sum_{n=0}^{N-1} x[n] \\cdot e^{-j \\frac{2\\pi}{N} k n}, \\quad k = 0, \\dots, N-1$$
    * **Signal-to-Noise Ratio (SNR):**
      $$\\text{SNR}_{\\text{dB}} = 10 \\log_{10} \\left( \\frac{P_{\\text{signal}}}{P_{\\text{noise}}} \\right) = 10 \\log_{10} \\left( \\frac{\\sum s^2[n]}{\\sum (x[n]-s[n])^2} \\right)$$

    ---

    #### 4. ECE Viva Voce Oral Defense Questions & Answers
    * **Q1: Why do we use zero-phase bidirectional filtering (`sosfiltfilt`)?**  
      *Answer:* Standard IIR filters introduce a non-linear phase shift that distorts waveform shapes. Zero-phase forward-backward filtering processes the sequence in both directions, yielding an effective phase distortion $\\theta(\\omega) = 0$ while doubling filter order attenuation.
    * **Q2: Why do Gaussian noise and high-frequency noise exhibit mutual misclassifications?**  
      *Answer:* White Gaussian noise possesses a flat Power Spectral Density (PSD) spanning from DC to the Nyquist limit $f_s/2$. Consequently, at lower SNRs, its spectral centroid and high-frequency band energy closely resemble bandlimited high-frequency noise.
    * **Q3: What determines the frequency resolution of the FFT?**  
      *Answer:* The frequency resolution is strictly determined by record length: $\\Delta f = \\frac{f_s}{N} = \\frac{1}{T}$, where $T$ is signal observation duration in seconds.
    * **Q4: Why does sampling at exactly $f = f_s/2$ fail?**  
      *Answer:* At the exact Nyquist frequency, depending on the phase of the continuous sinusoid, all samples can land precisely on zero-crossings, completely annihilating the signal. Hence, strict inequality $f_s > 2 f$ is required.

    ---

    #### 5. Verification Commands
    ```powershell
    # Run full unit and regression test suite
    python -m pytest tests/ -v

    # Run mathematical correctness and ECE audit
    python strict_audit.py

    # Run comprehensive multi-scenario pipeline audit
    python final_pipeline_audit.py
    ```
    """)

# ── Academic Footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#64748b;font-size:0.8rem;padding-bottom:1rem;'>"
    "SigNexis &nbsp;|&nbsp; ECE Undergraduate Capstone Demonstration &nbsp;|&nbsp; "
    "Digital Signal Processing &amp; Machine Learning Laboratory"
    "</div>",
    unsafe_allow_html=True,
)
