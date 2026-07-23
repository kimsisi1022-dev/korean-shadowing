import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import soundfile as sf
from audio_recorder_streamlit import audio_recorder

# --- Page Configuration (Dark theme & Title) ---
st.set_page_config(
    page_title="Korean Pronunciation Shadowing",
    page_icon="🗣️",
    layout="wide"
)

# --- Custom Styling (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #111827; color: #F3F4F6; }
    .stMarkdown h1 { color: #F3F4F6; font-weight: 700; }
    .sentence-box {
        background-color: #1F2937;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0EA5E9;
        margin-bottom: 25px;
    }
    .sentence-text {
        color: #0EA5E9;
        font-size: 22px;
        font-weight: bold;
    }
    /* 사이드바 버튼 스타일 */
    .stButton>button {
        text-align: left;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sentence Data Fallback ---
try:
    from sentences import sentence_level1
except ImportError:
    sentence_level1 = [
        "안녕하세요. 만나서 반가워요.",
        "오늘 날씨가 정말 화창하고 좋네요.",
        "한국어 발음 연습을 시작해 봅시다."
    ]

# --- Audio Analysis Function (Pitch & Intensity) ---
def analyze_audio(audio_data, fs):
    frame_size = int(fs * 0.03)  
    hop_size = int(fs * 0.01)    
    pitches, intensities, times = [], [], []
    
    for idx in range(0, len(audio_data) - frame_size, hop_size):
        frame = audio_data[idx:idx + frame_size]
        rms = np.sqrt(np.mean(frame**2)) if len(frame) > 0 else 0
        intensity = 20 * np.log10(rms + 1e-5) + 80  
        intensity = max(0, intensity) 
        intensities.append(intensity)
        
        if rms < 0.012: 
            pitches.append(np.nan) 
        else:
            autocorr = np.correlate(frame, frame, mode='full')[len(frame)-1:]
            min_lag, max_lag = int(fs / 400), int(fs / 80)
            if max_lag < len(autocorr):
                lag = np.argmax(autocorr[min_lag:max_lag]) + min_lag
                pitch = fs / lag if lag > 0 else np.nan
                pitches.append(pitch)
            else:
                pitches.append(np.nan)
        times.append(idx / fs)
    return np.array(times), np.array(pitches), np.array(intensities)

# --- Session State Management (Current Sentence Index) ---
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# --- 📁 Sidebar: Script List View ---
with st.sidebar:
    st.title("📋 Script List")
    st.caption("Select or click a sentence below to jump directly to it.")
    
    # 1. Dropdown Selector
    selected_idx = st.selectbox(
        "🎯 Select Sentence",
        options=range(len(sentence_level1)),
        index=st.session_state.current_idx,
        format_func=lambda x: f"Q{x+1}. {sentence_level1[x][:20]}..." if len(sentence_level1[x]) > 20 else f"Q{x+1}. {sentence_level1[x]}"
    )
    
    if selected_idx != st.session_state.current_idx:
        st.session_state.current_idx = selected_idx
        st.rerun()
        
    st.markdown("---")
    
    # 2. Clickable Sentence Buttons
    st.markdown("**All Sentences Overview**")
    for i, sentence in enumerate(sentence_level1):
        short_text = sentence[:18] + "..." if len(sentence) > 18 else sentence
        
        # 현재 선택된 대사는 별표(⭐)와 함께 비활성화 상태 버튼으로 표시
        if i == st.session_state.current_idx:
            st.button(f"⭐ Q{i+1}. {short_text}", key=f"btn_{i}", disabled=True, use_container_width=True)
        else:
            # 클릭 시 해당 문장 번호로 이동 후 화면 새로고침
            if st.button(f"Q{i+1}. {short_text}", key=f"btn_{i}", use_container_width=True):
                st.session_state.current_idx = i
                st.rerun()

# --- Main Header ---
st.title("Korean Pronunciation Shadowing Analyzer 🗣️")
st.caption("Compare the waveform and pitch between the Native Reference and your own voice side-by-side.")

# --- Sentence Navigation & Display ---
col_nav1, col_nav2, _ = st.columns([1, 1, 8])
with col_nav1:
    if st.button("◀ Previous", use_container_width=True):
        if st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
            st.rerun()
with col_nav2:
    if st.button("Next ▶", use_container_width=True):
        if st.session_state.current_idx < len(sentence_level1) - 1:
            st.session_state.current_idx += 1
            st.rerun()

idx = st.session_state.current_idx
st.markdown(f"""
    <div class="sentence-box">
        <p style="color: #9CA3AF; margin: 0; font-size: 14px;">Current Sentence</p>
        <p class="sentence-text">Q{idx + 1}. {sentence_level1[idx]}</p>
    </div>
""", unsafe_allow_html=True)

# --- Main Dashboard Split ---
col_left, col_right = st.columns(2)

# --- 👨‍🏫 Left Column: Native Reference ---
with col_left:
    st.subheader("👨‍🏫 Native Reference")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(BASE_DIR, "level", "level1", "audio")
    
    audio_path = os.path.join(audio_dir, f"{idx}.mp3")
    audio_path_wav = os.path.join(audio_dir, f"{idx}.wav")
    
    teacher_audio = None
    teacher_fs = 16000
    raw_audio_source = None

    # Check file availability
    if os.path.exists(audio_path):
        st.audio(audio_path)
        raw_audio_source = audio_path
    elif os.path.exists(audio_path_wav):
        st.audio(audio_path_wav)
        raw_audio_source = audio_path_wav
    else:
        st.info("No default reference file found. Please upload an audio file.")
        uploaded_t = st.file_uploader("Upload Teacher Audio", type=["wav", "mp3"], key="teacher_upload")
        if uploaded_t:
            st.audio(uploaded_t)
            raw_audio_source = uploaded_t

    # Audio Decoding Safeguard
    if raw_audio_source is not None:
        try:
            if isinstance(raw_audio_source, str):
                data, teacher_fs = sf.read(raw_audio_source)
            else:
                raw_audio_source.seek(0)
                data, teacher_fs = sf.read(io.BytesIO(raw_audio_source.read()))
            
            if len(data.shape) > 1: data = data[:, 0]
            teacher_audio = data.flatten()
        except Exception as e:
            st.error("Failed to decode teacher audio file.")

    # Render Teacher Plot
    fig_t, ax_t = plt.subplots(figsize=(5, 3), facecolor='#1F2937')
    ax_t.set_facecolor('#111827')
    
    if teacher_audio is not None and len(teacher_audio) > 0:
        t_times, t_pitches, t_ints = analyze_audio(teacher_audio, teacher_fs)
        ax_t.plot(t_times, t_pitches, color='#0EA5E9', linewidth=2, label="Pitch")
        ax_t.set_ylabel("Pitch (Hz)", color='#0EA5E9')
        ax_t.set_ylim(80, 350)
        ax_t.tick_params(colors='#9CA3AF')
        
        ax_t_int = ax_t.twinx()
        ax_t_int.plot(t_times, t_ints, color='purple', linestyle='--', alpha=0.4)
        ax_t_int.set_ylabel("Intensity (dB)", color='purple')
        ax_t_int.set_ylim(0, 90)
        ax_t_int.tick_params(colors='#9CA3AF')
    else:
        ax_t.text(0.5, 0.5, "Awaiting Audio Data", color='#9CA3AF', ha='center', va='center')
        ax_t.set_axis_off()
        
    st.pyplot(fig_t)


# --- 🎧 Right Column: User Audio (Student) ---
with col_right:
    st.subheader("🎧 User Audio")
    st.write("🎙️ 마이크 버튼을 눌러 직접 녹음하거나, 파일(.wav, .mp3)을 업로드하세요.")
    
    # 1. 라이브 녹음
    student_audio_bytes = audio_recorder(
        text="Record / Stop",
        recording_color="#EF4444",
        neutral_color="#9CA3AF",
        pause_threshold=30.0
    )
    
    student_audio = None
    student_fs = 16000
    
    # 2. 파일 업로드
    uploaded_s = st.file_uploader("Upload Audio File (Optional)", type=["wav", "mp3"], key="student_upload")
    
    # [우선순위 1] 녹음 데이터가 있는 경우
    if student_audio_bytes:
        st.audio(student_audio_bytes, format="audio/wav")
        try:
            data, student_fs = sf.read(io.BytesIO(student_audio_bytes))
            if len(data.shape) > 1: data = data[:, 0]
            student_audio = data.flatten()
        except Exception as e:
            st.error("Failed to process recorded audio.")

    # [우선순위 2] 파일이 업로드된 경우
    elif uploaded_s is not None:
        st.audio(uploaded_s)
        try:
            uploaded_s.seek(0)
            data, student_fs = sf.read(io.BytesIO(uploaded_s.read()))
            if len(data.shape) > 1: data = data[:, 0]
            student_audio = data.flatten()
        except Exception as e:
            st.error("Failed to read uploaded audio file. (Supported formats: WAV, MP3)")

    # Render Student Plot
    fig_s, ax_s = plt.subplots(figsize=(5, 3), facecolor='#1F2937')
    ax_s.set_facecolor('#111827')
    
    if student_audio is not None and len(student_audio) > 0:
        s_times, s_pitches, s_ints = analyze_audio(student_audio, student_fs)
        ax_s.plot(s_times, s_pitches, color='#EF4444', linewidth=2)
        ax_s.set_ylabel("Pitch (Hz)", color='#EF4444')
        ax_s.set_ylim(80, 350)
        ax_s.tick_params(colors='#9CA3AF')
        
        ax_s_int = ax_s.twinx()
        ax_s_int.plot(s_times, s_ints, color='orange', linestyle='--', alpha=0.4)
        ax_s_int.set_ylabel("Intensity (dB)", color='orange')
        ax_s_int.set_ylim(0, 90)
        ax_s_int.tick_params(colors='#9CA3AF')
    else:
        ax_s.text(0.5, 0.5, "Awaiting Recording or Upload...", color='#9CA3AF', ha='center', va='center')
        ax_s.set_axis_off()
        
    st.pyplot(fig_s)

# --- 📊 Bottom Column: Integrated Comparison Graph ---
st.markdown("---")
st.subheader("📊 Integrated Comparison (Intonation Overlay)")

fig_b, plt_ax_b = plt.subplots(figsize=(11, 2.5), facecolor='#1F2937')
plt_ax_b.set_facecolor('#111827')
plt_ax_b.tick_params(colors='#9CA3AF')

if teacher_audio is not None and len(teacher_audio) > 0:
    t_times, t_pitches, _ = analyze_audio(teacher_audio, teacher_fs)
    plt_ax_b.plot(t_times, t_pitches, color='#0EA5E9', linewidth=2.5, label="Reference (Teacher)", alpha=0.6)

if student_audio is not None and len(student_audio) > 0:
    s_times, s_pitches, _ = analyze_audio(student_audio, student_fs)
    plt_ax_b.plot(s_times, s_pitches, color='#EF4444', linewidth=2.5, label="User (Student)", alpha=0.9)

if (teacher_audio is not None and len(teacher_audio) > 0) or (student_audio is not None and len(student_audio) > 0):
    plt_ax_b.set_ylim(80, 350)
    plt_ax_b.legend(loc="upper right", facecolor='#1F2937', edgecolor='#9CA3AF', labelcolor='#F3F4F6')
else:
    plt_ax_b.text(0.5, 0.5, "Awaiting Input Data to Compare", color='#9CA3AF', ha='center', va='center')

st.pyplot(fig_b)
