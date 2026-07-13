import streamlit as st
import numpy as np
import plotly.graph_objects as px
from streamlit_plotly_events import plotly_events
import os
import soundfile as sf
from audio_recorder_streamlit import audio_recorder

# --- Page Configuration ---
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
    .guide-box {
        background-color: #1F2937;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-inline-start: 4px solid #10B981;
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

# --- Audio Analysis Function ---
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

# --- Interactive Plot Generator ---
def create_interactive_plot(times, pitches, intensities, color_line, title):
    fig = px.Figure()
    fig.add_trace(px.Scatter(
        x=times, y=pitches, mode='lines',
        line=dict(color=color_line, width=3), name="Pitch (Hz)",
        hovertemplate="Time: %{x:.2f}s<br>Pitch: %{y:.1f}Hz<extra></extra>"
    ))
    fig.add_trace(px.Scatter(
        x=times, y=intensities, mode='lines',
        line=dict(color='rgba(156, 163, 175, 0.3)', width=1.5, dash='dash'),
        name="Intensity (dB)", yaxis="y2",
        hovertemplate="Time: %{x:.2f}s<br>Intensity: %{y:.1f}dB<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color='#F3F4F6', size=14)),
        paper_bgcolor='#1F2937', plot_bgcolor='#111827',
        margin=dict(l=40, r=40, t=40, b=40), height=300, showlegend=False, clickmode='event',
        xaxis=dict(title=dict(text="Time (seconds)", font=dict(color='#9CA3AF')), tickfont=dict(color='#9CA3AF'), gridcolor='#1F2937'),
        yaxis=dict(title=dict(text="Pitch (Hz)", font=dict(color=color_line)), tickfont=dict(color='#9CA3AF'), range=[80, 350], gridcolor='#1F2937'),
        yaxis2=dict(title=dict(text="Intensity (dB)", font=dict(color='#9CA3AF')), tickfont=dict(color='#9CA3AF'), range=[0, 90], overlaying="y", side="right", showgrid=False)
    )
    return fig

# --- State Management ---
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# --- 📁 Sidebar: Script List & 💡 NEW: Pronunciation Guide ---
with st.sidebar:
    st.title("📋 Navigation & Guides")
    
    # 탭 메뉴로 깔끔하게 분리 (1 탭: 대사 선택, 2 탭: 조음 가이드)
    tab_nav, tab_guide = st.tabs(["🎯 Script List", "💡 Mouth & Tongue Guide"])
    
    with tab_nav:
        st.caption("Select a sentence from the list below.")
        selected_idx = st.selectbox(
            "Move to Sentence",
            options=range(len(sentence_level1)),
            index=st.session_state.current_idx,
            format_func=lambda x: f"Q{x+1}. {sentence_level1[x][:15]}..." if len(sentence_level1[x]) > 15 else f"Q{x+1}. {sentence_level1[x]}"
        )
        if selected_idx != st.session_state.current_idx:
            st.session_state.current_idx = selected_idx
            st.rerun()
            
        st.markdown("---")
        st.markdown("**Overview**")
        for i, sentence in enumerate(sentence_level1):
            if i == st.session_state.current_idx:
                st.markdown(f"**⭐ Q{i+1}. {sentence}**")
            else:
                st.markdown(f"Q{i+1}. {sentence}")
                
    with tab_guide:
        st.caption("Learn the correct mouth shapes and tongue positions for Korean.")
        
        guide_type = st.radio("Choose Category", ["Vowels (모음)", "Consonants (자음)"])
        
        if guide_type == "Vowels (모음)":
            st.markdown("""
            <div class="guide-box">
                <strong>👄 ㅏ [a] vs ㅓ [ㅓ]</strong><br>
                • <b>ㅏ:</b> Open your mouth wide vertically.<br>
                • <b>ㅓ:</b> Keep your tongue low, relax your lips, open slightly less than 'ㅏ'.
            </div>
            <div class="guide-box">
                <strong>👄 ㅗ [o] vs ㅜ [u]</strong><br>
                • <b>ㅗ:</b> Round your lips forward tightly (O-shape).<br>
                • <b>ㅜ:</b> Push your lips further out, making a smaller circle.
            </div>
            <div class="guide-box">
                <strong>👄 ㅡ [ɯ] vs ㅣ [i]</strong><br>
                • <b>ㅡ:</b> Pull your lips wide to the sides, flat mouth.<br>
                • <b>ㅣ:</b> Smile shape, tongue high up near the roof.
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class="guide-box">
                <strong>👅 ㄱ, ㅋ, ㄲ (Velar)</strong><br>
                • The back of your tongue blocks and releases air from the soft palate.
            </div>
            <div class="guide-box">
                <strong>👅 ㄴ, ㄷ, ㅌ, ㄹ (Alveolar)</strong><br>
                • The tip of your tongue must touch the ridge behind your upper front teeth.
            </div>
            <div class="guide-box">
                <strong>👅 ㅁ, ㅂ, ㅍ, ㅃ (Bilabial)</strong><br>
                • Completely close and pop your upper and lower lips together.
            </div>
            <div class="guide-box">
                <strong>💡 Tip for Lax vs Aspirated vs Tense</strong><br>
                • <b>ㄱ / ㅂ:</b> Normal breath.<br>
                • <b>ㅋ / ㅍ:</b> Strong burst of air (Aspirated).<br>
                • <b>ㄲ / ㅃ:</b> Tense up throat muscles, NO air escapes (Tense).
            </div>
            """, unsafe_allow_html=True)

# --- Main Header ---
st.title("Korean Pronunciation Shadowing Analyzer 🗣️")
st.caption("Compare the waveform and pitch between the Native Reference and your own voice side-by-side. You can record live or upload audio files to practice.")

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
            if hasattr(raw_audio_source, 'seek'):
                raw_audio_source.seek(0)
            data, teacher_fs = sf.read(raw_audio_source)
            if len(data.shape) > 1: data = data[:, 0]
            teacher_audio = data.flatten()
        except Exception as e:
            try:
                if hasattr(raw_audio_source, 'read'):
                    raw_audio_source.seek(0)
                    bytes_data = raw_audio_source.read()
                else:
                    with open(raw_audio_source, 'rb') as f:
                        bytes_data = f.read()
                audio_np = np.frombuffer(bytes_data, dtype=np.int16) / 32768.0
                if len(audio_np) > 44:
                    teacher_audio = audio_np[44:].flatten()
                    teacher_fs = 16000
            except:
                st.error("Failed to decode audio codec. Please try a different audio format.")

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
    
    st.write("🎙️ Click the microphone icon to start recording. Speak clearly, and click it again to stop when you are done!")
    
    # Custom audio recorder configuration
    student_audio_bytes = audio_recorder(
        text="Click to Record/Stop",
        recording_color="#EF4444",
        neutral_color="#9CA3AF",
        pause_threshold=30.0  # Safe threshold to avoid unwanted automatic stops
    )
    
    student_audio = None
    student_fs = 16000
    
    if student_audio_bytes:
        st.audio(student_audio_bytes, format="audio/wav")
        with open("temp_student.wav", "wb") as f:
            f.write(student_audio_bytes)
        data, student_fs = sf.read("temp_student.wav")
        if len(data.shape) > 1: data = data[:, 0]
        student_audio = data.flatten()
    else:
        uploaded_s = st.file_uploader("Upload Your Audio (Optional)", type=["wav", "mp3"], key="student_upload")
        if uploaded_s:
            data, student_fs = sf.read(uploaded_s)
            if len(data.shape) > 1: data = data[:, 0]
            student_audio = data.flatten()

    # Render Student Plot
    fig_s, ax_s = plt.subplots(figsize=(5, 3), facecolor='#1F2937')
    ax_s.set_facecolor('#111827')
    
    if student_audio is not None:
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
        ax_s.text(0.5, 0.5, "Awaiting Recording...", color='#9CA3AF', ha='center', va='center')
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
