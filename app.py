import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
from audio_recorder_streamlit import audio_recorder

# --- 페이지 기본 설정 (다크 모드풍 테마 및 타이틀) ---
st.set_page_config(
    page_title="Korean Pronunciation Shadowing",
    page_icon="🗣️",
    layout="wide"
)

# --- 스타일링 (CSS) ---
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
    </style>
""", unsafe_allow_html=True)

# --- 더미 문장 데이터 (기존 데이터 연동용) ---
try:
    from sentences import sentence_level1
except ImportError:
    sentence_level1 = [
        "안녕하세요. 만나서 반가워요.",
        "오늘 날씨가 정말 화창하고 좋네요.",
        "한국어 발음 연습을 시작해 봅시다."
    ]

# --- 오디오 분석 함수 (Pitch & Intensity) ---
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

# --- 상태 관리 (현재 문장 번호) ---
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# --- 상단 헤더 ---
st.title("Korean Pronunciation Shadowing Analyzer 🗣️")
st.caption("Compare the waveform and pitch between the Native Reference and your own voice side-by-side. You can record live or upload audio files to practice.")

# --- 문장 내비게이션 및 표시 ---
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

# --- 메인 분석 패널 (좌우 분할) ---
col_left, col_right = st.columns(2)

# --- 왼쪽: Native Reference (선생님) ---
with col_left:
    st.subheader("👨‍🏫 Native Reference")
    
    # 기본 음원 로드 시도
    audio_path = f"level/level1/audio/{idx}.mp3"
    teacher_audio = None
    teacher_fs = 16000
    
    if os.path.exists(audio_path):
        st.audio(audio_path)
        data, teacher_fs = sf.read(audio_path)
        if len(data.shape) > 1: data = data[:, 0]
        teacher_audio = data.flatten()
    else:
        st.info("기본 제공 음원 파일이 없습니다. 파일을 업로드해 주세요.")
        uploaded_t = st.file_uploader("선생님 파일 업로드", type=["wav", "mp3"], key="teacher_upload")
        if uploaded_t:
            data, teacher_fs = sf.read(uploaded_t)
            if len(data.shape) > 1: data = data[:, 0]
            teacher_audio = data.flatten()

    # 선생님 그래프 그리기
    fig_t, ax_t = plt.subplots(figsize=(5, 3), facecolor='#1F2937')
    ax_t.set_facecolor('#111827')
    
    if teacher_audio is not None:
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

# --- 오른쪽: User Audio (학생) ---
with col_right:
    st.subheader("🎧 User Audio")
    
    # 웹용 간편 녹음 부품 (마이크 아이콘 클릭 시 녹음 시작/중지)
    st.write("클릭하여 녹음 시작/종료:")
    student_audio_bytes = audio_recorder(text="", recording_color="#EF4444", neutral_color="#9CA3AF")
    
    student_audio = None
    student_fs = 16000
    
    if student_audio_bytes:
        st.audio(student_audio_bytes, format="audio/wav")
        # 임시 파일로 저장 후 읽기
        with open("temp_student.wav", "wb") as f:
            f.write(student_audio_bytes)
        data, student_fs = sf.read("temp_student.wav")
        if len(data.shape) > 1: data = data[:, 0]
        student_audio = data.flatten()
    else:
        uploaded_s = st.file_uploader("학생 파일 업로드(선택)", type=["wav", "mp3"], key="student_upload")
        if uploaded_s:
            data, student_fs = sf.read(uploaded_s)
            if len(data.shape) > 1: data = data[:, 0]
            student_audio = data.flatten()

    # 학생 그래프 그리기
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

# --- 하단: 통합 비교 그래프 ---
st.markdown("---")
st.subheader("📊 Integrated Comparison (억양 겹쳐보기)")

fig_b, ax_b = plt.subplots(figsize=(11, 2.5), facecolor='#1F2937')
ax_b.set_facecolor('#111827')
ax_b.tick_params(colors='#9CA3AF')

if teacher_audio is not None:
    t_times, t_pitches, _ = analyze_audio(teacher_audio, teacher_fs)
    ax_b.plot(t_times, t_pitches, color='#0EA5E9', linewidth=2.5, label="Reference (Teacher)", alpha=0.6)

if student_audio is not None:
    s_times, s_pitches, _ = analyze_audio(student_audio, student_fs)
    ax_b.plot(s_times, s_pitches, color='#EF4444', linewidth=2.5, label="User (Student)", alpha=0.9)

if teacher_audio is not None or student_audio is not None:
    ax_b.set_ylim(80, 350)
    ax_b.legend(loc="upper right", facecolor='#1F2937', edgecolor='#9CA3AF', labelcolor='#F3F4F6')
else:
    ax_b.text(0.5, 0.5, "Awaiting Input Data to Compare", color='#9CA3AF', ha='center', va='center')

st.pyplot(fig_b)