import streamlit as st
import numpy as np
import plotly.graph_objects as px  # 인터랙티브 그래프용
from streamlit_plotly_events import plotly_events  # 클릭 이벤트 감지용
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

# --- 문장 데이터 연동 (Fallback 포함) ---
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

# --- [중요] 인터랙티브 그래프 생성 함수 (정의 추가) ---
def create_interactive_plot(times, pitches, intensities, color_line, title):
    """Builds a Plotly figure with click events and custom hover tooltips."""
    fig = px.Figure()
    
    # Pitch (Left Y-axis)
    fig.add_trace(px.Scatter(
        x=times, y=pitches,
        mode='lines',
        line=dict(color=color_line, width=3),
        name="Pitch (Hz)",
        hovertemplate="Time: %{x:.2f}s<br>Pitch: %{y:.1f}Hz<extra></extra>"
    ))
    
    # Intensity (Right Y-axis)
    fig.add_trace(px.Scatter(
        x=times, y=intensities,
        mode='lines',
        line=dict(color='rgba(156, 163, 175, 0.3)', width=1.5, dash='dash'),
        name="Intensity (dB)",
        yaxis="y2",
        hovertemplate="Time: %{x:.2f}s<br>Intensity: %{y:.1f}dB<extra></extra>"
    ))
    
    # Theme Layout Styling
    fig.update_layout(
        title=dict(text=title, font=dict(color='#F3F4F6', size=14)),
        paper_bgcolor='#1F2937',
        plot_bgcolor='#111827',
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        showlegend=False,
        clickmode='event',
        xaxis=dict(
            title=dict(text="Time (seconds)", font=dict(color='#9CA3AF')),
            tickfont=dict(color='#9CA3AF'),
            gridcolor='#1F2937'
        ),
        yaxis=dict(
            title=dict(text="Pitch (Hz)", font=dict(color=color_line)),
            tickfont=dict(color='#9CA3AF'),
            range=[80, 350],
            gridcolor='#1F2937'
        ),
        yaxis2=dict(
            title=dict(text="Intensity (dB)", font=dict(color='#9CA3AF')),
            tickfont=dict(color='#9CA3AF'),
            range=[0, 90],
            overlaying="y",
            side="right",
            showgrid=False
        )
    )
    return fig

# --- 상태 관리 (현재 문장 번호) ---
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# --- 📁 왼쪽 사이드바 대사 목록화 영역 ---
with st.sidebar:
    st.title("📋 Script List")
    st.caption("Select a sentence from the list below to jump directly to it.")
    
    # 1. 드롭다운 선택 상자
    selected_idx = st.selectbox(
        "🎯 Select Sentence",
        options=range(len(sentence_level1)),
        index=st.session_state.current_idx,
        format_func=lambda x: f"Q{x+1}. {sentence_level1[x][:20]}..." if len(sentence_level1[x]) > 20 else f"Q{x+1}. {sentence_level1[x]}"
    )
    
    # 사이드바에서 문장을 고르면 현재 화면도 해당 문장으로 즉시 동기화
    if selected_idx != st.session_state.current_idx:
        st.session_state.current_idx = selected_idx
        st.rerun()
        
    st.markdown("---")
    
    # 2. 텍스트 리스트로 전체 대사 오버뷰
    st.markdown("**All Sentences Overview**")
    for i, sentence in enumerate(sentence_level1):
        if i == st.session_state.current_idx:
            st.markdown(f"**⭐ Q{i+1}. {sentence}**")
        else:
            st.markdown(f"Q{i+1}. {sentence}")

# --- 메인 화면 상단 헤더 ---
st.title("Korean Pronunciation Shadowing Analyzer 🗣️")
st.caption("💡 **NEW feature:** Click any point on the graphs below to play the audio starting directly from that specific moment!")

# --- 메인 화면 문장 내비게이션 및 표시 ---
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

# --- 👨‍🏫 왼쪽: Native Reference (선생님) ---
with col_left:
    st.subheader("👨‍🏫 Native Reference")
    
    # 실행 중인 파일 기준 절대 경로 자동 계산
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(BASE_DIR, "level", "level1", "audio")
    
    audio_path = os.path.join(audio_dir, f"{idx}.mp3")
    audio_path_wav = os.path.join(audio_dir, f"{idx}.wav")
    
    teacher_audio = None
    teacher_fs = 16000
    raw_audio_source = None

    # 1. 저장소 파일 확인 및 데이터 소스 지정
    if os.path.exists(audio_path):
        raw_audio_source = audio_path
    elif os.path.exists(audio_path_wav):
        raw_audio_source = audio_path_wav
    else:
        st.info("No default reference file found. Please upload an audio file.")
        uploaded_t = st.file_uploader("Upload Teacher Audio", type=["wav", "mp3"], key="teacher_upload")
        if uploaded_t:
            st.audio(uploaded_t) # 업로드 파일 재생기 노출
            raw_audio_source = uploaded_t

    # 2. 오디오 파일 읽기 및 데이터 디코딩 (안전장치 포함)
    if raw_audio_source is not None:
        try:
            if hasattr(raw_audio_source, 'seek'): raw_audio_source.seek(0)
            data, teacher_fs = sf.read(raw_audio_source)
            if len(data.shape) > 1: data = data[:, 0]
            teacher_audio = data.flatten()
        except:
            # 특수 코덱 WAV/MP3 디코딩 실패 시 바이트 강제 변환 시도
            try:
                if hasattr(raw_audio_source, 'read'):
                    raw_audio_source.seek(0)
                    bytes_data = raw_audio_source.read()
                else:
                    with open(raw_audio_source, 'rb') as f: bytes_data = f.read()
                audio_np = np.frombuffer(bytes_data, dtype=np.int16) / 32768.0
                if len(audio_np) > 44:
                    teacher_audio = audio_np[44:].flatten()
                    teacher_fs = 16000
            except:
                st.error("Failed to decode reference audio codec.")

    # 3. 그래프 클릭 재생 인터랙션 로직
    if teacher_audio is not None and len(teacher_audio) > 0:
        t_times, t_pitches, t_ints = analyze_audio(teacher_audio, teacher_fs)
        fig_t = create_interactive_plot(t_times, t_pitches, t_ints, '#0EA5E9', "Teacher's Intonation (Click to play from here)")
        
        # 그래프 클릭 이벤트 가로채기
        selected_point_t = plotly_events(fig_t, click_event=True, hover_event=False, override_height=300, key=f"t_plot_{idx}")
        
        if selected_point_t:
            # 클릭 지점부터 오디오 슬라이싱 재생
            click_time_t = selected_point_t[0]['x']
            start_sample_t = int(click_time_t * teacher_fs)
            st.success(f"🎵 Playing Reference from {click_time_t:.2f} seconds")
            st.audio(teacher_audio[start_sample_t:], sample_rate=teacher_fs)
        else:
            # 클릭이 없을 땐 전체 오디오 플레이어 연결
            if not isinstance(raw_audio_source, (str, os.PathLike)) and uploaded_t:
                # 이미 st.audio로 띄웠으므로 중복 노출 방지
                pass
            else:
                st.audio(raw_audio_source)
    else:
        st.info("Awaiting Reference Audio Data...")


# --- 🎧 오른쪽: User Audio (학생) ---
with col_right:
    st.subheader("🎧 User Audio")
    st.write("🎙️ Click to record your voice, speak, and click again to finish.")
    
    # 1. 무음 감지 없는 수동 조작 스타일 실시간 녹음기
    student_audio_bytes = audio_recorder(
        text="Click to Record/Stop",
        recording_color="#EF4444",
        neutral_color="#9CA3AF",
        pause_threshold=30.0
    )
    
    student_audio = None
    student_fs = 16000
    student_audio_source = None
    
    # 2. 데이터 유입 경로 파악 및 파일 포인터 리셋
    if student_audio_bytes:
        student_audio_source = student_audio_bytes
        with open("temp_student.wav", "wb") as f:
            f.write(student_audio_bytes)
        try:
            data, student_fs = sf.read("temp_student.wav")
            if len(data.shape) > 1: data = data[:, 0]
            student_audio = data.flatten()
        except:
            pass
    else:
        uploaded_s = st.file_uploader("Upload Your Audio (Optional)", type=["wav", "mp3"], key="student_upload")
        if uploaded_s:
            student_audio_source = uploaded_s
            try:
                # [중요] 파일 업로더 객체의 읽기 포인터를 처음으로 리셋!
                uploaded_s.seek(0)
                data, student_fs = sf.read(uploaded_s)
                if len(data.shape) > 1: data = data[:, 0]
                student_audio = data.flatten()
            except:
                st.error("Failed to decode uploaded file. Please make sure it's a valid WAV or MP3 file.")

    # 3. 학생 인터랙티브 오디오 재생 및 그래프 제어
    if student_audio is not None and len(student_audio) > 0:
        s_times, s_pitches, s_ints = analyze_audio(student_audio, student_fs)
        fig_s = create_interactive_plot(s_times, s_pitches, s_ints, '#EF4444', "Your Intonation (Click to play from here)")
        
        # 그래프 클릭 이벤트 가로채기
        selected_point_s = plotly_events(fig_s, click_event=True, hover_event=False, override_height=300, key=f"s_plot_{idx}")
        
        if selected_point_s:
            # 클릭 지점부터 오디오 슬라이싱 재생
            click_time_s = selected_point_s[0]['x']
            start_sample_s = int(click_time_s * student_fs)
            st.success(f"🎵 Playing your voice from {click_time_s:.2f} seconds")
            st.audio(student_audio[start_sample_s:], sample_rate=student_fs)
        else:
            # [중요 수정] 마우스 클릭이 없을 때는 업로드된 전체 오디오 플레이어를 화면에 정상 노출!
            if student_audio_bytes:
                st.audio(student_audio_bytes, format="audio/wav")
            elif uploaded_s:
                st.audio(student_audio_source)
    else:
        st.info("Awaiting Student Recording or File Upload...")


# --- 📊 하단: 통합 비교 그래프 (단순 보기용 정적 Plotly) ---
st.markdown("---")
st.subheader("📊 Integrated Comparison (Intonation Overlay)")

fig_b = px.Figure()

if teacher_audio is not None and len(teacher_audio) > 0:
    fig_b.add_trace(px.Scatter(x=t_times, y=t_pitches, mode='lines', line=dict(color='#0EA5E9', width=2.5), name="Teacher"))
if student_audio is not None and len(student_audio) > 0:
    fig_b.add_trace(px.Scatter(x=s_times, y=s_pitches, mode='lines', line=dict(color='#EF4444', width=2.5), name="Student"))

# 비교 그래프 스타일링 (클릭 모드 비활성화)
fig_b.update_layout(
    paper_bgcolor='#1F2937', plot_bgcolor='#111827',
    margin=dict(l=40, r=40, t=10, b=40), height=220,
    xaxis=dict(tickfont=dict(color='#9CA3AF'), gridcolor='#1F2937'),
    yaxis=dict(range=[80, 350], tickfont=dict(color='#9CA3AF'), gridcolor='#1F2937'),
    legend=dict(font=dict(color='#F3F4F6'), bgcolor='rgba(31, 41, 55, 0.7)')
)

st.plotly_chart(fig_b, use_container_width=True)
