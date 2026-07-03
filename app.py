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

# --- 문장 데이터 연동 ---
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

# --- 📁 [새 기능] 왼쪽 사이드바 대사 목록화 영역 ---
with st.sidebar:
    st.title("📋 대사 목록 전체보기")
    st.caption("연습하고 싶은 문장을 아래 목록에서 바로 선택할 수 있습니다.")
    
    # 1. 드롭다운 선택 상자로 목록화
    selected_idx = st.selectbox(
        "🎯 이동할 문장 선택",
        options=range(len(sentence_level1)),
        index=st.session_state.current_idx,
        format_func=lambda x: f"Q{x+1}. {sentence_level1[x][:20]}..." if len(sentence_level1[x]) > 20 else f"Q{x+1}. {sentence_level1[x]}"
    )
    
    # 사이드바에서 문장을 고르면 현재 화면도 해당 문장으로 즉시 동기화
    if selected_idx != st.session_state.current_idx:
        st.session_state.current_idx = selected_idx
        st.rerun()
        
    st.markdown("---")
    
    # 2. 텍스트 리스트로 전체 대사 한눈에 훑어보기 안내
    st.markdown("**전체 대사 요약 리스트**")
    for i, sentence in enumerate(sentence_level1):
        # 현재 선택된 대사에는 별(⭐) 표시로 강조
        if i == st.session_state.current_idx:
            st.markdown(f"**⭐ Q{i+1}. {sentence}**")
        else:
            st.markdown(f"Q{i+1}. {sentence}")

# --- 메인 화면 상단 헤더 ---
st.title("Korean Pronunciation Shadowing Analyzer 🗣️")
st.caption("Compare the waveform and pitch between the Native Reference and your own voice side-by-side. You can record live or upload audio files to practice.")

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
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(BASE_DIR, "level", "level1", "audio")
    
    audio_path = os.path.join(audio_dir, f"{idx}.mp3")
    audio_path_wav = os.path.join(audio_dir, f"{idx}.wav")
    
    teacher_audio = None
    teacher_fs = 16000
    raw_audio_source = None

    # 1. 파일 경로 확인 및 데이터 소스 지정
    if os.path.exists(audio_path):
        st.audio(audio_path)
        raw_audio_source = audio_path
    elif os.path.exists(audio_path_wav):
        st.audio(audio_path_wav)
        raw_audio_source = audio_path_wav
    else:
        st.info("기본 제공 음원 파일이 없습니다. 파일을 업로드해 주세요.")
        uploaded_t = st.file_uploader("선생님 파일 업로드", type=["wav", "mp3"], key="teacher_upload")
        if uploaded_t:
            st.audio(uploaded_t)
            raw_audio_source = uploaded_t

    # 2. 오디오 파일 읽기 및 그래프 디코딩 안전장치
    if raw_audio_source is not None:
        try:
            if hasattr(raw_audio_source, 'seek'):
                raw_audio_source.seek(0)
            data, teacher_fs = sf.read(raw_audio_source)
            if len(data.shape) > 1: data = data[:, 0]
            teacher_audio = data.flatten()
        except Exception as e:
            # soundfile로 읽기 실패 시, 스트림릿 내장 바이트 변환 시도
            try:
                if hasattr(raw_audio_source, 'read'):
                    raw_audio_source.seek(0)
                    bytes_data = raw_audio_source.read()
                else:
                    with open(raw_audio_source, 'rb') as f:
                        bytes_data = f.read()
                # 8비트/16비트 오디오 데이터를 넘파이 배열로 강제 역직렬화
                audio_np = np.frombuffer(bytes_data, dtype=np.int16) / 32768.0
                # 앞쪽 헤더 영역 제외하고 오디오 알맹이만 추출
                if len(audio_np) > 44:
                    teacher_audio = audio_np[44:].flatten()
                    teacher_fs = 16000
            except:
                st.error("오디오 파일 코덱 분석에 실패했습니다. 다른 포맷이나 기기로 녹음된 파일 사용을 권장합니다.")

    # 선생님 그래프 그리기
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

# --- 🎧 오른쪽: User Audio (학생) ---
with col_right:
    st.subheader("🎧 User Audio")
    st.write("🔽 아래 버튼을 눌러 발음을 연습해 보세요.")

    # HTML5/JavaScript 기반 커스텀 녹음기 컴포넌트
    import streamlit.components.v1 as components

    # 세션 상태 초기화
    if "recorded_bytes" not in st.session_state:
        st.session_state.recorded_bytes = None

    # 자바스크립트 기반 녹음 인터페이스 (기존 마이크 아이콘 대체)
    recorder_html = """
    <div style="display: flex; gap: 10px; margin-bottom: 15px;">
        <button id="startBtn" style="padding: 10px 20px; background-color: #10B981; color: white; border: none; border-radius: 5px; font-weight: bold; cursor: pointer;">🎙️ 녹음 시작</button>
        <button id="stopBtn" style="padding: 10px 20px; background-color: #EF4444; color: white; border: none; border-radius: 5px; font-weight: bold; cursor: pointer;" disabled>⏹️ 녹음 중지</button>
        <span id="status" style="color: #9CA3AF; margin-top: 10px; font-size: 14px;">대기 중...</span>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');

        startBtn.onclick = async () => {
            audioChunks = [];
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    const base64Audio = reader.result.split(',')[1];
                    // 스트림릿 서버로 데이터 전송
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: base64Audio
                    }, '*');
                };
                status.innerText = "녹음 완료! 분석 중...";
            };

            mediaRecorder.start();
            startBtn.disabled = true;
            stopBtn.disabled = false;
            status.innerText = "🔴 녹음 중...";
        };

        stopBtn.onclick = () => {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            startBtn.disabled = false;
            stopBtn.disabled = true;
        };
    </script>
    """

    # HTML 컴포넌트를 화면에 렌더링하고 결과 데이터 받기
    components_output = components.html(recorder_html, height=60)

    # 녹음 데이터 처리 (Base64 -> Bytes)
    import base64
    if components_output:
        st.session_state.recorded_bytes = base64.b64decode(components_output)

    student_audio_bytes = st.session_state.recorded_bytes
    student_audio = None
    student_fs = 16000
    
    if student_audio_bytes:
        st.audio(student_audio_bytes, format="audio/wav")
        with open("temp_student.wav", "wb") as f:
            f.write(student_audio_bytes)
        try:
            data, student_fs = sf.read("temp_student.wav")
            if len(data.shape) > 1: data = data[:, 0]
            student_audio = data.flatten()
        except:
            # 예외 처리 안전장치
            pass
    else:
        uploaded_s = st.file_uploader("학생 파일 업로드(선택)", type=["wav", "mp3"], key="student_upload")
        if uploaded_s:
            data, student_fs = sf.read(uploaded_s)
            if len(data.shape) > 1: data = data[:, 0]
            student_audio = data.flatten()

    # 학생 그래프 그리기
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
        ax_s.text(0.5, 0.5, "Awaiting Recording...", color='#9CA3AF', ha='center', va='center')
        ax_s.set_axis_off()
        
    st.pyplot(fig_s)

# --- 📊 하단: 통합 비교 그래프 ---
st.markdown("---")
st.subheader("📊 Integrated Comparison (억양 겹쳐보기)")

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
