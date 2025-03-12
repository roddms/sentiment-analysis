from dotenv import load_dotenv
import os
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
import time

# OpenAI GPT 모델 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# 페이지 설정
st.set_page_config(
    page_title="Advice Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 구성
with st.sidebar:
    #st.image("https://www.svgrepo.com/show/353655/discord-icon.svg", width=100)
    st.title("🤖 Advice Generator")
    st.markdown("---")
    st.markdown("### ⚙️ 설정")
    model = st.selectbox(
        "사용 모델",
        ("gpt-3.5-turbo", "gpt-4")
    )
    st.markdown("---")
    st.markdown("### 🪄 도움말")
    st.info("고객의 의견을 입력하시면 AI가 분석하여 맞춤형 비즈니스 인사이트를 제공합니다.")
    st.markdown("---")
    st.markdown("Made by ..")

# 메인 페이지
st.markdown("<h1 style='text-align: center; color: #4B89DC;'>Advice Generator</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #666666;'>고객의 의견을 입력하시면, AI가 맞춤형 인사이트를 제공합니다</h4>", unsafe_allow_html=True)


# 탭 생성
tab1, tab2 = st.tabs(["📝 조언 요청", "ℹ️ 사용 방법"])

with tab1:
    st.markdown("### 의견 입력")
    # 카드 스타일의 입력 영역
    with st.container():
        st.markdown("""
        <style>
        .input-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        </style>
        <div class="input-container">
        """, unsafe_allow_html=True)
        
        contents = st.text_area("", height=150, placeholder="의견을 입력해주세요 ...")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("🔍 조언 받기", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 버튼을 눌렀을 때 AI 조언 생성
    if analyze_button:
        if contents.strip():
            # 프로그레스 바로 로딩 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                # 진행 상태 업데이트
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text("고객 의견 분석 중...")
                elif i < 70:
                    status_text.text("인사이트 생성 중...")
                elif i < 99.5:
                    status_text.text("최종 결과 준비 중...")
                else:
                    status_text.text("")
                time.sleep(0.03)
            
            # 프롬프트 구성
            prompt = f"""
            "{contents}"

            위 의견은 감정 분석을 통해 "negative" 감정을 나타내는 것으로 분석되었습니다.

            당신의 역할:  
            - 서비스 개선을 위한 전략적 인사이트를 제공합니다.  
            - 감정을 고려하되, 감정적인 반응보다는 비즈니스적 해결책을 제안합니다.  
            - 고객의 불만 사항이 회사의 운영에 미칠 영향을 분석하고, 실용적인 대응 전략을 제공합니다.  
            - 필요하면 업계 사례, 데이터 기반 인사이트를 포함하여 답변하세요.  

            응답 예시: 
            - 감정이 "negative"이면: 문제의 원인을 분석하고 회사가 개선할 수 있는 전략적 조언을 제공합니다.  
            - 감정이 "positive"이면: 고객 경험을 더욱 강화할 방법을 제안하세요.  
            - 감정이 "neutral"이면: 추가적인 고객 피드백을 유도하고, 서비스 개선의 기회를 찾아 제안하세요.
            - 반드시 아래와 같은 형식으로 출력하세요:
              1. 문제의 원인 분석
              2. 실질적인 해결책 제안
              3. 관련된 업계 사례나 참고할 만한 전략

            이제 분석적인 인사이트와 적용 가능하고 구체적이며 실용적인 전략을 포함하여 조언을 작성하세요:
            """
            
            response = chat_model.predict(prompt)
            
            # 결과 출력
            st.markdown("### 💡 AI Advice")
            with st.container():
                st.markdown("""
                <style>
                .result-container {
                    background-color: #f0f7ff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    border-left: 5px solid #4B89DC;
                }
                </style>
                <div class="result-container">
                """, unsafe_allow_html=True)
                
                st.markdown(response)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # 결과 피드백
                st.markdown("#### 위 조언이 마음에 드셨나요?")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button("👍 유용해요", use_container_width=True)
                with col2:
                    st.button("👎 아쉬워요", use_container_width=True)
                with col3:
                    st.button("📋 저장하기", use_container_width=True)
        else:
            st.error("❗ 내용을 입력해주세요")

with tab2:
    st.markdown("### 사용 방법")
    st.markdown("""
    1. **고객 의견 입력**: 분석하고 싶은 고객의 의견이나 피드백을 입력창에 작성합니다.
    2. **조언 받기 클릭**: 버튼을 클릭하면 AI가 의견을 분석하고 맞춤형 조언을 제공합니다.
    3. **결과 확인**: 분석 결과는 다음 세 가지 항목으로 구성됩니다:
       - 문제의 원인 분석
       - 실질적인 해결책 제안
       - 관련된 업계 사례나 참고할 만한 전략
    4. **피드백**: 결과에 대한 피드백을 제공하여 서비스 개선에 도움을 줄 수 있습니다.
    """)
    
    st.info("💡 더 나은 결과를 얻기 위해 입력란에 상세하게 작성해 주세요.")

# 푸터
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888888;'>© 2025 AI Advice Generator</p>", unsafe_allow_html=True)

# 커스텀 CSS 추가
st.markdown("""
<style>
    .stButton>button {
        background-color: #4B89DC;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 15px;
    }
    .stButton>button:hover {
        background-color: #3A70B9;
    }
    .stTextArea>div>div>textarea {
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: #4B89DC;
        box-shadow: 0 0 0 2px rgba(75, 137, 220, 0.2);
    }
    .stProgress>div>div>div {
        background-color: #4B89DC;
    }
</style>
""", unsafe_allow_html=True)