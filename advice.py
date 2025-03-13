from dotenv import load_dotenv
import os
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
import time
import torch
import numpy as np
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# 페이지 설정
st.set_page_config(
    page_title="Advice Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    model_path = "sentiment-bert-model2" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
model.config.label2id = {"negative": 0, "neutral": 1, "positive": 2}
sentiment_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    framework="pt",
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=True
)
explainer = shap.Explainer(sentiment_pipeline)

# OpenAI GPT 모델 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)


# 사이드바 구성
with st.sidebar:
    st.title("MENU")
    st.markdown("---")
    st.markdown("### ⚙️ 설정")
    st.markdown("")
    model_type = st.selectbox(
        "사용 모델",
        ("gpt-3.5-turbo", "gpt-4")
    )
    st.markdown("")
    business_type = st.selectbox(
        "도메인 선택",
        ("항공", "숙박", "식당", "금융", "의료", "교육")
    )
    st.markdown("---")
    st.markdown("### 🪄 도움말")
    st.info("고객의 의견을 입력란에 입력하시면 AI가 맞춤형 비즈니스 인사이트를 제공합니다.")

# 메인 페이지
st.markdown("<h1 style='text-align: center; color: #4B89DC;'>Advice Generator</h1>", unsafe_allow_html=True)
#st.markdown("<h4 style='text-align: center; color: #666666;'>고객의 의견을 입력하시면, AI가 맞춤형 인사이트를 제공합니다</h4>", unsafe_allow_html=True)


# 탭 생성
tab1, tab2 = st.tabs(["📝 조언 요청", "ℹ️ 사용 방법"])

with tab1:
    st.markdown("### 의견 입력")
    with st.container():        
        contents = st.text_area("", height=150, placeholder="고객의 의견을 입력해주세요 ...")
        
        def predict(contents):
            inputs = tokenizer(contents, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = outputs.logits.softmax(dim=-1)
            return probabilities.cpu().numpy()

        def model_forward(contents):
            if isinstance(contents, list):
                contents = [text if isinstance(text, str) else "" for text in contents]
            else:
                contents = [contents]

            encoded_inputs = tokenizer(contents, padding=True, truncation=True, max_length=128, return_tensors="pt")
            encoded_inputs = {key: tensor.to(device) for key, tensor in encoded_inputs.items()}
            with torch.no_grad():
                output = model(**encoded_inputs)
                return output.logits.cpu().numpy()
        
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
                    status_text.text("🧐 고객 의견 분석 중 ...")
                elif i < 70:
                    status_text.text("🫨 인사이트 생성 중 ...")
                elif i < 99:
                    status_text.text("🤩 최종 결과 준비 중 ...")
                else:
                    status_text.text("")
                time.sleep(0.03)

            # inputs = tokenizer(contents, return_tensors="pt", truncation=True, max_length=128)
            # input_ids = inputs["input_ids"].to(device)
            # attention_mask = inputs["attention_mask"].to(device)
            # with torch.no_grad():
            #     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            #     probabilities = outputs.logits.softmax(dim=-1).cpu().numpy()
            # pred_label = np.argmax(probabilities)
            # label_map = {0: "negative", 1: "neutral", 2: "positive"}

            ### 수정 부분
            result = sentiment_pipeline(contents)
            pred_label = max(result[0], key=lambda x: x["score"])["label"]
            # shap 시각화
            shap_values = explainer([contents])
            #shap.plots.text(shap_values[0])  # 첫 번째 문장
            def st_shap(plot_html, height=None, width=None):
            # plot_html은 이제 HTML 문자열이므로 그대로 전달합니다.
                full_html = f"<head>{shap.getjs()}</head><body>{plot_html}</body>"
                components.html(full_html, height=height, width=width)
            shap_html = shap.plots.text(shap_values)
            st_shap(shap_html, height=400)
            
            # 프롬프트
            prompt = f"""
            "{contents}"

            위 의견은 감정 분석을 통해 "{pred_label}"  감정을 나타내는 것으로 분석되었습니다.

            당신의 역할:  
            - 입력된 {business_type}를 고려하여 답변을 작성합니다.
            - 서비스 개선을 위한 전략적 인사이트를 제공합니다.  
            - 감정을 고려하되, 감정적인 반응보다는 비즈니스적 해결책을 제안합니다.  
            - 고객의 불만 사항이 회사의 운영에 미칠 영향을 분석하고, 실용적인 대응 전략을 제공합니다.  
            - 필요하면 업계 사례, 데이터 기반 인사이트를 포함하여 답변하세요.

            응답 예시: 
            - 감정이 "negative"이면: 문제의 원인을 분석하고 회사가 개선할 수 있는 전략적 조언을 제공합니다.  
            - 감정이 "positive"이면: 고객 경험을 더욱 강화할 방법을 제안하세요.  
            - 감정이 "neutral"이면: 추가적인 고객 피드백을 유도하고, 서비스 개선의 기회를 찾아 제안하세요.
            - 반드시 아래와 같은 형식으로 출력하세요:
              1. 문제의 원인 분석 :
              2. 실질적인 해결책 제안 :
              3. 관련된 업계 사례나 참고할 만한 전략 :

            이제 분석적인 인사이트와 적용 가능하고 구체적이며 실용적인 전략을 포함하여 조언을 작성하세요:
            """
            
            response = chat_model.predict(prompt)
            
            # 결과 출력
            with st.container():
                st.markdown("### 💡 AI Advice")
                
                # 결과 컨테이너 시작
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                # 결과 내용 표시
                st.markdown(response)
                
                # 결과 컨테이너 종료
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 결과 피드백
                st.markdown("")
                st.markdown("##### 답변이 마음에 드셨나요?")
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
       - 문제 원인 분석
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
    .stProgress>div>div>div {
        background-color: #4B89DC;
    }
    *:hover {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)
