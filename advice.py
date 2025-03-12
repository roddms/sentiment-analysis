from dotenv import load_dotenv
import os
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
# OpenAI GPT 모델 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
st.set_page_config(layout="wide")
st.title("🤖 Advice Generator")
st.markdown("<h4>고객의 의견을 입력하시면, AI가 맞춤형 인사이트를 제공합니다</h4>", unsafe_allow_html=True)
contents = st.text_area("", height=100)
# 버튼을 눌렀을 때 AI 조언 생성
if st.button("조언 받기"):
    if contents.strip():
        with st.spinner("AI가 분석 중입니다..."):
            # 프롬프트 감정부분 {sentiment}와 같은 형식으로 수정 필요
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
        st.subheader("💡 AI Advice")
        st.write(response)
    else:
        st.warning("내용을 입력해주세요")