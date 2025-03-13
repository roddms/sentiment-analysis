import streamlit as st
import pandas as pd
import ast  # 리스트 변환용
import math
import html  # HTML 이스케이프용
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# 페이지 설정
st.set_page_config(
    page_title="🛅 Airline Issue Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/twitter_with_topic.csv")  
    topic_info = pd.read_csv("data/topic_info.csv")  
    return df, topic_info

df, topic_info = load_data()

# 사이드 바 
with st.sidebar:
    st.title("MENU")
    st.markdown("---")
    st.markdown("### ✈️ 항공사 선택")
    selected_airline = st.sidebar.selectbox("항공사를 선택하세요", df['airline'].unique())
    st.markdown("---")
    st.markdown("### 🪄 도움말")
    st.info("항공사 관련 고객 리뷰를 한눈에 확인할 수 있는 플랫폼입니다.")

st.markdown("<h1 style='text-align: center; color: #4B89DC;'>Airline Issue Monitoring</h1>", unsafe_allow_html=True)

# 선택한 항공사의 감정 분석 결과
airline_sentiment_counts = (
    df[df['airline'] == selected_airline]['airline_sentiment']
    .value_counts(normalize=True) * 100
)

sentiment_colors = {
    "positive": "#4CAF50",  # 초록색 (긍정)
    "neutral": "#FFC107",   # 노랑색 (중립)
    "negative": "#F44336"   # 빨간색 (부정)
}

#  감정별 주요 토픽 찾기
top_topics = {}
for sentiment in ["positive", "neutral", "negative"]:
    topic_counts = df[(df['airline'] == selected_airline) & (df['airline_sentiment'] == sentiment)]['topic'].value_counts()
    if not topic_counts.empty:
        top_topic_id = topic_counts.idxmax()  # 가장 많이 등장한 토픽 ID
        top_topic_name = topic_info.loc[topic_info['topic'] == top_topic_id, 'topic_name'].values[0]
        top_topics[sentiment] = top_topic_name
    else:
        top_topics[sentiment] = "관련 토픽 없음"

# 탭 생성
tab1, tab2 = st.tabs(["📝 항공사 만족도 분석", "🗂️ 고객 리뷰 분석"])

# 📌 탭1 - 전반적인 감성 분석 결과 
with tab1:
    st.markdown(f"### {selected_airline} 항공사의 만족도 분석")

    col1, col2, col3 = st.columns(3)

    # 긍정 비율
    with col1:
        positive_value = airline_sentiment_counts.get("positive", 0)
        st.markdown(
            f"""
            <div style="padding:12px; border-radius:10px; background-color:{sentiment_colors['positive']}; text-align:center;">
                <h3 style="color:white;">긍정</h3>
                <h2 style="color:white;">{positive_value:.1f}%</h2>
                <p style="color:white;"><b>주요 이슈: {top_topics['positive']}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(positive_value / 100)

    # 중립 비율
    with col2:
        neutral_value = airline_sentiment_counts.get("neutral", 0)
        st.markdown(
            f"""
            <div style="padding:12px; border-radius:10px; background-color:{sentiment_colors['neutral']}; text-align:center;">
                <h3 style="color:white;">중립</h3>
                <h2 style="color:white;">{neutral_value:.1f}%</h2>
                <p style="color:white;"><b>주요 이슈: {top_topics['neutral']}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(neutral_value / 100)

    # 부정 비율
    with col3:
        negative_value = airline_sentiment_counts.get("negative", 0)
        st.markdown(
            f"""
            <div style="padding:12px; border-radius:10px; background-color:{sentiment_colors['negative']}; text-align:center;">
                <h3 style="color:white;">부정</h3>
                <h2 style="color:white;">{negative_value:.1f}%</h2>
                <p style="color:white;"><b>주요 이슈: {top_topics['negative']}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(negative_value / 100)

# 📌 탭2 - 리뷰 모음 
with tab2:
    st.markdown("### 고객 리뷰")

    st.markdown("""
        <style>
            .box {
                padding: 12px;
                border-radius: 10px;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                margin-bottom: 10px;
            }
            .title {
                font-size: 18px;
                font-weight: bold;
                color: #333;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("🔴 **토픽 선택**")
            selected_topic_name = st.selectbox("", topic_info['topic_name'].unique(), key="topic_select")

        with col2:
            topic_keywords_str = topic_info[topic_info['topic_name'] == selected_topic_name]['top_words'].values[0]
            topic_keywords = ast.literal_eval(topic_keywords_str)  # 문자열을 리스트로 변환
            
            st.markdown("🔍 **관련 키워드 선택**")
            selected_keyword = st.selectbox("", topic_keywords)

        with col3:
            st.markdown("💬 **감정 선택**")
            selected_sentiments = st.multiselect("", ["negative", "neutral", "positive"], default=[], key="sentiment_select")

    
    if selected_keyword and selected_sentiments:
        # 📌 선택 정보 요약
        st.markdown(f"""
            <div class="box">
                <p><b>📝 선택한 토픽:</b> {selected_topic_name}</p>
                <p><b>🔑 선택한 키워드:</b> {selected_keyword}</p>
                <p><b>📊 선택한 감정:</b> {', '.join(selected_sentiments) if selected_sentiments else '선택 없음'}</p>
            </div>
        """, unsafe_allow_html=True)

        with st.spinner("로딩 중입니다...⏳"):
            filtered_reviews = df[df['airline'] == selected_airline]
            filtered_reviews = filtered_reviews[
                filtered_reviews['text'].str.contains(selected_keyword, case=False, na=False)
            ]

            filtered_reviews = filtered_reviews[
                filtered_reviews['airline_sentiment'].isin(selected_sentiments)
            ]

            filtered_reviews = filtered_reviews.drop_duplicates(subset=['tweet_id'])
            english_reviews = filtered_reviews['text'].tolist()

            # 페이지네이션 상태 초기화
            if "page_number" not in st.session_state:
                st.session_state.page_number = 1

            reviews_per_page = 5
            max_reviews = 20
            total_reviews = min(len(filtered_reviews), max_reviews)
            total_pages = max(1, math.ceil(total_reviews / reviews_per_page))  # 최소 1페이지 유지

            # 현재 페이지 범위를 벗어나면 조정
            st.session_state.page_number = min(st.session_state.page_number, total_pages)

            # 최대 20개까지만 표시
            filtered_reviews = filtered_reviews.iloc[:max_reviews]

            start_idx = (st.session_state.page_number - 1) * reviews_per_page
            end_idx = start_idx + reviews_per_page
            displayed_reviews = filtered_reviews.iloc[start_idx:end_idx]

            # 📌 리뷰 게시판
            st.markdown("----")
            st.markdown("### 📑 고객 리뷰")

            if total_reviews > 0:
                sentiment_tags = {
                    "positive": "🟢 Positive",
                    "neutral": "🟡 Neutral",
                    "negative": "🔴 Negative"
                }

                table_html = """
                <style>
                    .review-table {
                        width: 100%;
                        border-collapse: collapse;
                    }
                    .header {
                        background-color: rgb(218, 231, 255);
                        text-align: center;
                    }
                    .review-table th, .review-table td {
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                        vertical-align: top;
                    }
                    .review-table th {
                        background-color: #f2f2f2;
                    }
                </style>
                <table class='review-table'>
                    <tr>
                        <th>내용</th>
                        <th>레이블</th>
                        <th>복사</th>
                    </tr>
                """

                for _, row in displayed_reviews.iterrows():
                    sentiment_label = sentiment_tags.get(row["airline_sentiment"], "❓ Unknown")
                    tweet_text = html.escape(str(row['text']))  # HTML 인코딩 처리

                    table_html += f"""
                    <tr>
                        <td>{tweet_text}</td>
                        <td>{sentiment_label}</td>
                        <td>
                            <button class="copy-btn" onclick="navigator.clipboard.writeText('{tweet_text}');">
                                📋 복사
                            </button>
                        </td>
                    </tr>
                    """

                table_html += "</table>"
                table_html += f"<p>📄 페이지 {st.session_state.page_number} / {total_pages} (총 {total_reviews}개의 리뷰, 최대 20개 제한)</p>"
                st.html(table_html)

            else:
                st.write("⚠️ 리뷰가 없습니다.")

            col_prev, col_next = st.columns([1, 8]) 
            
            with col_prev:
                if st.button("⬅️ 이전 페이지") and st.session_state.page_number > 1:
                    st.session_state.page_number -= 1
                    st.rerun()  # 페이지 변경 즉시 반영

            with col_next:
                if st.button("다음 페이지 ➡️") and st.session_state.page_number < total_pages:
                    st.session_state.page_number += 1
                    st.rerun()  # 페이지 변경 즉시 반영 
