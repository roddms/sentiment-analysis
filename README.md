# 💬 Sentiment Analysis & AI Advice Dashboard

딥러닝 기반 감정 분석 프로젝트

## 📌 Overview
Kaggle의 **Twitter US Airline Sentiment dataset**을 활용하여 **트윗 내용을 기반으로 감정을 분류하는 딥러닝 모델을 훈련**하고, 이를 바탕으로 맞춤형 솔루션을 제공하는 서비스로 구현했습니다. **입력된 의견에 담긴 감정을 분류**한 후, **GPT를 활용하여 해당 감정 및 의견에 맞는 조언을 제공**합니다.   

## 🎯 Features
✅ **텍스트 감정 분석** (`positive`, `neutral`, `negative`)  
✅ **GPT 기반 조언 제공** (예: 부정적인 감정의 경우 해결책 제안)  
✅ **SHAP 시각화**를 통한 감정 분류 근거 제공  
✅ **Streamlit UI**를 통한 배포

## 📊 Dataset
- **출처**: [Twitter US Airline Sentiment (Kaggle)](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **구성**:  
  - 항공사 관련 트윗 데이터  
  - 감정(label): `positive`, `neutral`, `negative`  
  - 총 14,640개의 샘플 데이터

## 🏗 Model Architecture
### 🔹 자연어 전처리
- HTML 태그, URL, 멘션, 해시태그 제거
- 불필요한 특수문자 및 숫자 제거
- `thx/thanks` → `thank`와 같이 단어 정규화
- 불용어 및 표제어 제거 : NLTK (Natural Language Toolkit), Word Cloud 시각화 사용

### 🔹 딥러닝 모델
- BERT (Bidirectional Encoder Representations from Transformers)
- 모델 성능 평가 : Accuracy

## 🔧 Trouble shooting

## 🚀 Try it!
🔗 [스트림릿 배포 링크 추가]
