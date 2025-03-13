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
  - 2015년 2월 여행객들의 트윗 데이터
  - 감정(label): `positive`, `neutral`, `negative`  
  - 총 14,640개의 샘플 데이터

## 🏗 Model Architecture
### 🔹 자연어 전처리
- HTML 태그, URL, 멘션, 해시태그 제거
- 불필요한 특수문자 및 숫자 제거
- `thx/thanks` → `thank`와 같이 단어 정규화
- 불용어 및 표제어 제거 : NLTK (Natural Language Toolkit), Word Cloud 시각화 사용

# 🔹 딥러닝 기반 감성 분석 모델

## **📌 사용된 딥러닝 모델**
### **1️⃣ BERT (Bidirectional Encoder Representations from Transformers)**
- Google에서 개발한 **자연어 처리(NLP) 모델**  
- **양방향(Bidirectional) 문맥 이해**를 통해 문장의 의미를 정확하게 파악  
- **사전 학습(Pre-training)**된 모델로, 감성 분석을 위해 **파인튜닝(Fine-tuning)** 진행  

### **2️⃣ 사용한 Hugging Face Transformer 모델**
| 모델 | 설명 |
|------|------|
| **bert-base-uncased** | 기본 BERT 모델 (소문자 변환) |
| **bert-large-uncased** | BERT-Base보다 더 깊은 모델 (레이어 수 증가) |
| **roberta-base** | **BERT보다 10배 많은 데이터**로 학습된 모델 (동적 Masking 적용) |
| **roberta-large** | RoBERTa-Base보다 더 크고 강력한 모델 |
| **deberta-v3-large** | **Microsoft에서 개발한 15억 개의 파라미터를 가진 모델** |
| **twitter-roberta-base-sentiment** | **Twitter 감정 분석에 특화된 모델** |

✔ **이 중 가장 적합한 모델을 선정하여 감성 분석을 진행함.** 🚀  

---

## **📌 모델 성능 평가 지표**
- **Accuracy (정확도)** : 예측한 감성이 실제 감성과 얼마나 일치하는지 측정  
- **Loss (손실 함수 값)** : 모델의 예측 오류 정도를 나타내며, 낮을수록 성능이 좋음  

---

### 🔹 종류별 BERT 모델 성능지표
| Model                               | Train Accuracy | Validation Accuracy | Test Accuracy | Train Loss | Validation Loss | Test Loss |
|-------------------------------------|---------------|---------------------|--------------|-----------|----------------|----------|
| **BERT-Base**                       | 0.8730        | 0.8300              | 0.8220       | 0.3400    | 0.4150         | 0.4050   |
| **BERT-Large**                      | 0.8950        | 0.8500              | 0.8600       | 0.2850    | 0.4100         | 0.3980   |
| **RoBERTa-Base**                    | 0.8861        | 0.8402              | 0.8324       | 0.3241    | 0.4013         | 0.3961   |
| **RoBERTa-Base (Dropout 0.2)**      | 0.8659        | 0.8443              | 0.8369       | 0.3505    | 0.4184         | 0.4112   |
| **RoBERTa-Large**                   | 0.8928        | 0.8524              | 0.8551       | 0.2695    | 0.4023         | 0.3997   |
| **Twitter-RoBERTa-Base-Sentiment**  | 0.8786        | 0.8497              | 0.8465       | 0.3118    | 0.4037         | 0.3875   |
| **DeBERTa-V3-Large**                | 0.8816        | 0.8370              | 0.8493       | 0.3201    | 0.4351         | 0.4081   |

---		

## **📌 모델 선정 기준**
💡 **모델 선택 시 과적합(Overfitting)과 일반화 성능을 고려하여 아래 조건을 만족하는 모델을 선정함.**  

1️⃣ **Train Accuracy vs Test Accuracy 차이 ≤ 5%**  
   - Train(훈련) 데이터와 Test(테스트) 데이터에서 **정확도의 차이가 5% 이상이면 과적합 가능성이 있음.**  
   - 일반화 성능이 좋은 모델을 선택하기 위해 이 기준을 적용.  
   
2️⃣ **Test Loss ≤ 0.4**  
   - Test 데이터에서 **Loss가 0.4 이하인 모델을 우선적으로 고려**하여 안정적인 성능 확보.  

## 결과적으로 RoBERTa-Large모델 사용

---

## **📌 하이퍼파라미터 튜닝**
💡 **최적의 성능을 위해 여러 하이퍼파라미터를 실험하며 조정함.**  

| 하이퍼파라미터 | 설정 값 |
|---------------|--------|
| **Dropout (hidden, attention)** | `0.1 ~ 0.2` (과적합 방지) |
| **Epochs** | `2 ~ 4` (적절한 학습 횟수) |
| **Batch Size (Train, Eval)** | `16 ~ 32` (학습 안정성을 고려) |
| **Learning Rate** | `1e-5 ~ 2e-5` (AdamW 옵티마이저 사용) |
| **Warmup Steps** | 훈련 초반 `500~1000` 스텝 동안 작은 학습률 유지 |
| **Weight Decay** | `0.001 ~ 0.01` (과적합 방지 및 일반화 성능 개선) |

---


## 🔧 Trouble Shooting

### **📌 Twitter US Airline Sentiment Dataset 구성**
이 데이터셋은 미국 항공사에 대한 트윗을 감성 분석한 것으로, **부정적인 감성이 압도적으로 많아 데이터 불균형(imbalanced data)이 발생**하는 특징이 있음.

| Sentiment | 비율 (%) |
|-----------|---------|
| **Negative (부정적)** | **62%** |
| **Neutral (중립적)** | **21%** |
| **Positive (긍정적)** | **16%** |

💡 **이러한 불균형 문제를 해결하지 않으면, 모델이 negative 클래스에 편향되어 예측 성능이 저하될 가능성이 있음.**  

---
### **✅ 해결 방법: 손실 함수 변경 (CrossEntropyLoss → Focal Loss)**

#### **🔹 기존 Loss Function: CrossEntropyLoss 문제점**
- 모든 클래스(positive/neutral/negative)를 **동일한 중요도로 학습**  
- **데이터 불균형이 심한 경우**, 다수 클래스(negative) 중심으로 학습 진행  
- 결과적으로 **소수 클래스(positive, neutral)의 예측 정확도가 낮아질 가능성이 큼**  

#### **🔹 대안: Focal Loss 적용**
Focal Loss는 **자주 등장하는 쉬운 샘플(negative)에 대한 가중치를 낮추고, 어려운 샘플(positive, neutral)에 더 집중하도록 유도하는 손실 함수**

| Loss Function | 특징 | 데이터 불균형 해결 |
|--------------|------|----------------|
| **CrossEntropyLoss** | 모든 클래스 동일 가중치 | ❌ 다수 클래스(negative)에 편향될 가능성 |
| **Focal Loss** | 어려운 샘플에 더 집중 | ✅ 소수 클래스(positive, neutral)도 학습 가능 |

✔ **적용 효과**  
- **소수 클래스(positive, neutral)의 예측 성능 향상**  
- **과적합 방지 및 모델의 일반화 성능 개선** 

## 🏆 최종 모델 성능

학습된 모델의 성능은 **Train / Validation / Test 데이터에서의 정확도(Accuracy)로 평가**

| 데이터셋 | Accuracy (%) |
|----------|-------------|
| **Train (훈련 데이터)** | **90.39%** |
| **Validation (검증 데이터)** | **85.70%** |
| **Test (테스트 데이터)** | **86.89%** |

✔ **Train과 Test 성능 차이가 5% 이내로 유지되어, 과적합 없이 안정적인 성능**  
✔ **Validation과 Test 정확도가 비슷하여 일반화 성능이 우수**  

---

### **🚀 최종 결론**
**Focal Loss 적용 및 하이퍼파라미터 튜닝을 통해 최적의 성능을 달성하였으며, 모델이 훈련 데이터에 과적합하지 않고 새로운 입력 데이터에서도 분류를 잘 해냄을 확인** 🚀🔥

---

## 🚀 Try it!
🔗 [https://advicegenerator.streamlit.app/]
