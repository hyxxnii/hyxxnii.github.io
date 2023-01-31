---
layout: post
title: "[논문리뷰] LoCo-VAE: Modeling Short-Term Preference as Joint Effect of Long-Term Preference and Context-Aware Impact in Recommendation (PRICAI, 2021)"
date: 2022-12-11
categories:
- Recommender System
tags:
- paper review
- recommender system
- variational autoencoder
use_math: true
comments: true
---

#### 논문 출처: [LoCo-VAE: Modeling Short-Term Preference as Joint Effect of Long-Term Preference and Context-Aware Impact in Recommendation](https://link.springer.com/chapter/10.1007/978-3-030-89363-7_37)


이번 포스팅은 
이번 포스팅은 `Mult-VAE`를 통해 user의 short-term preference와 long-term preference를 함께 고려한  논문을 읽고 정리한 것이다.

📍 ***연구 IDEA 정리***

- Context-aware Encoder와 Long-term preference Encoder를 합쳐서 유저의 Short-term Preference를 캡쳐
- 데이터에서 observed된 context (e.g. 낮/밤, 봄/여름/가을/겨울) 별로 Context-aware Encoder를 각각 태움 (⇒ 이미 관측된 context를 사용해서 각 encoder를 태운다는 건 큰 develop은 아닌 것 같다)
- Long-term과 Context-aware Encoder의 파라미터를 공유

<br>

---

# Abstract

- 기존의 user preference 모델링 → short-term 기간 내에 적용
- 하지만 **short-term user preference는 더 복잡한 contexts와 관련되어 있다**
    - **ex) 계절 또는 하루 중의 시간**
- **연구의 가설: 유저의 short-term preferences는 실제로 그 유저의 안정적인 long-term preferences와 context-aware 영향의 공동 효과(joint effect)**
- **LoCo-VAE 제안: VAE를 이용한 공동 효과의 통합 모델**
    1. long-term user preferences를 캡쳐하기 위한 MLP 활용
    2. context-aware 영향을 도입하기 위해 다양한 contexts와 관련하여 user의 상호 작용을 분산시킴으로써 전통적인 VAE를 개선
    3. short-term user preferences 임베딩을 생성하기 위해 long-term preferences와 context-aware 영향을 결합
- real-world datasets
    - Amazon consumption, music selection

<br>

# 1 Introduction

- user preferences를 모델링하는 가장 인기있는 두 가지 방법: static 모델링, dynamic 모델링
- Static 모델링
    - user의 long-term preferences를 학습하는 것을 목표로 한다
    - Collaborative Filtering (CF) ⇒ user-item 상호작용 matrix의 latent space를 찾음
    - Variational Autoencoders (VAEs)와 같은 딥러닝 발달 ⇒ non-linear한 user-item 관계 캡쳐
- Dynamic 모델링
    - temporal-based 추천시스템 ⇒ dynamic한 short-term preferences를 모델링
    - RNN 및 변형, LSTM ⇒ 시간에 따른 user의 preferences의 시간적 변화를 캡쳐
- 현재의 temporal-based 모델은 sequence of interactions 중에 items 들간의 이동에 더 집중
    
    ⇒ 하지만 user의 short-term interests를 결정하는데 여전히 다른 요소들이 존재한다
    
- 유저 행동의 관련성은 매우 맥락적이고, 음악 소비 분야와 같이 하루의 시간대나 감정에 따라 short-term preferences가 결정된다

- **연구 초기에 아마존 데이터셋의 Beauty 카테고리에서 계절을 context로 정하고 조사 (Section 5.1 참고)**
    - 모든 유저에 대해, **같은 context를 공유하는 short-term 상호작용이** 다른 context로 인한 상호작용보다 **서로 더 유사함**을 발견
    - 반면, 주어진 user에 대해 **그 user의 long-term preference는 안정적**이었다 (e.g. 특정 브랜드나 타입을 선호)
    
    **⇒ user의 높은 만족도를 보장하기위해 각 user의 short-term preference는 그 user의 long-term preference와 특정 context의 공동 효과이다 =  연구의 주요 가설**
    

- **2가지 문제에 집중: long-term preferences modeling, context-aware short-term preferences modeling**
    - Long-term preferences modeling: 각 user의 구체적이고 안정적인 preferences에 집중
    - Context-aware short-term preferences modeling: long-term preferences와 특정 context 정보의 통합에 집중

- **연구의 일반적인 idea:** user의 short-term preferences는 그들의 안정적인 long-term preferences에 기반하여 어떤 context에 의한 변동의 결과
    - ex) Taylor Swift를 좋아하는 user는 하루 중의 다양한 시간이나 다양한 mood에 따라 Taylor Swift의 다른 스타일의 노래를 아마도 들을 것이다
    
- **연구의 동기: [[link]](https://arxiv.org/abs/1910.14238) Learning Disentangled Representations for Recommendation, NeurIPS 2017**
    - 다양한 concepts에 관한 user의 preferences를 캡쳐하는 향상된 VAE 제안 (향후 포스팅 예정)
    

- **LoCo-VAE**
    - long-term preferences: 모든 contexts에서의 user의 behavior records
    - short-term preferences: 각 context로부터 user의 선호도를 별도로 학습
    - 모든 contexts의 prior distribution은 Gaussian distribution으로 설정 ⇒ 이때 평균은 user의 long-term preference
    - user의 context-aware short-term preferences는 posterior distribution으로써 모델링

- **Main Contributions**
    - LoCo-VAE 제안 ⇒ long-term user preferences와 특정 context 영향의 공동 효과로서 context-aware short-term preferences를 모델링하는 통합된 프레임워크
    - VAE의 latent space에서 user의 long-term preferences에 기반한 prior를 통합하고, context 영향에 따라 short-term preferences를 fine-tune
        - 이때 long-term preferences를 얻기 위해 MLP를 사용하고, variational distribution으로부터 context-aware preferences를 얻음

<br>

# 2 Related Work
생략
# 3 Methodology

## 3.1 Notations and Problem Formulation

- set of $N$ users, set of $M$ items, $K$ contexts
- implicit feedback ⇒ user-item interaction is binary
- $x_{ui}^{(k)}=1$:  $k^{th}$ context에서 user $u$와 item $i$의 interaction이 존재
    - $X_u^{(k)}=\{ x^{(k)}_{ui}: x^{(k)}_{ui}=1 \}$
- input matrix: $X_u=[X_u^{(1)}, X_u^{(2)},..., X_u^{(k)}]\in \mathbb N^{K\times M}$
- 목표: context $k$에서 top-k items 추천

<br>

## 3.2 Model of LoCo-VAE

![PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5](https://user-images.githubusercontent.com/48899040/215712271-b44ef176-5177-4c37-aaae-d711bab1cc5f.png)

### Context-Aware Modeling

- latent representation $Z_u^{s_k}$는 short-term preference와 분리된 context-aware preference
- input $X_u^{(k)}$를 variational distribution $q_\phi(Z_u^{s_k}|X_u^{
(k)})$의 평균과 분산으로 mapping

### Long-Term Modeling

- prior distribution에서 얻는 것과 달리 (Giannis [9] 참고), context-aware preferences와 **동시에 최적화될 수 있도록 모델링**
    
    **⇒ MLP를 적용해서 CDAE의 인코더와 유사한 구조를 공유 (user-specific nodes)**
    
    > CDAE
    ![PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5 1](https://user-images.githubusercontent.com/48899040/215712478-41001291-fdb2-454c-b518-5101e326b553.png)
    - User Node: user-specific vector ⇒ 유저마다 고유한 값을 가지고 있는 노드
    - ❓질문: 구현할 때는 데이터셋의 고유한 user_id를 임베딩한 벡터를 사용하는것인가?
        
<br>

- input을 $Z_u^{(k)}$와 **동일한 dimension을 같도록** latent representation으로 변환
    - $Z_u^l=f_\theta^l(x_u^l), \text{where}~x_u^l=\sum_kX_u^{(k)}$
- 실제로 long-term과 context-aware preference 인코더는 **파라미터를 공유**함 $(\theta$가 $\phi$로 대체 가능) ⇒ 효과적으로 overfitting 완화

- context-aware와 long-term preferences의 두 분포를 합쳐서 short-term preferences $Z_u^{(k)}$를 나타내고 이를 decoder의 input으로
- decoder는 대응하는 context에서 item set에 대한 user의 확률 분포를 output ⇒ 가장 높은 확률의 k item을 추천 결과로 선택

<br>

## 3.3 Objective Function

![PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5 2](https://user-images.githubusercontent.com/48899040/215712723-f5788ade-9364-4185-a14b-b786f9641d65.png)

- 기존 Mult-VAE와 동일한 Loss

<br>

# 4 Experiments

## 4.1 Datasets

- 2개의 real-word user-item datasets
    - Million Musical Tweets dataset (MMTD)
    - Amazon consumption dataset (→ Beauty 카테고리로 선정)
- **time 피쳐를** context 정보로 선택
    - MMTD: periods of the day
    - Amazon Beauty: seasons
- 4 이상의 ratings을 가진 item에 대해 binarized, 최소 5개의 item에 대해 듣거나 구매한 유저

![PNG_%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%E1%85%B5 3](https://user-images.githubusercontent.com/48899040/215712817-f00eea13-b8ad-494c-93f5-ff58e4277abf.png)


## 4.2 Baselines

- Weighted Matrix Factorization (WMF)
- Neural Collaborative Filtering (NCF)
- Multinomial-Denoising Autoencoder (Mult-DAE)
- Multinomial-Variational Autoencoder (Mult-VAE)

## 4.3 Metrics

- 2개의 ranking-based metrics
    - **Hit Ratio (HR@R)**
        
        
    - **Normalized Discounted cumulative gain(NDCG@R)**
- test set의 각 positive item에 대해, user가 negative item으로 상호작용하지 않은 99개의 샘플링된 item과 pair를 만듦
- 각 user에 대해, 두 metric은 샘플링된 negative items와 positive items의 예측 순위를 실제 순위와 비교

## 4.4 Parameter Settings

- latent representation dimension d = 100
- MLP with hidden layer dimension = 200
- Loss → Regularization term $\beta$=0.5
- Adam optimizer
- learning rate = 0.001
- batch size of user = 500

<br>

# 5 Results and Analysis

## 5.1 Exploratory Analysis

- 연구 가설의 합리성을 증명하기 위해 Amazon Beauty Dataset을 조사
    - *가설: user의 short-term preference는 long-term preference와 특정 context의 joint effect*
- word2vec을 통해 user-item interaction record를 임베딩
    - 100차원으로 학습 ⇒ 상품 리뷰 임베딩의 평균을 계산
- 랜덤으로 두 set의 interactions을 선택 ⇒ 각각 400개의 record (계절당 100개의 record)
- 두 임베딩 사이의 cosine similarity를 계산하여 유사도를 구함 ⇒ heatmap으로 결과 표현

![Untitled](https://user-images.githubusercontent.com/48899040/215712869-8d224d9b-84cd-44b9-8148-2ce138df446e.png)


- 유사도 비교 방법 참고 → **Contextual and Sequential User Embeddings for Large-Scale Music Recommendation, RecSys 2020**
    
    [https://labtomarket.files.wordpress.com/2020/08/recsys2020.pdf](https://labtomarket.files.wordpress.com/2020/08/recsys2020.pdf)
    
- **동일한 context (=같은 계절) 를 공유하는 records가 더 높은 유사도를 보임**
- 가설 강조 → 동일한 context를 가진 user interactions은 비슷한 유사도를 공유한다
- Hansen [4] 에서도 음악 분야에서의 비슷한 결론을 보임

## 5.2 Performance Comparison

![Untitled 1](https://user-images.githubusercontent.com/48899040/215712923-773bb672-d759-4a6c-b2b4-07f38ffc0f18.png)

- LoCo-VAE의 상대적인 improvement
    - HR보다 NDCG에서 더 높다
    - MMTD 데이터셋보다 Beauty 데이터셋에서 더 높다
    
- **모든 baseline에 대한 LoCo-VAE의 개선은 두가지 측면에 기여**
    
    1) context에 의해 변하는 short-term preferences에 long-term preference를 결합 ⇒ context를 고려하지 않는 baseline들 보다 더 나은 성능
    
    2) context-aware preference와 함께 훈련된 encoder를 통해 long-term preference를 모델링 ⇒ context 없이 user의 short-term preference를 캡쳐할 수 있는 능력이 있다
    

## 5.3 Case Study

- 위의 2) 측면을 검증하기 위해, 랜덤으로 두 유저를 샘플링하고 낮과 밤의 context에서 그들의 추천 아이템을 가져와서 비교

![Untitled 2](https://user-images.githubusercontent.com/48899040/215712907-556cd6e3-a7ff-4d2e-8699-10c35b1b36fc.png)

- 서로 다른 context에서 동일한 user의 추천된 음악 스타일을 비교
    
    ⇒ LoCo-VAE가 밤에는 위로하는 음악을 추천, 반면에 낮에는 리듬이 더 강한 음악을 추천
    
    ⇒ 각 user의 short-term 관심사의 context-aware 영향을 반영
    
- 또한 두 user의 추천된 음악 스타일도 다르다 ⇒ user의 long-term preference를 반영
    - user B ⇒ 활기찬 음악
- **case study를 통한 LoCo-VAE의 장점**
    - 보다 정확한 추천
    - 추천의 해석 가능성
    
    ⇒ short-term preference를 long-term preference와 context-aware 영향의 공동 효과로 모델링함으로써 얻을 수 있다