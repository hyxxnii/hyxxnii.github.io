---
layout: post
title: "[논문리뷰] Linear Variational Autoencoder for Top-N Recommendation (IEEE, 2022)"
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

#### 논문 출처: [Linear Variational Autoencoder for Top-N Recommendation](https://ieeexplore.ieee.org/document/9760352)


이번 포스팅은 `Mult-VAE` ([논문 리뷰](https://hyxxnii.github.io/recommender%20system/2022/09/14/paper-review-VAE/)) 의 선형 버전인 Linear VAE를 제안한 논문을 읽고 정리한 것이다.


<br>

## Abstract

- VAE와 Mult-VAE는 multinomial likelihood와 ELBO의 KL divergence term에 대한 추가 하이퍼파라미터 $\beta$를 도입하여 큰 성능을 이끌었다
- 그러나 Mult-VAE는 **비선형 신경망을 인코더와 디코더로 사용**하여 유저-아이템 상호 작용 데이터를 인코딩하고 재구성 ⇒ **예측 정확도를 저하시키기 때문에 sparse 데이터 셋에서 불필요하다는 것을 증명**
- 또한 VAE 기반 협업 필터링 방법의 대부분의 변형에서는 **비정규화된 유저-아이템 상호 작용 데이터**를 사용 ⇒ **상호 작용 데이터의 학습 과정을 방해할 것이다**
- **본 논문에서는 implicit feedback에서 유저-아이템 상호 작용 데이터에 대한 추가 정규화를 고려하는 Mult-VAE의 선형 버전인 LVA(Linear Variaitional Autoencoder)를 제안한다**

<br>

## I. INTRODUCTION

- 비선형 신경망을 이용한 기존 VAE와 그에 대한 많은 변형 모델 ⇒ **sparse dataset의 추천시스템 성능을 향상시키는 데 필요한가라는 질문**
- 저자들이 Mult-VAE의 인코더와 디코더를 single-layer 선형 구조로 단순화했을 때 sparse dataset에서 성능이 더 향상된다는 것을 발견 (in Table 3) ⇒ 비선형 인코더, 디코더가 오히려 성능 떨어뜨린다

⇒ 생성 모델이고 원래 이미지 생성을 위해 설계된 VAE가 **sparse한 유저-아이템 상호 작용 데이터에 인코더와 디코더로 복잡한 비선형 신경망을 사용할 경우 CF task에 맞지 않는다고 주장!**

**📍 Motivation**

- LightGCN(Graph Convolution Network): GCN에서 상속된 feature transformation과 비선형 활성함수를 사용할 때 더 나쁜 결과를 얻는다 ⇒ sparse한 유저-아이템 상호작용 데이터를 처리하기 위해 복잡한 모델을 사용하면 성능이 저하되고 학습 프로세스가 왜곡될 수 있다
    - GCN의 2가지 특징, feature transformation과 nonlinear activation들이 협업필터링 성능에 크게 기여하지 않는다는 것을 확인 (오히려 이들을 추가했을 때 학습이 더 어려워지고 추천 성능이 더 떨어지는 경우도 있었음)
    - 그래서 GCN을 더 간소화해서 user-item interaction graph에서 선형적으로 전파
    
    **⇒ VAE도 훨씬 단순하게 학습하도록 동기 부여**


<br>

📍 **LVA(Linear Variational Autoencoder)**

비선형 활성함수가 없는 one-layer linear structure(= one-layer MLP)를 인코더와 디코더로 제안
  - 인코더와 디코더가 linear regression 모델과 유사
  - 다른 linear 방식도 더 좋은 추천 성능에 쓰일 수 있지만 (e.g., pure matrix multiplication without extra bias) future work로 남겼다
  - **이러한 간단한 선형 인코더와 디코더는 posterior collapse 문제를 완화하고 상호 작용 데이터를 더 잘 맞출 수 있도록하며, ranking accuracy를 더 높인다** (이후 실험에서 보여줌)
    

### Main contributions

- Top-N recommendation을 위해 **선형 인코더와 디코더**를 가진 단순화된 VAE 기반 모델 LVA를 제안
- LVA의 **선형 구조에 맞춘** 유저-아이템 상호 작용 데이터에 사용자와 항목에 대한 **추가 정규화를 채택**할 것
- LVA가 LightGCN 및 다른 복잡한 VAE 기반 변형 모델과 비교하여 **더 나은 또는 경쟁력 있는 성능** 달성
- 향후 효과적인 추천 알고리즘을 더 잘 설계하기 위해 **실험 및 이론적 분석** 제시

<br>

# II. PRELIMINARY

- notations, 문제 정의, Mult-VAE의 기본 소개


<br>


### A. notations

![Untitled](https://user-images.githubusercontent.com/48899040/215703919-f633471a-01b9-4e07-9aaa-5ac583d2d01d.png)


<br>

### B. Problem Definition

- Sparse binary implicit feedback setting가 주어질 때
- personalized recommender that can recommend the top-N items

<br>

### C. Basics of Mult-VAE

![Untitled 1](https://user-images.githubusercontent.com/48899040/215704121-8a9b5534-fc13-40e3-8fb1-0e5b60dbe54d.png)

![Untitled 2](https://user-images.githubusercontent.com/48899040/215704129-8172d8a3-3401-4475-b04d-4cd90229193b.png)


<br>

# III. LINEAR VARIATIONAL AUTOENCODER

## A. Linear Encoder and Decoder

- linear transformation을 가진 선형 디코더
    
    ![Untitled 3](https://user-images.githubusercontent.com/48899040/215704131-82ae1f83-ca2f-431c-be07-e4337d2e410d.png)
    
    - $W_\theta\in \mathbb R^{K\times|I|}$, $b_\theta \in \mathbb R^{|I|}$: 디코더의 weight와 bias, $K$는 latent dimension
    - $z_u\in \mathbb R^K$: 유저 $u$의 latent representation
  
- 최근 연구에서는 유저와 아이템 간의 상호작용을 모델링할 때 단순한 dot product가 복잡한 MLP보다 낫다는 것을 보여주었기에 타당하다
    - MLP는 충분한 hidden states를 가지고 있는 compact set에서 모든 연속 함수를 근사할 수 있는 universal approximator ⇒ MLP로 dot product를 근사하는 것은 큰 모델 용량과 훈련 셋이 필요하기때문에 어렵다

- 인코더: variational distribution $q_\phi(z_u|x_u)$의 평균과 분산을 계산하는 데 사용되는 두 함수로 구성됨
    
    ![Untitled 4](https://user-images.githubusercontent.com/48899040/215704136-25ce1952-a58c-445d-96e0-4d3148ce1059.png)
    
    - $W_\mu\in \mathbb R^{|I|\times K}$, $b_\mu \in \mathbb R^{K}$: mean function의 weight와 bias
    - $W_\sigma\in \mathbb R^{|I|\times K}$, $b_\sigma \in \mathbb R^{K}$: variance function의 weight와 bias

![Untitled 5](https://user-images.githubusercontent.com/48899040/215704140-d2cf02de-0e01-4a7a-9056-524648cb4a58.png)

![Untitled 6](https://user-images.githubusercontent.com/48899040/215704143-3cc61c81-0f0b-49f3-9323-22f7d64e2809.png)

- ELBO식은 기존 Mult-VAE와 동일
    
    ![Untitled 7](https://user-images.githubusercontent.com/48899040/215704146-2a197110-2ee2-4d2c-9d37-88863a93b0da.png)

    - 차이점: variational distribution $q_\phi(z_u|x_u)$의 파라미터가 (평균, 분산) 식 (5)을 사용하여 계산된다는 것
    - 나머지 학습과정도 동일
    

<br>

## B. Normalization

- 유저-아이템 상호작용 matrix의 행, 열 벡터 모두에 대한 정규화 추가
    - 열 벡터에 대한 추가 정규화를 추가하는 이유: 행렬 $X$의 item의 인기가 추천시스템 성능에 영향을 줄 수 있기 때문 ⇒ 직관적으로 인기 아이템이 과도하게 추천되고, 덜 인기 있는 아이템은 거의 추천되지 않는 문제를 피할 수 있다 (fairness를 고려)
- denote
    - all one column vector: $1$ ($1$ can be any dimension)
    - The user degree matrix: $D_U=Diag(X\cdot1)$
    - The item degree matrix: $D_I=Diag(1^T\cdot X)$
- 정규화된 user-item 상호작용 matrix
    
    ![Untitled 8](https://user-images.githubusercontent.com/48899040/215704153-3541f916-be02-4219-bd38-85375cd2755f.png)
    
    - 유저-아이템 상호 작용 matrix의 행과 열 벡터 모두에 정규화를 추가하면 Section $IV-D$에 나타난 바와 같이 LDA가 훈련 데이터에 더 잘 적합하게 된다는 것을 발견했다
    

<br>

## C. Interpretation

- 선형 인코더와 디코더를 사용한 LVA의 효과를 간략하게 제시

1. 인코더와 디코더의 단순한 선형구조 → 이미지 데이터보다 비교적 더 단순한 유저-아이템 상호 작용 matrix에 더 잘맞는다
    - [15]에서 언급한 바와 같이, linear VAE의 ELBO는 local maixma를 introduce하지 않고, variational inference를 통해 훈련하면 주성분 방향에 해당하는 identifiable global maximum을 recover한다
    - As is mentioned in [15], the ELBO of linear VAE does not introduce local maxima and training a linear VAE with variational inference recovers an identifiable global maximum corresponding to the principle component directions.
    
    **⇒ 즉, linear VAE는 VAE에서의 posterior collapse 문제를 완화할 수 있다**
    
2. LVA는 one-layer LightGCN과 밀접한 관련이 있다
    - 인코더의 weight 행과 디코더의 weight 열을 두 세트의 아이템 임베딩(two sets of item embeddings)으로 간주할 때
    - 인코더는 one-layer graph convolution, 디코더는 각각의 유저의 아이템 선호도 점수를 계산하는 데 사용되는 dot product에 해당
    - 차이점: LVA의 인코더와 디코더에는 biases term이 있고, LDA에는 **self-connection이 없다**는 것
        
        **⇒ LVA가 이웃을 활용해 유저의 representations을 강화하고, 단순하지만 효과적인 dot product를 유저와 아이템의 상호작용 function으로 적용하고 있음을 나타냄**
        
    - 또한 LVA에 사용된 베이지안 추론은 sparse 데이터셋에 더 적합하게 만든다

<br>

# IV. EXPERIMENTS

## A. Experimental Settings

*1) Datasets and Evaluation Metrics*

![Untitled 9](https://user-images.githubusercontent.com/48899040/215704159-43c0d6ad-325a-48c5-a891-ce5d6cf5689e.png)

- evaluation metrics: recall@20, ndcg@20

*2) Baseline Methods*

- 2개의 전통적인 추천시스템 방법 ⇒ **LightGCN, variants of VAE-based recommendation methods**
    - LightGCN
        - NGCF (Neural Graph Collaborative Filtering) 보다 더 개선된 성능
        - NGCF는 NCF와 같은 많은 신경망 추천 방법보다 우수한 것으로 나타남
    - VAE-based variants
        - **Mult-VAE, EVCF(Enhancing VAEs for Collaborative Filtering), RecVAE**
        
- The baseline method
    - Item-Pop
    - MF-BRP
    - LightGCN
    - Mult-VAE
    - EVCF
    - RecVAE
    

*3) Hyperparameter Settings*

- 공정한 비교 ⇒ LightGCN의 embedding size, LVA의 latent dimension는 64로 고정
- Mult-VAE: 600 → 200 → 600 (기존 모델 아키텍쳐)
- RecVAE에 따라 우리의 모델도 hidden dimension=600, latent dimension=200으로 설정
- Adam optimizer
- learning rates: 0.001
    - RecVAE, EVCF: $5\cdot 10^-4$

<br>

## B. Performance Comparison

![Untitled 10](https://user-images.githubusercontent.com/48899040/215704165-200ca33f-fedf-453c-9e10-3e2996a523f5.png)

- 특히 Mult-VAE 성능보다 큰 개선을 보였음
    - recall@20에 대해선 15.3%, ndcg@20에 대해선 18.0%
- Yelp2018, Amazon-Book, Video-Games 데이터셋에 대해 LightGCN보다 더 나은 성능
    
    **⇒ sparse 데이터셋과 linear encoder/decoder, normalization on the user-item interaction matrix에서 LVA의 효과를 증명한 결과**
    
<br>

## C. Ablation Study

- LVA-norm: 제안된 정규화를 유저-아이템 matrix이 아닌, **유저에 대해서만 정규화를 적용한 변형**
    - 비선형 구조를 가진 VAE 기반 변형 모델을 능가
- LVA의 성능은 LVA-norm 보다 더욱 향상된 퍼포먼스 ⇒ 유저와 아이템 모두에 대해 정규화하는 것의 효과 검증

<br>

## D. Discussion

![Untitled 11](https://user-images.githubusercontent.com/48899040/215704175-f97a68c6-3381-4c8b-a36f-1fbe0c635227.png)

- reconstruction error와 KL divergence에 관한 학습 절차 그래프
- LVA가 더 낮은 reconstruction error ⇒ training data를 더 잘 fit할 수 있고, local minima를 피할 수 있다
- 더 높은 KLD 값 ⇒ posterior collapse를 겪을 가능성이 더 낮다
    - prior 분포와 inference 분포가 최대한 일치하도록 KLD 값을 minimize *(⇒ 근데 개인적으로 이렇게 해석하는 것은 위험한 발언으로 보임..)*
    

![Untitled 12](https://user-images.githubusercontent.com/48899040/215704182-81a2ca2d-0ef0-45a1-9a57-dc2b61a39a08.png)

- 모든 test user에 대해 top 20에 추천되는 아이템 빈도수를 기준으로 순위를 매긴 그래프
- LVA가 LVA-norm보다 더 자주 long-tail 아이템을 추천할 수 있음을 보여준다
    - 아이템에 대한 추가 정규화의 효과 입증

<br>

# V. RELATED WORKS
생략

## *A. VAE-based CF methods*

## *B. Latent Factor models*

<br>

# VI. CONCLUSION

- Mult-VAE의 단순화된 버전 제안 ⇒ LVA
- sparse한 상호 작용 데이터를 학습하는 데 도움이 되도록 유저와 아이템 모두에서 user-item interaction matrix에 대한 정규화 사용 제안