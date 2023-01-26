---
layout: post
title: "[논문리뷰] Variational Autoencoders for Collaborative Filtering (WWW, 2018)"
date: 2022-09-14
categories:
- Recommender System
tags:
- paper review
- recommender system
- variational autoencoder
use_math: true
comments: true
---

#### 논문 출처: [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/pdf/1802.05814.pdf)


추천시스템에 VAE(Variational AutoEncoder)를 적용한 논문인 "Variational Autoencoders for Collaborative Filtering"에 대해 정리하였다. `Mult-VAE` 라고도 불리는 이 논문은 현재도 추천시스템 모델을 연구하는데 많이 응용되고 있다.

<br>

💡 ***Contribution***

1. 추천시스템에 **Variational AutoEncoder(VAEs)** 도입
2. 표준 VAEs의 **Decoder**에 베르누이 분포가 아닌 **Multinomial** 분포(다항 분포)를 가정
3. **ELBO의 KL 식**(regularization)에 파라미터 $\mathbb\beta \in [0,1]$ 도입

---
<br>


# Abstract

본 논문은 암시적 피드백(implicit feedback)에 대한 **협업 필터링(Collaborative Filtering, CF)에 variational autoencoders(VAEs)를 확장**하였다. 이러한 **비선형 확률 모델**은 여전히 CF 연구에 지배적인 linear factor 모델의 제한된 모델링 능력을 넘어서게 하였다.

다음의 세 가지를 통해 논문에서 제안하는 `Mult-VAE`가 기존의 VAE와 다른 점을 설명한다.

- **multinomial likelihood를 가진 생성 모델**을 소개하고 파라미터 추정을 위한 **Bayesian inference**를 사용한다.
- 다른 **정규화 파라미터**를 도입하여 더욱 경쟁력 있는 성능을 보였다.
- **annealing**을 사용한 **파라미터 튜닝**을 사용한다.

제안된 접근 방식은 real-world datasets에 대해 최근에 제안된 신경망 접근 방식을 포함한 몇몇 SOTA baselines보다 상당히 뛰어난 성능을 보인다. 또한 multinomial likelihood와 잠재 요인 협업 필터링에서 보통 사용되는 다른 likelihood 함수를 비교하는 확장된 실험을 통해 더 유리한 결과를 보였다.

<br>

# 1 Introduction

전형적인 추천시스템은 유저들이 아이템과 어떻게 상호 작용하는지 관찰한다. 이때 유저-아이템 상호 작용 데이터를 사용하여 유저에게 그들이 좋아할 만한 이전에 보지 못한 아이템을 추천하고자 한다.

CF는 추천시스템에서 가장 널리 적용되는 접근 방식 중 하나이다. CF는 유저와 아이템 간의 유사성 패턴을 발견하고 활용하여 유저가 어떤 아이템을 선호할 것인지 예측한다. 잠재 요인 모델(Latent Factor Model)은 단순하고 효율적이여서 여전히 CF 연구에 지배적이다. 그러나, 이러한 모델들은 **본질적으로 선형이기 때문에** **모델링 능력을 제한**한다. 이전 연구에서는 **선형 잠재 요인 모델에 정교하게 조작된 비선형 features를 추가하는 것**이 추천 성능을 크게 향상시킬 수 있다는 것을 증명했다. 최근에는 **CF에 신경망을 적용**하는 연구가 증가하고 있다.

본 논문에서는 **암시적 피드백에 대해 협업 필터링에 variational autoencoders(VAEs)를 연장**하였다. VAEs는 이미지 모델링과 생성에 광범위하게 연구되어왔음에도 불구하고, 추천시스템에 VAEs를 적용하는 연구는 적었다. 따라서 VAEs는 선형 잠재적 요인 모델을 일반화하고, 대규모 추천 데이터 세트에서 신경망에 의해 구동되는 비선형 확률적 잠재-변수 모델을 탐구할 수 있게 한다. 

추천시스템에 VAEs를 적용하여 SOTA 결과를 얻기 위해 다음의 두 가지를 제안한다.

1. **데이터 분포에 multinomial likelihood 사용**
    
    이러한 단순한 선택이 일반적으로 사용되는 Gaussian 및 logistic likelihood보다 모델이 더 뛰어난 성능을 실현하게 한다.
    
2. **지나치게 정규화(over-regularized)되었다고 생각하는 Standard VAE의 목적함수를 재해석하고 조정**

<br>

# 2 Method

이 연구에서는 암시적 피드백에 대한 학습을 고려한다.

- $u\in \{1, ...~,U\}$: 유저 인덱스
- $i\in \{1, ...~,I\}$: 아이템 인덱스
- $X\in \mathbb N^{U\times I}$: **click matrix** ⇒ **유저-아이템 상호작용 행렬**
- $x_u=[x_{u1}, ...,x_{uI}]^T\in \mathbb N^I$: **각 아이템 $i$에 대한 유저 $u$의 클릭 횟수**를 나타내는 **bag-of-words vector**

 단순함을 위해 click matrix를 binary화하였다.

<br>

## 2.1 Model

본 논문에서 고려하는 생성 과정은 *deep latent Gaussian model* 과 비슷하다.

각 유저 $u$에 대해, 모델은 **prior standard Gaussian(정규분포)으로부터** 잠재 표현 $z_u$를 샘플링한다. 잠재 표현 $z_u$은 $I$에 대한 확률 분포 $\pi(z_u)$를 생성하기 위해 비선형 함수 $f_\theta(\cdot)\in\mathbb R^I$를 통해 변형된다.

이때 $\pi(z_u)$는 click history $x_u$가 도출되는 것으로 가정한다.

![Untitled](https://user-images.githubusercontent.com/48899040/214767475-ba10c8d0-8c0f-4c27-bc6c-443df4c3d5c1.png)

- (부가설명) *표준 VAE는 encoder에서는 gaussian 분포, decoder은 bernouii 분포를 가정하는데
논문에서 제안하는 VAE는 decoder 부분이 bernouii가 아닌 multinomial 분포를 따른다고 하는 것 ⇒ 즉 생성되는 $x_u$는 multinomial 분포로부터 생성된다.*
    
    <img src="https://user-images.githubusercontent.com/48899040/214767249-a712787d-6381-4663-8027-07005c2b1076.png" width="70%" height="70%">


<br>

비선형 함수 $f_\theta(\cdot)$는 파라미터 $\theta$를 가지는 multilayer perceptron (=FCNN)이다. 아이템셋 전체에 대한 확률 벡터 $\pi(z_u)\in\mathbb S^{I-1}$를 생산하기 위해 softmax 함수를 통해 정규화된다.

유저의 전체 클릭 수 $N_u=\sum_ix_{ui}$가 주어지면, **관측된 bag-of-words 벡터 $x_u$는 확률 $\pi(z_u)$를 가지는 multinomial distribution으로 부터 샘플링된다고 가정한다.** 

이러한 생성 모델은 잠재 요인 모델을 일반화하는데, $f_\theta(\cdot)$을 선형적으로 설정하고 Gaussian likelihood를 사용하여 고전적인 matrix factorization을 recover할 수 있다.

유저 $u$에 대한 **log-likelihood** (잠재 표현에 따른 조건) 는 아래와 같다.

![Untitled 2](https://user-images.githubusercontent.com/48899040/214767262-9b6dbbd1-5446-47ac-8f55-d03533db53f7.png)

click matrix(식 2)의 likelihood는 $x_u$의 0이 아닌 엔트리에 확률 질량(probability mass)을 가한 모델을 보상한다. 하지만 $\pi(z_u)$의 합이 반드시 1이 되어야하기 때문에 모델은 확률 질량의 제한된 budget을 가진다. 


**따라서 모델은 클릭될 가능성이 높은 아이템들에 더 많은 확률 질량을 지정해주어야 한다.**  일반적으로 추천시스템을 평가하는 **top-N ranking loss**에서 좋은 성능을 보일 것이다.

<br>

📍 **Gaussian, Logistic 함수**

잠재 요인 협업 필터링에 사용되는 두 가지 likelihood 함수(Gaussian, Logistic) 를 통해 논문에서 제안하는 방법과 비교하였다.

- $f_\theta(z_u)\equiv [f_{u1}, ...~,f_{uI}]^T$: generative 함수 $f_\theta(\cdot)$의 출력
- **유저 $u$에 대한 Gaussian log-likelihood**
    
    ![Untitled 3](https://user-images.githubusercontent.com/48899040/214767266-b3f512fe-e7b8-468d-8fce-af3101fabff8.png)
    
    - “confidence” weight: $c_{x_{ui}}\equiv c_{ui}$ where $c1 > c0$
        - 대부분의 click data에서 관찰된 1보다 훨씬 많은 관찰되지 않은 0의 균형을 맞추기 위해 사용
    - 이는 unweighted Gaussian likelihood과 negative sampling을 가진 모델을 훈련하는 것과 동일하다.
    
- **유저 $u$에 대한 log-likelihood**
    
    ![Untitled 4](https://user-images.githubusercontent.com/48899040/214767270-0976de16-8e0d-4198-a8aa-f6f7759fd563.png)
    
    - $\sigma(x)=1/(1+exp(-x))$: logistic 함수

Section 4에서 비교해보자.

<br>

## 2.2 Variatioanl inference
식 1의 생성 모델을 학습하기 위해, $f_\theta(\cdot)$의 파라미터인 $\theta$를 추정해야한다. 이를 위해 **variational inference(변분 추론)**를 사용하여 각 data point마다 다루기 힘든 **posterior 분포 $p(z_u|x_u)$를 근사할 필요**가 있다.

Variational inference는 다루기 힘든 실제 posterior을 더 단순한 variational 분포 $q(z_u)$로 근사한다. **논문에서는 $q(z_u)$를** fully factorized (diagonal) Gaussian distribution**으로 설정하였다.**

![Untitled 5](https://user-images.githubusercontent.com/48899040/214767271-f46dce73-3b3d-4ec2-9f6b-ce67aabeb45d.png)
**variational inference의 목적은 Kullback-Leiber divergence $KL(q(z_u)||p(z_u|x_u))$를 최소화하도록 파라미터 $\{\mu_u, \sigma^2_u \}$를 최적화하는 것이다.** 

<br>

### 2.2.1 Amortized inference and the variational autoencoder

variational inference를 통해 $\{\mu_u, \sigma^2_u \}$를 최적화할 파라미터의 수는 dataset의 유저 수와 아이템 수에 따라 증가한다. 이것은 수백만명의 유저와 아이템을 가진 상업 추천시스템에 대한 병목 현상이 될 수 있다. VAE는 개별 variational 파라미터를 inference 모델이라고 보통 불리는 아래의 데이터 의존 함수로 교체하고,

![Untitled 6](https://user-images.githubusercontent.com/48899040/214767273-b891970f-c4b5-4ad6-8ad1-5bb647daed34.png)

- $\mu_\phi(x_u)$와 $\sigma_\phi(x_u)$가 모두 $K$차원의 벡터인 $**\phi$에 의해 파라미터화된 inference 모델**

variational 분포를 아래와 같이 설정한다.

![Untitled 7](https://user-images.githubusercontent.com/48899040/214767275-d490eaf8-25d1-44d0-bd05-a68db5bc5765.png)
즉, **입력으로써 관측된 데이터 $x_u$를 사용하는 inference 모델은 variational 분포 $q_\phi(z_u|x_u)$에 해당하는 varitional 파라미터를 출력하는데, 이 $q_\phi(z_u|x_u)$는 최적화될 때 다루기 힘든 posterior $p(z_u|x_u)$를 근사한다.**

- 구현 시에는 inference 모델은 variational 분포의 variance의 log값이 출력된다.
그림 2에서, $q_\phi(z_u|x_u)$와 생성 모델 $p_\theta(x_u|z_u)$를 함께 넣으면, autoencoder와 유사한 뉴럴 아키텍처가 만들어진다. 따라서 variational autoencoder라고 불린다.

<br>

📍 **Learning VAEs**

variational inference을 사용하여 잠재 변수를 사용할 때, 표준 VAE처럼 **데이터의 marginal likelihood(로그 한계 우도)를 낮출 수 있다(lower-bound)**. 

![Untitled 8](https://user-images.githubusercontent.com/48899040/214767277-26353609-b1ba-4129-b263-099a987699cd.png)

이를 일반적으로 **Evidence Lower Bound (ELBO)** 라고 불린다. 

- **ELBO**: $\theta$와 $\phi$ 에 대한 함수

$z_u$를 $q_\phi$로부터 샘플링($z_u\sim q_\phi$)함으로써 **ELBO의** unbiased한 추정을 얻을 수 있고, 그것을 최적화하기 위해 확률적 경사 하강법(SGD)를 수행할 수 있다.

이때 **reparametrization trick**을 통해 $**\phi$에 관한 gradient가 샘플링된 $z_u$를 통해 역전파된다.**

- $\epsilon\sim N(0,I_K)$를 샘플링하고 $z_u=\mu_\phi(x_u)+\epsilon \odot \sigma_\phi(x_u)$를 재파라미터화한다.

이러한 학습 절차는 알고리즘 1에 요약되어 있다.

<img src="https://user-images.githubusercontent.com/48899040/214767279-144c509f-fbcf-4292-a559-8c1541e3fbf1.png" width="70%" height="70%">

<br>

### 2.2.2 Alternative interpretation of ELBO

식 5에서 정의된 ELBO를 다른 관점에서 볼 수 있다.

**첫 번째 term**은 **(negative) reconstruction error**로 해석될 수 있고, 반면 **KL term은 regularization** (정규화)로 보여질 수 있다. 본 논문에서는 정규화의 강도를 제어하는 파라미터 $\beta$를 통해 **ELBO**를 확장하여 둘의 trade-off를 다루었다.

![Untitled 10](https://user-images.githubusercontent.com/48899040/214767282-fe1f53ee-21d1-446b-b782-b19072c637c9.png)

- $\beta$: 표준 VAE에서의 regularization term의 over-regularization을 제어하는 파라미터

**ELBO의 정규화 식은 data를 얼마나 잘 fit할 수 있는지와 학습하는 동안 approximate posterior가 얼마나 prior에 가깝게 머무르는지 사이에 대한 trade-off를 도입한다.**

<br>

📍 **Selecting $\mathbb\beta$**

$\beta$ 설정을 위해 단순한 휴리스틱을 제안한다. ⇒ $\beta=0$에서 훈련을 시작하고, 점점 $\beta$를 1까지 증가시킨다.

$θ, \phi$ 에 대한 gradient의 업데이트에 걸쳐 선형적으로 **KL term을 천천히 annealing하고,** *(anneal: 천천히 식혀 강화시키다),* **성능이 최고조에 달했을 때 가장 좋은 $\beta$ 를 기록**한다.

![Untitled 11](https://user-images.githubusercontent.com/48899040/214767451-1b6d27ff-a866-4c6d-815a-772070e915bc.png)

Figure 1은 basic idea를 설명한다. (데이터셋 전체에서 동일한 추세를 일관적으로 관찰) 

KL annealing이 없는 방법(blue solid)과 $\beta=1$까지 KL annealing이 있는 (green dashed, 약 80 에폭에서 $\beta$가 1에 도달) validation ranking metric을 그린 것이다. 여기서 볼 수 있듯이, KL annealing이 없는 방법은 좋은 성능을 보이지 못했다. annealing을 추가했을 때, validation 성능은 훈련이 진행됨에 따라 우선 성능이 증가했다가, $\beta$가 1에 가까워질수록 annealing을 전혀 하지 않는 것보다 약간 더 나은 값으로 떨어진다.

peak validation metric으로 최상의 $\beta$를 식별한 후에, 동일한 annealing schedule을 가진 모델을 재훈련 시킬 수 있지만, $\beta$가 peak에 도달한 이후에는 증가하는 것을 멈춘다 (red dot-dahsed)

이는 훈련 전체에 걸쳐 최상의 $\beta$ 값을 유지하는 것이 약간 더 나은 결과를 주는 것으로 해석할 수 있다.

<br>

### 2.2.3 Computational Burden
논문의 접근 방식에 따른 계산상의 어려움은 아이템의 수가 많을 때, 정규화(normalization)을 위해 모든 아이템의 대해 예측 계산이 필요하기 때문에 **multinomial 확률 $\pi(z_u)$를 계산하는 것이 비용적으로 부담이 될 수 있다.**

50K개 아이템보다 적은 medium-to-large 데이터셋에 대한 실험에서는 (Section 4.1) 아직까지 계산적 병목 현상으로 발생하지 않았다. 만약 더 큰 아이템 셋으로 작업할 때 병목 현상이 된다면, Botev et al.이 제안한 간단하고 효과적인 방법을 쉽게 적용할 수 있다.

<br>

## 2.3 A taxonomy of autoencoders

저자들은 `Mult-VAE`에 추가적으로 multinomial likelihood를 가진 denoising autoencoder (`Mult-DAE`)를 연구하여 성능을 비교하였다.

![Untitled 12](https://user-images.githubusercontent.com/48899040/214767455-e68a9c1d-7a3d-471c-98ea-c3e9284498d8.png)

Figure 2는 Autoencoder의 다양한 변형에 대한 통일된 view를 나타내는 그림이다.

각각에 대해, 모델을 구체화하고(*표본 화살표는 샘플링 작업을 나타냄*), 파라미터 추정에 사용되는 학습 objective을 설명한다.

- **Figure 2a - Autoencoder**
    - 식 7에서와 동일한 objectvie를 가진 입력을 재구성하도록 학습된다.
 
- **Figure 2b - Denoising Autoencoder**
    - autoencoder의 입력에 노이즈를 추가하는 것 (또는 중간 hidden representation)
    - 학습 objective는 autoencoder의 것과 동일하다.
    - `Mult-DAE`는 이 모델 class에 속한다.
  
- **Figure 2c - Variational Autoencoder**
    - $\phi$로 파라미터화된 inference 모델을 사용하여 근사 variational 분포의 평균과 분산을 생성한다.
    - 학습 objective는 식 6에서 주어진다.
    - $\beta$를 1로 설정하면 기존 VAE와 동일한 loss 식이 된다.
    - `Mult-VAE`는 $\beta\in[0,1]$을 가진 VAEs를 학습하는 것과 대응된다.

<br>

## 2.4 Prediction

`Mult-VAE` 와 `Mult-DAE` 둘다 동일한 방식으로 예측을 한다. 유저의 click history $x$가 주어지면, 정규화되지 않은 예측 multinomial 확률 $f_\theta(z)$에 기반하여 모든 아이템의 순위를 매긴다.

$x$에 대한 잠재 표현 $z$는 다음과 같이 구성된다.

- `Mult-VAE`: variational 분포 $z=\mu_\phi(x)$의 평균을 가져온다.
- `Mult-DAE`: 출력 $z=g_\phi(x)$를 가져온다.

<br>

# 3 Related Work

(생략)

# 4 Empirical Study
source code: [GitHub](https://github.com/dawenl/vae_cf) 

## 4.1 Datasets

다양한 도메인으로부터 3개의 medium-to large-scale 유저-아이템 소비 데이터를 연구했다.

- **MovieLens-20M (ML-20M)**: 영화 추천 서비스로부터 수집한 유저-영화 평점 데이터. 평점 4 이상만 유지하여 명시적 데이터를 이진화(binarize)하고,*최소 5편의 영화를 시청한 유저들만 유지한다.
- **Netflix Prize (Netflix)**: [Netflix Prize](http://www.netflixprize.com/)의 유저-영화 평점 데이터. ML-20M과 유사하고 평점 4 이상만 유지하여 명시적 데이터를 이진화하고, 최소 5편의 영화를 시청한 유저들만 유지한다.
- **Million Song Dataset (MSD)**: Million Song Dataset 일부로 공개된 유저-곡 재생 횟수가 포함된 데이터. 재생 횟수를 이진화하고, 유저의 청취 기록에서 최소 20곡을 이상의 노래가 있는 유저와 최소 200명 이상의 유저가 청취한 곡만 유지한다.

<img src="https://user-images.githubusercontent.com/48899040/214767456-7b3fd057-1e87-4ff7-bde9-27e42a372dc1.png" width="50%" height="50%">


Table 1은 전처리 후 데이터셋의 속성을 나타낸다. 상호작용은 0이 아닌 엔트리이고, % of interactions는 유저-아이템 click matrix X의 밀집도를 나타낸다. # of held-out users는 첫 번째 행의 총 사용자 수 중 검증/테스트 유저 수이다.

<br>

## 4.2 Metrics

2개의 ranking-based metrics를 사용한다.

1. **$Recall@R$**
2. **$NDCG@R$**: truncated normalized discounted cumulative gain
    
    <img src="https://user-images.githubusercontent.com/48899040/214767457-d2728b0b-f5a7-49a7-a8da-1230a370fb4b.png" width="50%" height="50%">
    

각 유저에 대해, 두 metrics 모두 held-out된 아이템의 rank와 실제 rank를 비교했다. `Mult-VAE`와 `Mult-DAE`에 대해, 정규화되지 않은 multinomial 확률 $f_\theta(z)$를 정렬하여 예측 rank를 가져왔다. 

$Recall@R$은 첫번째 $R$ 내에서 rank가 매겨진 모든 아이템을 동등하게 중요하게 고려하는 반면, $NDCG@R$은 낮은 rank보다 더 높은 ranks의 중요성을 강조하기 위해 단조롭게 증가하는 discount를 사용한다. 

- $\omega(r)$: rank $r$의 아이템
- $\mathbb I[\cdot]$: 지시 함수(indicator funciton)
- $I_u$: 유저 $u$가 클릭한 held-out 아이템 셋

### Recall@R for user $u$

![Untitled 15](https://user-images.githubusercontent.com/48899040/214767459-6219c1c2-6335-47b2-8822-b94282800c1e.png)

분모의 표현은 $R$의 최솟값과 유저 $u$가 클릭한 아이템의 수를 나타낸다. 이것은 $Recall@R$을 최댓값이 1이 되도록 정규화하고, 이는 상위 $R$의 위치에 있는 모든 관련 아아템의 rank를 매기는 것에 해당한다.

### NDCG@R

![Untitled 16](https://user-images.githubusercontent.com/48899040/214767460-2f92cacc-4601-4366-af7d-34acda5a402a.png)

Truncated discounted cumulative gain (DCG@R)은 다음과 같다.

$NDCG@R$은 가능한 최선(possible best)의 DCG@R로 나눈 후 $[0,1]$로 선형 정규화된 DCG@R로, 모든 held-out 아이템이 상위에 rank된다.

<br>

## 4.3 Experimental setup

모든 유저를 훈련/검증/테스트 셋으로 분리한 후, 훈련셋 유저의 전체 click history를 사용하여 모델을 훈련한다.

평가를 위해서는 모델에 필요한 유저-레벨 표현을 학습하고, held-out 유저의 보지 못한 click history의 나머지를 모델이 얼마나 잘 ranking 할 수 있는지 본다. 이때 검증셋 유저에 대해 $NDCG@100$으로 평가하여 모델의 하이퍼파라미터와 아키텍쳐를 결정하였다. `Mult-VAE`와 `Mult-DAE` 둘 다에 대해, 생성 모델 $f_\theta(\cdot)$과 추론 모델 $g_\phi(\cdot)$을 대칭으로 유지하고 0-2개의 hidden layers를 가진 multilayer perceptron(MLP)를 실험했다. 

**잠재 표현 $K$의 차원은 200, hidden layer는 600까지 설정했다.**

- ex) 전체 아이템 수 $I$에 대해 1-hidden-layer MLP 생성 모델이 있는 `Mult-VAE`/`Mult-DAE` 아키텍쳐는 $[I → 600 → 200 → 600 → I]$ 이 된다.

또한 layer가 더 깊어질수록 성능이 증가하지 않았고, 최상의 성능을 내는 아키텍쳐는 0 또는 1개의 hidden layer를 가진 MLPs라고 한다. layers 사이에 활성 함수로는 tanh를 사용하였다. (in Mult-DAE)

`Mult-VAE`의 경우, $g_\phi(\cdot)$의 출력이 Gaussian 랜덤 변수의 평균과 분산으로 사용되기때문에 출력에 활성 함수를 적용하지 않았다. 따라서, 0개의 hidden layer MLP인 `Mult-VAE`는 사실상 로그-선형(log-linear) 모델이다. Section 2.2.2에 설명된 절차에 따라 Mult-VAE의 정규화 파라미터 $\beta$를 튜닝했다.

- `Mult-VAE`와 `Mult-DAE` 모델 설정
    - 입력 layer에 0.5 확률의 dropout
    - 0.01의 weight decay (Mult-DAE에만)
    - batch size=500
    - Adam optimizer
    - ML-20M은 200 epochs, 나머지 데이터셋은 100 epochs

<br>

## 4.4 Baselines

### Weighted matrix factorization (WMF)

- linear low-rank factorization model
- Alternating Least Squares (ALS)로 훈련
- click matrix에서 모든 0에 대한 가중치를 1로 설정하고, 검증 유저에 대한 $NDCG@100$을 평가하여 잠재 표현 차원  $K\in\{100, 200\}$뿐만 아니라 $\{2, 5, 10, 30, 50, 100\}$ 중에서 click matrix에 있는 모든 1에 대한 가중치를 조정

### SLIM

- 제한된 $l_1$-정규화된 최적화 문제를 해결하여 희소 아이템 간 (sparse item-to-item) 유사성 행렬을 학습하는 선형 모델
- $\{0.1, 0.5, 1, 5\}$ 에 대해 정규화 파라미터 모두를 grid-search, $NDCG@100$으로 평가
- MSD 데이터셋에 대해선 SLIM을 평가하지 않았음 ⇒ 데이터셋이 너무 커서 시간 상의 문제로 완료 X (Netflix 데이터셋에 대해서는 병렬화된 grid-search가 약 2주 걸렸다.)
- SLIM의 더 빠른 근사치로도 경쟁력있는 성능 X

### Collaborative denoising autoencoder (CDAE)

- 입력에 유저별 잠재 요소를 추가하여 표준 DAE를 보강한 모델
- SGD 훈련의 (유저, 아이템) 엔트리의 subsampling 전략을 `Mult-VAE 및 `Mult-DAE`에서 그랬던 것처럼 유저-레벨 subsampling으로 변경 ⇒ 더 안정된 수렴과 더 나은 성능
- bottleneck layer의 차원은 200까지 설정하고, 원래 논문에서 negative sampling을 이용한 sqaure loss(가중 손실)과 동등한 weighted square loss(가중 제곱 손실)을 사용
- 출력 layer뿐만 아니라 bottleneck layer에서도 tanh 활성 함수를 적용
- weight decay를 적용하여 과적합을 제어 ⇒ validation $NDCG@100$을 검사하여 $\{0.01, 0.1,...,100\}$에 대한 weight decay 파라미터를 선택

### Neural collaborative filtering (NCF)

- 유저와 아이템의 잠재 요소 사이에서 신경망을 통해 비선형 상호작용을 탐구
- 저자들이 제공한 공식 소스 코드를 사용하였지만 본 논문에서 사용한 데이터셋에 대해 경쟁력있는 성능 X
- pre-training을 포함하거나 포함하지 않은 가장 최고의 성능을 낸 하이브리드 NeuCF 모델을 비교

<br>

## 4.5 Experimental result and analysis

- 3개의 real-world 데이터셋에 대해 SOTA(state-of-the-art) 를 달성
- denoising/variational autoencoder에 대해 multinomial likelihood는 일반적인 Gaussian과 logistic likelihood보다 더 좋은 성능
  
<br>


📍 **Research Question**

1. **Multinomial likelihood를 협업 필터링에서 일반적으로 사용되는 다른 likelihood와 어떻게 비교하는가?**
2. **`Mult-VAE`는 언제 `Mult-DAE`보다 더 성능이 좋고 나쁜가?**

<br>

### Quantitative results

<img src="https://user-images.githubusercontent.com/48899040/214767463-2af448cb-3857-4f2e-a1a4-66106920868d.png" width="50%" height="50%">

Table 2는 `Mult-VAE`/`Mult-DAE`과 baseline model을 비교한 것이다. 각 metric은 모든 테스트 유저에 대한 평균값이다. `Mult-VAE`와 `Mult-DAE` 모두 데이터셋과 metrics에 대해 상당히 더 나은 성능을 보였으며, `Mult-VAE`는 ML-20M과 Netflix 데이터셋에서 `Mult-DAE`보다 월등한 성능을 보였다. 

<br>

<img src="https://user-images.githubusercontent.com/48899040/214767464-ea19fcc3-d871-4786-8d5f-3ee786db9fa6.png" width="50%" height="50%">

Table 3은 Mult-DAE와 NCF의 결과를 요약한 것이다. Mult-DAE는 두 데이터셋에 대해 pre-training을 하지않은 NCF보다 월등한 성능을 보였다. 더 큰 Pinterest 데이터셋에 대해서, Mult-DAE는 pre-trained NCF 모델보다 더 큰 마진의 성능 개선을 보였다.

<br>

### How well does multinomial likelihood perform?

multinomial likelihood가 top-N ranking loss에 대한 좋은 proxy이고, 암시적 피드백 데이터에 대해 적합하다는 것을 증명하기 위해, 각 데이터셋에 대해 최고의 성능을 가진 `Mult-VAE`와 `Mult-DAE` 모델을 가져와 다른 모든 것들은 동일하게 유지한 채 데이터마다 likelihood distribution model을 swap하였다.

<img src="https://user-images.githubusercontent.com/48899040/214767465-57288d8b-081f-4b0d-b6f9-6340854f2435.png" width="50%" height="50%">


Table 4는 ML-20M 데이터셋에 대한 다양한 likelihood의 결과를 요악한 것이다. 각 likelihood마다 별개로 하이퍼파라미터를 튜닝했다. 이때 부분 정규화(parital regularization)는 Gaussian과 logistic에 대해 덜 효과적인 것으로 보였다.

Multinomial likelihood는 다른 likelihood보다 더 좋은 성능을 낸다. 저자들은 **likelihood의 선택이 데이터에 의존적**이라는 것을 강조한다. 협업 필터링 task에 대해서는 multinomial likelihood가 훌륭한 결과를 얻을 수 있다. 

<br>

### When does Mult-VAE perform better/worse than Mult-DAE?

직관적으로 `Mult-VAE`는 보다 강력한 모델링 가정을 강요하므로 유저-아이템 상호작용 데이터가 희소(scarce)할 때 더 강력해질 수 있다. 이를 연구하기 위해 아래의 두 데이터셋을 활용했다.

- ML-20M: `Mult-DAE`보다 더 큰 마진(성능 차이)을 가지는 `Mult-VAE`
- MSD: `Mult-VAE`와 `Mult-DAE`가 비슷한 성능을 보임

inference 모델 $g_\phi(\cdot)$의 입력으로서 제공되는 fold-in 셋에 있는 유저의 활동 수준에 기반하여 테스트 유저를 5분위로 분류하였다.

`Mult-VAE`와 `Mult-DAE`를 사용하여 각 유저 그룹에 대해 $NDCG@100$을 계산하였고, ****다양한 수준의 활동을 하는 유저마다 성능이 어떻게 다른가를 Figure 3을 통해 볼 수 있다.

![Untitled 20](https://user-images.githubusercontent.com/48899040/214767466-6457bb72-5854-4a71-bdef-fb0efc0c7d11.png)

Figure 3은 증가하는 유저 활동에 대한 성능을 보여준다. 오류 막대는 하나의 표준 오류(standard error)를 표현한다. 각 subplot마다, paired t-test가 수행되고 통계적 유의성이 표시된다. **데이터셋에 따라 약간의 차이가 있지만, `Mult-VAE`는 적은 아이템 수만 클릭한 사용자를 위해 지속적으로 좋은 추천 성능을 보였다.** (ML-20M, Figure 3a)

`Mult-DAE`는 실제로 가장 활동적인 유저에 대해 Mult-VAE보다 성능이 더 뛰어나다. 이는 prior 가정이 더 강할 경우 유저가 많은 데이터를 사용할 수 있을 때 잠재적으로 성능이 저하될 수 있음을 나타낸다. 

MSD (Figure 3b)에서, **가장 활동적이지 않은 유저들은 `Mult-VAE`와 `Mult-DAE` 모두에서 비슷한 성능**을 가진다. 

전반적으로, **principled Bayesian inference approach의 관점에서 볼 수 있는 `Mult-VAE`가 데이터의 scarcity(희소성)에 상관없이 `Mult-DAE`의 point estimate 접근법보다 더 강력하다**는 것을 발견했다. 

반면, `Mult-DAE`도 아래의 장점을 가진다.

1. bottleneck layer에서 더 적은 파라미터를 요구한다. 
Mult-VAE는 잠재 표현 $z$를 얻기 위해 두 파라미터셋 (variational mean $\mu_\phi(\cdot)$, variational variance $\theta_\phi(\cdot)$) 을 요구한다.
1. Mult-DAE는 실무에서 더 간단하게 쓰일 수 있다.

<br>

# 5 Conclusion

본 논문은 암시적 피드백 데이터에 대해 collaborative filtering을 위한 VAE의 변형 모델을 제안하였고, 이를 통해 제한된 모델링 용량을 가진 선형 요인 모델을 넘어설 수 있었다. 특히 **multinomial likelihood** 함수를 가진 생성 모델을 도입하여 유저-아이템의 암시적 피드백 데이터를 모델링하는데 더욱 적합하다는 것을 보였다.

또한 `Mult-VAE`를 부분적으로 정규화하기 위해 추가적인 정규화 파라미터 $\beta$를 도입했다. 이때 **KL annealing**을 사용하여 $\beta$를 실용적이고 효과적으로 조정하였다.
논문에서 제안한 두 모델 `Mult-VAE`와 `Mult-DAE` 모두 real-world 데이터셋에 대해 SOTA 베이스라인보다 월등한 성능을 보였다.
마지막으로 `Mult-VAE`와 `Mult-DAE`의 장단점을 확인하여 pniicpied Bayesian approach를 사용하는 것이 더 강건하다는 것을 보였다.