---
title: "[Paper Review] Matrix Factorization Techniques for Recommender Systems (IEEE, 2009)"
date: 2022-09-07
categories:
- Recommender System
tags:
- paper review
- recommender system
- matrix factorization
use_math: true
comments: true
---

##### 논문 출처: [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)


본 포스팅은 Netflix Prize에서 우승을 거둔, 2009년에 발표된 Matrix Factorization Techniques for Recommender Systems 논문을 정리한 글이다.

<br>

> Netflix Prize 대회에서 증명되었듯이, matrix factorization 모델은 상품을 추천하면서 implicit 피드백와 일시적 효과 및 신뢰도에 대해 고전적인 nearest-neighbor 기술보다 월등하다

<br>

# Recommender system strategies

대체적으로, 추천 시스템은 두 가지 전략 중 하나에 기반한다.

## 1. Contents-based Filtering (CBF)

각 유저 또는 아이템에 대해 성질(nature)을 특징짓기 위해 `profile`을 생성한다.
  
- ex) 영화 profile: 장르, 참여 배우, 영화의 box office 인기도 등 관련 속성
- ex) 유저 profile: 인구 통계 정보 또는 적절한 설문지에 제공된 답변이 포함될 수 있다.

## 2. Collaborative Filtering (CF)

Content Filtering의 대안은 명시적(explicit) profiles를 작성할 필요 없이, **`과거 유저의 행동`**(e.g. , 이전 거래 또는 product ratings)에만 의존한다
    
⇒ **협업 필터링(colaborative filtering)**
    
협업 필터링은 새로운 유저-아이템 관계를 발견하기 위해 **유저 간의 관계**와 **상품 간의 상호 의존성**을 분석한다.
    
CF의 주요 매력은 domain free 하다는 것 ⇒ 종종 이해할 수 없거나 CBF을 사용하여 profile하기 어려운 데이터의 측면들을 잘 설명할 수 있다. 일반적으로 CBF 기술보다 더 정확한 반면, **CF는 시스템이 새로운 상품과 유저들을 다루지 못하는, 소위 `cold start` 라고 불리는 문제에 시달리고 있다.** 이러한 측면에서는 CBF는 더 우수하다고 할 수 있다.

CF의 두 가지 주요 areas는 인접 방법(neighborhood)과 잠재 요인 모델(latent factor model)이다. 이제부터 이 두 가지에 대해 자세히 알아보자.

### Neighborhood methods

인접 방법은 **아이템 사이(또는 유저들 사이)의 관계를 계산하는 것에 중점**을 둔 방법이다. item-oriented의 접근은 동일한 유저가 **“인접(Neighboring)" 아이템의 ratings**을 기준으로 하나의 아이템에 대한 유저의 선호도를 평가한다. 
이때 한 아이템의 이웃은 동일한 유저에 대해 비슷한 평가를 받는 다른 아이템들이다.
- ex) 유저 A의 영화 M의 평점을 예측하려고 할 때, 유저 A가 실제로 평가한 영화의 인접 이웃을 찾는다.

![user-oriented neighborhood method](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/687b6263-ff79-47e8-904e-fd19c1fcfe7d/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T101032Z&X-Amz-Expires=86400&X-Amz-Signature=0f816007fa9a6d1a2d02c78af2b27e39afb72134de6a14a50978aa54e1aef83f&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)


Figure 1) user-oriented 접근은 서로의 ratings을 보완할 수 있는, 생각이 비슷한 유저들을 식별한다.


### Latent Factor Model

잠재 요인 모델은 **평점 패턴으로부터 추론된 20~100가지 요인에 대해 아이템과 유저 둘 다 특성화함으로써** 평점을 설명하려고 하는 대안적 방식이다. 
- 영화의 경우, 발견되는 요인들이 코미디 vs 드라마, 액션의 양, 어린이의 성향(orientation)과 같은 **분명한 차원**, 또는 character development의 깊이 또는 기묘함과 같은 **잘 정의되지 않은 차원**, 또는 **완전히 해석할 수 없는 차원**을 측정할 수 있다고 한다. 
- 유저의 경우, 각 요인(factor)은 **해당되는 영화 요인(factor)에서 높은 점수를 가진 영화를 유저가 얼마나 좋아하는지** 측정할 수 있다.

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f2d006e2-52b8-423a-baed-45da01505f03/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T101213Z&X-Amz-Expires=86400&X-Amz-Signature=5edb451b3b37f0f33833f787a854154aae50c826a7c7e25c009f827bfc0a4881&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

Figure 2는 2차원에서 이 아이디어에 대한 간단한 예시를 보여준다.

여성-지향적 vs 남성-지향적과 Serious vs Escapist로 특징 지어지는 두 가상 차원을 고려해보자. 
Figure 2는 몇개의 잘 알려진 영화와 몇 명의 가상 유저가 이러한 2차원에 떨어진 것을 보여준다. 영화의 평균 평점과 비교하여, 영화에 대한 유저의 예측 평점은 그래프에서 영화와 유저 위치의 내적(dot product)과 동일하다.

- ex) Gus가 ‘Dumb and Dumber’를 좋아하고, ‘The Color Puple’은 싫어하고, ‘Braveheart’를 평균적으로 평가할 것으로 기대한다. ‘Ocean’s 11’과 같은 몇가지 영화들과 Dave와 같은 유저들은 2차원 상에서 공정하게 중립적인 것으로 특징지어 질 것이다.

# Matrix Factorization (MF) Models

잠재 요인 모델의 가장 성공적인 구현의 일부는 **Matrix Factorization(MF)** 에 기반한다. MF의 기본적인 형태는 **아이템 평점 패턴으로부터 추론된 요인들의 벡터에 의해 아이템과 유저 둘 다 특징 짓는다.** 추천시스템은 유저를 나타내는 1차원과 아이템의 관심을 나타내는 다른 차원을 가진 행렬에 놓여진 다양한 유형의 입력 데이터에 의존한다. 
가장 간편한 데이터는 높은 퀄리티의 암시적 피드백(explicit feedback)으로, 제품에 대한 유저의 관심의 명시적인 input을 포함한다.
  - ex) Netflix는 영화의 별점을 수집하고, TiVo 유저들은 up/down 버튼을 눌러 TV 프로그램에 대한 그들의 선호도를 나타낸다.

우리는 명시적인 유저의 피드백을 **평점**이라고 부른다. 보통, **명시적 피드백은 sparse matrix를 구성**하는데, 어떤한 싱글 유저든 possible items의 작은 비율만큼만 평가하기 때문이다.

MF의 한 가지 장점은 **추가 정보를 통합**할 수 있다는 것이다.
명시적 피드백을 이용하지 못할 때, 추천시스템은 암시적 피드백(implicit feedback)을 통해 유저의 선호도를 추론할 수 있고, 그것은 구매 이력, 브라우징 이력, 검색 패턴, 심지어 마우스 이동 등을 포함하는 유저의 행동을 관찰함으로써 간접적으로 의견을 반영할 수 있다. 암시적 피드백은 보통 이벤트의 유무를 나타내므로, 따라서 일반적으로 **dense matrix**로 표현된다.
    

# A Basic Matrix Factorization Model

MF 모델은 유저와 아이템 모두 $f$ 차원의 공동(joint) 잠재 요인 공간에 매핑하여 **유저-아이템 상호 작용이 해당 공간에서 내적**되어 모델링된다. 따라서, 각 아이템 $i$은 벡터 $q_i\in\mathbb{R}^f$와 연관되고, 각 유저 $u$는 벡터 $p_u\in\mathbb{R}^f$와 연관된다. 
- 주어진 아이템 $i$의 경우, $q_i$의 요소들은 **해당** **아이템이 긍/부정의 요인(factors)을 가지고 있는 정도**를 측정한다.
- 주어진 유저 $u$의 경우, $p_i$의 요소들은 **긍/부정의 높은 아이템에 대해 유저가 관심을 가지는 정도**를 측정한다.

이때 내적 $q_i^Tp_u$ 의 결과는 유저 $u$와 아이템 $i$ 사이의 상호 작용을 캡쳐한다. 즉, 아이템의 특징에 대한 유저의 전반적인 관심을 나타낸다. 이것은 유저 $u$의 아이템 $i$의 평점을 근사하고, $r_{ui}$라고 표기된다.
    
![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/2c2aceae-88ce-41b4-adcd-e3d80732f708/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T101542Z&X-Amz-Expires=86400&X-Amz-Signature=342b4b11b0b595629d2aa1d786bc61c723aef8a6802bd2f720cba7567e2d9297&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)
    

<br>

MF의 major challenge는 아이템과 유저를 요인 벡터 $q_i,p_u\in \mathbb{R}^f$의 매핑을 계산하는 것이다. 추천시스템이 이 매핑을 완료한 후에, 식 1을 통해 유저가 어떠한 아이템에 줄 평점을 쉽게 추정할 수 있다. 
이러한 모델은 **singualr value decomposition (SVD, 특이값 분해)** 와 밀접한 관련이 있다. 이는 정보 검색(information retrieval)에서 잠재 의미적 요인(latent seantic factors)를 식별하기 위해 잘 만들어진 기술이다.

CF 도메인에서 SVD를 적용하려면, 유저-아이템 평점 행렬을 인수분해(factoring) 해야 한다. 이는 **유저-아이템 평점 행렬에서의 sparseness에 의해 야기된 결측값의 높은 비율**때문에 종종 어려움이 생긴다. 전통적인 SVD는 **행렬이 불완전할 때 정의되지 않고**, 게다가 상대적으로 덜 알려진 엔트리에 대해서만 무심코 나타내는 것은 **과적합**하기 매우 쉽다.

이전 시스템은 누락된 평점을 채우고 평점 행렬을 dense하게 만들기 위해 **Imputation**에 의존한다. 그러나, imputation은 데이터의 양이 증가할수록 크게 증가하여 비용적으로 비싸질 수 있다. 또, 부정확한 imputation은 데이터를 상당히 왜곡할 수도 있다.
따라서, 더 최신 연구에서는 **오직 관측된 평점에 대해서만 직접적으로 모델링하고, 반면 정규화된 모델을 통해 과적합을 방지한다.**

요인 벡터 ($p_u, q_i)$을 학습하기 위해, 시스템은 알려진 평점 셋에서 **정규화 제곱 오차를 최소화**한다.
    
![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/e0f64ffd-2e3e-48ce-bec7-1b067871c1ec/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T101738Z&X-Amz-Expires=86400&X-Amz-Signature=aa346178572ce4a41da36e47aad4489b52e15dfc1be05c78f34bcdc66e46e551&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)
    
- $k$: (훈련 데이터셋에서) 알려진 평점 $r_{ui}$$(u,i)$ 이 있는 $(u,i)$ 쌍의 집합

시스템은 이전에 관측된 평점을 fitting함으로써 학습한다. 그러나, **목표는 알려지지 않은 평점, 즉 미래를 예측하는 방식으로 이전 평점을 일반화**하는 것이다.따라서 학습된 파라미터를 정규화함으로써 관측된 데이터의 과적합을 방지해야 한다. 이때 학습된 파라미터는 규모에 대해 패널티를 받는다.

- 상수항 $\lambda$는 정규화의 정도를 제어하고 보통 cross-validation에 의해 값이 결정된다.

# Learning Algorithms

식 (2)를 최소화하는 2가지 접근 방식은 `stochastic gradient descent(SGD)`와 `alternating least squares(ALS)`이다. 두 가지에 대해 자세히 알아보자.

## Stochastic gradient descent (SGD)
SGD는 훈련 셋에 있는 **모든 평점에 대해 loop를 도는** 알고리즘으로, 각각 주어진 training case에 대해, 시스템은 $r_{ui}$를 예측하고 아래 식의 연관된 예측 에러를 계산한다.

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/56eb98ac-8122-496a-8cd0-cd5d358aa154/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T101907Z&X-Amz-Expires=86400&X-Amz-Signature=a163189d76392ac0c12cdb2c4c9462ede8e53cafc0e62be7b34ab2c0d970dcd8&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

그 후 크기에 비례하여 파라미터를 **gradient의 반대 방향**으로 수정하고, 다음을 산출한다.

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/3b694f4e-126f-4ec5-aa1e-77ebe514848e/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T101951Z&X-Amz-Expires=86400&X-Amz-Signature=1d4c6141b687c8eb8de1f1d15e725c15cb46d0cdc1db0edd15aa00a9c80f04f6&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

이러한 SGD 방식의 접근은 구현 용이성(implement ease)과 비교적 빠른 실행 시간을 결합한다. 

하지만 어떤 경우에선, ALS 최적화를 사용하는 것이 유익하다.

## Alternaing least squares (ALS)
$q_i$와 $p_u$ 둘 다 알려져있지 않기 때문에, 식 (2)는 convex 하지 않다. 하지만, **이들 중 하나를 고정** 한다면, 최적화 문제는 **2차(Quadratic)** 가 되어 최적으로 풀 수가 있다.
따라서, ALS 기술은 $q_i$의 고정과 $p_u$의 고정을 번갈아 진행한다. (EM 알고리즘처럼) 모든 $p_u$가 고정된다면, 시스템은 반대로 $q_i$에 대해 **최소-제곱 문제**를 푸는 것으로써 재계산 ⇒ 각 단계가 수렴할 때까지 식 (2)를 감소시키는 것을 보장한다.

일반적으로 **SGD가** ALS보다 **더 쉽고 빠른** 반면, **ALS는 최소 2가지 경우에 대해서 유리**하다.

  1. **시스템이 병렬화를 사용할 수 있을 때**
      
      ALS에서는 시스템은 다른 아이템 요인들과 독립적으로 $q_i$를 계산하고, 다른 유저 요인들과 독립적으로 $p_u$를 계산한다. 이것은 잠재적으로 **알고리즘의 대규모(massive) 병렬화**를 발생시킨다.
      
  2. **암시적 데이터(implicit data) 중심의 시스템의 경우**
      
      훈련 셋이 sparse하다고 볼 수 없기 때문에, 각 단일 훈련 케이스에 대해 gradient descent처럼 looping하는 것은 실용적이지 않을 수 있다. ⇒ ALS는 이러한 경우에 대해 효율적으로 다룰 수 있다.
        

# Adding Biases

CF에 대한 MF 접근 방식의 한 가지 이익은 다양한 데이터 측면과 기타 애플리케이션 요구 사항을 다루는 유연함이다. 이는 동일한 학습 프레임워크에 머무르는 동안 식 (1)의 축적이 필요하다. 식 (1)은 다양한 평점을 만들어내는 유저와 아이템 사이의 상호 작용을 포착하려고 한다. 그러나, 평점에서 관측된 변동의 상당 부분은 *biases* (편향) 또는 *intercepts* (절편)로 알려진 유저 또는 아이템과 관련된 효과 때문이고, 다른 어떤 상호작용과는 무관하다.
    
- ex) 전형적인 CF 데이터는 일부 유저가 다른 유저들보다 더 높은 평점을 주고, 일부 아이템이 다른 아이템들보다 더 높은 평점을 받는 큰 체계적 경향을 나타낸다. 결국 일부 아이템들은 다른 아이템들보다 더 좋게 혹은 더 나쁘게 인식되고 있다.
    

따라서, $q_i^Tp_u$ 형식의 상호작용으로 전체 평점(full rating value)을 설명하는 것은 현명하지 못할 수 있다. 대신, **요인 모델링(factor modeling)에 데이터의 실제 상호작용 일부만 적용하여, 개별 사용자 또는 개별 아이템이 설명할 수 있는 평점들의 일부를 식별**하려고 노력한다.

평점 $r_{ui}$에 포함된 bias의 1차 (first-order) 근사치는 다음과 같다.
    
![해당 유저와 해당 영화 각각의 bias 값을 평균 평점에 더한다](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/6136676c-9bc8-4388-a6ce-f5bbb3c06d47/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T102241Z&X-Amz-Expires=86400&X-Amz-Signature=fdb152c3a18056b3521c7160b9ec607cfa76970a2e01e42dfbbdc5c9a65a4c5c&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

해당 유저와 해당 영화 각각의 bias 값을 평균 평점에 더한다

- 파라미터 설명
    - $b_{ui}$: 평점 $r_{ui}$과 연관된 bias로 유저와 아이템의 효과를 설명
    - $\mu$: 전체적인 평균 평점
    - $b_u$와 $b_i$: 각각 유저 $u$와 아이템 $i$의 관측된 편차

    ex) 유저 Joe의 영화 ‘Titanic’에 대한 평점의 1차식을 추정한다고 가정해보자.
  - 이때 모든 영화에 대한 평균 평점 $\mu$는 3.7점이다.
  - 게다가 ‘Titanic’은 평균 영화들보다 더 나아서 평균보다 0.5점 이상 평가되는 경향이 있다.
  - 반면에, Joe는 평균보다 0.3점 이하로 평점을 매기는 경향이 있다.
  - 따라서, Joe의 영화 ‘Titanic’의 예측 평점은 3.9점일 것이다 (3.7 + 0.5 - 0.3)


Biases는 다음과 같이 Equation 1을 확장한다.
   
![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b183420c-8476-4343-9b9b-359332b68c63/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T102322Z&X-Amz-Expires=86400&X-Amz-Signature=0a330ba247412b3b457183b46bb93ae0bec31d27873be23d387646d6391d6d96&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)
    

- 관측된 평점의 4가지 구성 요소
    - global 평균
    - 아이템 bias
    - 유저 bias
    - 유저-아이템 상호작용

    ⇒ **각 구성 요소가 자신과 관련된 signal 부분만을 설명**할 수 있게 한다.

- 학습: 아래의 오차 제곱 함수를 최소화
    
    ![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/63c14b0b-7a1d-4b6d-a095-165c1d180b53/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T102406Z&X-Amz-Expires=86400&X-Amz-Signature=83c3c38246d17f825c7ded14651d24c185199e1437a275609bafb05edcae400f&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)
    

biases가 관측된 signal의 대부분을 포착하려는 경향이 있어 정확한 모델링은 필수적이다. 따라서, 다른 작업들이 더 정교한 bias 모델을 제공한다.

# Additional Input Sources

종종 추천 시스템은 많은 유저가 아주 적은 평점을 제공하여 그들의 취향에 대해 일반적인 결론을 내기 어렵게 하는 **cold start 문제**를 처리해야 한다. 이를 해결하는 한 가지 방법은 **유저에 대한 추가적인 정보를 통합하는 것**이다. 

추천 시스템은 **유저의 선호도에 대한 통찰**을 얻기 위해 **암시적 피드백**을 사용할 수 있다. 실제로, 명시적 평점을 제공하려는 **유저의 의지와는 무관하게 행동 정보**를 모을 수 있다.

- ex) 소매업자는 고객이 제공하는 평점(등급)과 더불어 고객의 경향성을 학습하기 위해 고객의 구매 또는 브라우징 이력을 사용할 수 있다.
        

단순함을 위해, Boolean 암시적 피드백의 경우에 대해 고려해보자
- $N(u)$: **유저 $u$가 암시적 선호도를 표현하는 아이템 집합**
    - 유저가 암시적으로 선호하는 아이템을 통해 유저를 프로파일링
    - 여기서 아이템 $i$가 $x_i\in\mathbb{R}^f$와 관련된 아이템 요인의 새로운 집합이 필수적
    - 따라서, $N(u)$를 통해 아이템의 선호도를 보여준 유저는 벡터 $\sum_{i\in N(u)}x_i$에 의해 특징 지어진다

다음과 같이 합계를 정규화하는 것은 종종 유익하다
    ![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b019e8b2-dcbd-44c1-9324-3ac485de2300/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T102535Z&X-Amz-Expires=86400&X-Amz-Signature=314914caa093bc3af2712d7ebdbb6d26c758cd0bb97dbbd06255f6b24151c3b6&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)
    

또다른 정보는 **인구 통계와 같은 유저의 특성**으로 알려진다. 단순함을 위해, 유저 $u$가 성별, 연령대, zip code, 수입 정도 등과 같은 특성 집합 $A(u)$에 해당하는 Boolean 특성을 고려해보자. 구별되는 요인 벡터 $y_a\in\mathbb{R}^f$는 다음과 같은 유저 관련 특성 집합을 통해 유저를 설명하는 각 특성에 해당한다.

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a93660bc-3abe-40c5-b7a6-078723a1146c/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T102616Z&X-Amz-Expires=86400&X-Amz-Signature=ca40b592417c4e3b616a9799f0414577a6d3233188694db76324f02e9c7b0c0a&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

최종적으로 matrix factorization 모델은 모든 signal sources를 통합해야 하기 때문에, 더욱 향상된 user representation이 필요하다.
    
![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/48557d75-a9ef-4318-92a7-59cc812a3903/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T102645Z&X-Amz-Expires=86400&X-Amz-Signature=f8ed243a09b0bcfcfc4c0ccdb503baee1cbb693d7394bf6b5ca2d3082880b0b4&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)
    

## Temporal Dynamics
지금까지의 모델들은 static(동적)하지만, 실제로 제품에 대한 인식과 인기는 새로운 selection이 나타남에 따라 계속해서 변한다.
  - ex) 고객의 성향이 진화 ⇒ 고객의 취향을 재정의

따라서 **유저-아이템 상호작용의 동적이고 시간적인 특성을 반영하는 temporal effect(시간적 효과)를 고려**해야 한다. 
MF 방식은 시간적 효과를 모델링하는데 적합하여 정확도를 크게 향상시킬 수 있다. ratings을 특정한 terms으로 분해하는 것은 다양한 시간적 측면을 분리하여 다룰 수 있게 한다.
- 특히, 다음과 같은 term들은 시간에 따라 변화한다
    - $b_i(t)$: item biases
    - $b_u(t)$: user biases
    - $p_u(t)$: user preferences

1. **첫 번째 temporal effect: 아이템의 인기는 시간이 지남에 따라 변할 수 있다**
    - ex) 영화는 배우의 새로운 영화 출연과 같은 외부 사건에 의해 트리거된 것처럼 인기에 들락날락할 수 있다
    - 따라서 이러한 모델들은 **item bias $b_i$를 function of time**으로 다뤄야 한다

2. **두 번째 temporal effect: 유저는 시간이 지남에 따라 기준 ratings을 변경할 수 있다**
    - ex) 한 유저가 영화를 평균적으로 4점을 주는 경향이 있었는데, 지금은 3점을 준다
    - 이는 여러가지 요인을 반영할지도 모른다 ← natural drift, 다른 최근 ratings과 비교하여 할당, 시간이 지남에 따라 가구 내에서 rating의 정체성이 변경될 수 있다는 사실
    - 따라서 이러한 모델들은 파라미터 $b_u$를 function of time으로 다뤄야한다
    

- Temporal dynamics은 다음을 넘어선다.

     - user preference와 유저와 아이템 사이의 상호작용 또한 영향을 미친다
     - 유저는 시간이 지남에 따라 그들의 선호도가 변화한다
     - 이때 모델은 user factors ($p_u$)를 function of time으로 가져와 이러한 효과를 설명한다
     - 반면에 static 아이템 특성을 명시 ($q_i$) ⇒ 왜냐하면 인간과 달리 아이템은 본질적으로 static이기 때문
    
- time-varying 파라미터의 정확한 파라미터화는 식 4를 time t에서 dynamic prediction rule로 변경
    
    ![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d26f16d4-f9ad-4dcd-8956-56ad3d663773/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T102832Z&X-Amz-Expires=86400&X-Amz-Signature=c5d4a811016c6cfad55e1fd9b1715cd19213a7b50dc85bd1e1752802c52932bc&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)
    
