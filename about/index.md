---
layout: page
type: about
date: 2023-01-15
---


안녕하세요. 저는 추천시스템에 관심있는 AI 대학원 석사과정(2023~) 김영현입니다.

우리의 일상과 밀접한 인공지능, 일상에 녹여든 자연스러운 인공지능을 좋아합니다.

Contact) ✉️ khyeon0819@gmail.com

---

## Projects

**Fake news detection and web page platform creation (March 2022 - June 2022)**

- A project to detect if the news is fake news with a different title and body when users enter the news.
- Word embedding with FastText and developed fake news detection model using bi-LSTM with attention model with 1 input combined title and body.
- Building webpage using Flask and providing the probability of fake news and attention visualization of which words the model focused more on to detect fake news.
    - In Visualization, the redder the background, the more focused the word in the fake news detection process.

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c3d6444b-2463-4ada-a23d-ec773518363a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T094844Z&X-Amz-Expires=86400&X-Amz-Signature=86448abbf2f39b18ede4e9d49d15ef6d131eeb27ec6c5e112835abcc619f8b59&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

* *Web page demonstration → [[link]](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0c5c78ec-9063-4472-aee4-81a36ff67be8/99%ED%8E%98%EC%9D%B4%EC%A7%80%EC%8B%9C%EC%97%B0.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T093828Z&X-Amz-Expires=86400&X-Amz-Signature=b117c3b27cee9d3520a5d7a575d670f68542809a6c0d62539c2ed39e050874c3&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%2299%25ED%258E%2598%25EC%259D%25B4%25EC%25A7%2580%25EC%258B%259C%25EC%2597%25B0.mp4%22&x-id=GetObject)*

<br>

**Simple Wine Recommendation System using Content-based Filtering (March 2022 - April 2022)**

- Recommendation system using content-based filtering to recommend wines similar to those consumed by users.
- Collected wine data through crawling and analyzed wine data such as wine type, aroma, food matching, flavor, and variety, etc.
    - Calculate the number of varieties according to the proportion of varieties belonging to the wine and convert them into strings
    - Convert a numerical data with 0-5 of flavor column to category data based the degree of [no/very weak/weak/normal/strong/very strong]
    - Vectorize wine features using CountVectorzier and compare the cosine similarity in various combinations
- The top 10 wines most similar to the wine consumed by the user were provided as recommendation results.

<br>

**Creation a timeline for each news event through text analysis and a web page platform (March 2021 - June 2021)**

- Provided a timeline of the progress from occurrence to termination of the particular news event.
- Collected news data for each category through crawling and developed the clustering of news events using HDBSCAN by tf-idf vectorization.
- Extraction summary for each news article by timeline using KR-WordRank.

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/3b2feb90-83cb-47f8-86a2-ef40f9928dda/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230115%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230115T093930Z&X-Amz-Expires=86400&X-Amz-Signature=cad99f101ddc41e73c72b267e3816167f5f18aef9122e4eff70f4e9ab2cc5c5e&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

* *Web page Demonstration link → [[link]](https://www.youtube.com/watch?v=Xd-rYflqYoE)*

---

## Research Experience

**Analysis public perception and emotion for climate change and its adaptation policy
(March 2022 – June 2022)**

- Analyzed the public perception and emotion of climate change and related policies using social big data for the past three years(2019-2021).
- Collected Naver blog and Twitter data through crawling.
- Keyword analysis using Doc2Vec model and subject analysis using LDA topic modeling.
    
    
    * Conference Presentation
    
        A Study on the Public Perception and Emotion for Climate Change and Its Adaptation Policy (Yeonghyeon Kim, Woosuk Choi), in KCC 2022