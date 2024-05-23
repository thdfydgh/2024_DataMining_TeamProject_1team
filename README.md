# 서울시 교통사고 상해예측 & 위험요인 분석
---
## 폴더 설명
<pre>
data : 2022년도, 2023년도 상해 DataSet (TAAS)
data_prepro: 데이터 전처리 파일 모음
predict_model : 모델링 파일 모음
  </pre>
## 파일 설명
---
### data_prepro 폴더
<pre>

death_classifier.py : 사망자수 예측(Binary classfier)을 위한 전처리 코드
total_data_prepro.py : ECLO 예측을 위한 전처리 코드
two_model : 사망자수와 중상자수,경상자수,부상신고자수 따로 예측하기 위한 통합 전처리 코드
</pre>
### predict_model 폴더
<pre>
deathnum_classfier.ipynb : 사망자수 예측 
modeling : ECLO 예측
trial_modeling : 시도중인 연습 코드
시도중인 것 : 중상자수, 경상자수, 부상신고자수를 한 번에 예측해보자

--> 현재 시도중인 것.
사망자수 classfier, 중상자수 classfier, 경상자수+부상신고자수 regression
-> feature selection.
</pre>
---
