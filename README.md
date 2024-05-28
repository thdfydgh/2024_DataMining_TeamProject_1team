# 서울시 교통사고 상해예측 & 위험요인 분석
---
## 폴더 설명
<pre>
data : 2022년도, 2023년도 상해 DataSet (TAAS)
data_prepro: 데이터 전처리 파일 모음
predict_model : 모델링 파일 모음
  </pre>
## 파일 설명
### data_prepro 폴더
<pre>

death_classifier.py : 사망자수 예측(Binary classfier)을 위한 전처리 코드
total_data_prepro.py : ECLO 예측을 위한 전처리 코드
two_model : 사망자수와 중상자수,경상자수,부상신고자수 따로 예측하기 위한 통합 전처리 코드
행정동 통합 및 데이터합산.csv : 데이터 별 행정동명 통일 및 변환 후 동일한 이름의 행정동 행들을 합산한 데이터
</pre>
### predict_model 폴더
<pre>
deathnum_classfier.ipynb : 사망자수 예측 
modeling : ECLO 예측
</pre>
### Main
<pre>
교통사고 피해예측 EDA.ipynb : 각 feature별 ECLO값 EDA
deathnum_classfier.ipynb
modeling.ipynb
predict.ipynb
</pre>

## 활용 Data : TAAS(Traffic Accident Analysis System) 교통사고 분석 시스템 Data
https://taas.koroad.or.kr/web/shp/sbm/initGisAnals.do?menuId=WEB_KMP_GIS_TAS 

![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/6e613ca4-6ed4-467d-9441-91bf0873628d)
교통사고분석시스템에서 구 별 사고 data를 수집해 서울시 전체 사고 Dataset을  구축하였다.
Data의 형태는 다음과 같다.
![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/588dc067-23f7-42d1-a9ff-9a84cd77541b)
2022년도, 2023년도 Dataset을 구축 하였으며, 2022년도 데이터로 2023년도를 예측하는 것이 큰 흐름이다.
우리는 이 Dataset을 기반으로, Equivalent Casualty Loss Only(ECLO) 즉, 교통사고 상해정도를 예측하고 위험 요인 분석을 한다.
(ECLO = 사망자수 * 10 + 중상자수 * 5 + 경상자수 * 3 + 부상자수 * 1)
<pre>
사망자분류모델 -> feature selection -> 차대 사람 유의미 -> ECLO예측 --> 동별 상해정도 차이 유의미  -> 동별 예측 -> merge -> 동별 대책 수립(결론)
</pre>


---
## 모델링
### 1. 사망사고 여부 이진 분류 예측 모델
<pre>
사고내용
경상사고      23723
중상사고       6940
부상신고사고     1862
사망사고        180
Name: count, dtype: int64
서울 내 2022년도의 33699건 사고 중 사망사고의 비율은 0.5503745604647607% 이다.
</pre>

#### 데이터 전처리 과정

1. **사망사고 라벨링**:
   - `사고내용`이 '사망사고'인 경우를 판별하여 `사망사고여부` 컬럼을 생성하고, boolean 값으로 라벨링하였다.

   ```python
   index1 = df['사고내용'] == '사망사고'
   df['사망사고여부'] = index1
   ```

2. **불필요한 컬럼 제거**:
   - 분석에 불필요하거나 중복되는 정보를 담고 있는 컬럼들을 삭제하였다. 삭제된 컬럼들은 `사고번호`, `시군구`, `사망자수`, `중상자수`, `경상자수`, `부상신고자수`, `노면상태`, `가해운전자 상해정도`, `피해운전자 상해정도`, `사고내용`, `가해운전자 차종`, `피해운전자 차종`이다.

   ```python
   df.drop(['사고번호', '시군구', '사망자수', '중상자수', '경상자수', '부상신고자수', '노
   면상태', '가해운전자 상해정도', '피해운전자 상해정도', '사고내용', '가해운전자 차종', '피해운전자 차종'], axis=1, inplace=True)
   ```

3. **사고 시간 정보 추출**:
   - `사고일시` 컬럼에서 시간을 추출하여 `사고시각` 컬럼을 생성하였다.

   ```python
   time_lst = list(df['사고일시'])
   hour_lst = []
   for time in time_lst:
       hour_lst.append((str(time)[-3:-1]))
   df['사고시각'] = hour_lst
   ```

4. **사고일시 컬럼 삭제**:
   - `사고일시` 컬럼을 삭제하였다.

   ```python
   df.drop(['사고일시'], axis=1, inplace=True)
   ```

5. **야간 여부 이진분류**:
   - `사고시각`을 기준으로 22시부터 06시까지를 야간으로, 그 외 시간을 주간으로 분류하여 `야간여부` 컬럼을 생성하였다.

   ```python
   index2 = df['사고시각'] == '22', '23', '24', '1', '2', '3', '4', '5', '6'
   df['야간여부'] = index2[0]
   df.drop(['사고시각'], axis=1, inplace=True)
   ```

6. **주말 여부 이진분류**:
   - `요일`을 기준으로 토요일과 일요일을 주말로, 그 외를 평일로 분류하여 `주말여부` 컬럼을 생성하였다.

   ```python
   index3 = df['요일'] == '토요일', '일요일'
   df['주말여부'] = index3[0]
   df.drop('요일', axis=1, inplace=True)
   ```

7. **성별 이진분류**:
   - `가해운전자 성별`과 `피해운전자 성별`을 남성과 여성으로 구분하여 각각 `가해운전자 성별- 여성0 남성1`과 `피해운전자 성별- 여성0 남성1` 컬럼을 생성하였다.

   ```python
   index4 = df['가해운전자 성별'] == '남'
   df['가해운전자 성별- 여성0 남성1'] = index4
   df.drop('가해운전자 성별', axis=1, inplace=True)

   index5 = df['피해운전자 성별'] == '남'
   df['피해운전자 성별- 여성0 남성1'] = index5
   df.drop('피해운전자 성별', axis=1, inplace=True)
   ```

8. **Boolean 값을 정수형으로 변환**:
   - `사망사고여부`, `야간여부`, `주말여부`, `가해운전자 성별- 여성0 남성1`, `피해운전자 성별- 여성0 남성1` 컬럼의 값을 boolean에서 정수형으로 변환하였다.

   ```python
   df['야간여부'] = df['야간여부'].astype(int)
   df['주말여부'] = df['주말여부'].astype(int)
   df['사망사고여부'] = df['사망사고여부'].astype(int)
   df['가해운전자 성별- 여성0 남성1'] = df['가해운전자 성별- 여성0 남성1'].astype(int)
   df['피해운전자 성별- 여성0 남성1'] = df['피해운전자 성별- 여성0 남성1'].astype(int)
   ```

9. **연령 전처리**:
   - `가해운전자 연령`과 `피해운전자 연령` 컬럼에서 '미분류'와 '98세 이상' 데이터를 삭제하였다.
   - `가해운전자 연령`과 `피해운전자 연령`에서 나이 숫자만 추출하고, 이를 정수형으로 변환하였다.
   - `가해운전자 연령`과 `피해운전자 연령` 컬럼을 정규화하여 각각 `가해운전자 연령(정규화 됨)`과 `피해운전자 연령(정규화 됨)`으로 변환하였다.

   ```python
   idx = df[df['가해운전자 연령'] == '98세 이상'].index
   df.drop(idx, inplace=True)
   idx = df[df['피해운전자 연령'] == '98세 이상'].index
   df.drop(idx, inplace=True)

   idx2 = df[df['가해운전자 연령'] == '미분류'].index
   df.drop(idx2, inplace=True)
   idx2 = df[df['피해운전자 연령'] == '미분류'].index
   df.drop(idx2, inplace=True)

   suspect_lst = list((df['가해운전자 연령']))
   suspect_old = []
   for old in suspect_lst:
       suspect_old.append((old)[:-1])
   df['가해운전자 연령'] = suspect_old

   victim_lst = list(df['피해운전자 연령'])
   victim_old = []
   for old in victim_lst:
       victim_old.append((old)[:-1])
   df['피해운전자 연령'] = victim_old

   df['가해운전자 연령'] = df['가해운전자 연령'].astype('int')
   df['피해운전자 연령'] = df['피해운전자 연령'].astype('int')

   olds = df[['피해운전자 연령', '가해운전자 연령']]
   scaler = StandardScaler()
   scaled_olds = scaler.fit_transform(olds)
   df[['피해운전자 연령', '가해운전자 연령']] = scaled_olds
   df.rename({'피해운전자 연령': '피해운전자 연령(정규화 됨)', '가해운전자 연령': '가해운전자 연령(정규화 됨)'}, axis=1, inplace=True)
   ```

10. **컬럼 재배치**:
    - 종속변수인 `사망사고여부`가 데이터프레임의 마지막에 오도록 컬럼을 재배치하였다.

    ```python
    df = df.reindex(columns=['주말여부', '야간여부', '사고유형', '법규위반', '기상상태', '도로형태', '가해운전자 성별- 여성0 남성1', '가해운전자 연령(정규화 됨)', '피해운전자 성별- 여성0 남성1', '피해운전자 연령(정규화 됨)', '사망사고여부'])
    ```
  


### 모델링 결과 요약

#### 모델링 1: GBC와 SVM을 이용한 Binary Classifier

초기 모델링에서는 Gradient Boosting Classifier(GBC)와 Support Vector Machine(SVM)을 이용하여 이진 분류를 수행하였다. 두 모델 모두 정확도는 매우 높았으나, 소수 클래스인 사망사고(1)에 대한 성능이 매우 낮았다.

- **Gradient Boosting Classifier**
  - 정확도(Accuracy): 0.9942
  - 정밀도(Precision): 0.99
  - 재현율(Recall): 1.00
  - F1-점수(F1-score): 1.00
  - 사망사고(1) 클래스에 대한 성능은 매우 낮음

- **Support Vector Machine**
  - 정확도(Accuracy): 0.9944
  - 정밀도(Precision): 0.99
  - 재현율(Recall): 1.00
  - F1-점수(F1-score): 1.00
  - 사망사고(1) 클래스에 대한 성능은 매우 낮음

#### 모델링 2: 데이터 불균형 문제 해결을 위한 오버샘플링과 언더샘플링

데이터 불균형 문제를 해결하기 위해 SMOTE를 사용하여 소수 클래스에 대한 오버샘플링을, RandomUnderSampler를 사용하여 다수 클래스에 대한 언더샘플링을 수행하였다. 이를 통해 데이터의 균형을 맞추었으며, GBC와 SVM 모델의 성능이 향상되었다.

- **Gradient Boosting Classifier (Resampled Data)**
  - 정확도(Accuracy): 0.89
  - 정밀도(Precision): 0.90 (0), 0.86 (1)
  - 재현율(Recall): 0.94 (0), 0.79 (1)
  - F1-점수(F1-score): 0.92 (0), 0.82 (1)

- **Support Vector Machine (Resampled Data)**
  - 정확도(Accuracy): 0.94
  - 정밀도(Precision): 0.94 (0), 0.92 (1)
  - 재현율(Recall): 0.96 (0), 0.89 (1)
  - F1-점수(F1-score): 0.95 (0), 0.90 (1)

#### 모델링 3: 오버샘플링과 언더샘플링을 적용한 후 GBC와 SVM의 앙상블 모델

최종 모델링에서는 오버샘플링과 언더샘플링을 적용한 후, GBC와 SVM의 앙상블 모델을 사용하였다. 이 모델은 개별 모델들의 장점을 결합하여 더 나은 성능을 보여주었다.

- **Ensemble Classifier**
  - 정확도(Accuracy): 0.93
  - 정밀도(Precision): 0.94 (0), 0.91 (1)
  - 재현율(Recall): 0.96 (0), 0.88 (1)
  - F1-점수(F1-score): 0.95 (0), 0.90 (1)

### 성능 변화 요약

- **초기 모델링**: 높은 정확도에도 불구하고, 소수 클래스(사망사고)에 대한 성능이 매우 낮았다.
- **오버샘플링 + 언더샘플링**: 데이터 균형을 맞추어 모델의 전반적인 성능이 향상되었다. 특히, 소수 클래스에 대한 성능이 크게 개선되었다.
- **앙상블 모델**: 개별 모델의 장점을 결합하여 전반적으로 가장 높은 성능을 기록하였다. 

이 과정을 통해 데이터 불균형 문제를 효과적으로 해결하고, 이진 분류 모델의 성능을 최적화할 수 있었다.


앙상블 모델 내 Gradient Boosting Classifier의 Feature Importance를 분석한 결과, 사망자수 예측에 중요한 영향을 미치는 주요 변수를 확인하였다. 이 정보를 바탕으로 데이터셋 전체에 대한 예측을 수행할 계획이다.

![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/b8569772-5470-4e68-9dcb-8f4da2de95b1)

--- 
### 2. 교통사고 건당 ECLO 예측 모델
주어진 Feature importance를 기반으로, 사고유형의 (차대차, 차대 사람)의 차이가 유의미하다는 인사이트를 얻었다. 이를 기반으로, 사고유형의 세 가지 분류인 차대차, 차대사람, 차량 단독으로 데이터셋을 3개로 나누어 예측을 진행하였다.

#### 데이터 EDA(Exploratory Data Analysis)

#### 데이터 확인

#### 도로 형태별 사고 유형 (count plot)


![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/debd6e39-0f52-4abc-a1b3-34bd103cca47)
도로 형태에 따른 사고 유형을 시각화하였다. 터널 및 고가도로와 같은 도로 형태에서 주목할 만한 사고 유형이 관찰되었다.


#### 도로 형태/사고유형별 ECLO (violin plot)
![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/c743278a-9a77-4224-a97a-0dbfdde36923)

도로 형태와 사고 유형에 따른 ECLO (Expected Casualties and Loss of Life)를 시각화하였다. 특히 터널 및 고가도로와 같은 도로 형태에서 높은 ECLO가 관찰되었다.

#### 노면상태별 ECLO 평균 (bar plot)
![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/9f924956-b08b-443c-8472-09c51228c162)

노면 상태에 따른 ECLO 평균을 분석하였다. 특히 '해빙' 상태에서 높은 ECLO가 관찰되었다.


#### 요일 / ECLO
![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/93cb883c-7b20-4a2c-8aab-1d08edd16558)

ECLO 값이 8.00 이상인 경우 주말에 발생하는 비율이 높고, 8.00 이하인 경우 주말 사고 발생 비율이 낮은 경향이 있다.

#### 시간 / ECLO
![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/664b32f0-036b-41c7-8d2f-4cdf993be622)
![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/5b83d25f-7ed9-471f-818f-0799b50b3d39)

사고 발생 시간대와 ECLO 간의 관계를 분석하였다.
ECLO가 높은 사고의 다른 시간대와의 비율은 밤 사이가 낮은 사고보다 높은걸 확인 할 수 있었다.

EDA에서 얻은 인사이트를 기반으로 feature engineering를 진행하였다.


---

### 데이터 전처리

#### 전처리 과정

1. **데이터 로드**:
    - `train_path`와 `test_path`에서 데이터를 불러온다.
    - 데이터는 EUC-KR 인코딩을 사용하여 읽어들인다.

    ```python
    import pandas as pd
    import numpy as np
    from pandas import Timestamp
    from workalendar.asia import SouthKorea

    def preprocess_data(train_path, test_path):
        # 데이터 로드
        train = pd.read_csv(train_path, encoding='euc-kr')
        test = pd.read_csv(test_path, encoding='euc-kr')
    ```

2. **불필요한 컬럼 제거**:
    - 학습 데이터와 테스트 데이터에서 `사고번호` 컬럼을 제거한다.
    - `사고유형` 컬럼에서 세부 유형을 제거하여 단일 유형으로 만든다.

    ```python
        # 불필요한 컬럼 제거
        train.drop(columns=['사고번호'], inplace=True)
        test.drop(columns=['사고번호'], inplace=True)
        train['사고유형'] = train['사고유형'].str.split(' - ').str[0]
        test['사고유형'] = test['사고유형'].str.split(' - ').str[0]
    ```

3. **예측 시점에 알 수 없는 정보 제거**:
    - 테스트 데이터에서 예측 시점에 알 수 없는 정보(`사고내용`, `사망자수`, `중상자수`, `경상자수`, `부상신고자수`, `법규위반`, `가해운전자 차종`, `가해운전자 성별`, `가해운전자 연령`, `가해운전자 상해정도`, `피해운전자 차종`, `피해운전자 성별`, `피해운전자 연령`, `피해운전자 상해정도`)를 제거한다.

    ```python
        # 예측 시점에 알 수 없는 정보들 제거
        test_drop = ['사고내용', '사망자수', '중상자수', '경상자수', '부상신고자수', '법규위반',
                     '가해운전자 차종', '가해운전자 성별', '가해운전자 연령', '가해운전자 상해정도',
                     '피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도']
        test.drop(columns=test_drop, inplace=True)
    ```

4. **날짜 및 시간 정보 생성**:
    - 사고 일시에서 연, 월, 일, 시간을 추출하여 각각의 컬럼으로 분리한다.

    ```python
        # 날짜, 시간 정보 생성
        time_pattern = r'(\d{4})년 (\d{1,2})월 (\d{1,2})일 (\d{1,2})시'
        train[['연', '월', '일', '시간']] = train['사고일시'].str.extract(time_pattern)
        train[['연', '월', '일', '시간']] = train[['연', '월', '일', '시간']].apply(pd.to_numeric)
        test[['연', '월', '일', '시간']] = test['사고일시'].str.extract(time_pattern)
        test[['연', '월', '일', '시간']] = test[['연', '월', '일', '시간']].apply(pd.to_numeric)
    ```

5. **공휴일 컬럼 생성**:
    - `SouthKorea` 모듈을 사용하여 사고 발생일이 공휴일인지 여부를 판단한다.
    - 평일(공휴일 아님)은 0, 공휴일 또는 주말은 1로 표시한다.

    ```python
        # 공휴일 컬럼 생성
        cal = SouthKorea()
        def is_holiday(year, month, day):
            return cal.is_holiday(Timestamp(year, month, day))

        def classify_day(year, month, day):
            date = Timestamp(year, month, day)
            if date.dayofweek < 5 and not is_holiday(year, month, day):
                return 0
            else:
                return 1

        train['Holiday'] = train.apply(lambda row: classify_day(row['연'], row['월'], row['일']), axis=1)
        train.drop(columns=['사고일시', '일', '연'], inplace=True)
        test['Holiday'] = test.apply(lambda row: classify_day(row['연'], row['월'], row['일']), axis=1)
        test.drop(columns=['사고일시', '일', '연'], inplace=True)
    ```

6. **계절 컬럼 생성**:
    - 월 정보를 기반으로 봄, 여름, 가을, 겨울로 구분하여 계절 컬럼을 생성한다.

    ```python
        # 계절 컬럼 생성
        def categorize_season(month):
            if month in [3, 4, 5]:
                return '봄'
            elif month in [6, 7, 8]:
                return '여름'
            elif month in [9, 10, 11]:
                return '가을'
            else:
                return '겨울'

        train['계절'] = train['월'].apply(categorize_season)
        test['계절'] = test['월'].apply(categorize_season)
    ```

7. **출퇴근 시간 컬럼 생성**:
    - 출퇴근 시간(평일: 7-9시, 18-20시 / 주말: 18-22시)을 구분하여 출퇴근 시간 컬럼을 생성한다.

    ```python
        # 출퇴근 시간 컬럼 생성
        def rush_hour(hour, holiday):
            if (holiday == 0 and hour in [7, 8, 9, 18, 19, 20]) or (holiday == 1 and hour in [18, 19, 20, 21, 22]):
                return "Rush"
            else:
                return "NoRush"

        train['출퇴근'] = train.apply(lambda row: rush_hour(row['시간'], row['Holiday']), axis=1)
        test['출퇴근'] = test.apply(lambda row: rush_hour(row['시간'], row['Holiday']), axis=1)
    ```

8. **시군구 분리**:
    - `시군구` 컬럼을 `도시`, `구`, `동`으로 분리한다.

    ```python
        # 시군구 분리
        location_pattern = r'(\S+) (\S+) (\S+)'
        train[['도시', '구', '동']] = train['시군구'].str.extract(location_pattern)
        train.drop(columns=['시군구'], inplace=True)
        test[['도시', '구', '동']] = test['시군구'].str.extract(location_pattern)
        test.drop(columns=['시군구'], inplace=True)
    ```

9. **도로형태 분리**:
    - `도로형태` 컬럼을 `도로형태1`, `도로형태2`로 분리한다.

    ```python
        # 도로형태 분리
        road_pattern = r'(.+) - (.+)'
        train[['도로형태1', '도로형태2']] = train['도로형태'].str.extract(road_pattern)
        train.drop(columns=['도로형태'], inplace=True)
        test[['도로형태1', '도로형태2']] = test['도로형태'].str.extract(road_pattern)
        test.drop(columns=['도로형태'], inplace=True)
    ```

10. **나이 변수 전처리**:
    - `가해운전자 연령` 및 `피해운전자 연령` 컬럼에서 연령 정보를 숫자로 변환한다.
    - 연령 정보가 없는 경우 NaN으로 처리한다.

    ```python
        # 나이 변수 전처리
        def age_transform(x):
            try:
                ret = int(x.split('세')[0])
            except:
                ret = np.NaN
            return ret

        train['가해운전자 연령'] = train['가해운전자 연령'].apply(lambda x: age_transform(x))
        train['피해운전자 연령'] = train['피해운전자 연령'].apply(lambda x: age_transform(x))
    ```

11. **지역별 가해운전자 및 피해운전자 평균 연령 추출**:
    - `도시`, `구`, `동` 단위로 가해운전자와 피해운전자의 평균 연령을 계산하여 추가한다.

    ```python
        # 지역별 가해운전자 & 피해운전자 평균 연령 추출
        age_mean = train[['도시', '구', '동', '가해운전자 연령', '피해운전자 연령']].groupby(['도시', '구', '동']).mean()
        age_mean.columns = ['가해운전자 평균연령','피해운전자 평균연령']

        train = pd.merge(train, age_mean, how='left', on=['도시', '구', '동'])
        test = pd.merge(test, age_mean, how='left', on=['도시', '구', '동'])
    ```

12. **불필요한 컬럼 제거**:
    - 학습 데이터에서 예측에 불필요한 정보를 제거한다.

    ```python
        train_drop = ['사고내용', '사망자수', '중상자수', '경상자수', '부상신고자수',
                      '가해운전자 차종', '가해운전자 성별', '가해운전자 연령', '가해운전자 상해정도',
                      '피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도',
                      '월', '시간', '도시', '법규위반']
        train.drop(columns=train_drop, inplace=True)

        test_drop = ['월', '시간', '도시']
        test.drop(columns=test_drop, inplace=True)

        return train, test
    ```


---

### CatBoost 선정 이유

CatBoost는 범주형 변수를 효과적으로 처리할 수 있는 능력과 강력한 트리 기반 모델링 기능을 제공하기 때문에 선택되었다. CatBoost의 주요 장점은 다음과 같다:

1. **트리 분개(Tree Splitting)의 장점**:
    - 트리 분개는 복잡한 데이터 패턴을 효과적으로 캡처하고 비선형 관계를 잘 모델링할 수 있다.
    - 여러 변수 간의 상호작용을 자동으로 고려하며, 데이터의 스케일에 민감하지 않다.

2. **Ordered Target Statistics (TS)의 카테고리 처리 장점**:
    - CatBoost는 카테고리 데이터를 처리할 때, 표준적인 one-hot 인코딩 대신 순서가 있는 대상 통계(Ordered TS)를 사용하여 과적합을 방지하고 학습 속도를 향상시킨다.
    - 이 방법은 데이터 누출을 방지하고, 범주형 변수의 고유값이 많아도 효과적으로 학습할 수 있게 한다.

#### 데이터셋 준비

#### 결측값 처리
```python
# 결측값 처리 (결측값이 있는 경우)
train = train.fillna(0)
test = test.fillna(0)
```
결측값을 0으로 채운다.

#### 사고 유형별 데이터셋 분할 및 샘플링
```python
# 사고 유형별로 데이터셋 분할 및 샘플링
train1 = train[train["사고유형"] == "차량단독"]  # 차량단독
train2 = train[train["사고유형"] == "차대차"]  # 차대차
train3 = train[train["사고유형"] == "차대사람"]  # 차대사람

test1 = test[test["사고유형"] == "차량단독"]
test2 = test[test["사고유형"] == "차대차"]
test3 = test[test["사고유형"] == "차대사람"]
```
사고 유형에 따라 데이터를 세 개의 그룹으로 분리한다: 차량단독, 차대차, 차대사람.

#### 데이터셋 분할 함수 정의
```python
# 데이터셋 분할 함수 정의
def split_data(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

X_train1, y_train1 = split_data(train1, 'ECLO')
X_train2, y_train2 = split_data(train2, 'ECLO')
X_train3, y_train3 = split_data(train3, 'ECLO')

X_test1, y_test1 = split_data(test1, 'ECLO')
X_test2, y_test2 = split_data(test2, 'ECLO')
X_test3, y_test3 = split_data(test3, 'ECLO')
```
데이터셋을 특징(X)과 목표 변수(y)로 분리하는 함수를 정의하고 이를 통해 학습 및 테스트 데이터를 분할한다.

#### CatBoost 모델 학습 및 평가

#### RMSLE 계산 함수 정의
```python
# RMSLE 계산 함수 정의
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
```
모델의 예측 성능을 평가하기 위해 RMSLE (Root Mean Squared Log Error)를 계산하는 함수를 정의한다.

#### CatBoost 모델 학습 및 평가 함수 정의
```python
# CatBoost 모델 학습 및 평가 함수 정의
def train_and_evaluate(X_train, y_train, X_test, y_test, params, cat_features):
    model = CatBoostRegressor(random_seed=42, thread_count=-1, **params)
    model.fit(X_train, y_train, cat_features=cat_features, verbose=0)
    y_pred_test = model.predict(X_test)

    rmsle_value = rmsle(y_test, y_pred_test)
    r2_value = r2_score(y_test, y_pred_test)

    return model, rmsle_value, r2_value
```
CatBoost 모델을 학습하고 평가하는 함수를 정의한다. 모델은 주어진 파라미터와 범주형 특징을 사용하여 학습된다.

#### 범주형 피처 인덱스 추출
```python
# 범주형 피처 인덱스 추출
cat_features1 = np.where(X_train1.dtypes == object)[0]
cat_features2 = np.where(X_train2.dtypes == object)[0]
cat_features3 = np.where(X_train3.dtypes == object)[0]
```
범주형 특징의 인덱스를 추출하여 CatBoost 모델에 제공한다.

#### CatBoost 파라미터 설정

```python
# CatBoost 파라미터 설정
catboost_params1 = {
    'iterations': 6111,
    'od_wait': 609,
    'learning_rate': 0.04119744178592561,
    'reg_lambda': 5.488592836608496,
    'subsample': 0.9203846250162546,
    'random_strength': 14.883401811234684,
    'depth': 5,
    'min_data_in_leaf': 3,
    'leaf_estimation_iterations': 10,
    'bagging_temperature': 0.01789129283627206,
    'colsample_bylevel': 0.5972283837388589
}

catboost_params2 = {
    'iterations': 13411,
    'od_wait': 1144,
    'learning_rate': 0.021727882868213363,
    'reg_lambda': 21.015616790374914,
    'subsample': 0.878372685297051,
    'random_strength': 36.58060974949341,
    'depth': 7,
    'min_data_in_leaf': 7,
    'leaf_estimation_iterations': 9,
    'bagging_temperature': 0.05180927511974106,
    'colsample_bylevel': 0.579406682782964
}

catboost_params3 = {
    'iterations': 6500,
    'od_wait': 1641,
    'learning_rate': 0.039883471636645535,
    'reg_lambda': 8.723928583044282,
    'subsample': 0.8317293182421713,
    'random_strength': 22.481544108255296,
    'depth': 5,
    'min_data_in_leaf': 9,
    'leaf_estimation_iterations': 9,
    'bagging_temperature': 0.022614197973049137,
    'colsample_bylevel': 0.7882803255907027
}
```
각 사고 유형별로 CatBoost 모델의 하이퍼파라미터를 설정한다.

#### 모델 학습 및 평가
```python
# 모델 학습 및 평가
model1, rmsle1, r21 = train_and_evaluate(X_train1, y_train1, X_test1, y_test1, catboost_params1, cat_features1)
model2, rmsle2, r22 = train_and_evaluate(X_train2, y_train2, X_test2, y_test2, catboost_params2, cat_features2)
model3, rmsle3, r23 = train_and_evaluate(X_train3, y_train3, X_test3, y_test3, catboost_params3, cat_features3)

print(f'차량단독 사고 유형 - RMSLE: {rmsle1}, R²: {r21}')
print(f'차대차 사고 유형 - RMSLE: {rmsle2}, R²: {r22}')
print(f'차대사람 사고 유형 - RMSLE: {rmsle3}, R²: {r23}')
```
각 사고 유형별로 CatBoost 모델을 학습하고 평가한다. RMSLE와 R² 값을 출력한다.

## 예측 및 시각화

### 예측값을 테스트 데이터프레임에 추가
```python
# 예측값을 test 데이터프레임에 추가
test1['predict'] = model1.predict(X_test1)
test2['predict'] = model2.predict(X_test2)
test3['predict'] = model3.predict(X_test3)
```
각 모델의 예측값을 테스트 데이터프레임에 추가한다.



![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/7eec64e8-b8a4-4ecc-8b94-bc4dc9a7a25d)

![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/986696a1-5d73-4b96-9c88-b012ced61228)

![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/b1a7b379-91e0-4de1-a156-b0c2578ed765)


#### Feature Importance 분석
각 예측에서 얻을 수 있는 주요 위험 요인은 다음과 같다.
동, 구 : 서울시 내 지역별 공간적 특징이 교통사고 상해정도에 영향을 끼친다.
-> 동 별로 위험요인 분석을 추가적으로 진행해 구체적인 방안을 도출할 예정이다.
이 feature importance를 기반으로, 동 별 위험요인 분석에 사용할 feature selection을 진행한다.

[도로형태, 가해운전자 차종, 가해운전자 평균 연령 ]

동 별 예측을 하기 위해 추가적인 공공데이터를 수집했다.


## 공공데이터 전처리 (행정동, 법정동 단위 맞추기)
---
### 배경

- Raw data 의 사고 위치 → 법정동 기준
- 사용할 feature → 행정동 기준
- 따라서, 법정동 → 행정동으로의 변환이 필요.

### 변환 기준
#### 서울시 법정동, 행정동 살펴보기
기준 : 행정안전부 행정구역코드


![Untitled](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/161555261/ccdc5139-980b-4c32-b116-f01de332a891)

모든 동은 다음과 같은 기준으로 분류 가능
1. 유형 1 - 1. - 하나의 행정동에 여러 법정동을 완전히 포함시킬 수 있는 경우
- 용신동(행정동) $\supset$ 용두동, 신설동 (법정동)     
![d5f7fa10-a2f6-4e9a-af84-6529240e1a63](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/161555261/0ae637e1-bae4-4728-9864-59df2f172db4)
![Untitled 1](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/161555261/bc6187ae-c63e-47f0-a442-16d6e6aa5e67)
- 이러한 유형에 속하는 법정동은 ‘동리명’ 열에서 카운트시 1의 값을 가짐
- 이 경우 법정동 → 관할 행정동으로 변환
2. 유형 2 - 법정동 내 여러 행정동 포함시킬 수 있는 경우
![Untitled 2](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/161555261/4980440a-03a2-4d06-99a5-50b09c9cdaf6)
![Untitled 3](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/161555261/490a995c-cd2c-4bde-854c-64ef2fe58132)
- 노원구의 모든 동이 해당
- 공릉1동, 공릉2동(행정동) $\subset$ 공릉동(법정동)
- 이러한 유형에 속하는 행정동은 ‘읍면동명’ 열에서 카운트시 1의 값을 가짐
- 이 경우 모든 행정동을 합쳐 하나의 동으로 사용
3. 여러 경계에 걸치는 경우
- 세종로(법정동) - 세종대로를 기준으로 서쪽은 사직동, 동쪽은 종로1.2.3.4가동, 북쪽(경복궁, 청와대)은 청운효자동에 속함
- 행정동 경계(빨간색 선)에 세종로가 3개 동네로 나누어지는 것 확인
![ee86defb-154d-43cf-9fb7-b424ab2e0b81](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/161555261/8ecb729d-0154-4922-915c-a57d419ba79d)
- 신사동, 압구정동(강남구) : 서로의 경계가 어긋나있음
![cfa2e03d-d86f-479f-9dc7-a83c7c4355e5](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/161555261/7a7c975a-26dc-4acd-b258-231f776923d6)
![119bf0aa-51ce-4408-8ac6-177f3cbdb99e](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/161555261/3a21ccc1-a27b-4ffd-8abf-07f1469f086d)
4. 완전히 일치하는 경우
- ‘읍면동명’, ‘동리명’ 열에서 모두 1의 count 값을 가짐
- 예시 : 길동(강동구)
- 경계, 이름 완전 일치함
- 변환 없이 그대로 사용 가능
5. 동일한 이름을 가진 동
- 신사동(은평구), 신사동(관악구), 신사동(강남구)
- 삼성동(강남구), 삼성동(관악구)
- 신정동(양천구), 신정동(마포구)
- ‘시군구’에서 ‘00구’에 해당하는 문자열을 참조해 신사동(은평구)와 같이 변환한 뒤 사용
6. 기타
- 청운효자동(행정동)에 속하는 세종로를 제외한 모든 법정동 → 경계가 온전히 행정동 내에 속하므로 세종로를 제외하고 모두 청운효자동으로 변환하여 사용
- 이러한 특징의 동은 유형 1번과 같은 변환 방법을 따름
---
### 변환 과정
변환용 딕셔너리를 유형에 따라 두 개 제작
- 1번 : 법정동 $\subset$ 행정동
<pre>
<code>
dong_dict_1 = {

#종로구
'청운효자동': ['청운동', '신교동', '궁정동', '효자동', '창성동', '통인동', '누상동', '누하동', '옥인동'], 
'사직동': ['통의동', '적선동', '체부동', '필운동', '내자동', '사직동', '도렴동', '당주동', '내수동', '신문로1가', '신문로2가'], 
'삼청동': ['팔판동', '삼청동', '안국동', '소격동', '화동', '사간동', '송현동'], 
... #중략
#강동구
'길동': ['길동']
}
  
</code>
</pre>

- 2번 : 행정동 $\subset$ 법정동
<pre>
<code>
dong_dict_2 = {

#종로구
'창신동': ['창신제1동', '창신제2동', '창신제3동'], 
'숭인동': ['숭인제1동', '숭인제2동'], 
...#중략

'천호동': ['천호제1동', '천호제2동', '천호제3동'], 
'성내동': ['성내제1동', '성내제2동', '성내제3동'], 
'둔촌동': ['둔촌제1동', '둔촌제2동']

}
</code>
</pre>
포함 관계를 key와 value로 표현, 변환해야 할 동네 대입 시 key값으로 변환해주는 함수 제작하여 사용
<pre>
<code>
def find_hang(convert_dict, dong_convert):
    for dong, dong_list in convert_dict.items():
        if dong_convert in dong_list:
            return dong
    return "not found"

find_hang(dong_dict_1, '용두동') #용신동
</code>
</pre>
유형에 따라 각각 사용해야 하므로, raw 데이터셋에서는 새로운 행을 만들어 변환하고, 2 유형에 속하는 동(not found로 반환됨)에 대해서는 한번 더 참조하여 변환해줄 필요가 있음
<pre>
<code>
#1번에서 행정동 not found인 동 재변환
def replace_administrative_dong(dong_dict, row):
    if row['행정동'] == 'not found':
        beopjeongdong = row['dong_name']
        if beopjeongdong in dong_dict:
            return row['dong_name']
    return 'not found'
</code>
</pre>
위 함수를 이용해 로우 데이터셋의 ‘행정동’ 행을 참조해 값이 ‘not found’인 행에서 ‘법정동’ 행을 참조해 그 값을 그대로 반환함
---
### 데이터 전처리 과정
- 사용한 데이터 : 행정동 인구수, 노인인구수, 노면주차장 수
- 데이터 별 행정동 표기 상이 (가락1동, 가락제1동과 같은 식)
- 숫자 앞의 ‘제’자가 있다면 제거하고 사용
    - 위 방법을 사용할 경우 문제되는 ‘홍제동’의 경우 ‘홍동’으로 변환되므로 ‘홍동’은 ‘홍제동’으로 인지하고, 차후 처리를 거침
<pre>
<code>
#표기법 통일을 위한 함수
import re

def unify_dong_names(dong_name):
    # 숫자 바로 앞에 '제'자를 제거
    dong_name = re.sub(r'제(\d)', r'\1', dong_name)
    return dong_name

test_dong = ['묵1동', '홍제제1동', '가양제2동'] #묵1동, 홍1동, 가양2동
for x in test_dong :
    print(unify_dong_names(x))
</code>
</pre>
- 위에서 사용한 변환 함수 이용해 2번 유형에 속하는 행정동 변환
- 변환 이후 동일한 이름을 가진 행 여러 개 생김(공릉1동 → 공릉동, 공릉2동 → 공릉동) ⇒ 동일한 이름 가진 행의 값을 합쳐서 사용
<pre>
<code>
df_hjd_total['dong_name'] = df_hjd_total['dong_name'].apply(find_bub, args=(unified_dong_dict,))
df_grouped_2 = df_hjd_total.groupby('dong_name').agg({
    'senior': 'sum',
    'total_population': 'sum',
    'street parking': 'sum'
}).reset_index()
</code>
</pre>

- 변환 뒤 데이터
![Untitled 6](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/161555261/5198034d-f0b5-4d0e-9c35-1b3173335a16)
- senior_rate : 노인인구비율(합산 후 노인인구수/총인구수로 계산하여 추가)
- raw data에서 ‘행정동’ 컬럼 값을 참조해 그 옆에 해당하는 행정동의 total_population, street parking, senior_ratio 값을 추가함
---
### 동별 평균 ECLO 예측 모델



#### "평균 ECLO" 예측을 위한 RandomForestRegressor 및 Feature Selection

 `RandomForestRegressor` 모델을 사용하여 "평균 ECLO"를 예측하고, feature importances를 기반으로 중요한 특징을 선택하여 모델 성능을 향상시키는 것을 목표로 한다.


#### 1. 데이터 로드 및 준비

데이터를 로드하고, "행정동" 열을 제거하여 전처리한다. (행정동 컬럼은 공공데이터 merge 용이기 때문)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
data1 = pd.read_csv(r"C:\Users\thdfy\OneDrive\바탕 화면\Datamining_Teampro\dong_predict\data\2022_final_차대사람.csv", encoding = 'euc-kr')
data2 = pd.read_csv(r"C:\Users\thdfy\OneDrive\바탕 화면\Datamining_Teampro\dong_predict\data\2022_final_차대차.csv", encoding = 'euc-kr')
# 데이터 전처리: "행정동" 열 제거
data1.drop(columns='행정동', inplace=True)
data2.drop(columns='행정동', inplace=True)
```

#### 2. 한국어 폰트 설정

시각화를 위해 한국어 폰트를 설정한다.

```python
# 한국어 폰트 설정
font_path = "c:/Windows/Fonts/malgun.ttf"  # Windows의 경우
fontprop = fm.FontProperties(fname=font_path, size=12)
plt.rc('font', family=fontprop.get_name())
```

#### 3. 데이터 준비 함수

데이터를 학습용과 테스트용으로 분리한다.

```python
# 모델을 위한 데이터 준비 함수
def prepare_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 4. 모델 학습 및 평가 함수

`RandomForestRegressor` 모델을 학습시키고, 예측 성능을 평가한다. 또한, feature importances를 반환한다.

```python
# 모델 학습 및 평가 함수
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    feature_importances = model.feature_importances_
    return r2, feature_importances, X_train.columns
```

#### 5. 중요한 특징 선택 함수

초기 모델에서 얻은 feature importances를 기준으로 중요한 특징만을 선택한다.

```python
# 중요한 특징 선택 함수
def select_important_features(X_train, X_test, importances, threshold=0.02):
    important_features = [feature for feature, importance in zip(X_train.columns, importances) if importance > threshold]
    return X_train[important_features], X_test[important_features]
```

#### 6. 초기 모델 학습 및 feature importances 얻기

초기 모델을 학습시키고 feature importances를 얻는다.

```python
# 초기 모델 학습 및 feature importances 얻기
X_train_data1, X_test_data1, y_train_data1, y_test_data1 = prepare_data(data1, '평균 ECLO')
r2_data1, feature_importances_data1, feature_names_data1 = train_and_evaluate_model(X_train_data1, X_test_data1, y_train_data1, y_test_data1)

X_train_data2, X_test_data2, y_train_data2, y_test_data2 = prepare_data(data2, '평균 ECLO')
r2_data2, feature_importances_data2, feature_names_data2 = train_and_evaluate_model(X_train_data2, X_test_data2, y_train_data2, y_test_data2)
```

#### 7. 초기 R-squared 값 출력

초기 모델의 R-squared 값을 출력한다.

```python
# 초기 R-squared 값 출력
print(f'Initial R2 Score for 차대사람 dataset: {r2_data1}')
print(f'Initial R2 Score for 차대차 dataset: {r2_data2}')
```

#### 8. 중요한 특징 선택

초기 feature importances를 기반으로 중요한 특징을 선택한다.

```python
# 초기 feature importances를 기반으로 중요한 특징 선택
X_train_data1_selected, X_test_data1_selected = select_important_features(X_train_data1, X_test_data1, feature_importances_data1)
X_train_data2_selected, X_test_data2_selected = select_important_features(X_train_data2, X_test_data2, feature_importances_data2)
```

#### 9. 선택된 특징으로 모델 재학습 및 평가

선택된 특징을 사용하여 모델을 재학습하고 성능을 평가한다.

```python
# 선택된 특징으로 모델 재학습 및 평가
r2_data1_selected, feature_importances_data1_selected, _ = train_and_evaluate_model(X_train_data1_selected, X_test_data1_selected, y_train_data1, y_test_data1)
r2_data2_selected, feature_importances_data2_selected, _ = train_and_evaluate_model(X_train_data2_selected, X_test_data2_selected, y_train_data2, y_test_data2)
```

#### 10. feature selection 후 R-squared 값 출력

feature selection 후 R-squared 값을 출력한다.

```python
# feature selection 후 R-squared 값 출력
print(f'R2 Score for 차대사람 dataset after feature selection: {r2_data1_selected}')
print(f'R2 Score for 차대차 dataset after feature selection: {r2_data2_selected}')
```

#### 11. 중요 특징들 데이터프레임으로 정리

중요 특징들을 데이터프레임으로 정리한다.

```python
# 중요 특징들 데이터프레임으로 정리
feature_importances_df1 = pd.DataFrame({'Feature': X_train_data1_selected.columns, 'Importance': feature_importances_data1_selected}).sort_values(by='Importance', ascending=False)
feature_importances_df2 = pd.DataFrame({'Feature': X_train_data2_selected.columns, 'Importance': feature_importances_data2_selected}).sort_values(by='Importance', ascending=False)
```

#### 12. 중요 특징 시각화 함수

중요 특징을 시각화하는 함수를 정의하고, 이를 사용하여 그래프를 그린다.

```python
# 중요 특징 시각화 함수
def plot_feature_importances(feature_importances_df, title):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances_df['Feature'][:10], feature_importances_df['Importance'][:10])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

# 중요 특징 시각화
plot_feature_importances(feature_importances_df1, 'Top 10 Feature Importances for 차대사람 Dataset')
plot_feature_importances(feature_importances_df2, 'Top 10 Feature Importances for 차대차 Dataset')
```

#### 13. 상위 10개 중요한 특징 출력

각 데이터셋에서 상위 10개의 중요한 특징을 출력한다.

```python
# 상위 10개 중요한 특징 출력
print('Top 10 Feature Importances for 차대사람 Dataset:')
print(feature_importances_df1.head(10))
print('Top 10 Feature Importances for 차대차 Dataset:')
print(feature_importances_df2.head(10))
```

### 결과 분석

- 초기 모델의 R-squared 값:
  - 차대사람 데이터셋: 0.425
  - 차대차 데이터셋: 0.261

- feature selection 후 모델의 R-squared 값:
  - 차대사람 데이터셋: 0.427
  - 차대차 데이터셋: 0.243

feature selection을 통해 차대사람 데이터셋의 성능이 소폭 향상되었다.


![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/ed39cb12-f9a9-495c-8934-7087f6088098)

![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/4a685c81-5301-4ab6-9dde-220eebc4adc4)











