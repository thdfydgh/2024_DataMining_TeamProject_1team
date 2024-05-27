# 서울시 교통사고 상해예측 & 위험요인 분석
---
## 활용 Data : TAAS(Traffic Accident Analysis System) 교통사고 분석 시스템 Data
https://taas.koroad.or.kr/web/shp/sbm/initGisAnals.do?menuId=WEB_KMP_GIS_TAS 
교통사고분석시스템에서 제공하는 서울시 지역별 Dataset을 구축하였다.
![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/6e613ca4-6ed4-467d-9441-91bf0873628d)
Data의 형태는 다음과 같다.
![image](https://github.com/thdfydgh/2024_DataMining_TeamProject_1team/assets/126649413/588dc067-23f7-42d1-a9ff-9a84cd77541b)
우리는 이 Dataset을 기반으로, Equivalent Casualty Loss Only(ECLO) 즉, 교통사고 상해정도를 예측하고 위험 요인 분석을 한다.
ECLO = 사망자수 * 10 + 중상자수 * 5 + 경상자수 * 3 + 부상자수 * 1
<pre>
사망자분류모델 -> feature selection -> 차대 사람 유의미 -> ECLO예측 --> 동별 상해정도 차이 유의미  -> 동별 예측 -> merge -> 동별 대책 수립(결론)
</pre>

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

교통사고 피해예측 EDA.ipynb : 각 feature별 ECLO값 EDA
---


































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
- raw data에서 ‘행정동’ 컬럼 값을 참조해 그 옆에 해당하는 행정동의 senior, total_population, street parking, senior_ratio 값을 추가함
---

## Modeling 
































