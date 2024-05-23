import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({'font.family':'MalgunGothic'})
mpl.rc('axes', unicode_minus=False)

def preprocess_data(df):
    # 결측값 제거
    df.dropna(inplace=True)
    
    # 사망사고 여부 라벨링
    df['사망사고여부'] = df['사고내용'] == '사망사고'
    
    # 불필요한 컬럼 제거
    df.drop(['사고번호', '시군구', '사망자수', '중상자수', '경상자수', '부상신고자수', '노면상태',
             '가해운전자 상해정도', '피해운전자 상해정도', '사고내용', '가해운전자 차종', '피해운전자 차종'], axis=1, inplace=True)
    
    # 사고일시에서 시간 정보 추출
    df['사고시각'] = df['사고일시'].str.extract(r'(\d{2})(?=:00)').astype(int)
    df.drop(['사고일시'], axis=1, inplace=True)
    
    # 야간 여부 분류
    df['야간여부'] = df['사고시각'].apply(lambda x: 1 if 22 <= x <= 23 or 0 <= x <= 6 else 0)
    df.drop(['사고시각'], axis=1, inplace=True)
    
    # 주말 여부 분류
    df['주말여부'] = df['요일'].apply(lambda x: 1 if x in ['토요일', '일요일'] else 0)
    df.drop('요일', axis=1, inplace=True)
    
    # 성별 이진 분류
    df['가해운전자 성별- 여성0 남성1'] = df['가해운전자 성별'] == '남'
    df['피해운전자 성별- 여성0 남성1'] = df['피해운전자 성별'] == '남'
    df.drop(['가해운전자 성별', '피해운전자 성별'], axis=1, inplace=True)
    
    # 데이터 타입 변환
    df['야간여부'] = df['야간여부'].astype(int)
    df['주말여부'] = df['주말여부'].astype(int)
    df['사망사고여부'] = df['사망사고여부'].astype(int)
    df['가해운전자 성별- 여성0 남성1'] = df['가해운전자 성별- 여성0 남성1'].astype(int)
    df['피해운전자 성별- 여성0 남성1'] = df['피해운전자 성별- 여성0 남성1'].astype(int)
    
    # 연령 전처리
    df = df[df['가해운전자 연령'] != '98세 이상']
    df = df[df['피해운전자 연령'] != '98세 이상']
    df = df[df['가해운전자 연령'] != '미분류']
    df = df[df['피해운전자 연령'] != '미분류']
    
    df['가해운전자 연령'] = df['가해운전자 연령'].str.extract(r'(\d+)').astype(int)
    df['피해운전자 연령'] = df['피해운전자 연령'].str.extract(r'(\d+)').astype(int)
    
    # 연령 정규화
    scaler = StandardScaler()
    df[['가해운전자 연령', '피해운전자 연령']] = scaler.fit_transform(df[['가해운전자 연령', '피해운전자 연령']])
    df.rename(columns={'가해운전자 연령': '가해운전자 연령(정규화 됨)', '피해운전자 연령': '피해운전자 연령(정규화 됨)'}, inplace=True)
    
    # 컬럼 재배치
    df = df[['주말여부', '야간여부', '사고유형', '법규위반', '기상상태', '도로형태',
             '가해운전자 성별- 여성0 남성1', '가해운전자 연령(정규화 됨)',
             '피해운전자 성별- 여성0 남성1', '피해운전자 연령(정규화 됨)', '사망사고여부']]
    
    # 원-핫 인코딩
    features = ['사고유형', '법규위반', '기상상태', '도로형태']
    df = pd.get_dummies(df, columns=features, prefix=features)
    
    return df