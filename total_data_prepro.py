import pandas as pd
import numpy as np
from pandas import Timestamp
from workalendar.asia import SouthKorea

def preprocess_data(train_path, test_path):
    # 데이터 로드
    train = pd.read_csv(train_path, encoding='euc-kr')
    test = pd.read_csv(test_path, encoding='euc-kr')

    # 불필요한 컬럼 제거
    train.drop(columns=['사고번호'], inplace=True)
    test.drop(columns=['사고번호'], inplace=True)
    train['사고유형'] = train['사고유형'].str.split(' - ').str[0]
    test['사고유형'] = test['사고유형'].str.split(' - ').str[0]

    # 예측 시점에 알 수 없는 정보들 제거
    test_drop = ['사고내용', '사망자수', '중상자수', '경상자수', '부상신고자수', '법규위반',
                 '가해운전자 차종', '가해운전자 성별', '가해운전자 연령', '가해운전자 상해정도',
                 '피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도']
    test.drop(columns=test_drop, inplace=True)

    # 날짜, 시간 정보 생성
    time_pattern = r'(\d{4})년 (\d{1,2})월 (\d{1,2})일 (\d{1,2})시'
    train[['연', '월', '일', '시간']] = train['사고일시'].str.extract(time_pattern)
    train[['연', '월', '일', '시간']] = train[['연', '월', '일', '시간']].apply(pd.to_numeric)

    test[['연', '월', '일', '시간']] = test['사고일시'].str.extract(time_pattern)
    test[['연', '월', '일', '시간']] = test[['연', '월', '일', '시간']].apply(pd.to_numeric)

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

    # 출퇴근 시간 컬럼 생성
    def rush_hour(hour, holiday):
        if (holiday == 0 and hour in [7, 8, 9, 18, 19, 20]) or (holiday == 1 and hour in [18, 19, 20, 21, 22]):
            return "Rush"
        else:
            return "NoRush"

    train['출퇴근'] = train.apply(lambda row: rush_hour(row['시간'], row['Holiday']), axis=1)
    test['출퇴근'] = test.apply(lambda row: rush_hour(row['시간'], row['Holiday']), axis=1)

    # 시군구 분리
    location_pattern = r'(\S+) (\S+) (\S+)'
    train[['도시', '구', '동']] = train['시군구'].str.extract(location_pattern)
    train.drop(columns=['시군구'], inplace=True)

    test[['도시', '구', '동']] = test['시군구'].str.extract(location_pattern)
    test.drop(columns=['시군구'], inplace=True)

    # 도로형태 분리
    road_pattern = r'(.+) - (.+)'
    train[['도로형태1', '도로형태2']] = train['도로형태'].str.extract(road_pattern)
    train.drop(columns=['도로형태'], inplace=True)

    test[['도로형태1', '도로형태2']] = test['도로형태'].str.extract(road_pattern)
    test.drop(columns=['도로형태'], inplace=True)

    # 나이 변수 전처리
    def age_transform(x):
        try:
            ret = int(x.split('세')[0])
        except:
            ret = np.NaN
        return ret

    train['가해운전자 연령'] = train['가해운전자 연령'].apply(lambda x: age_transform(x))
    train['피해운전자 연령'] = train['피해운전자 연령'].apply(lambda x: age_transform(x))

    # 지역별 가해운전자 & 피해운전자 평균 연령 추출
    age_mean = train[['도시', '구', '동', '가해운전자 연령', '피해운전자 연령']].groupby(['도시', '구', '동']).mean()
    age_mean.columns = ['가해운전자 평균연령', '피해운전자 평균연령']

    train = pd.merge(train, age_mean, how='left', on=['도시', '구', '동'])
    test = pd.merge(test, age_mean, how='left', on=['도시', '구', '동'])

    train_drop = ['사고내용', '사망자수', '중상자수', '경상자수', '부상신고자수',
                  '가해운전자 차종', '가해운전자 성별', '가해운전자 연령', '가해운전자 상해정도',
                  '피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도',
                  '월', '시간', '도시', '법규위반']
    train.drop(columns=train_drop, inplace=True)

    test_drop = ['월', '시간', '도시']
    test.drop(columns=test_drop, inplace=True)

    return train, test
