import pandas as pd
import numpy as np

df = pd.read_csv('diabetes_data.csv') # csv 파일 불러기 pd.read_csv
group_mean = df.groupby("Outcome").mean() # Outcome 별 평균값 계산
mean_diff = group_mean.loc[1] - group_mean.loc[0] # Outcome이 1인 그룹의 평균 Outcome이 0인 그룹의 평균의 차이

print(mean_diff.sort_values(ascending=False)) # 차이값 내림차순 정렬해서 print




df = pd.read_csv('PowerConsumption.csv')
df['DateTime'] = pd.to_datetime(df['DateTime']) # 문자열 데이터를 datetime으로 변경
df['Month'] = df['DateTime'].dt.month # df['DateTime'].dt.month를 사용해서 DateTime 열에서 month만 추출
# df['Month'] 에 저정
month_counts = df.groupby('Month').size().reset_index(name='Count')
# df['Month']별로 나눠서 사이즈를 재고, 숫자 인덱스로 초기화하고, 각 그룹마다 해당 size를 열 이름 Count로 지정함
print(month_counts)


df = pd.read_csv("worldcupgoals.csv")

def two_years(years): # 임의의 함수 생성
    return len(str(years).split("-"))
# 문자열로 되어있는 year를 -를 제외하고 리스트로 만듦

df["two"] = df["Years"].apply(two_years)
# 각 Years 마다 two_years 함수 적용 (for문 대용), 그 값을 df["two"]에 넣음
two_caps = df[df["two"] == 2]
# df["two"] 가 2 인 값만 two_caps에 넣음
top_scorer = two_caps.sort_values(by="Goals", ascending=False).iloc[0]
# two_caps를 내림차순으로 정렬해서 맨 위 값을 뽑아 냄 (2번 출전하고, Goals가 제일 높은 행)
print(top_scorer)
