import pandas as pd

# 파일 불러오기
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
submission_df = pd.read_csv("./data/sample_submission.csv")

train_df.columns
train_df.head()
train_df.tail()

# 날짜 변환
train_df['측정일시'] = pd.to_datetime(train_df['측정일시'])
test_df['측정일시'] = pd.to_datetime(test_df['측정일시'])

# 날짜 정보 추출
train_df['month'] = train_df['측정일시'].dt.month
train_df['day'] = train_df['측정일시'].dt.day
train_df['hour'] = train_df['측정일시'].dt.hour
train_df['minute'] = train_df['측정일시'].dt.minute

test_df['day'] = test_df['측정일시'].dt.day
test_df['hour'] = test_df['측정일시'].dt.hour
test_df['minute'] = test_df['측정일시'].dt.minute

# 1~11월 데이터만 사용
jan_to_nov = train_df[train_df['month'] < 12]

# 동일 시각 평균 계산
grouped_avg = (
    jan_to_nov
    .groupby(['day', 'hour', 'minute'])['전기요금(원)']
    .mean()
    .reset_index()
    .rename(columns={'전기요금(원)': '예측_전기요금'})
)

# test와 병합
predicted_df = test_df.merge(grouped_avg, on=['day', 'hour', 'minute'], how='left')

# 예측값 삽입
submission_df['target'] = predicted_df['예측_전기요금']

# 저장
submission_df.to_csv("./data/sample_submission_filled.csv", index=False)