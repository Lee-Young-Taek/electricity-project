import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# 한글 폰트 설정 (Windows 기본 값: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 경로
BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_DATA_PATH = BASE_DIR / 'data' / 'train.csv'
TEST_DATA_PATH = BASE_DIR / 'data' / 'test.csv'
WEATHER_DATA_PATH = BASE_DIR / 'data' / '청주_기상_2024년도.csv'

# 데이터 로드
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
weather_df = pd.read_csv(WEATHER_DATA_PATH, encoding='cp949')

# 데이터 기본 정보
train_df.info()
train_df.isna().sum()
train_df.head()

test_df.info()
test_df.isna().sum()
test_df.head()

weather_df.info()
weather_df.isna().sum()
weather_df.head()

# 범주형 컬럼 확인 (동일 확인)
train_df["작업유형"].unique()
train_df["작업유형"].value_counts()

test_df["작업유형"].unique()
test_df["작업유형"].value_counts()

# 측정일시 데이터 타입 변환
train_df["측정일시"] = pd.to_datetime(train_df["측정일시"], format="%Y-%m-%d %H:%M:%S")
test_df["측정일시"] = pd.to_datetime(test_df["측정일시"], format="%Y-%m-%d %H:%M:%S")

# 모든 측정일시 데이터 년도 18년도로 변경
train_df["측정일시"] = train_df["측정일시"].apply(lambda dt: dt.replace(year=2018))
test_df["측정일시"] = test_df["측정일시"].apply(lambda dt: dt.replace(year=2018))

####################################################################################################
# 시간축 (15분 간격) 무결성 확인
####################################################################################################
for label, df in [("train", train_df), ("test", test_df)]:
    df_sorted = df.sort_values("측정일시").copy()

    diffs = df_sorted["측정일시"].diff().dropna()
    vc = diffs.value_counts(normalize=True).sort_values(ascending=False).head(5)
    print("Top time diffs (ratio):")
    print(vc)

    expected = pd.Timedelta(minutes=15)
    pct_expected = (diffs == expected).mean() * 100.0
    print(f"Share of expected interval ({15} min): {pct_expected:.2f}%")
    
    dup_ts = df_sorted.duplicated(subset=["측정일시"]).sum()
    print("Duplicate timestamps:", dup_ts)
    
    per_day = df_sorted.groupby(df_sorted["측정일시"].dt.date).size().describe()
    print("Per-day row count describe:", per_day)


####################################################################################################
# 단가 패턴 (TOU) 확인
####################################################################################################
# 음수 요금/0 이하 사용량은 데이터 오류 또는 정전 등 특수 상황으로 간주할 수 있음.
neg_target = (train_df["전기요금(원)"] < 0).sum()
zero_kwh = (train_df["전력사용량(kWh)"] <= 0).sum()

# 사용량이 0 이하인 데이터 확인 (1건)
train_df.loc[train_df["전력사용량(kWh)"] == 0, :]

# 주변 데이터 확인
train_df.iloc[29850:29860, :]

# 관측 단가(원/kWh) 분포 요약
eps = 1e-9  # 0 나눗셈 방지용 작은 수
unit_price = train_df["전기요금(원)"] / (train_df["전력사용량(kWh)"] + eps)
print("\nUnit price describe:\n", unit_price.describe(percentiles=[.1,.5,.9]))

# 시간대(시)별 중앙값: 시간이 바뀌면 단가가 체계적으로 달라지는지(TOU) 확인
hour = train_df["측정일시"].dt.hour
hourly_med = unit_price.groupby(hour).median()
print("\nHourly median unit price (first 12):\n", hourly_med)
print("→ 시간대별 중앙값 차이가 크면 TOU 신호가 존재. 과거 통계로 '추정단가' 피처를 만들어 사용.")
# TOU가 존재한다고 판단

# 월별로 시간대(시)별 중앙값
month = train_df["측정일시"].dt.month
hourly_monthly_med = unit_price.groupby([train_df["측정일시"].dt.hour, month]).median().unstack()
print("\nHourly-Monthly median unit price (first 3 months):\n", hourly_monthly_med)

# 시간대의 월별 단가 변화 시각화
plt.figure(figsize=(10,6))
for m in range(1,12):
    plt.plot(hourly_monthly_med.index, hourly_monthly_med[m], label=f'Month {m}')
plt.xlabel('Hour of Day')
plt.ylabel('Median Unit Price (원/kWh)')
plt.legend(title='월', bbox_to_anchor=(1.02, 1), loc='upper left')

# 월별로 시간대(시)별 평균
hourly_monthly_mean = unit_price.groupby([train_df["측정일시"].dt.hour, month]).mean().unstack()
print("\nHourly-Monthly median unit price (first 3 months):\n", hourly_monthly_mean)

# 시간대의 월별 단가 변화 시각화
plt.figure(figsize=(10,6))
for m in range(1,12):
    plt.plot(hourly_monthly_mean.index, hourly_monthly_mean[m], label=f'Month {m}')
plt.xlabel('Hour of Day')
plt.ylabel('Median Unit Price (원/kWh)')
plt.legend(title='월', bbox_to_anchor=(1.02, 1), loc='upper left')

# 21시 파생변수 고려해보면 좋을거같음.

####################################################################################################
# 전처리
####################################################################################################

####################################################################################################
# 시간 관련 파생변수 생성
####################################################################################################
train_df["year"] = train_df["측정일시"].dt.year
train_df["month"] = train_df["측정일시"].dt.month
train_df["day"] = train_df["측정일시"].dt.day
train_df["hour"] = train_df["측정일시"].dt.hour
train_df["minute"] = train_df["측정일시"].dt.minute
train_df["second"] = train_df["측정일시"].dt.second
train_df["weekday"] = train_df["측정일시"].dt.weekday
train_df["is_weekend"] = train_df["weekday"].isin([5,6]).astype(int)
pd.set_option('display.max_rows', None)
train_df.loc[(train_df["hour"] == 12), :]

highlight_periods_2018 = [
    ("2018-01-01", "2018-01-01"),
    ("2018-02-15", "2018-02-17"),
    ("2018-03-01", "2018-03-01"),
    ("2018-05-05", "2018-05-05"),
    ("2018-05-07", "2018-05-07"),
    ("2018-05-22", "2018-05-22"),
    ("2018-06-06", "2018-06-06"),
    ("2018-06-13", "2018-06-13"),
    ("2018-08-15", "2018-08-15"),
    ("2018-09-23", "2018-09-25"),
    ("2018-09-26", "2018-09-26"),
    ("2018-10-03", "2018-10-03"),
    ("2018-10-09", "2018-10-09"),
    ("2018-12-25", "2018-12-25"),
]

highlight_dates_2018 = set()
for start, end in highlight_periods_2018:
    highlight_dates_2018.update(pd.date_range(start=start, end=end, freq="D").date)

train_df["is_special_day"] = train_df["측정일시"].dt.date.apply(lambda d: int(d in highlight_dates_2018))

for month_value in sorted(train_df["month"].unique()):
    month_slice = train_df[train_df["month"] == month_value]
    if month_slice.empty:
        continue

    daily_profile = (
        month_slice
        .groupby("day")
        .agg(
            mean_kwh=("전력사용량(kWh)", "mean"),
            weekend_flag=("is_weekend", "mean"),
            holiday_flag=("is_special_day", "mean")
        )
        .reset_index()
    )

    bar_colors = []
    for is_holiday, is_weekend in zip(daily_profile["holiday_flag"], daily_profile["weekend_flag"]):
        if is_holiday:
            bar_colors.append("gold")
        elif is_weekend:
            bar_colors.append("red")
        else:
            bar_colors.append("steelblue")

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        daily_profile["day"],
        daily_profile["mean_kwh"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5
    )
    ax.set_title(f"{month_value}월 일별 평균 전력사용량")
    ax.set_xlabel("일")
    ax.set_ylabel("평균 전력사용량(kWh)")

    legend_handles = [
        Patch(facecolor="steelblue", edgecolor="black", label="평일"),
        Patch(facecolor="red", edgecolor="black", label="주말"),
        Patch(facecolor="gold", edgecolor="black", label="공휴일")
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_xticks(daily_profile["day"])
    ax.set_xticklabels(daily_profile["day"], rotation=0)

    value_labels = [f"{val:.1f}" for val in daily_profile["mean_kwh"]]
    ax.bar_label(bars, labels=value_labels, padding=3, fontsize=9)
    fig.tight_layout()
    plt.show()


    # add(pd.to_datetime("2018-01-01").date())  # 신정
    # add_range("2018-02-15", "2018-02-17")     # 설날 연휴
    # add(pd.to_datetime("2018-03-01").date())  # 삼일절
    # add(pd.to_datetime("2018-05-07").date())  # 어린이날 대체공휴일
    # add(pd.to_datetime("2018-05-22").date())  # 부처님오신날
    # add(pd.to_datetime("2018-06-06").date())  # 현충일
    # add(pd.to_datetime("2018-06-13").date())  # 지방선거
    # add(pd.to_datetime("2018-08-15").date())  # 광복절
    # add_range("2018-09-22", "2018-09-26")     # 추석 연휴 및 대체공휴일
    # add(pd.to_datetime("2018-10-03").date())  # 개천절
    # add(pd.to_datetime("2018-10-09").date())  # 한글날
    # add(pd.to_datetime("2018-12-25").date())  # 성탄절

# 2024년 연휴 여부 컬럼 생성
holiday_periods = [
	("2024-01-01", "2024-01-01"),
	("2024-02-09", "2024-02-12"),
	("2024-03-01", "2024-03-01"),
	("2024-04-10", "2024-04-10"),
	("2024-05-05", "2024-05-05"),
	("2024-05-06", "2024-05-06"),
	("2024-05-15", "2024-05-15"),
	("2024-06-06", "2024-06-06"),
	("2024-08-15", "2024-08-15"),
	("2024-09-17", "2024-09-19"),
	("2024-10-03", "2024-10-03"),
	("2024-10-09", "2024-10-09"),
	("2024-12-25", "2024-12-25"),
]

holiday_dates = set()
for start_date, end_date in holiday_periods:
	holiday_dates.update(pd.date_range(start=start_date, end=end_date, freq="D").date)

train_df["is_holiday"] = train_df["측정일시"].dt.date.apply(lambda d: int(d in holiday_dates))

# 계절 컬럼 생성
season_mapping = {
    11: 'winter', 12: 'winter', 1: 'winter',
    2: 'spring', 3: 'spring', 4: 'spring',
    5: 'summer', 6: 'summer', 7: 'summer', 8: 'summer',
    9: 'autumn', 10: 'autumn'
}
train_df["season"] = train_df["month"].map(season_mapping)

# year, second 컬럼 제거 (단일값)
train_df["year"].nunique()
train_df["second"].nunique()
train_df.drop(columns=["year", "second"], inplace=True)

####################################################################################################
# 기상 데이터 병합
####################################################################################################
weather_df.drop(columns=["지점", "지점명"], inplace=True)
weather_df["일시"] = pd.to_datetime(weather_df["일시"], format="%Y-%m-%d %H:%M")

train_df["측정일시_분"] = train_df["측정일시"].dt.floor("H")
train_df = pd.merge(train_df, weather_df, left_on="측정일시_분", right_on="일시", how="left")
train_df.drop(columns=["측정일시_분", "일시"], inplace=True)

# 결측치 재확인
train_df.isna().sum()
# 강수량 결측치는 0으로 대체
train_df["강수량(mm)"].fillna(0, inplace=True)

# 기상 데이터에서 누락 (9월19일, 9월20일)
train_df.loc[train_df.isna().sum(axis=1)>0, :]

# 결측치 처리 - 전날 데이터로 채우기
for idx in train_df[train_df["기온(°C)"].isna()].index:
    prev_day = train_df.loc[idx - 24*3]
    train_df.loc[idx, ["기온(°C)", "습도(%)", "지면온도(°C)"]] = prev_day[["기온(°C)", "습도(%)", "지면온도(°C)"]]

####################################################################################################
# 전력량 0인 데이터 선형 보간
####################################################################################################
zero_kwh_indices = train_df[train_df["전력사용량(kWh)"] <= 0].index
train_df.loc[zero_kwh_indices, :]
train_df.info()

prev_val = train_df.loc[zero_kwh_indices - 1,:]
next_val = train_df.loc[zero_kwh_indices + 1,:]
train_df.loc[zero_kwh_indices, "전력사용량(kWh)"] = (prev_val["전력사용량(kWh)"].values + next_val["전력사용량(kWh)"].values) / 2
train_df.loc[zero_kwh_indices, "지상무효전력량(kVarh)"] = (prev_val["지상무효전력량(kVarh)"].values + next_val["지상무효전력량(kVarh)"].values) / 2
train_df.loc[zero_kwh_indices, "진상무효전력량(kVarh)"] = (prev_val["진상무효전력량(kVarh)"].values + next_val["진상무효전력량(kVarh)"].values) / 2
train_df.loc[zero_kwh_indices, "탄소배출량(tCO2)"] = (prev_val["탄소배출량(tCO2)"].values + next_val["탄소배출량(tCO2)"].values) / 2
train_df.loc[zero_kwh_indices, "지상역률(%)"] = (prev_val["지상역률(%)"].values + next_val["지상역률(%)"].values) / 2
train_df.loc[zero_kwh_indices, "진상역률(%)"] = (prev_val["진상역률(%)"].values + next_val["진상역률(%)"].values) / 2
train_df.loc[zero_kwh_indices, "전기요금(원)"] = (prev_val["전기요금(원)"].values + next_val["전기요금(원)"].values) / 2

train_df.to_csv(BASE_DIR / 'data' / 'processed' / 'yt' / 'v1_train_full.csv', index=False)

# 11월 데이터 기준 train/valid 분리 저장
train_processed = train_df[train_df["month"] != 11].copy()
valid_processed = train_df[train_df["month"] == 11].copy()

train_processed.to_csv(BASE_DIR / 'data' / 'processed' / 'yt' / 'v1_train_split_.csv', index=False)
valid_processed.to_csv(BASE_DIR / 'data' / 'processed' / 'yt' / 'v1_valid_split.csv', index=False)