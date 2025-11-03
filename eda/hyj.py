import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/train.csv")
df['측정일시'] = pd.to_datetime(df['측정일시'])

df.head()
df.tail()

df.info()
df.isna().sum() # 결측치 없음

df.describe() # 기초 통계

### 시계열 기준 데이터 시각화

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

df = df.sort_values('측정일시')

cols = [
    '전력사용량(kWh)',
    '지상무효전력량(kVarh)',
    '진상무효전력량(kVarh)',
    '탄소배출량(tCO2)',
    '지상역률(%)',
    '진상역률(%)'
]

n = len(cols)
plt.figure(figsize=(14, 3 * n))  # 변수 개수에 따라 높이 자동 조절

for i, col in enumerate(cols, start=1):
    plt.subplot(n, 1, i)
    plt.plot(df['측정일시'], df[col], label=col, color='tab:blue')
    plt.title(col)
    plt.xlabel('시간')
    plt.ylabel('값')
    plt.grid(True)
    plt.tight_layout()

plt.show()


### 범주형 변수 확인
df["작업유형"].unique()
df["작업유형"].value_counts()
# Light_Load
# Medium_Load
# Maximum_Load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 0) 로드 & 기본 확인 =====
df = pd.read_csv("./data/train.csv")
df['측정일시'] = pd.to_datetime(df['측정일시'], errors='coerce')

# === 자정(00:00) 오표기 보정: 00:00 이면서 직전 행과 '같은 날짜'로 찍힌 경우만 +1일 ===
midnight = (df['측정일시'].dt.hour == 0) & (df['측정일시'].dt.minute == 0)
same_day_as_prev = df['측정일시'].dt.normalize().eq(df['측정일시'].shift(1).dt.normalize())
bug_mask = midnight & same_day_as_prev
df.loc[bug_mask, '측정일시'] = df.loc[bug_mask, '측정일시'] + pd.Timedelta(days=1)

# 정렬
df = df.sort_values('측정일시').reset_index(drop=True)

# ===== 기본 출력 =====
print("=== head ==="); print(df.head(), "\n")
print("=== tail ==="); print(df.tail(), "\n")
print("=== info ==="); print(df.info(), "\n")
print("=== 결측치 합계 ==="); print(df.isna().sum(), "\n")
print("=== 기술통계(수치) ===")
print(df.select_dtypes(include=[np.number]).describe(percentiles=[.01,.25,.5,.75,.99]).T, "\n")

# ===== 1) 15분 간격/결측/중복 체크 =====
df_idx = df.set_index('측정일시')
full_idx = pd.date_range(df_idx.index.min(), df_idx.index.max(), freq='15T')

missing_times = full_idx.difference(df_idx.index)
dup_count = df_idx.index.duplicated().sum()

print(f"[간격체크] 빠진 타임스탬프 수: {len(missing_times)}")
print(f"[간격체크] 중복 타임스탬프 수: {dup_count}\n")

# ===== 2) 시계열 시각화(핵심 변수) =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

cols = [
    '전력사용량(kWh)',
    '지상무효전력량(kVarh)',
    '진상무효전력량(kVarh)',
    '탄소배출량(tCO2)',
    '지상역률(%)',
    '진상역률(%)',
    '전기요금(원)'
]
cols_plot = [c for c in cols if c in df.columns]

n = len(cols_plot)
if n > 0:
    plt.figure(figsize=(14, 2.6 * n))
    for i, col in enumerate(cols_plot, start=1):
        plt.subplot(n, 1, i)
        plt.plot(df['측정일시'], df[col], label=col)
        plt.title(col); plt.xlabel('시간'); plt.ylabel('값'); plt.grid(True)
    plt.tight_layout(); plt.show()

# ===== 3) 범주형(작업유형) 확인 =====
if '작업유형' in df.columns:
    print("=== 작업유형 .unique() ==="); print(df["작업유형"].unique(), "\n")
    print("=== 작업유형 .value_counts() ==="); print(df["작업유형"].value_counts(), "\n")
else:
    print("[알림] '작업유형' 컬럼이 없습니다.\n")

# ===== 4) 캘린더 피처(테스트에서도 만들 수 있는 것만) =====
df['hour'] = df['측정일시'].dt.hour
df['dow'] = df['측정일시'].dt.dayofweek      # 월=0 … 일=6
df['month'] = df['측정일시'].dt.month
df['is_weekend'] = (df['dow'] >= 5).astype(int)

def season(m):
    return 'summer' if m in [6,7,8] else ('winter' if m in [11,12,1,2] else 'spring_fall')
df['season'] = df['month'].map(season)

# ===== 5) 파생 지표(요금 분해에 꼭 필요한 3가지) =====
# (1) 단가(원/kWh) = 전기요금 / 전력사용량
safe = ('전력사용량(kWh)' in df.columns) & ('전기요금(원)' in df.columns)
if safe:
    mask = df['전력사용량(kWh)'] > 1e-6
    df['단가(원/kWh)'] = np.where(mask, df['전기요금(원)'] / df['전력사용량(kWh)'], np.nan)

# (2) 무효합 & 무효비 = (지상+진상)/사용량
if {'지상무효전력량(kVarh)','진상무효전력량(kVarh)'}.issubset(df.columns):
    df['무효합(kVarh)'] = df['지상무효전력량(kVarh)'] + df['진상무효전력량(kVarh)']
    if '전력사용량(kWh)' in df.columns:
        mask = df['전력사용량(kWh)'] > 1e-6
        df['무효비'] = np.where(mask, df['무효합(kVarh)'] / df['전력사용량(kWh)'], np.nan)

# (3) PF(%) = 지상/진상 중 큰 값(효율감)
if {'지상역률(%)','진상역률(%)'}.issubset(df.columns):
    df['PF(%)'] = df[['지상역률(%)','진상역률(%)']].max(axis=1)

# 빠르게 시계열 확인
quick_cols = ['전기요금(원)','전력사용량(kWh)','단가(원/kWh)','무효비','PF(%)']
quick_cols = [c for c in quick_cols if c in df.columns]

if len(quick_cols) > 0:
    plt.figure(figsize=(14, 2.6 * len(quick_cols)))
    for i, col in enumerate(quick_cols, start=1):
        plt.subplot(len(quick_cols), 1, i)
        plt.plot(df['측정일시'], df[col])
        plt.title(f"{col} (시계열)"); plt.xlabel('시간'); plt.ylabel(col); plt.grid(True)
    plt.tight_layout(); plt.show()

# ===== 6) 시간/요일/월/작업유형별 평균 요금 =====
def bar_mean(by, target='전기요금(원)'):
    base_cols = [c for c in [target,'전력사용량(kWh)','단가(원/kWh)','무효비','PF(%)'] if c in df.columns]
    if (by not in df.columns) or (target not in df.columns) or len(base_cols) == 0:
        return None
    g = df.groupby(by)[base_cols].mean()
    ax = g[target].plot(kind='bar', rot=0, figsize=(10,3), title=f'{by}별 평균 {target}')
    ax.set_xlabel(by); ax.set_ylabel('평균 요금(원)'); plt.tight_layout(); plt.show()
    return g

g_hour = bar_mean('hour')
g_dow  = bar_mean('dow')
g_mon  = bar_mean('month')

# 작업유형 상위만 보기(많을 수 있어서)
if '작업유형' in df.columns and '전기요금(원)' in df.columns:
    top_types = df['작업유형'].value_counts().head(10).index
    g_type = df[df['작업유형'].isin(top_types)].groupby('작업유형')[[c for c in ['전기요금(원)','전력사용량(kWh)'] if c in df.columns]] \
             .mean().sort_values('전기요금(원)')
    ax = g_type['전기요금(원)'].plot(kind='barh', figsize=(8,4), title='작업유형별 평균 요금(상위 10개)')
    ax.set_xlabel('평균 요금(원)'); ax.set_ylabel('작업유형'); plt.tight_layout(); plt.show()

# ===== 7) 피크 구간(상위 5%) — 비용 리스크 확인 =====
if '전기요금(원)' in df.columns:
    q95 = df['전기요금(원)'].quantile(0.95)
    peak = df[df['전기요금(원)'] >= q95]
    print(f"[피크] 상위 5% 개수: {len(peak)}")
    if len(peak) > 0:
        peak['hour'].value_counts().sort_index().plot(kind='bar', figsize=(10,3), title='피크(상위 5%) 발생 시간대')
        plt.xlabel('hour'); plt.ylabel('count'); plt.tight_layout(); plt.show()
        if '작업유형' in peak.columns:
            print("[피크 시간대 작업유형 Top5]\n", peak['작업유형'].value_counts().head(5), "\n")

# ===== 8) 간단 상관 — 타깃과 주요 수치 변수 (스피어만) =====
num_for_corr = [c for c in [
    '전력사용량(kWh)','지상무효전력량(kVarh)','진상무효전력량(kVarh)',
    '탄소배출량(tCO2)','지상역률(%)','진상역률(%)',
    '무효합(kVarh)','무효비','단가(원/kWh)','전기요금(원)'
] if c in df.columns]

if '전기요금(원)' in num_for_corr:
    corr = df[num_for_corr].corr(method='spearman')['전기요금(원)'].sort_values(ascending=False)
    print("=== Spearman 상관(타깃 vs 변수) ==="); print(corr, "\n")
    corr.drop('전기요금(원)').plot(kind='bar', figsize=(10,3), title='Spearman 상관 (전기요금 vs 변수)')
    plt.ylabel('상관계수'); plt.tight_layout(); plt.show()
else:
    print("[알림] 상관 분석을 위한 '전기요금(원)' 컬럼이 없습니다.\n")
    
# -*- coding: utf-8 -*-
"""
전기요금(원) 분포 탐색 대시 플롯 (월/일/시/분/계절)
입력 CSV 컬럼 예:
id,측정일시,전력사용량(kWh),지상무효전력량(kVarh),진상무효전력량(kVarh),탄소배출량(tCO2),지상역률(%),진상역률(%),작업유형,전기요금(원)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# ====== 설정 ======
CSV_PATH = "./data/train.csv"   # <- 너의 파일 경로로 바꿔줘
DT_COL   = "측정일시"
TARGET   = "전기요금(원)"

# 한글 폰트(윈도우: 맑은 고딕)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set(style="whitegrid")

# ====== 로드 & 전처리 ======
df = pd.read_csv(CSV_PATH)
if DT_COL not in df.columns or TARGET not in df.columns:
    raise ValueError(f"CSV에 '{DT_COL}', '{TARGET}' 컬럼이 필요합니다.")

df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce")
df = df.dropna(subset=[DT_COL]).sort_values(DT_COL).reset_index(drop=True)

# 날짜·시간 파생
df["year"]      = df[DT_COL].dt.year
df["month"]     = df[DT_COL].dt.month
df["day"]       = df[DT_COL].dt.day
df["hour"]      = df[DT_COL].dt.hour
df["minute"]    = df[DT_COL].dt.minute
df["dayofweek"] = df[DT_COL].dt.dayofweek  # 0=월, 6=일

# 계절 맵(한국형 4계절)
def season_kr(m):
    if m in (3,4,5):   return "봄"
    if m in (6,7,8):   return "여름"
    if m in (9,10,11): return "가을"
    return "겨울"      # 12,1,2
df["season"] = df["month"].apply(season_kr)

# 15분 데이터의 minute은 보통 {0,15,30,45}
df["minute_cat"] = df["minute"].astype(str).str.zfill(2)  # '00','15','30','45'

# 기본 통계(참고용)
print("=== 전기요금(원) 기본 통계 ===")
print(df[TARGET].describe(percentiles=[.1,.25,.5,.75,.9]).to_string())
print()

# ====== 헬퍼: 안전한 y축 범위(이상치로 스케일 깨지는 것 방지) ======
q1, q99 = np.nanpercentile(df[TARGET], [1, 99])
ymin, ymax = max(0, q1*0.8), q99*1.2

# ====== 1. 전체 분포(히스토그램 + KDE) ======
plt.figure(figsize=(10,5))
sns.histplot(df[TARGET], bins=60, kde=True)
plt.ylim(0, None)
plt.title("전기요금(원) 분포 (전체)")
plt.xlabel(TARGET)
plt.tight_layout()
plt.show()

# ====== 2. 월별 박스플롯 ======
plt.figure(figsize=(12,5))
sns.boxplot(data=df, x="month", y=TARGET, showfliers=False)
plt.title("월별 전기요금(원) 분포 (Boxplot)")
plt.ylim(ymin, ymax)
plt.xlabel("월")
plt.tight_layout()
plt.show()

# ====== 3. 일별 라인(월 구분 색상) ======
# 한 달만 볼 수도 있으니, year-month 단위로 그룹핑
df["ym"] = df[DT_COL].dt.to_period("M").astype(str)
plt.figure(figsize=(12,5))
sns.lineplot(data=df.groupby([df[DT_COL].dt.date])[[TARGET]].mean().reset_index(),
             x=DT_COL, y=TARGET)
plt.title("일별 전기요금(원) 평균 (전체 기간)")
plt.xlabel("날짜")
plt.ylabel("평균 전기요금(원)")
plt.tight_layout()
plt.show()

# ====== 4. 시간대별(0~23시) 박스플롯/바이올린 ======
fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)
sns.boxplot(data=df, x="hour", y=TARGET, showfliers=False, ax=axes[0])
axes[0].set_title("시간별 전기요금(원) 분포 (Boxplot)")
axes[0].set_xlabel("시(hour)")
axes[0].set_ylim(ymin, ymax)

sns.violinplot(data=df, x="hour", y=TARGET, inner="quartile", cut=0, ax=axes[1])
axes[1].set_title("시간별 전기요금(원) 분포 (Violin)")
axes[1].set_xlabel("시(hour)")
axes[1].set_ylim(ymin, ymax)

plt.tight_layout()
plt.show()

# ====== 5. 분(00/15/30/45) 박스플롯 ======
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="minute_cat", y=TARGET, showfliers=False)
plt.title("분(00/15/30/45)별 전기요금(원) 분포")
plt.xlabel("분(minute)")
plt.ylim(ymin, ymax)
plt.tight_layout()
plt.show()

# ====== 6. 계절별 박스플롯 + 시간×계절 피벗 히트맵 ======
plt.figure(figsize=(8,5))
order = ["봄","여름","가을","겨울"]
sns.boxplot(data=df, x="season", y=TARGET, order=order, showfliers=False)
plt.title("계절별 전기요금(원) 분포")
plt.xlabel("계절")
plt.ylim(ymin, ymax)
plt.tight_layout()
plt.show()

# 시간×계절 히트맵(평균)
pivot = df.pivot_table(index="hour", columns="season", values=TARGET, aggfunc="mean")
pivot = pivot[order]  # 계절 순서 맞춤
plt.figure(figsize=(7,6))
sns.heatmap(pivot, cmap="YlOrRd", annot=False, fmt=".0f")
plt.title("시간×계절 전기요금(원) 평균 히트맵")
plt.ylabel("시(hour)")
plt.xlabel("계절")
plt.tight_layout()
plt.show()

# ====== 7. 월×시간 히트맵(평균) ======
pivot_mh = df.pivot_table(index="hour", columns="month", values=TARGET, aggfunc="mean")
plt.figure(figsize=(10,6))
sns.heatmap(pivot_mh, cmap="YlGnBu", annot=False, fmt=".0f")
plt.title("월×시간 전기요금(원) 평균 히트맵")
plt.ylabel("시(hour)")
plt.xlabel("월")
plt.tight_layout()
plt.show()

# ====== 8. (선택) 주중/주말 비교 라인 ======
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
avg_by_hour_week = df.groupby(["is_weekend","hour"])[TARGET].mean().reset_index()
plt.figure(figsize=(10,5))
sns.lineplot(data=avg_by_hour_week, x="hour", y=TARGET, hue="is_weekend", marker="o")
plt.title("시간대별 전기요금(원) 평균: 평일(0) vs 주말(1)")
plt.xlabel("시(hour)")
plt.ylabel("평균 전기요금(원)")
plt.legend(title="주말 여부")
plt.tight_layout()
plt.show()

# ---------- 2024 공휴일(대체공휴일 제외) 목록 ----------
# 이미지/사용자 제공 기준: 신정, 설연휴(2/10~2/12), 삼일절, 국회의원 선거, 어린이날, 부처님오신날,
# 현충일, 광복절, 추석연휴(9/16~9/18), 개천절, 한글날, 크리스마스
# ※ '근로자의날(5/1)'은 법정 공휴일이 아니므로 기본 제외. 필요 시 리스트에 추가하세요.
HOLIDAYS_2024 = set()
def add(d): HOLIDAYS_2024.add(pd.to_datetime(d).date())
def add_range(s, e):
    for d in pd.date_range(s, e, freq="D"):
        HOLIDAYS_2024.add(d.date())

add("2024-01-01")                     # 신정
add_range("2024-02-10", "2024-02-12") # 설 연휴
add("2024-03-01")                     # 삼일절
add("2024-04-10")                     # 22대 국회의원 선거
add("2024-05-05")                     # 어린이날  (※ 5/6 대체공휴일은 제외)
add("2024-05-15")                     # 부처님 오신 날
add("2024-06-06")                     # 현충일
add("2024-08-15")                     # 광복절
add_range("2024-09-16", "2024-09-18") # 추석 연휴
add("2024-10-03")                     # 개천절
add("2024-10-09")                     # 한글날
add("2024-12-25")                     # 크리스마스

# 필요하면 근로자의날 포함: HOLIDAYS_2024.add(pd.to_datetime("2024-05-01").date())

# ---------- 공휴일 플래그 ----------
df["is_holiday"] = df["date"].isin(HOLIDAYS_2024).astype(int)

# ---------- EDA: 요약 통계 ----------
def summarize(group_col="is_holiday", target="전기요금(원)"):
    g = (df
         .dropna(subset=[target])
         .groupby(group_col)[target]
         .agg(["count", "mean", "median", "std", "min", "max"])
         .rename(index={0:"평일", 1:"공휴일"}))
    print("\n[공휴일 여부별 전기요금 요약 통계]")
    print(g.round(2))
    return g

summary_table = summarize()

# ---------- 시각화: 박스플롯 ----------
plt.figure(figsize=(6, 5))
data0 = df.loc[df["is_holiday"]==0, "전기요금(원)"].dropna().values
data1 = df.loc[df["is_holiday"]==1, "전기요금(원)"].dropna().values
plt.boxplot([data0, data1], labels=["평일", "공휴일"], showfliers=False)
plt.title("공휴일 여부에 따른 전기요금(원) 분포 (박스플롯)")
plt.ylabel("전기요금(원)")
plt.tight_layout()
plt.show()

# ---------- 시각화: 히스토그램(정규화) ----------
plt.figure(figsize=(7, 5))
bins = 50
plt.hist(data0, bins=bins, density=True, alpha=0.5, label="평일")
plt.hist(data1, bins=bins, density=True, alpha=0.5, label="공휴일")
plt.title("공휴일 vs 평일 전기요금(원) 히스토그램(정규화)")
plt.xlabel("전기요금(원)")
plt.ylabel("밀도")
plt.legend()
plt.tight_layout()
plt.show()

# (선택) 월·요일 조합으로도 공휴일 영향 확인
# 피벗: 월-요일별 평균 요금(공휴일/평일 분리)
pivot = (df.dropna(subset=["전기요금(원)"])
           .assign(month=df["측정일시"].dt.month,
                   dow=df["측정일시"].dt.dayofweek)
           .pivot_table(index=["month","dow"], columns="is_holiday",
                        values="전기요금(원)", aggfunc="mean"))
pivot.columns = ["평일평균", "공휴일평균"]
print("\n[월-요일 조합별 평일/공휴일 평균 요금(일부 미존재 조합은 NaN)]")
print(pivot.round(1).head(12))

### 상관관계

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 0) 로드 & 기본 확인 =====
df = pd.read_csv("./data/train.csv")
df['측정일시'] = pd.to_datetime(df['측정일시'], errors='coerce')

# === 자정(00:00) 오표기 보정: 00:00 이면서 직전 행과 '같은 날짜'로 찍힌 경우만 +1일 ===
midnight = (df['측정일시'].dt.hour == 0) & (df['측정일시'].dt.minute == 0)
same_day_as_prev = df['측정일시'].dt.normalize().eq(df['측정일시'].shift(1).dt.normalize())
bug_mask = midnight & same_day_as_prev
df.loc[bug_mask, '측정일시'] = df.loc[bug_mask, '측정일시'] + pd.Timedelta(days=1)

# 정렬
df = df.sort_values('측정일시').reset_index(drop=True)

# ===== 기본 출력 =====
print("=== head ==="); print(df.head(), "\n")
print("=== tail ==="); print(df.tail(), "\n")
print("=== info ==="); print(df.info(), "\n")
print("=== 결측치 합계 ==="); print(df.isna().sum(), "\n")
print("=== 기술통계(수치) ===")
print(df.select_dtypes(include=[np.number]).describe(percentiles=[.01,.25,.5,.75,.99]).T, "\n")

# ===== 1) 15분 간격/결측/중복 체크 =====
df_idx = df.set_index('측정일시')
full_idx = pd.date_range(df_idx.index.min(), df_idx.index.max(), freq='15T')

missing_times = full_idx.difference(df_idx.index)
dup_count = df_idx.index.duplicated().sum()

print(f"[간격체크] 빠진 타임스탬프 수: {len(missing_times)}")
print(f"[간격체크] 중복 타임스탬프 수: {dup_count}\n")

# ===== 2) 시계열 시각화(핵심 변수) =====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

cols = [
    '전력사용량(kWh)',
    '지상무효전력량(kVarh)',
    '진상무효전력량(kVarh)',
    '탄소배출량(tCO2)',
    '지상역률(%)',
    '진상역률(%)',
    '전기요금(원)'
]
cols_plot = [c for c in cols if c in df.columns]

n = len(cols_plot)
if n > 0:
    plt.figure(figsize=(14, 2.6 * n))
    for i, col in enumerate(cols_plot, start=1):
        plt.subplot(n, 1, i)
        plt.plot(df['측정일시'], df[col], label=col)
        plt.title(col); plt.xlabel('시간'); plt.ylabel('값'); plt.grid(True)
    plt.tight_layout(); plt.show()

# ===== 3) 범주형(작업유형) 확인 =====
if '작업유형' in df.columns:
    print("=== 작업유형 .unique() ==="); print(df["작업유형"].unique(), "\n")
    print("=== 작업유형 .value_counts() ==="); print(df["작업유형"].value_counts(), "\n")
else:
    print("[알림] '작업유형' 컬럼이 없습니다.\n")

# ===== 4) 캘린더 피처(테스트에서도 만들 수 있는 것만) =====
df['hour'] = df['측정일시'].dt.hour
df['dow'] = df['측정일시'].dt.dayofweek      # 월=0 … 일=6
df['month'] = df['측정일시'].dt.month
df['is_weekend'] = (df['dow'] >= 5).astype(int)

def season(m):
    return 'summer' if m in [6,7,8] else ('winter' if m in [11,12,1,2] else 'spring_fall')
df['season'] = df['month'].map(season)

# ===== 5) 파생 지표(요금 분해에 꼭 필요한 3가지) =====
# (1) 단가(원/kWh) = 전기요금 / 전력사용량
safe = ('전력사용량(kWh)' in df.columns) & ('전기요금(원)' in df.columns)
if safe:
    mask = df['전력사용량(kWh)'] > 1e-6
    df['단가(원/kWh)'] = np.where(mask, df['전기요금(원)'] / df['전력사용량(kWh)'], np.nan)

# (2) 무효합 & 무효비 = (지상+진상)/사용량
if {'지상무효전력량(kVarh)','진상무효전력량(kVarh)'}.issubset(df.columns):
    df['무효합(kVarh)'] = df['지상무효전력량(kVarh)'] + df['진상무효전력량(kVarh)']
    if '전력사용량(kWh)' in df.columns:
        mask = df['전력사용량(kWh)'] > 1e-6
        df['무효비'] = np.where(mask, df['무효합(kVarh)'] / df['전력사용량(kWh)'], np.nan)

# (3) PF(%) = 지상/진상 중 큰 값(효율감)
if {'지상역률(%)','진상역률(%)'}.issubset(df.columns):
    df['PF(%)'] = df[['지상역률(%)','진상역률(%)']].max(axis=1)

# 빠르게 시계열 확인
quick_cols = ['전기요금(원)','전력사용량(kWh)','단가(원/kWh)','무효비','PF(%)']
quick_cols = [c for c in quick_cols if c in df.columns]

if len(quick_cols) > 0:
    plt.figure(figsize=(14, 2.6 * len(quick_cols)))
    for i, col in enumerate(quick_cols, start=1):
        plt.subplot(len(quick_cols), 1, i)
        plt.plot(df['측정일시'], df[col])
        plt.title(f"{col} (시계열)"); plt.xlabel('시간'); plt.ylabel(col); plt.grid(True)
    plt.tight_layout(); plt.show()

# ===== 6) 시간/요일/월/작업유형별 평균 요금 =====
def bar_mean(by, target='전기요금(원)'):
    base_cols = [c for c in [target,'전력사용량(kWh)','단가(원/kWh)','무효비','PF(%)'] if c in df.columns]
    if (by not in df.columns) or (target not in df.columns) or len(base_cols) == 0:
        return None
    g = df.groupby(by)[base_cols].mean()
    ax = g[target].plot(kind='bar', rot=0, figsize=(10,3), title=f'{by}별 평균 {target}')
    ax.set_xlabel(by); ax.set_ylabel('평균 요금(원)'); plt.tight_layout(); plt.show()
    return g

g_hour = bar_mean('hour')
g_dow  = bar_mean('dow')
g_mon  = bar_mean('month')

# 작업유형 상위만 보기(많을 수 있어서)
if '작업유형' in df.columns and '전기요금(원)' in df.columns:
    top_types = df['작업유형'].value_counts().head(10).index
    g_type = df[df['작업유형'].isin(top_types)].groupby('작업유형')[[c for c in ['전기요금(원)','전력사용량(kWh)'] if c in df.columns]] \
             .mean().sort_values('전기요금(원)')
    ax = g_type['전기요금(원)'].plot(kind='barh', figsize=(8,4), title='작업유형별 평균 요금(상위 10개)')
    ax.set_xlabel('평균 요금(원)'); ax.set_ylabel('작업유형'); plt.tight_layout(); plt.show()

# ===== 7) 피크 구간(상위 5%) — 비용 리스크 확인 =====
if '전기요금(원)' in df.columns:
    q95 = df['전기요금(원)'].quantile(0.95)
    peak = df[df['전기요금(원)'] >= q95]
    print(f"[피크] 상위 5% 개수: {len(peak)}")
    if len(peak) > 0:
        peak['hour'].value_counts().sort_index().plot(kind='bar', figsize=(10,3), title='피크(상위 5%) 발생 시간대')
        plt.xlabel('hour'); plt.ylabel('count'); plt.tight_layout(); plt.show()
        if '작업유형' in peak.columns:
            print("[피크 시간대 작업유형 Top5]\n", peak['작업유형'].value_counts().head(5), "\n")

# ===== 8) 간단 상관 — 타깃과 주요 수치 변수 (스피어만) =====
num_for_corr = [c for c in [
    '전력사용량(kWh)','지상무효전력량(kVarh)','진상무효전력량(kVarh)',
    '탄소배출량(tCO2)','지상역률(%)','진상역률(%)',
    '무효합(kVarh)','무효비','단가(원/kWh)','전기요금(원)'
] if c in df.columns]

if '전기요금(원)' in num_for_corr:
    corr = df[num_for_corr].corr(method='spearman')['전기요금(원)'].sort_values(ascending=False)
    print("=== Spearman 상관(타깃 vs 변수) ==="); print(corr, "\n")
    corr.drop('전기요금(원)').plot(kind='bar', figsize=(10,3), title='Spearman 상관 (전기요금 vs 변수)')
    plt.ylabel('상관계수'); plt.tight_layout(); plt.show()
else:
    print("[알림] 상관 분석을 위한 '전기요금(원)' 컬럼이 없습니다.\n")



# 1) 변수들끼리 상관행렬 (Spearman)
cols = num_for_corr  # 이미 만든 리스트 사용
corr_mat = df[cols].corr(method='spearman')

print("=== 변수들끼리 Spearman 상관행렬 ===")
print(corr_mat.round(3))

# 2) 히트맵 (상삼각 마스킹 = 중복 제거)
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(np.where(mask, np.nan, corr_mat), vmin=-1, vmax=1, cmap='coolwarm')
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right')
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
ax.set_title('변수들끼리 Spearman 상관 히트맵')
cbar = plt.colorbar(im, ax=ax, shrink=0.85); cbar.set_label('상관계수')
plt.tight_layout(); plt.show()

# (Seaborn 사용 가능하면 더 깔끔)
# import seaborn as sns
# sns.heatmap(corr_mat, mask=mask, vmin=-1, vmax=1, annot=True, fmt=".2f",
#             cmap="coolwarm", linewidths=.5, cbar_kws={'label': '상관계수'})
# plt.title('변수들끼리 Spearman 상관 히트맵'); plt.tight_layout(); plt.show()

# 3) 절댓값 기준 상위 상관쌍 TOP-N 출력 (자기자신/중복 제외)
upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
top_pairs = (upper.stack()
                  .reindex(upper.stack().abs().sort_values(ascending=False).index))
N = 10
print(f"\n=== 절댓값 기준 상위 {N}개 상관쌍 ===")
print(top_pairs.head(N).round(3))






weather = pd.read_csv("./data/청주_기상_2024년도.csv", encoding="cp949")
weather.head(10)
weather.tail(10)

weather.info()


# ==============================
# 0) 라이브러리 & 한글 폰트
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트: 가능한 후보들 중 설치된 것 자동 선택
from matplotlib import font_manager, rcParams
candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
available = {f.name for f in font_manager.fontManager.ttflist}
rcParams["font.family"] = next((f for f in candidates if f in available), rcParams.get("font.family", "sans-serif"))
rcParams["axes.unicode_minus"] = False

# ==============================
# 1) 데이터 로드 (경로만 바꿔)
# ==============================
# 원본(전기요금 등) + 날씨
df_main = pd.read_csv("./data/train.csv")      # or 통합 데이터
weather = pd.read_csv("./data/청주_기상_2024년도.csv", encoding="cp949")    # 질문에서 준 포맷

# ==============================
# 2) datetime 파싱 & 기본 키 컬럼
# ==============================
DT_MAIN = "측정일시"
DT_WTHR = "일시"
TARGET  = "전기요금(원)"
ADD_YEAR_KEY = False   # 여러 해가 섞여 있으면 True 로!

df = df_main.copy()
w  = weather.copy()

df[DT_MAIN] = pd.to_datetime(df[DT_MAIN], errors="coerce")
w[DT_WTHR]  = pd.to_datetime(w[DT_WTHR],  errors="coerce")
if df[DT_MAIN].isna().any() or w[DT_WTHR].isna().any():
    bad_df = df[df[DT_MAIN].isna()].head()
    bad_w  = w[w[DT_WTHR].isna()].head()
    raise ValueError(f"[에러] 날짜 파싱 실패 row 존재\nmain 예시:\n{bad_df}\n\nweather 예시:\n{bad_w}")

# 메인 쪽 시간 파생(조인 키 및 참고용)
df["year"]  = df[DT_MAIN].dt.year
df["month"] = df[DT_MAIN].dt.month
df["day"]   = df[DT_MAIN].dt.day
df["dayofweek"] = df[DT_MAIN].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# 날씨 쪽 키
w["year"]  = w[DT_WTHR].dt.year
w["month"] = w[DT_WTHR].dt.month
w["day"]   = w[DT_WTHR].dt.day

# ==============================
# 3) 날씨 일단위 집계 (기온/습도/지면온도=평균, 강수량=합)
# ==============================
# 원본 컬럼명 가정: '기온(°C)', '강수량(mm)', '습도(%)', '지면온도(°C)'
for col in ["기온(°C)", "강수량(mm)", "습도(%)", "지면온도(°C)"]:
    if col not in w.columns:
        raise ValueError(f"[에러] 날씨 데이터에 '{col}' 컬럼이 없습니다.")

w["강수량(mm)"] = w["강수량(mm)"].fillna(0.0)  # 강수량 결측은 0으로

# 연도 키 포함 여부 결정
group_keys = ["month","day"] if not ADD_YEAR_KEY else ["year","month","day"]
use_cols   = group_keys + ["기온(°C)", "강수량(mm)", "습도(%)", "지면온도(°C)"]

daily = (
    w[use_cols]
      .groupby(group_keys, dropna=False)
      .agg({
          "기온(°C)": "mean",
          "습도(%)": "mean",
          "지면온도(°C)": "mean",
          "강수량(mm)": "sum"
      })
      .reset_index()
      .rename(columns={
          "기온(°C)"  : "일평균_기온(°C)",
          "습도(%)"   : "일평균_습도(%)",
          "지면온도(°C)": "일평균_지면온도(°C)",
          "강수량(mm)": "일강수량(mm)"
      })
)

# ==============================
# 4) 메인과 조인
# ==============================
join_keys = ["month","day"] if not ADD_YEAR_KEY else ["year","month","day"]
merged = df.merge(daily, on=join_keys, how="left")

# ==============================
# 5) 날씨 파생변수 생성
# ==============================
# 5-1) 강수 유무/강도
merged["강수_유무"] = (merged["일강수량(mm)"] > 0).astype(int)
merged["강수_중강"] = (merged["일강수량(mm)"] >= 10).astype(int)  # 예시 경계
merged["강수_강함"] = (merged["일강수량(mm)"] >= 30).astype(int)  # 예시 경계

# 5-2) 고습/저습 플래그
merged["고습(>=80%)"] = (merged["일평균_습도(%)"] >= 80).astype(int)
merged["저습(<=40%)"] = (merged["일평균_습도(%)"] <= 40).astype(int)

# 5-3) 온도 구간 원-핫(비선형 효과 캡처)
bins   = [-1e9, 5, 20, 27, 1e9]   # 매우 춥/선선/보통/더움
labels = ["T1_<=5", "T2_6~20", "T3_21~27", "T4_>27"]
merged["온도구간"] = pd.cut(merged["일평균_기온(°C)"], bins=bins, labels=labels).astype("category")
for lab in labels:
    merged[f"온도구간_{lab}"] = (merged["온도구간"] == lab).astype(int)

# 5-4) 월평균 대비 편차(이상치 성격)
monthly_mean = merged.groupby("month")[["일평균_기온(°C)","일평균_습도(%)"]].transform("mean")
merged["월평균대비_기온편차"] = merged["일평균_기온(°C)"] - monthly_mean["일평균_기온(°C)"]
merged["월평균대비_습도편차"] = merged["일평균_습도(%)"] - monthly_mean["일평균_습도(%)"]

# 5-5) 상호작용(주말 x 강수)
merged["주말x강수"] = merged["is_weekend"] * merged["강수_유무"]

# ==============================
# 6) Spearman 상관 (타깃 vs 날씨 원/파생)
# ==============================
weather_cols = [
    "일평균_기온(°C)","일평균_습도(%)","일평균_지면온도(°C)","일강수량(mm)",
    "강수_유무","강수_중강","강수_강함",
    "고습(>=80%)","저습(<=40%)",
    "월평균대비_기온편차","월평균대비_습도편차",
    "주말x강수"
] + [f"온도구간_{lab}" for lab in labels]

exist_cols = [c for c in weather_cols if c in merged.columns]
if TARGET not in merged.columns:
    raise ValueError(f"[에러] 메인 데이터에 타깃 '{TARGET}' 컬럼이 없습니다.")

corr_ser = (
    merged[exist_cols + [TARGET]]
      .corr(method="spearman")[TARGET]
      .drop(labels=[TARGET])
      .sort_values(ascending=False)
)

print("=== Spearman 상관 (전기요금 vs 날씨 원/파생) ===")
print(corr_ser, "\n")

plt.figure(figsize=(11, 3.2))
ax = corr_ser.plot(kind="bar", title="전기요금 vs 날씨(원/파생) — Spearman 상관")
ax.set_ylabel("상관계수")
plt.tight_layout()
plt.show()

# ==============================
# 7) 날씨 변수들끼리 상관 히트맵
# ==============================
if len(exist_cols) >= 2:
    corr_mat = merged[exist_cols].corr(method="spearman")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_mat, annot=False, cmap="coolwarm", center=0)
    plt.title("날씨(원/파생) 간 Spearman 상관 히트맵")
    plt.tight_layout()
    plt.show()
else:
    print("[안내] 날씨(원/파생) 컬럼이 2개 미만이라 히트맵 생략")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator

# ===== 폰트(윈도/맥/리눅스 순차 시도) =====
import matplotlib
for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = f
        break
    except:
        pass
plt.rcParams["axes.unicode_minus"] = False

# ===== 데이터 준비 =====
# train: (id, 측정일시, 전기요금(원), 작업유형, ...)
# 측정일시는 datetime으로 변환
df = train.copy()
df["측정일시"] = pd.to_datetime(df["측정일시"])

# 월, 시/분 파생
df["month"] = df["측정일시"].dt.month
df["hour"] = df["측정일시"].dt.hour
df["minute"] = df["측정일시"].dt.minute

# 하루 내 분(minute of day) → 참조일(예: 2000-01-01)에 더해 x축을 "시간"으로 사용
df["minute_of_day"] = df["hour"]*60 + df["minute"]
ref_date = pd.Timestamp("2000-01-01")
df["time_of_day"] = ref_date + pd.to_timedelta(df["minute_of_day"], unit="m")

# ===== 월별-시각 평균 (15분 단위 자동 집계) =====
# 같은 시각(HH:MM)에 대한 여러 날의 값을 월별 평균으로 축약
agg = (df
       .groupby(["month", "time_of_day"], as_index=False)["전기요금(원)"]
       .mean())

# 피벗: index=시각, columns=월, values=전기요금
pivot = agg.pivot(index="time_of_day", columns="month", values="전기요금(원)").sort_index()

# ===== 플롯 =====
fig, ax = plt.subplots(figsize=(12, 5))

for m in sorted(pivot.columns.dropna()):
    ax.plot(pivot.index, pivot[m], label=f"{int(m)}월", linewidth=1.8)

# x축을 HH:MM로 예쁘게
ax.xaxis.set_major_locator(HourLocator(byhour=range(0,24,1)))   # 1시간 간격 눈금
ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

# 보기 좋게 라벨 간격 줄이기(혼잡하면 2시간 간격 권장)
for label in ax.get_xticklabels()[1::2]:  # 라벨 절반만 표시(가독성)
    label.set_visible(False)

ax.set_title("월별 전기요금(원) — 시각(HH:MM) 기준 겹쳐보기")
ax.set_xlabel("시각 (HH:MM)")
ax.set_ylabel("전기요금(원)")
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(ncol=4, title="월", frameon=False)
plt.tight_layout()
plt.show()
