# viz/appendix_plots.py
from __future__ import annotations
import numpy as np, pandas as pd
from shiny import ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= Color / Layout palette (tone-matched) =================
_PALETTE = {
    "primary": "#3B82F6",   # blue-500
    "accent":  "#10B981",   # emerald-500
    "warn":    "#F59E0B",   # amber-500
    "danger":  "#EF4444",   # red-500
    "muted":   "#6B7280",   # gray-500
    "line":    "#111827",   # near-black for lines
}

def _apply_layout(fig: go.Figure, title: str = "", height: int = 420):
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Noto Sans KR, Inter, Arial", size=12, color=_PALETTE["line"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

# ==========================================================
# 공통 유틸 (DatetimeIndex 호환 + 안전 변환)
# ==========================================================

def _to_dt(s) -> pd.Series:
    if isinstance(s, (pd.DatetimeIndex, pd.Index)):
        s = pd.Series(s)
    else:
        s = pd.Series(s)
    return pd.to_datetime(s, errors="coerce")


def _safe_replace_year(dt_like, year: int) -> pd.Series:
    s = _to_dt(dt_like)
    out = []
    for ts in s:
        if pd.isna(ts):
            out.append(pd.NaT); continue
        m, d, h, mi, se = ts.month, ts.day, ts.hour, ts.minute, ts.second
        if m == 2 and d == 29:
            d = 28
        try:
            out.append(pd.Timestamp(year, m, d, h, mi, se))
        except Exception:
            last = pd.Timestamp(year, m, 1) + pd.offsets.MonthEnd(0)
            out.append(pd.Timestamp(year, m, last.day, h, mi, se))
    return pd.Series(pd.to_datetime(out))


def _weekend_flag(dt_like) -> pd.Series:
    s = _to_dt(dt_like)
    return (s.dt.dayofweek >= 5).astype(int)


def _season_of_month(m: int) -> str:
    return {12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"}.get(m, "unknown")


# ==========================================================
# 공휴일 세트 (대체/임시 제외) — 사용자 제공 리스트 반영
# ==========================================================

def _holidays_by_year(year: int) -> set:
    d = set()
    def add_range(a, b):
        for dt in pd.date_range(a, b, freq="D"):
            d.add(dt.date())

    if year == 2018:
        d.add(pd.Timestamp("2018-01-01").date())
        add_range("2018-02-15","2018-02-17")
        d.add(pd.Timestamp("2018-03-01").date())
        d.add(pd.Timestamp("2018-05-05").date())
        d.add(pd.Timestamp("2018-05-22").date())
        d.add(pd.Timestamp("2018-06-06").date())
        d.add(pd.Timestamp("2018-08-15").date())
        add_range("2018-09-23","2018-09-25")
        d.add(pd.Timestamp("2018-10-03").date())
        d.add(pd.Timestamp("2018-10-09").date())
        d.add(pd.Timestamp("2018-12-25").date())
    elif year == 2019:
        d.add(pd.Timestamp("2019-01-01").date())
        add_range("2019-02-04","2019-02-06")
        d.add(pd.Timestamp("2019-03-01").date())
        d.add(pd.Timestamp("2019-05-05").date())
        d.add(pd.Timestamp("2019-05-12").date())
        d.add(pd.Timestamp("2019-06-06").date())
        d.add(pd.Timestamp("2019-08-15").date())
        add_range("2019-09-12","2019-09-14")
        d.add(pd.Timestamp("2019-10-03").date())
        d.add(pd.Timestamp("2019-10-09").date())
        d.add(pd.Timestamp("2019-12-25").date())
    elif year == 2021:
        d.add(pd.Timestamp("2021-01-01").date())
        add_range("2021-02-11","2021-02-13")
        d.add(pd.Timestamp("2021-03-01").date())
        d.add(pd.Timestamp("2021-05-05").date())
        d.add(pd.Timestamp("2021-05-19").date())
        d.add(pd.Timestamp("2021-06-06").date())
        d.add(pd.Timestamp("2021-08-15").date())
        add_range("2021-09-20","2021-09-22")
        d.add(pd.Timestamp("2021-10-03").date())
        d.add(pd.Timestamp("2021-10-09").date())
        d.add(pd.Timestamp("2021-12-25").date())
    elif year == 2022:
        d.add(pd.Timestamp("2022-01-01").date())
        add_range("2022-01-31","2022-02-02")
        d.add(pd.Timestamp("2022-03-01").date())
        d.add(pd.Timestamp("2022-03-09").date())
        d.add(pd.Timestamp("2022-05-05").date())
        d.add(pd.Timestamp("2022-05-08").date())
        d.add(pd.Timestamp("2022-06-01").date())
        d.add(pd.Timestamp("2022-06-06").date())
        add_range("2022-09-09","2022-09-11")
        d.add(pd.Timestamp("2022-10-03").date())
        d.add(pd.Timestamp("2022-10-09").date())
        d.add(pd.Timestamp("2022-12-25").date())
    elif year == 2023:
        d.add(pd.Timestamp("2023-01-01").date())
        add_range("2023-01-21","2023-01-23")
        d.add(pd.Timestamp("2023-03-01").date())
        d.add(pd.Timestamp("2023-05-05").date())
        d.add(pd.Timestamp("2023-05-27").date())
        d.add(pd.Timestamp("2023-06-06").date())
        d.add(pd.Timestamp("2023-08-15").date())
        add_range("2023-09-28","2023-09-30")
        d.add(pd.Timestamp("2023-10-03").date())
        d.add(pd.Timestamp("2023-10-09").date())
        d.add(pd.Timestamp("2023-12-25").date())
    else:
        return set()
    return d


# ==========================================================
# 개요 탭: 스키마/헤드
# ==========================================================

def render_data_schema():
    schema = pd.DataFrame({
        "컬럼명": [
            "id", "측정일시", "전력사용량(kWh)", "지상무효전력량(kVarh)",
            "진상무효전력량(kVarh)", "탄소배출량(tCO2)", "지상역률(%)",
            "진상역률(%)", "작업유형", "전기요금(원)"
        ],
        "타입": [
            "int", "datetime", "float", "float", "float",
            "float", "float", "float", "object", "float"
        ],
        "설명": [
            "고유 식별자",
            "측정 시각 (15분 간격)",
            "전력 사용량",
            "무효 전력량(지상)",
            "무효 전력량(진상)",
            "탄소 배출량",
            "역률(지상)",
            "역률(진상)",
            "부하 유형",
            "예측 대상 값"
        ],
    })
    html = schema.to_html(classes="table table-sm table-hover", index=False, border=0)
    return ui.HTML(f'<div style="max-height:500px; overflow-y:auto;">{html}</div>')


def render_data_head(df: pd.DataFrame, n: int = 10):
    d = df.head(n).copy()
    html = d.to_html(classes="table table-sm table-striped table-bordered", index=False, border=0)
    return ui.HTML(f'<div style="overflow-x:auto; overflow-y:auto;">{html}</div>')


# ==========================================================
# EDA: 표/그래프 기본
# ==========================================================

def render_basic_stats(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    stats = num.describe().T
    stats["결측수"] = num.isnull().sum()
    stats["결측률(%)"] = (num.isnull().sum() / len(num) * 100).round(2)
    stats = stats[["count","mean","std","min","25%","50%","75%","max","결측수","결측률(%)"]].round(2)
    stats.columns = ["개수","평균","표준편차","최소","25%","중앙값","75%","최대","결측수","결측률(%)"]
    html = stats.to_html(classes="table table-sm table-striped", border=0)
    return ui.HTML(f'<div style="max-height:420px; overflow-y:auto;">{html}</div>')


def render_missing_summary(df: pd.DataFrame):
    m = pd.DataFrame({
        "컬럼": df.columns,
        "결측수": df.isnull().sum(),
        "결측률(%)": (df.isnull().sum() / len(df) * 100).round(2),
    })
    m = m[m["결측수"] > 0].sort_values("결측수", ascending=False)
    if len(m) == 0:
        return ui.div(ui.tags.h5("결측치 없음", class_="text-center"), class_="p-3")
    html = m.to_html(classes="table table-sm table-striped", index=False, border=0)
    return ui.HTML(html)


def render_outlier_summary(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns
    rows = []
    for c in num_cols:
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        cnt = int(((df[c] < lo) | (df[c] > hi)).sum())
        if cnt:
            rows.append({"컬럼": c, "이상치수": cnt, "이상치율(%)": round(cnt/len(df)*100,2), "하한": round(lo,2), "상한": round(hi,2)})
    html = "<div>"
    html += '<div class="alert alert-info mb-3">'
    html += '<h6 class="mb-2">적용된 이상치 처리</h6>'
    html += '<ul class="mb-0">'
    html += '<li>타겟(전기요금) 상위 0.7% 제거</li>'
    html += '<li>특정 시점 제거: 2018-11-07 00:00:00 (정합성 이슈 보정)</li>'
    html += '</ul></div>'
    if rows:
        outlier_df = pd.DataFrame(rows)
        html += '<h6 class="mt-3">IQR 기준 이상치 분포</h6>'
        html += outlier_df.to_html(classes='table table-sm table-striped', index=False, border=0)
    else:
        html += '<p class="text-success mt-3">IQR 기준 이상치 없음</p>'
    html += '</div>'
    return ui.HTML(html)


def plot_distribution(df: pd.DataFrame):
    fig = make_subplots(rows=2, cols=2, subplot_titles=("전력사용량(kWh)", "전기요금(원)", "지상무효전력량(kVarh)", "지상역률(%)"))
    if "전력사용량(kWh)" in df:
        fig.add_histogram(x=df["전력사용량(kWh)"], nbinsx=50, showlegend=False, row=1, col=1)
    if "전기요금(원)" in df:
        fig.add_histogram(x=df["전기요금(원)"], nbinsx=50, showlegend=False, row=1, col=2)
    if "지상무효전력량(kVarh)" in df:
        fig.add_histogram(x=df["지상무효전력량(kVarh)"], nbinsx=50, showlegend=False, row=2, col=1)
    if "지상역률(%)" in df:
        fig.add_histogram(x=df["지상역률(%)"], nbinsx=50, showlegend=False, row=2, col=2)
    fig.update_layout(height=520, title_text="주요 변수 분포", margin=dict(l=40, r=20, t=60, b=40))
    return ui.HTML(go.Figure(fig).to_html(include_plotlyjs='cdn', full_html=False))


def plot_correlation_heatmap(df: pd.DataFrame):
    keys = ["전력사용량(kWh)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)", "지상역률(%)", "진상역률(%)", "탄소배출량(tCO2)", "전기요금(원)"]
    cols = [c for c in keys if c in df.columns]
    if len(cols) < 2:
        return ui.div("상관분석을 위한 수치형 변수 부족", class_="p-3 small-muted")
    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu_r', zmid=0, text=corr.values.round(2), texttemplate='%{text}', textfont={"size": 10}))
    fig.update_layout(title='주요 변수 상관관계', height=520, margin=dict(l=40, r=20, t=60, b=40))
    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))


def plot_time_trend(df: pd.DataFrame):
    d = df.copy()
    d['측정일시'] = _to_dt(d['측정일시'])
    daily = d.groupby(d['측정일시'].dt.date).agg({
        '전력사용량(kWh)': 'sum',
        '전기요금(원)': 'sum'
    }).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if '전력사용량(kWh)' in daily:
        fig.add_scatter(x=daily['측정일시'], y=daily['전력사용량(kWh)'], name='전력사용량(kWh)')
    if '전기요금(원)' in daily:
        fig.add_scatter(x=daily['측정일시'], y=daily['전기요금(원)'], name='전기요금(원)', secondary_y=True)
    fig.update_layout(title='일별 전력사용량/전기요금 추이', height=420, hovermode='x unified', margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(title_text='날짜')
    fig.update_yaxes(title_text='전력사용량 (kWh)', secondary_y=False)
    fig.update_yaxes(title_text='전기요금 (원)', secondary_y=True)
    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))


def plot_hourly_pattern(df: pd.DataFrame):
    d = df.copy()
    d['측정일시'] = _to_dt(d['측정일시'])
    d['hour'] = d['측정일시'].dt.hour
    hourly = d.groupby('hour')['전력사용량(kWh)'].mean().reset_index()
    fig = go.Figure()
    if not hourly.empty:
        fig.add_scatter(x=hourly['hour'], y=hourly['전력사용량(kWh)'], mode='lines+markers', name='평균')
    fig.update_layout(title='시간대별 평균 전력 사용량', xaxis_title='시간', yaxis_title='kWh', height=420, margin=dict(l=40, r=20, t=60, b=40))
    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))


def plot_weekday_pattern(df: pd.DataFrame):
    d = df.copy()
    d['측정일시'] = _to_dt(d['측정일시'])
    d['weekday'] = d['측정일시'].dt.dayofweek
    names = ['월','화','수','목','금','토','일']
    wk = d.groupby('weekday')['전력사용량(kWh)'].mean().reset_index()
    wk['요일'] = wk['weekday'].map(lambda x: names[x])
    fig = go.Figure()
    if not wk.empty:
        fig.add_bar(x=wk['요일'], y=wk['전력사용량(kWh)'], text=wk['전력사용량(kWh)'].round(2), textposition='outside', name='평균')
    fig.update_layout(title='요일별 평균 전력 사용량', xaxis_title='요일', yaxis_title='kWh', height=420, showlegend=False, margin=dict(l=40, r=20, t=60, b=40))
    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))


def plot_worktype_distribution(df: pd.DataFrame):
    if '작업유형' not in df.columns:
        return ui.div('작업유형 컬럼 없음', class_='p-3 small-muted')
    vc = df['작업유형'].value_counts()
    fig = go.Figure()
    if not vc.empty:
        fig.add_bar(x=vc.index, y=vc.values, text=vc.values, textposition='outside')
    fig.update_layout(title='작업유형별 분포', xaxis_title='작업유형', yaxis_title='건수', height=420, showlegend=False, margin=dict(l=40, r=20, t=60, b=40))
    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))


# ==========================================================
# 전처리 설명 블록 (모델링 코드와 정합)
# ==========================================================

def render_pipeline_accordion():
    return ui.accordion(
        ui.accordion_panel(
            "Step 0: 데이터 정제",
            ui.tags.ul(
                ui.tags.li(ui.tags.b("자정 롤오버 보정:"), " 00:00 → 다음날 00:00 교정"),
                ui.tags.li(ui.tags.b("이상 시점 제거:"), " 2018-11-07 00:00:00"),
                ui.tags.li(ui.tags.b("극단치 제거:"), " 전기요금 상위 0.7% 컷")
            )
        ),
        ui.accordion_panel(
            "Step 1: 기본 시간 변수",
            ui.tags.ul(
                ui.tags.li(ui.tags.b("시간 분해:"), " day/hour/minute/weekday/month"),
                ui.tags.li(ui.tags.b("주말 플래그:"), " is_weekend"),
                ui.tags.li(ui.tags.b("공휴일 플래그:"), " is_holiday/eve/after — 기준연도 사용"),
                ui.tags.li(ui.tags.b("시간대 구분:"), " is_peak_after, is_peak_even, is_night"),
                ui.tags.li(ui.tags.b("hour_of_week:"), " 0~167")
            )
        ),
        ui.accordion_panel(
            "Step 2: 주기성 인코딩",
            ui.tags.ul(
                ui.tags.li(ui.tags.b("1차/2차 고조파:"), " hour_sin/cos, hour_sin2/cos2"),
                ui.tags.li(ui.tags.b("요일 주기:"), " dow_sin/cos")
            )
        ),
        ui.accordion_panel(
            "Step 3: OOF 3변수 (5Fold TS)",
            ui.tags.ul(
                ui.tags.li("oof_kwh, oof_reactive, oof_pf"),
                ui.tags.li("미래 정보 차단 목적의 OOF 방식")
            )
        ),
        ui.accordion_panel(
            "Step 4: 시차(Lag)",
            ui.tags.ul(
                ui.tags.li("lag1, 1h, 6h, 24h, 48h, 7d"),
            )
        ),
        ui.accordion_panel(
            "Step 5: 롤링/EMA",
            ui.tags.ul(
                ui.tags.li("roll6h, roll24h, ema24h — shift(1) 후 집계")
            )
        ),
        ui.accordion_panel(
            "Step 6: 차분",
            ui.tags.ul(
                ui.tags.li("samehour_d1, samehour_w1, samehour_w2, diff1h")
            )
        ),
        ui.accordion_panel(
            "Step 7: 비율/프록시",
            ui.tags.ul(
                ui.tags.li("oof_ratio, oof_pf_proxy, oof_ratio_ema24h")
            )
        ),
        ui.accordion_panel(
            "Step 8: 프로필/잔차·변화율",
            ui.tags.ul(
                ui.tags.li("how_profile_kwh, how_resid_kwh, kwh_rate_w1, kwh_rate_w2 — OOF 누적평균 기반")
            )
        ),
        ui.accordion_panel(
            "Step 9: 카테고리화",
            ui.tags.ul(
                ui.tags.li("how_cat(=hour_of_week 문자열), 작업유형")
            )
        ),
        id="pipeline_accordion",
        open=False
    )


def render_feature_summary():
    html = """
    <div class="p-3" style="font-size: 1rem;">
      <table class="table table-hover mt-0" style="font-size: 0.95rem;">
        <thead class="table-light">
          <tr><th>피처 그룹</th><th>개수</th><th>예시</th></tr>
        </thead>
        <tbody>
          <tr><td><b>시간 기본</b></td><td>~15</td><td>hour, weekday, is_weekend, is_holiday, hour_of_week</td></tr>
          <tr><td><b>주기성 인코딩</b></td><td>6</td><td>hour_sin/cos, hour_sin2/cos2, dow_sin/cos</td></tr>
          <tr><td><b>OOF 기본</b></td><td>3</td><td>oof_kwh, oof_reactive, oof_pf</td></tr>
          <tr><td><b>Lag</b></td><td>~18</td><td>각 OOF에 1h/6h/24h/48h/7d</td></tr>
          <tr><td><b>롤링/EMA</b></td><td>~9</td><td>roll6h/24h, ema24h × 3</td></tr>
          <tr><td><b>차분</b></td><td>~12</td><td>samehour_d1/w1/w2, diff1h × 3</td></tr>
          <tr><td><b>비율/프록시</b></td><td>3</td><td>oof_ratio, oof_pf_proxy, oof_ratio_ema24h</td></tr>
          <tr><td><b>프로필/잔차·변화율</b></td><td>4</td><td>how_profile_kwh, how_resid_kwh, kwh_rate_w1/w2</td></tr>
          <tr><td><b>카테고리</b></td><td>2</td><td>작업유형, how_cat</td></tr>
        </tbody>
      </table>
    </div>
    """
    return ui.HTML(html)


def render_scaling_info():
    html = """
    <div class="p-4" style="font-size: 1rem;">
      <div class="alert alert-success">
        <h6 class="mb-2">스케일링 미적용</h6>
        <p class="mb-0">트리 기반 모델에 한해 스케일 영향 미미</p>
      </div>
      <h6 class="mt-3 mb-2">인코딩 전략</h6>
      <table class="table table-sm table-striped">
        <thead><tr><th>타입</th><th>처리</th><th>상세</th></tr></thead>
        <tbody>
          <tr><td>수치형</td><td>원본</td><td>값 그대로</td></tr>
          <tr><td>범주형(작업유형)</td><td>Categorical dtype</td><td>LGBM categorical_feature 사용</td></tr>
          <tr><td>범주형(how_cat)</td><td>String→Categorical</td><td>hour_of_week 168 클래스</td></tr>
          <tr><td>결측</td><td>Median</td><td>Train 기준</td></tr>
        </tbody>
      </table>
    </div>
    """
    return ui.HTML(html)


def render_leakage_check():
    html = """
    <div class="p-4" style="font-size: 1rem;">
      <h6 class="mb-2">데이터 누수 점검</h6>
      <ul>
        <li>TimeSeriesSplit(3-fold) — 시간 순서 보존</li>
        <li>OOF 3변수 — 미래 정보 차단</li>
        <li>lag/rolling — shift 후 집계</li>
        <li>프로필 — 누적평균의 shift(OoF) 방식</li>
        <li>스케일링/결측 — Train 통계로 transform</li>
      </ul>
      <div class="small-muted">주의: 공휴일은 기준연도 캘린더 사용, Test lag는 Train 말단 기준</div>
    </div>
    """
    return ui.HTML(html)


# ==========================================================
# 스토리라인 (월/일/시간/분/계절)
# ==========================================================

def render_eda_storyline_panels(df: pd.DataFrame):
    if "측정일시" not in df.columns:
        return ui.div("측정일시 컬럼 부재", class_="billx-panel p-3")

    d = df.copy()
    d["측정일시"] = _to_dt(d["측정일시"])
    d = d.dropna(subset=["측정일시"]).sort_values("측정일시")

    # 1) 월별
    d["month"] = d["측정일시"].dt.month
    m = d.groupby("month")["전기요금(원)"].sum().reset_index()
    fig_m = go.Figure()
    fig_m.add_bar(x=m["month"], y=m["전기요금(원)"], text=m["전기요금(원)"].round(0), textposition="outside")
    fig_m.update_layout(title="월별 전기요금 합계", height=380, margin=dict(l=40, r=20, t=60, b=40))

    # 2) 일별
    d["date"] = d["측정일시"].dt.date
    byday = d.groupby("date")["전기요금(원)"].sum().reset_index()
    fig_d = go.Figure()
    fig_d.add_scatter(x=byday["date"], y=byday["전기요금(원)"], mode="lines")
    fig_d.update_layout(title="일별 전기요금 추이", height=380, margin=dict(l=40, r=20, t=60, b=40))

    # 3) 시간별
    d["hour"] = d["측정일시"].dt.hour
    byhour = d.groupby("hour")["전기요금(원)"].mean().reset_index()
    fig_h = go.Figure()
    fig_h.add_scatter(x=byhour["hour"], y=byhour["전기요금(원)"], mode="lines+markers")
    fig_h.update_layout(title="시간별 평균 전기요금", height=380, margin=dict(l=40, r=20, t=60, b=40))

    # 4) 15분(분 단위)
    d["minute"] = d["측정일시"].dt.minute
    bymin = d.groupby(["hour","minute"], as_index=False)["전기요금(원)"].mean()
    bymin["time_in_hour"] = bymin["hour"].astype(str) + ":" + bymin["minute"].astype(str).str.zfill(2)
    fig_q = go.Figure()
    fig_q.add_scatter(x=bymin["time_in_hour"], y=bymin["전기요금(원)"], mode="lines", showlegend=False)
    fig_q.update_layout(title="분별(15분) 평균 전기요금", height=380, xaxis_tickangle=45, margin=dict(l=40, r=20, t=60, b=80))

    # 5) 계절별
    d["season"] = d["측정일시"].dt.month.map(_season_of_month)
    bys = d.groupby("season")["전기요금(원)"].mean().reindex(["spring","summer","autumn","winter"]).reset_index()
    fig_s = go.Figure()
    fig_s.add_bar(x=bys["season"], y=bys["전기요금(원)"], text=bys["전기요금(원)"].round(0), textposition="outside")
    fig_s.update_layout(title="계절별 평균 전기요금", height=380, margin=dict(l=40, r=20, t=60, b=40))

    return ui.div(
        ui.div(ui.h5("스토리라인", class_="billx-panel-title"),
               ui.tags.ol(
                   ui.tags.li("월/일/시간/분/계절 단위로 전기요금 패턴 확인"),
                   ui.tags.li("피크 시간대 대비 반영 — is_peak_after, is_peak_even, is_night"),
                   ui.tags.li("주기성 반영 — hour_sin/cos, hour_sin2/cos2, dow_sin/cos"),
               ),
               class_="billx-panel"),
        ui.layout_columns(
            ui.div(ui.HTML(fig_m.to_html(include_plotlyjs='cdn', full_html=False)), class_="billx-panel"),
            ui.div(ui.HTML(fig_d.to_html(include_plotlyjs='cdn', full_html=False)), class_="billx-panel"),
            col_widths=[6,6]
        ),
        ui.layout_columns(
            ui.div(ui.HTML(fig_h.to_html(include_plotlyjs='cdn', full_html=False)), class_="billx-panel"),
            ui.div(ui.HTML(fig_q.to_html(include_plotlyjs='cdn', full_html=False)), class_="billx-panel"),
            col_widths=[6,6]
        ),
        ui.div(ui.HTML(fig_s.to_html(include_plotlyjs='cdn', full_html=False)), class_="billx-panel"),
    )


# ==========================================================
# NEW — 00:00 롤오버 보정 리포트
# ==========================================================

def render_midnight_rollover_fix(df: pd.DataFrame):
    if "측정일시" not in df.columns:
        return ui.div("측정일시 컬럼 부재", class_="billx-panel p-3")

    d = df[["측정일시", "전기요금(원)"]].copy()
    d["측정일시"] = _to_dt(d["측정일시"]).dropna()
    d = d.sort_values("측정일시")

    ts = d["측정일시"]
    mask = (ts.dt.hour == 0) & (ts.dt.minute == 0)
    n_total = len(ts)
    n_fix = int(mask.sum())
    pct = round(n_fix / n_total * 100, 2) if n_total else 0.0

    # +1일 보정 시뮬레이션 (원본 데이터는 변경하지 않음)
    ts_fix = ts.copy()
    ts_fix.loc[mask] = ts_fix.loc[mask] + pd.Timedelta(days=1)

    # 일별 합계 비교 (원본 vs 보정)
    d_o = d.copy(); d_o["date"] = ts.dt.date
    daily_o = d_o.groupby("date", as_index=False)["전기요금(원)"].sum().rename(columns={"전기요금(원)":"원본"})

    d_f = d.copy(); d_f["date"] = ts_fix.dt.date
    daily_f = d_f.groupby("date", as_index=False)["전기요금(원)"].sum().rename(columns={"전기요금(원)":"보정"})

    daily = pd.merge(daily_o, daily_f, on="date", how="outer").fillna(0)
    daily["차이"] = daily["보정"] - daily["원본"]

    fig = go.Figure()
    fig.add_scatter(x=daily["date"], y=daily["원본"], name="원본", mode="lines", line=dict(width=2, color=_PALETTE["muted"]))
    fig.add_scatter(x=daily["date"], y=daily["보정"], name="보정(+1일)", mode="lines", line=dict(width=2, color=_PALETTE["accent"]))
    _apply_layout(fig, title="자정(00:00) 롤오버 보정 — 일별 합계 영향", height=360)
    fig.update_yaxes(title_text="전기요금(원)")

    # 샘플 10건 (원본 vs +1일)
    sample = pd.DataFrame({
        "원본": ts.astype(str),
        "+1일 보정": ts_fix.astype(str),
        "변경여부": np.where(mask, "SHIFT", "—")
    }).head(10)

    sample_html = sample.to_html(classes="table table-sm table-striped", index=False, border=0)

    html = f"""
    <div class=\"billx-panel\">
      <div class=\"row g-3\">
        <div class=\"col-md-5\">
          <ul class=\"mb-2\">
            <li>검출 수: <b>{n_fix:,}</b> / {n_total:,} ({pct}%)</li>
            <li>규칙: 시각이 00:00인 관측치는 <code>+1일</code>로 이동하여 날짜 경계 정합성 확보</li>
          </ul>
          <div class=\"small-muted\">※ EDA 보고용 시뮬레이션이며 원본 데이터는 수정하지 않습니다.</div>
          <div class=\"mt-3\">{sample_html}</div>
        </div>
        <div class=\"col-md-7\">{fig.to_html(include_plotlyjs='cdn', full_html=False)}</div>
      </div>
    </div>
    """
    return ui.HTML(html)


# ==========================================================
# 달력 정합성 스토리라인 (연 전체, 비윤년 5개년 비교)
# ==========================================================

def render_calendar_alignment_storyline(df: pd.DataFrame):
    if "측정일시" not in df.columns:
        return ui.div("측정일시 컬럼 부재로 정합성 진단 생략", class_="billx-panel small-muted p-3")

    d = df[["측정일시", "전기요금(원)"]].copy()
    d["측정일시"] = _to_dt(d["측정일시"])
    d = d.dropna(subset=["측정일시"]).sort_values("측정일시")

    # 0) 윤년 2/29 존재 여부
    has_feb29 = ((d["측정일시"].dt.month == 2) & (d["측정일시"].dt.day == 29)).any()
    leap_line = "윤년 2/29 관측" if has_feb29 else "윤년 2/29 미관측"

    # 1) 연 전체 범위의 주말 플래그
    full_dates = d["측정일시"]
    w_ref = _weekend_flag(full_dates).to_numpy()

    # 2) 후보 연도(비윤년 5개년) 비교
    candidates = [2018, 2019, 2021, 2022, 2023]
    bars_x, bars_y, details = [], [], []
    hol_hits_map = {}

    for yr in candidates:
        ts_y = _safe_replace_year(full_dates, yr)
        w_y = _weekend_flag(ts_y).to_numpy()
        mismatch = int((w_ref != w_y).sum())

        hols = _holidays_by_year(yr)
        hits = int(pd.Series(pd.to_datetime(ts_y)).dt.date.isin(hols).sum())

        bars_x.append(str(yr))
        bars_y.append(mismatch)
        details.append((yr, mismatch, hits, len(w_y)))
        hol_hits_map[yr] = hits

    details.sort(key=lambda x: x[1])
    best_year, best_mis, best_hol_hits, N = details[0]

    # 3) 막대그래프
    fig_mismatch = go.Figure()
    fig_mismatch.add_bar(x=bars_x, y=bars_y, name="주말 불일치 수", marker_color=_PALETTE["primary"])
    fig_mismatch.update_layout(
        title="연 전체 주말 플래그 불일치(원본 vs 연도치환)",
        yaxis_title="불일치 수",
        height=380,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    li_html = [
        f"<li>후보 {yr}: 주말 불일치 {mis:,} / {tot:,} | 공휴일 매칭 {hol_hits_map[yr]:,}</li>"
        for yr, mis, _, tot in details
    ]
    ul_html = "".join(li_html)
    fig_html = fig_mismatch.to_html(include_plotlyjs='cdn', full_html=False)

    html = f"""
    <div class="billx-panel">
      <h6 class="billx-panel-title">달력 정합성 — 기준연도 판별(연 전체)</h6>
      <ol class="mb-2">
        <li>윤년 유무 점검: {leap_line}</li>
        <li>연 전체 기간에서 후보 비윤년 5개년과 주말 플래그 비교</li>
        <li>오차 최소 연도 선택 → 기준연도 <b>{best_year}</b> 지정</li>
      </ol>
      <ul class="mb-2 small-muted">{ul_html}</ul>
      <div class="small-muted">공휴일 세트는 대체·임시 공휴일 제외, 사용자 제공 목록 기반</div>
    </div>
    <div class="billx-panel">{fig_html}</div>
    """
    return ui.HTML(html)


# ==========================================================
# 달력 정합성 오버레이(선택 연도) — 연 전체 주말/공휴일 하이라이트
# ==========================================================

def render_calendar_overlay(
    df: pd.DataFrame,
    year: int = 2018,
    highlight_weekend: bool = True,
    highlight_holiday: bool = True
):
    if "측정일시" not in df.columns:
        return ui.div("측정일시 컬럼 부재", class_="billx-panel p-3")

    d = df[["측정일시", "전기요금(원)"]].copy()
    d["측정일시"] = _to_dt(d["측정일시"])
    d = d.dropna(subset=["측정일시"]).sort_values("측정일시")

    # 연 전체 일별 합계
    d["date"] = d["측정일시"].dt.date
    daily = d.groupby("date", as_index=False)["전기요금(원)"].sum()

    # 선택 연도로 캘린더 치환 → 주말/공휴일 라벨 생성
    mapped = _safe_replace_year(pd.to_datetime(daily["date"]), year)
    flags = pd.DataFrame({
        "is_weekend": _weekend_flag(mapped).astype(bool),
        "is_holiday": pd.Series(mapped).dt.date.isin(_holidays_by_year(year)),
    })

    def label_row(i):
        w = bool(flags.loc[i, "is_weekend"]) if highlight_weekend else False
        h = bool(flags.loc[i, "is_holiday"]) if highlight_holiday else False
        if h:
            return "공휴일"
        if w:
            return "주말"
        return "평일"

    labels = [label_row(i) for i in range(len(daily))]

    fig = go.Figure()
    for key in ["공휴일", "주말", "평일"]:
        idx = [i for i, v in enumerate(labels) if v == key]
        if not idx:
            continue
        color = _PALETTE["danger"] if key == "공휴일" else (_PALETTE["warn"] if key == "주말" else _PALETTE["primary"])
        fig.add_scatter(
            x=daily["date"].iloc[idx],
            y=daily["전기요금(원)"].iloc[idx],
            mode="markers",
            name=f"{key}",
            marker=dict(color=color, size=7),
        )
    fig.add_scatter(
        x=daily["date"],
        y=daily["전기요금(원)"],
        mode="lines",
        name="전체 추이",
        line=dict(width=1),
        hoverinfo="skip"
    )

    fig.update_layout(
        title=f"연 전체 일별 전기요금 — {year} 기준 주말/공휴일 강조",
        xaxis_title="날짜",
        yaxis_title="전기요금(원)",
        height=460,
        hovermode='x unified',
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))