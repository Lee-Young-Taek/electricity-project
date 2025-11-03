# =============================
# viz/appendix_plots.py — Appendix 시각화/요약 유틸 (그래프 + 스토리라인)
# =============================
from __future__ import annotations
import uuid
import numpy as np
import pandas as pd
from shiny import ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────

def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "겨울"
    if m in (3, 4, 5):
        return "봄"
    if m in (6, 7, 8):
        return "여름"
    return "가을"


def _apply_plot_theme(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="simple_white",
        height=height,
        margin=dict(l=40, r=16, t=48, b=36),
        hovermode="x unified",
        font=dict(family="Noto Sans KR, system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=12),
        title=dict(font=dict(size=16), x=0),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


# ─────────────────────────────────────────────
# 개요 탭
# ─────────────────────────────────────────────

def render_data_schema() -> ui.HTML:
    schema = pd.DataFrame(
        {
            "컬럼명": [
                "id",
                "측정일시",
                "전력사용량(kWh)",
                "지상무효전력량(kVarh)",
                "진상무효전력량(kVarh)",
                "탄소배출량(tCO2)",
                "지상역률(%)",
                "진상역률(%)",
                "작업유형",
                "전기요금(원)",
            ],
            "타입": [
                "int",
                "datetime",
                "float",
                "float",
                "float",
                "float",
                "float",
                "float",
                "object",
                "float",
            ],
            "설명": [
                "각 측정값의 고유 식별자",
                "측정 시각 (15분 간격)",
                "실제 전력 사용량",
                "무효 전력량 (지상)",
                "무효 전력량 (진상)",
                "전력 사용으로 인한 탄소 배출량",
                "지상 방향 역률(%)",
                "진상 방향 역률(%)",
                "부하/운영 유형",
                "예측 대상 (전력×단가)",
            ],
        }
    )
    html = schema.to_html(classes="table table-sm table-hover", index=False, border=0)
    return ui.HTML('<div style="max-height:500px; overflow-y:auto;">{}</div>'.format(html))


def render_data_head(df: pd.DataFrame, n: int = 10) -> ui.HTML:
    display_df = df.head(n).copy()
    html = display_df.to_html(
        classes="table table-sm table-striped table-bordered",
        index=False,
        border=0,
        float_format=lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x,
    )
    return ui.HTML('<div style="overflow-x:auto; overflow-y:auto;">{}</div>'.format(html))


# ─────────────────────────────────────────────
# EDA (표/요약/분포/상관/패턴/시계열)
# ─────────────────────────────────────────────

def render_basic_stats(df: pd.DataFrame) -> ui.HTML:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return ui.div("수치형 컬럼 없음", class_="p-3 text-muted")

    stats = numeric_df.describe().T
    stats["결측수"] = numeric_df.isnull().sum()
    stats["결측률(%)"] = (numeric_df.isnull().sum() / len(numeric_df) * 100).round(2)
    stats = stats[["count","mean","std","min","25%","50%","75%","max","결측수","결측률(%)"]]
    stats.columns = ["개수","평균","표준편차","최소","25%","중앙값","75%","최대","결측수","결측률(%)"]
    html = stats.round(2).to_html(classes="table table-sm table-striped", border=0)
    return ui.HTML('<div style="max-height:400px; overflow-y:auto;">{}</div>'.format(html))


def render_missing_summary(df: pd.DataFrame) -> ui.Tag:
    missing = pd.DataFrame({"컬럼": df.columns, "결측수": df.isnull().sum(), "결측률(%)": (df.isnull().sum() / len(df) * 100).round(2)})
    missing = missing[missing["결측수"] > 0].sort_values("결측수", ascending=False)
    if len(missing) == 0:
        return ui.div(ui.tags.h5("결측치 없음", class_="text-success text-center mt-4"), class_="p-4")
    html = missing.to_html(classes="table table-sm table-striped", index=False, border=0)
    return ui.HTML(html)


def render_outlier_summary(df: pd.DataFrame) -> ui.HTML:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        cnt = ((df[col] < lower) | (df[col] > upper)).sum()
        rate = (cnt / len(df) * 100)
        if cnt > 0:
            outliers.append({"컬럼": col, "이상치수": cnt, "이상치율(%)": round(rate, 2), "하한": round(lower, 2), "상한": round(upper, 2)})

    result = ['<div class="alert alert-info mb-3">',
              "<h6 class=\"mb-2\">실제 적용 이상치 처리</h6>",
              "<ul class=\"mb-0\">",
              "<li><strong>타겟(전기요금)</strong> 상위 0.7% 제거 (99.3% 분위수 초과)</li>",
              "<li><strong>특정 시점</strong> 제거: 2024-11-07 00:00:00</li>",
              "</ul></div>"]

    if outliers:
        outlier_df = pd.DataFrame(outliers)
        result += ['<h6 class="mt-3">IQR 기준 이상치 분포</h6>', outlier_df.to_html(classes="table table-sm table-striped", index=False, border=0)]
    else:
        result += ["<p class=\"text-success mt-3\">IQR 기준 이상치 없음</p>"]

    return ui.HTML("".join(result))


def plot_distribution(df: pd.DataFrame) -> ui.HTML:
    cols = [("전력사용량(kWh)", None),("전기요금(원)", None),("지상무효전력량(kVarh)", None),("지상역률(%)", None)]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[c for c, _ in cols])
    r, c = 1, 1
    for col, _ in cols:
        if col not in df.columns:
            c = 1 if c == 2 else 2
            continue
        fig.add_trace(go.Histogram(x=df[col], nbinsx=50, showlegend=False), row=r, col=c)
        c = 1 if c == 2 else 2
        if c == 1:
            r += 1
    _apply_plot_theme(fig, height=520)
    fig.update_layout(title_text="주요 변수 분포")
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id=_uid("dist")))


def plot_correlation_heatmap(df: pd.DataFrame) -> ui.Tag:
    key_cols = ["전력사용량(kWh)","지상무효전력량(kVarh)","진상무효전력량(kVarh)","지상역률(%)","진상역률(%)","탄소배출량(tCO2)","전기요금(원)"]
    exist = [c for c in key_cols if c in df.columns]
    if len(exist) < 2:
        return ui.div("상관관계 계산을 위한 수치형 변수 부족", class_="p-3 text-muted")
    corr = df[exist].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu_r", zmid=0,
                                    text=corr.values.round(2), texttemplate="%{text}", textfont={"size": 10}))
    _apply_plot_theme(fig, height=520)
    fig.update_layout(title="주요 변수 간 상관관계")
    fig.update_xaxes(tickangle=45)
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id=_uid("corr")))


def plot_time_trend(df: pd.DataFrame) -> ui.HTML:
    d = df.copy()
    d["측정일시"] = _to_dt(d["측정일시"])
    d = d.dropna(subset=["측정일시"])  
    daily = d.groupby(d["측정일시"].dt.date).agg({"전력사용량(kWh)": "sum", "전기요금(원)": "sum"}).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=daily["측정일시"], y=daily["전력사용량(kWh)"], name="전력사용량(kWh)", line=dict(width=2)), secondary_y=False)
    if "전기요금(원)" in daily.columns:
        fig.add_trace(go.Scatter(x=daily["측정일시"], y=daily["전기요금(원)"], name="전기요금(원)", line=dict(width=2)), secondary_y=True)
    _apply_plot_theme(fig, height=420)
    fig.update_layout(title="일별 전력사용량 및 전기요금 추이")
    fig.update_xaxes(title_text="날짜")
    fig.update_yaxes(title_text="전력사용량 (kWh)", secondary_y=False)
    fig.update_yaxes(title_text="전기요금 (원)", secondary_y=True)
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id=_uid("trend")))


def plot_hourly_pattern(df: pd.DataFrame) -> ui.HTML:
    d = df.copy()
    d["측정일시"] = _to_dt(d["측정일시"]) ; d = d.dropna(subset=["측정일시"])  
    d["hour"] = d["측정일시"].dt.hour
    g = d.groupby("hour")["전력사용량(kWh)"].mean().reset_index()
    fig = go.Figure(go.Scatter(x=g["hour"], y=g["전력사용량(kWh)"], mode="lines+markers", marker=dict(size=7)))
    _apply_plot_theme(fig, height=360)
    fig.update_layout(title="시간대별 평균 전력 사용량")
    fig.update_xaxes(title="시간 (Hour)")
    fig.update_yaxes(title="평균 전력사용량 (kWh)")
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id=_uid("hourly")))


def plot_weekday_pattern(df: pd.DataFrame) -> ui.HTML:
    d = df.copy()
    d["측정일시"] = _to_dt(d["측정일시"]) ; d = d.dropna(subset=["측정일시"])  
    d["weekday"] = d["측정일시"].dt.dayofweek
    names = ["월", "화", "수", "목", "금", "토", "일"]
    g = d.groupby("weekday")["전력사용량(kWh)"].mean().reset_index()
    g["요일"] = g["weekday"].map(lambda x: names[x])
    fig = go.Figure(go.Bar(x=g["요일"], y=g["전력사용량(kWh)"]))
    _apply_plot_theme(fig, height=360)
    fig.update_layout(title="요일별 평균 전력 사용량 (주말 강조 없음)")
    fig.update_xaxes(title="요일")
    fig.update_yaxes(title="평균 전력사용량 (kWh)")
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id=_uid("weekday")))


def plot_worktype_distribution(df: pd.DataFrame) -> ui.Tag:
    if "작업유형" not in df.columns:
        return ui.div("작업유형 컬럼 없음", class_="p-3 text-muted")
    counts = df["작업유형"].value_counts()
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values))
    _apply_plot_theme(fig, height=360)
    fig.update_layout(title="작업유형별 데이터 분포")
    fig.update_xaxes(title="작업유형")
    fig.update_yaxes(title="빈도수")
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id=_uid("worktype")))


# ─────────────────────────────────────────────
# 전처리 탭 (텍스트 카드)
# ─────────────────────────────────────────────

def render_pipeline_accordion() -> ui.Tag:
    return ui.accordion(
        ui.accordion_panel(
            "Step 0: 데이터 정제",
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("자정 롤오버 보정:"), " 00:00:00 → 다음날 00:00:00로 수정"),
                ui.tags.li(ui.tags.strong("이상 시점 제거:"), " 2024-11-07 00:00:00"),
                ui.tags.li(ui.tags.strong("극단치 제거:"), " 타겟(전기요금) 상위 0.7% (99.3% 분위수 초과)"),
            ),
        ),
        ui.accordion_panel(
            "Step 1: 기본 시간 변수",
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("시간 분해:"), " day, hour, minute, weekday"),
                ui.tags.li(ui.tags.strong("주말 플래그:"), " is_weekend (weekday >= 5)"),
                ui.tags.li(ui.tags.strong("공휴일 플래그:"), " is_holiday, is_holiday_eve, is_holiday_after"),
                ui.tags.li(ui.tags.strong("시간대 구분:"), " is_peak_after (13-17h), is_peak_even (18-22h), is_night (23-5h)"),
                ui.tags.li(ui.tags.strong("hour_of_week:"), " weekday*24 + hour (0~167)"),
            ),
        ),
        ui.accordion_panel(
            "Step 2: 주기성 인코딩 (사이클릭)",
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("1차 고조파:"), " hour_sin, hour_cos (24시간 주기)"),
                ui.tags.li(ui.tags.strong("2차 고조파:"), " hour_sin2, hour_cos2 (12시간 주기)"),
                ui.tags.li(ui.tags.strong("요일 주기:"), " dow_sin, dow_cos (7일 주기)"),
            ),
        ),
        ui.accordion_panel(
            "Step 3: OOF 기반 3변수 (5-Fold TimeSeriesSplit)",
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("oof_kwh:"), " 전력사용량 예측 (LightGBM MAE)"),
                ui.tags.li(ui.tags.strong("oof_reactive:"), " 지상무효전력량 예측"),
                ui.tags.li(ui.tags.strong("oof_pf:"), " 지상역률(%) 예측"),
                ui.tags.li(ui.tags.em("목적: 미래 정보 없이 현재 시점 특성 추정")),
            ),
        ),
        ui.accordion_panel(
            "Step 4: 시차(Lag) 변수",
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("lag1:"), " 1스텝 전 (15분 전)"),
                ui.tags.li(ui.tags.strong("lag1h:"), " 4스텝 전 (1시간 전)"),
                ui.tags.li(ui.tags.strong("lag6h:"), " 24스텝 전 (6시간 전)"),
                ui.tags.li(ui.tags.strong("lag24h:"), " 96스텝 전 (1일 전 동시간)"),
                ui.tags.li(ui.tags.strong("lag48h:"), " 192스텝 전 (2일 전 동시간)"),
                ui.tags.li(ui.tags.strong("lag7d:"), " 672스텝 전 (7일 전 동시간)"),
            ),
        ),
        ui.accordion_panel(
            "Step 5: 롤링 통계",
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("roll6h:"), " 6시간 이동평균 (24스텝, min_periods=3)"),
                ui.tags.li(ui.tags.strong("roll24h:"), " 24시간 이동평균 (96스텝, min_periods=6)"),
                ui.tags.li(ui.tags.strong("ema24h:"), " 24시간 지수이동평균 (span=96)"),
            ),
        ),
        ui.accordion_panel(
            "Step 6: 차분 변수",
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("samehour_d1:"), " 전일 동시간 차분 (현재 - lag24h)"),
                ui.tags.li(ui.tags.strong("samehour_w1:"), " 전주 동시간 차분 (현재 - lag7d)"),
                ui.tags.li(ui.tags.strong("samehour_w2:"), " 전전주 동시간 차분 (현재 - lag14d)"),
                ui.tags.li(ui.tags.strong("diff1h:"), " 1시간 차분 (현재 - lag1h)"),
            ),
        ),
        ui.accordion_panel(
            "Step 7: 비율 & 프록시 변수",
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("oof_ratio:"), " reactive / kwh (무효전력 비율)"),
                ui.tags.li(ui.tags.strong("oof_pf_proxy:"), " kwh / (kwh + reactive) (역률 대리변수)"),
                ui.tags.li(ui.tags.strong("oof_ratio_ema24h:"), " oof_ratio의 24시간 EMA"),
            ),
        ),
        ui.accordion_panel(
            "Step 8: 시간대 프로필 & 잔차",
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("how_profile_kwh:"), " hour_of_week별 과거 평균 (누적평균 shift 방식)"),
                ui.tags.li(ui.tags.strong("how_resid_kwh:"), " 실제값 - 프로필 (이상 탐지용)"),
                ui.tags.li(ui.tags.strong("kwh_rate_w1:"), " (현재 - lag7d) / |lag7d| (주간 변화율, -3~3 클리핑)"),
                ui.tags.li(ui.tags.strong("kwh_rate_w2:"), " (현재 - lag14d) / |lag14d| (2주 변화율)"),
            ),
        ),
        ui.accordion_panel(
            "Step 9: 카테고리화",
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("how_cat:"), " hour_of_week를 문자열 카테고리로 (168개 클래스)"),
                ui.tags.li(ui.tags.strong("작업유형:"), " 결측치 → 'UNK', Categorical dtype"),
            ),
        ),
        id="pipeline_accordion",
        open=["Step 0: 데이터 정제", "Step 3: OOF 기반 3변수 (5-Fold TimeSeriesSplit)"],
    )


def render_feature_summary() -> ui.HTML:
    html = """
    <div class="p-3" style="font-size: 1rem;">
      <div class="alert alert-primary">
        <h5 class="mb-3">생성된 피처 통계</h5>
        <ul class="mb-0" style="font-size: 1rem;">
          <li><strong>총 피처 수:</strong> 100+ 개</li>
          <li><strong>수치형 피처:</strong> 95+ 개</li>
          <li><strong>범주형 피처:</strong> 2개 (작업유형, how_cat)</li>
        </ul>
      </div>
      <table class="table table-hover mt-3" style="font-size: 0.95rem;">
        <thead class="table-light">
          <tr><th>피처 그룹</th><th>개수</th><th>주요 피처 예시</th></tr>
        </thead>
        <tbody>
          <tr><td><strong>시간 기본</strong></td><td>~15개</td><td>hour, weekday, is_weekend, is_holiday, hour_of_week</td></tr>
          <tr><td><strong>주기성 인코딩</strong></td><td>6개</td><td>hour_sin/cos, hour_sin2/cos2, dow_sin/cos</td></tr>
          <tr><td><strong>OOF 기본</strong></td><td>3개</td><td>oof_kwh, oof_reactive, oof_pf</td></tr>
          <tr><td><strong>Lag 변수</strong></td><td>~18개</td><td>oof_kwh_lag1h/24h/48h/7d × 3개 OOF</td></tr>
          <tr><td><strong>롤링 통계</strong></td><td>~9개</td><td>oof_kwh_roll6h/24h/ema24h × 3개 OOF</td></tr>
          <tr><td><strong>차분 변수</strong></td><td>~12개</td><td>oof_kwh_samehour_d1/w1/w2/diff1h × 3개 OOF</td></tr>
          <tr><td><strong>비율/프록시</strong></td><td>3개</td><td>oof_ratio, oof_pf_proxy, oof_ratio_ema24h</td></tr>
          <tr><td><strong>프로필/잔차</strong></td><td>4개</td><td>how_profile_kwh, how_resid_kwh, kwh_rate_w1/w2</td></tr>
          <tr><td><strong>카테고리</strong></td><td>2개</td><td>작업유형, how_cat</td></tr>
        </tbody>
      </table>
    </div>
    """
    return ui.HTML(html)


def render_scaling_info() -> ui.HTML:
    html = """
    <div class="p-4" style="font-size: 1rem;">
      <div class="alert alert-success">
        <h6 class="mb-3">스케일링 미적용</h6>
        <p class="mb-2"><strong>이유:</strong> 트리 기반 모델은 변수 스케일 영향 적음</p>
        <ul class="mb-0"><li>트리는 <strong>분할 지점(threshold)</strong> 중심</li><li>절대적 값 크기 무관</li></ul>
      </div>
      <h6 class="mt-4 mb-3">인코딩 전략</h6>
      <table class="table table-sm table-striped">
        <thead><tr><th>변수 타입</th><th>처리 방법</th><th>상세</th></tr></thead>
        <tbody>
          <tr><td><strong>수치형</strong></td><td>미적용</td><td>원본 값 사용</td></tr>
          <tr><td><strong>범주형 (작업유형)</strong></td><td>Categorical dtype</td><td>LightGBM categorical_feature 자동 처리</td></tr>
          <tr><td><strong>범주형 (how_cat)</strong></td><td>String → Categorical</td><td>168개 클래스 (hour_of_week 0~167)</td></tr>
          <tr><td><strong>결측치</strong></td><td>Median Imputation</td><td>Train 중앙값 대체</td></tr>
        </tbody>
      </table>
    </div>
    """
    return ui.HTML(html)


def render_leakage_check() -> ui.HTML:
    html = """
    <div class="p-4" style="font-size: 1rem;">
      <h6 class="mb-3">데이터 누수 점검 체크리스트</h6>
      <div class="list-group">
        <div class="list-group-item"><h6 class="mb-2">타겟 변수</h6><p class="mb-0">전기요금(원) 단일 사용, 미래 정보 차단</p></div>
        <div class="list-group-item"><h6 class="mb-2">시간 분할</h6><p class="mb-0">TimeSeriesSplit 적용, 순서 보존</p></div>
        <div class="list-group-item"><h6 class="mb-2">OOF 변수</h6><p class="mb-0">5-Fold TS 기반 OOF 예측</p></div>
        <div class="list-group-item"><h6 class="mb-2">Lag 변수</h6><p class="mb-0">shift() 기반 과거 참조</p></div>
        <div class="list-group-item"><h6 class="mb-2">롤링 통계</h6><p class="mb-0">shift(1) 후 rolling, 현재 시점 제외</p></div>
        <div class="list-group-item"><h6 class="mb-2">시간대 프로필</h6><p class="mb-0">누적평균 shift 구조</p></div>
        <div class="list-group-item"><h6 class="mb-2">스케일링/결측 처리</h6><p class="mb-0">Train 중앙값 기준 변환</p></div>
        <div class="list-group-item list-group-item-warning"><h6 class="mb-2">주의사항</h6><ul class="mb-0"><li>공휴일 정보 2024년 달력 기준</li><li>Test lag는 Train 마지막 시점 기준 생성</li></ul></div>
        <div class="list-group-item list-group-item-success"><h6 class="mb-2">최종 검증</h6><p class="mb-0">Holdout MAE 확인</p></div>
      </div>
    </div>
    """
    return ui.HTML(html)


# ─────────────────────────────────────────────
# EDA 스토리라인 (요약 텍스트 + 그래프 묶음)
# ─────────────────────────────────────────────

def _story_metrics(df: pd.DataFrame) -> dict:
    d = df[["측정일시", "전기요금(원)"]].copy()
    d["측정일시"] = _to_dt(d["측정일시"]) ; d = d.dropna(subset=["측정일시"])  
    d = d[d["전기요금(원)"].notna()]
    if d.empty:
        return {}
    d["month"], d["date"], d["hour"] = d["측정일시"].dt.month, d["측정일시"].dt.date, d["측정일시"].dt.hour
    by_m = d.groupby("month")["전기요금(원)"].sum()
    top_m = int(by_m.idxmax()) if len(by_m) else None
    bot_m = int(by_m.idxmin()) if len(by_m) else None
    by_h = d.groupby("hour")["전기요금(원)"].mean()
    top_hours = ", ".join([f"{int(h)}시" for h in by_h.sort_values(ascending=False).head(3).index]) if len(by_h) else "-"
    low_hours = ", ".join([f"{int(h)}시" for h in by_h.sort_values(ascending=True).head(3).index]) if len(by_h) else "-"
    return {"top_m": top_m, "bot_m": bot_m, "top_hours": top_hours, "low_hours": low_hours}


def plot_monthly_cost(df: pd.DataFrame) -> ui.HTML:
    d = df[["측정일시", "전기요금(원)"]].copy()
    d["측정일시"] = _to_dt(d["측정일시"]) ; d = d.dropna()
    d["월"] = d["측정일시"].dt.to_period("M").dt.to_timestamp()
    g = d.groupby("월")["전기요금(원)"].sum().reset_index()
    fig = go.Figure(go.Bar(x=g["월"], y=g["전기요금(원)"]))
    _apply_plot_theme(fig, height=340)
    fig.update_layout(title="월별 전기요금 분포 (합계)")
    fig.update_xaxes(title="월")
    fig.update_yaxes(title="원")
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id=_uid("mon")))


def plot_daily_cost(df: pd.DataFrame) -> ui.HTML:
    d = df[["측정일시", "전기요금(원)"]].copy()
    d["측정일시"] = _to_dt(d["측정일시"]) ; d = d.dropna()
    d["일"] = d["측정일시"].dt.date
    g = d.groupby("일")["전기요금(원)"].sum().reset_index()
    fig = go.Figure(go.Scatter(x=g["일"], y=g["전기요금(원)"], mode="lines"))
    _apply_plot_theme(fig, height=340)
    fig.update_layout(title="일별 전기요금 분포 (합계)")
    fig.update_xaxes(title="일")
    fig.update_yaxes(title="원")
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id=_uid("day")))


def plot_hourly_cost(df: pd.DataFrame) -> ui.HTML:
    d = df[["측정일시", "전기요금(원)"]].copy()
    d["측정일시"] = _to_dt(d["측정일시"]) ; d = d.dropna()
    d["hour"] = d["측정일시"].dt.hour
    g = d.groupby("hour")["전기요금(원)"].mean().reset_index()
    fig = go.Figure(go.Scatter(x=g["hour"], y=g["전기요금(원)"], mode="lines+markers"))
    _apply_plot_theme(fig, height=320)
    fig.update_layout(title="시간별 전기요금 분포 (평균)")
    fig.update_xaxes(title="시")
    fig.update_yaxes(title="원")
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id=_uid("hour")))


def plot_quarter_cost(df: pd.DataFrame) -> ui.HTML:
    d = df[["측정일시", "전기요금(원)"]].copy()
    d["측정일시"] = _to_dt(d["측정일시"]) ; d = d.dropna()
    d["분"] = d["측정일시"].dt.minute  # (0, 15, 30, 45)
    g = d.groupby("분")["전기요금(원)"].mean().reset_index()
    fig = go.Figure(go.Box(x=g["분"], y=g["전기요금(원)"], boxpoints=False))
    _apply_plot_theme(fig, height=320)
    fig.update_layout(title="분별(15분) 전기요금 분포")
    fig.update_xaxes(title="분(0,15,30,45)")
    fig.update_yaxes(title="원")
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id=_uid("qtr")))


def plot_season_cost(df: pd.DataFrame) -> ui.HTML:
    d = df[["측정일시", "전기요금(원)"]].copy()
    d["측정일시"] = _to_dt(d["측정일시"]) ; d = d.dropna()
    d["계절"] = d["측정일시"].dt.month.map(_season_from_month)
    order = ["봄", "여름", "가을", "겨울"]
    g = d.groupby("계절")["전기요금(원)"].sum().reindex(order).reset_index()
    fig = go.Figure(go.Bar(x=g["계절"], y=g["전기요금(원)"]))
    _apply_plot_theme(fig, height=340)
    fig.update_layout(title="계절별 전기요금 분포 (합계)")
    fig.update_xaxes(title="계절")
    fig.update_yaxes(title="원")
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", div_id=_uid("season")))


def render_eda_storyline_panels(df: pd.DataFrame) -> ui.Tag:
    if not {"측정일시", "전기요금(원)"}.issubset(df.columns):
        return ui.div(ui.div("EDA 스토리 생성에 필요한 컬럼 부족 (측정일시, 전기요금(원))", class_="small-muted"), class_="billx-panel")

    m = _story_metrics(df)
    intro_html = ui.HTML(
        (
            "<div class=\"billx-panel\">"
            "<h5 class=\"billx-panel-title\">EDA 스토리라인</h5>"
            "<p class=\"mb-2\">월·일·시간·분·계절 단위 비용 패턴 점검을 통해 피처 설계 근거 확보</p>"
            f"<ul class=\"mb-0\">"
            f"<li>월별 분포: 고비용 {m.get('top_m','-')}월, 저비용 {m.get('bot_m','-')}월</li>"
            f"<li>시간대 평균: 상위 {m.get('top_hours','-')}, 하위 {m.get('low_hours','-')}</li>"
            "<li>분(15분) 분포: 운영 리듬 및 교대 타이밍 단서</li>"
            "<li>계절별 합계: 냉난방/요금 정책 영향 장주기 변동</li>"
            "</ul>"
            "</div>"
        )
    )

    panels = ui.div(
        ui.layout_columns(
            ui.div(ui.h6("① 월별 합계"), plot_monthly_cost(df), class_="billx-panel"),
            ui.div(ui.h6("② 일별 합계"), plot_daily_cost(df), class_="billx-panel"),
            col_widths=[6, 6],
        ),
        ui.layout_columns(
            ui.div(ui.h6("③ 시간별 평균"), plot_hourly_cost(df), class_="billx-panel"),
            ui.div(ui.h6("④ 분(15분) 분포"), plot_quarter_cost(df), class_="billx-panel"),
            col_widths=[6, 6],
        ),
        ui.div(ui.h6("⑤ 계절별 합계"), plot_season_cost(df), class_="billx-panel"),
    )

    return ui.div(intro_html, panels)