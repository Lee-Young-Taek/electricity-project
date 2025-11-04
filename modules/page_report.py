# modules/page_report.py
from shiny import ui, render, reactive
import datetime as dt
import calendar
import pandas as pd

from shared import report_df  # ✅ 실데이터
from viz.report_plots import mom_bar_chart, yearly_trend_chart  # ✅ 분리된 시각화

# 2024-01 ~ 2024-11 고정 선택지
MONTH_CHOICES = {f"2024-{m:02d}": f"2024년 {m}월" for m in range(1, 12)}
DEFAULT_MONTH = "2024-11"

NUM_COLS_COST = "전기요금(원)"
NUM_COLS_KWH  = "전력사용량(kWh)"
COL_TS        = "측정일시"


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[COL_TS] = pd.to_datetime(out[COL_TS], errors="coerce")
    out = out.dropna(subset=[COL_TS]).sort_values(COL_TS).reset_index(drop=True)
    return out


def _ym_to_year_month(ym: str) -> tuple[int, int]:
    y, m = ym.split("-")
    return int(y), int(m)


def report_ui():
    return ui.page_fluid(
        ui.tags.link(rel="stylesheet", href="report.css"),

        # ───────── 상단 리본 ─────────
        ui.div(
            ui.div(
                ui.div(
                    ui.span("전기요금 분석 고지서", class_="billx-title"),
                    ui.span(" · 생산설비 전력 사용 고지", class_="billx-sub"),
                    class_="billx-titlebox",
                ),
                ui.div(
                    ui.input_select("rep_month", "청구월", choices=MONTH_CHOICES, selected=DEFAULT_MONTH),
                    class_="billx-month",
                ),
                ui.div(
                    ui.div("청구금액", class_="billx-amt-label"),
                    ui.div(ui.output_text("amt_due"), class_="billx-amt-value"),
                    class_="billx-amount-pill",
                ),
                class_="billx-ribbon",
            ),
            class_="billx",
        ),

        # ───────── 본문 그리드 ─────────
        ui.div(
            # 좌측 컬럼
            ui.div(
                # 청구 요약 KPI
                ui.div(
                    ui.div("청구 요약", class_="billx-panel-title"),
                    ui.div(
                        ui.div(
                            ui.div("전력사용량(kWh)", class_="kpi-title"),
                            ui.div(ui.output_text("kpi_kwh"), class_="kpi-value"),
                            ui.div("당월 합계", class_="kpi-sub"),
                            ui.div(ui.output_ui("mom_kwh"), class_="kpi-delta"),
                            class_="kpi",
                        ),
                        ui.div(
                            ui.div("전기요금(원)", class_="kpi-title"),
                            ui.div(ui.output_text("kpi_cost"), class_="kpi-value"),
                            ui.div("합계", class_="kpi-sub"),
                            ui.div(ui.output_ui("mom_cost"), class_="kpi-delta"),
                            class_="kpi",
                        ),
                        ui.div(
                            ui.div("평균 단가(원/kWh)", class_="kpi-title"),
                            ui.div(ui.output_text("kpi_unit"), class_="kpi-value"),
                            ui.div("전기요금/사용량", class_="kpi-sub"),
                            ui.div(ui.output_ui("mom_unit"), class_="kpi-delta"),
                            class_="kpi",
                        ),
                        class_="kpi-row",
                    ),
                    class_="billx-panel",
                ),

                # 전력 품질 평균 요약
                ui.div(
                    ui.div("전력 품질 평균 요약", class_="billx-panel-title"),
                    ui.tags.ul(
                        {"class": "billx-list"},
                        ui.tags.li(ui.tags.b("지상역률(%)"), ui.span(ui.output_text("pf_lg"))),
                        ui.tags.li(ui.tags.b("진상역률(%)"), ui.span(ui.output_text("pf_ld"))),
                        ui.tags.li(ui.tags.b("지상무효전력량(kVarh)"), ui.span(ui.output_text("q_lg"))),
                        ui.tags.li(ui.tags.b("진상무효전력량(kVarh)"), ui.span(ui.output_text("q_ld"))),
                    ),
                    ui.div({"class": "billx-note"}, "※ 역률 저하는 요금 가산/설비 효율 저하로 이어질 수 있습니다."),
                    class_="billx-panel",
                ),

                # 탄소·환경
                ui.div(
                    ui.div("탄소·환경", class_="billx-panel-title"),
                    ui.div(
                        ui.div(
                            ui.div("총 탄소배출량(tCO₂)", class_="kpi-title"),
                            ui.div(ui.output_text("kpi_co2"), class_="kpi-value"),
                            class_="kpi",
                        ),
                        ui.div(  # ← 추가: 배출강도
                            ui.div("배출강도(kgCO₂/kWh)", class_="kpi-title"),
                            ui.div(ui.output_text("kpi_co2_intensity"), class_="kpi-value"),
                            ui.div("= (tCO₂×1000) / kWh", class_="kpi-sub"),
                            class_="kpi",
                        ),

                        ui.div(  # ← 추가: 일평균 배출량
                            ui.div("일평균 배출량(kgCO₂/일)", class_="kpi-title"),
                            ui.div(ui.output_text("kpi_co2_daily"), class_="kpi-value"),
                            ui.div("당월 기준", class_="kpi-sub"),
                            class_="kpi",
                        ),

                        class_="kpi-row",
                    ),
                    class_="billx-panel",
                ),

                # 월 달력 · 일일 요금
                ui.div(
                    ui.div(
                        ui.span("월 달력 · 일일 요금", class_="billx-panel-title"),
                        ui.input_radio_buttons(
                            "cal_metric", None, {"cost": "요금", "kwh": "사용량"},
                            selected="cost", inline=True, 
                        ),
                        class_="cal-metric-toggle"
                    ),
                    ui.output_ui("month_calendar"),
                    ui.div({"class": "small-muted mt-2"}, "※ 날짜 칸 수치는 선택 지표의 일일 합계입니다. ● = 월내 상위 10% 피크"),
                    class_="billx-panel",
                ),

                class_="billx-col",
            ),

            # 우측 컬럼
            ui.div(
                # 검침/사용 정보
                ui.div(
                    ui.div("검침/사용 정보", class_="billx-panel-title"),
                    ui.tags.ul(
                        {"class": "billx-list"},
                        ui.tags.li(ui.tags.b("검침기간"), ui.span(ui.output_text("period"))),
                        ui.tags.li(ui.tags.b("데이터 건수"), ui.span(ui.output_text("rows"))),
                        ui.tags.li(ui.tags.b("주요 작업유형"), ui.span(ui.output_text("worktypes"))),
                    ),
                    class_="billx-panel",
                ),

                # 전월 대비 비교 (MoM)
                ui.div(
                    ui.div("전월 대비 비교", class_="billx-panel-title"),
                    ui.output_ui("mom_chart"),
                    class_="billx-panel",
                ),

                # 2024년 추이
                ui.div(
                    ui.div("2024년 추이", class_="billx-panel-title"),
                    ui.output_ui("year_chart"),
                    class_="billx-panel",
                ),
                class_="billx-col",
            ),
            class_="billx-grid",
        ),

        # 하단 바
        ui.div(
            ui.div(
                ui.span(ui.output_text("issue_info"), class_="billx-issue"),
                ui.div(
                    ui.input_action_button("btn_export_pdf", "PDF 저장", class_="btn btn-primary"),
                    ui.input_action_button("btn_export_csv", "CSV 내보내기", class_="btn btn-outline-primary"),
                    class_="billx-actions",
                ),
                class_="billx-footer-inner",
            ),
            class_="billx-footer",
        ),
    )


def report_server(input, output, session):
    # ===== 데이터 준비 =====
    df_all = _ensure_datetime(report_df)
    # 2024년만 사용 (요구사항 월 선택과 맞춤)
    df_2024 = df_all[(df_all[COL_TS].dt.year == 2024)].copy()

    # 공통 파생: 날짜/월 문자열
    df_2024["date"] = df_2024[COL_TS].dt.date
    df_2024["ym"] = df_2024[COL_TS].dt.strftime("%Y-%m")

    # 월별 집계(합/평균 혼합: 비용/사용량은 합, 역률은 평균, 무효전력 합, CO2 합)
    def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
        out = df.groupby("ym").agg({
            NUM_COLS_COST: "sum",
            NUM_COLS_KWH: "sum",
            "지상역률(%)": "mean",
            "진상역률(%)": "mean",
            "지상무효전력량(kVarh)": "sum",
            "진상무효전력량(kVarh)": "sum",
            "탄소배출량(tCO2)": "sum",
            "id": "count",
        }).rename(columns={"id": "rows"}).reset_index()
        return out

    monthly_df_all = monthly_agg(df_2024)

    # 선택 월의 월간 DataFrame (reactive)
    @reactive.calc
    def month_key():
        return input.rep_month() or DEFAULT_MONTH

    @reactive.calc
    def df_month():
        ym = month_key()
        y, m = _ym_to_year_month(ym)
        mdf = df_2024[(df_2024[COL_TS].dt.year == y) & (df_2024[COL_TS].dt.month == m)].copy()
        return mdf

    # ===== KPI / 텍스트 바인딩 =====
    @output
    @render.text
    def amt_due():
        mdf = df_month()
        if mdf.empty or NUM_COLS_COST not in mdf:
            return "— 원"
        return f"{int(mdf[NUM_COLS_COST].sum()):,}원"

    @output
    @render.text
    def kpi_kwh():
        mdf = df_month()
        if mdf.empty or NUM_COLS_KWH not in mdf:
            return "—"
        return f"{mdf[NUM_COLS_KWH].sum():,.2f}"

    @output
    @render.text
    def kpi_cost():
        mdf = df_month()
        if mdf.empty or NUM_COLS_COST not in mdf:
            return "—"
        return f"{int(mdf[NUM_COLS_COST].sum()):,}"

    @output
    @render.text
    def kpi_unit():
        mdf = df_month()
        if mdf.empty or (NUM_COLS_COST not in mdf) or (NUM_COLS_KWH not in mdf):
            return "—"
        kwh = mdf[NUM_COLS_KWH].sum()
        cost = mdf[NUM_COLS_COST].sum()
        return f"{(cost / kwh):,.1f}" if kwh > 0 else "—"

    @output
    @render.text
    def pf_lg():
        mdf = df_month()
        if mdf.empty or "지상역률(%)" not in mdf:
            return "—"
        return f"{mdf['지상역률(%)'].mean():.1f}"

    @output
    @render.text
    def pf_ld():
        mdf = df_month()
        if mdf.empty or "진상역률(%)" not in mdf:
            return "—"
        return f"{mdf['진상역률(%)'].mean():.1f}"

    @output
    @render.text
    def q_lg():
        mdf = df_month()
        col = "지상무효전력량(kVarh)"
        if mdf.empty or col not in mdf:
            return "—"
        return f"{mdf[col].mean():,.3f}"

    @output
    @render.text
    def q_ld():
        mdf = df_month()
        col = "진상무효전력량(kVarh)"
        if mdf.empty or col not in mdf:
            return "—"
        return f"{mdf[col].mean():,.3f}"

    @output
    @render.text
    def kpi_co2():
        mdf = df_month()
        col = "탄소배출량(tCO2)"
        if mdf.empty or col not in mdf:
            return "—"
        return f"{mdf[col].sum():,.3f}"

    @output
    @render.text
    def period():
        mdf = df_month()
        if mdf.empty:
            return "—"
        start = mdf[COL_TS].min().date()
        end   = mdf[COL_TS].max().date()
        return f"{start} ~ {end}"

    @output
    @render.text
    def rows():
        mdf = df_month()
        return f"{len(mdf):,}"

    @output
    @render.text
    def worktypes():
        mdf = df_month()
        if mdf.empty or "작업유형" not in mdf:
            return "—"

        # 한글 맵
        def _ko(name: str) -> str:
            key = str(name).strip().lower().replace(" ", "_")
            MAP = {
                "light_load": "경부하",
                "medium_load": "중간부하",
                "maximum_load": "최대부하",
            }
            return MAP.get(key, str(name))

        vc = mdf["작업유형"].value_counts().head(3)
        parts = [f"{_ko(k)}({v:,})" for k, v in vc.items()]
        return " · ".join(parts) if parts else "—"

    @output
    @render.text
    def issue_info():
        today = dt.date.today().strftime("%Y-%m-%d")
        return f"발행일 {today} · 공장 전력 데이터 기반 자동 생성"
    
    def _prev_key(ym: str) -> str | None:
        try:
            y, m = _ym_to_year_month(ym)
            py, pm = (y, m-1) if m > 1 else (y-1, 12)
            return f"{py:04d}-{pm:02d}"
        except Exception:
            return None
        
    def _mom_fmt(cur: float | None, prev: float | None):
        """(표시문자열, 클래스명[na/pos/neg]) 반환"""
        if cur is None or prev is None or prev == 0:
            return "—", "na"
        delta = (cur - prev) / prev * 100.0
        sign = "+" if delta >= 0 else ""
        cls  = "pos" if delta > 0 else ("neg" if delta < 0 else "na")
        return f"{sign}{delta:.1f}%", cls

    
    @output
    @render.ui
    def mom_kwh():
        ym = month_key(); prev = _prev_key(ym)
        cur_df = df_month()
        cur = float(cur_df[NUM_COLS_KWH].sum()) if not cur_df.empty and NUM_COLS_KWH in cur_df else None
        prv = None
        if prev:
            y, m = _ym_to_year_month(prev)
            prv_df = df_2024[(df_2024[COL_TS].dt.year==y)&(df_2024[COL_TS].dt.month==m)]
            prv = float(prv_df[NUM_COLS_KWH].sum()) if not prv_df.empty and NUM_COLS_KWH in prv_df else None
        txt, cls = _mom_fmt(cur, prv)
        return ui.span(txt, class_=cls)

    @output
    @render.ui
    def mom_cost():
        ym = month_key(); prev = _prev_key(ym)
        cur_df = df_month()
        cur = float(cur_df[NUM_COLS_COST].sum()) if not cur_df.empty and NUM_COLS_COST in cur_df else None
        prv = None
        if prev:
            y, m = _ym_to_year_month(prev)
            prv_df = df_2024[(df_2024[COL_TS].dt.year==y)&(df_2024[COL_TS].dt.month==m)]
            prv = float(prv_df[NUM_COLS_COST].sum()) if not prv_df.empty and NUM_COLS_COST in prv_df else None
        txt, cls = _mom_fmt(cur, prv)
        return ui.span(txt, class_=cls)

    @output
    @render.ui
    def mom_unit():
        ym = month_key(); prev = _prev_key(ym)
        cur_df = df_month()
        cur_kwh = float(cur_df[NUM_COLS_KWH].sum()) if not cur_df.empty and NUM_COLS_KWH in cur_df else None
        cur_cost= float(cur_df[NUM_COLS_COST].sum()) if not cur_df.empty and NUM_COLS_COST in cur_df else None
        cur = (cur_cost/cur_kwh) if (cur_kwh and cur_kwh>0 and cur_cost is not None) else None

        prv = None
        if prev:
            y, m = _ym_to_year_month(prev)
            prv_df = df_2024[(df_2024[COL_TS].dt.year==y)&(df_2024[COL_TS].dt.month==m)]
            if not prv_df.empty and NUM_COLS_KWH in prv_df and NUM_COLS_COST in prv_df:
                prv_kwh  = float(prv_df[NUM_COLS_KWH].sum())
                prv_cost = float(prv_df[NUM_COLS_COST].sum())
                prv = (prv_cost/prv_kwh) if prv_kwh>0 else None

        txt, cls = _mom_fmt(cur, prv)
        return ui.span(txt, class_=cls)

    


    # ===== 달력(일일 요금 합계) =====
    WEEK_LABELS = ["일","월","화","수","목","금","토"]

    def _first_meta_from_str(ym: str):
        y, m = _ym_to_year_month(ym)
        first = dt.date(y, m, 1)
        first_weekday, ndays = calendar.monthrange(y, m)  # Mon=0..Sun=6
        offset = (first_weekday + 1) % 7  # 일요일 시작
        return first, ndays, offset

    @output
    @render.ui
    def month_calendar():
        ym = month_key()
        first, ndays, offset = _first_meta_from_str(ym)
    
        # 선택 지표
        metric = input.cal_metric() if hasattr(input, "cal_metric") else "cost"
        metric_col = NUM_COLS_COST if metric == "cost" else NUM_COLS_KWH
    
        # 일자별 합계 (둘 다 구해두기)
        mdf = df_month()
        if not mdf.empty:
            g = mdf.groupby(mdf[COL_TS].dt.date)
            daily_cost = g[NUM_COLS_COST].sum()
            daily_kwh  = g[NUM_COLS_KWH].sum() if NUM_COLS_KWH in mdf else None
        else:
            daily_cost, daily_kwh = pd.Series(dtype=float), None
    
        # 표시 값/피크 판정용 시리즈
        if metric == "cost":
            s_disp = daily_cost
        else:
            s_disp = daily_kwh if daily_kwh is not None else pd.Series(dtype=float)
    
        # 피크(상위 10%) threshold
        try:
            thresh = s_disp.quantile(0.9) if len(s_disp) else None
        except Exception:
            thresh = None
    
        header = ui.tags.div({"class": "cal-weekdays"}, *[ui.tags.div(x) for x in WEEK_LABELS])
    
        cells = []
        for _ in range(offset):
            cells.append(ui.tags.div({"class": "cal-cell empty"}))
    
        # 주간 합계 계산 준비
        week_pills = []
        cur = first
        # monthcalendar: 주간 단위 배열(0은 그 달 외의 날)
        weeks = calendar.MonthCalendar().monthdayscalendar(first.year, first.month) if hasattr(calendar, "MonthCalendar") \
            else calendar.monthcalendar(first.year, first.month)
    
        today = dt.date.today()
        for d in range(1, ndays + 1):
            date_obj = dt.date(first.year, first.month, d)
            col = (offset + (d - 1)) % 7
    
            val = s_disp.get(date_obj, None)
            txt = "—"
            if isinstance(val, (int, float)) and pd.notna(val):
                txt = f"{val:,.0f}" + ("원" if metric == "cost" else " kWh")
    
            classes = ["cal-cell"]
            if col == 0: classes.append("sun")
            if col == 6: classes.append("sat")
            if date_obj == today: classes.append("today")
            if (thresh is not None) and isinstance(val, (int, float)) and pd.notna(val) and (val >= thresh):
                classes.append("peak")
    
            # 툴팁: 요금/사용량/단가
            tip_cost = daily_cost.get(date_obj, None)
            tip_kwh  = daily_kwh.get(date_obj, None) if daily_kwh is not None else None
            tip_unit = (tip_cost / tip_kwh) if (tip_cost is not None and tip_kwh not in (None, 0)) else None
            tooltip = f"{date_obj} · 요금 {tip_cost:,.0f}원" if isinstance(tip_cost, (int,float)) else f"{date_obj}"
            if isinstance(tip_kwh, (int,float)):  tooltip += f" · 사용 {tip_kwh:,.0f} kWh"
            if isinstance(tip_unit, (int,float)): tooltip += f" · 단가 {tip_unit:,.1f} 원/kWh"
    
            cells.append(
                ui.tags.div(
                    {"class": " ".join(classes), "title": tooltip},
                    ui.tags.div(str(d), {"class": "date"}),
                    ui.tags.div(txt, {"class": "cost"}),
                )
            )
    
        # 주간 합계 pills (선택 지표 기준)
        # weeks는 각 주 [월..일] 숫자 리스트(0은 해당 월 아님)
        for i, wk in enumerate(weeks, start=1):
            day_objs = [dt.date(first.year, first.month, dd) for dd in wk if dd != 0]
            tot = float(sum([s_disp.get(d, 0.0) for d in day_objs])) if len(day_objs) else 0.0
            label = "원" if metric == "cost" else " kWh"
            week_pills.append(ui.tags.span(f"W{i}: {tot:,.0f}{label}", class_="pill"))
    
        grid = ui.tags.div({"class": "cal-grid"}, *cells)
        footer = ui.tags.div({"class": "cal-week-summary"}, *week_pills)
    
        return ui.tags.div({"class": "billx-cal"}, header, grid, footer)
    

    # ===== 시각화: MoM / 2024년 추이 (viz 분리) =====
    @output
    @render.ui
    def mom_chart():
        return mom_bar_chart(monthly_df_all, month_key())

    @output
    @render.ui
    def year_chart():
        return yearly_trend_chart(monthly_df_all, month_key())
    
    # 탄소
    @output
    @render.text
    def kpi_co2_intensity():
        mdf = df_month()
        if mdf.empty or ("탄소배출량(tCO2)" not in mdf) or (NUM_COLS_KWH not in mdf):
            return "—"
        co2_t  = float(mdf["탄소배출량(tCO2)"].sum())
        kwh    = float(mdf[NUM_COLS_KWH].sum())
        if kwh <= 0:
            return "—"
        val = (co2_t * 1000.0) / kwh  # kgCO2/kWh
        return f"{val:,.2f}"

    @output
    @render.text
    def kpi_co2_daily():
        mdf = df_month()
        if mdf.empty or "탄소배출량(tCO2)" not in mdf:
            return "—"
        days = mdf[COL_TS].dt.date.nunique()
        if days == 0:
            return "—"
        kg = float(mdf["탄소배출량(tCO2)"].sum() * 1000.0)
        return f"{(kg / days):,.1f}"

