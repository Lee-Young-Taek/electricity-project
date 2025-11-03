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
                            class_="kpi",
                        ),
                        ui.div(
                            ui.div("전기요금(원)", class_="kpi-title"),
                            ui.div(ui.output_text("kpi_cost"), class_="kpi-value"),
                            ui.div("합계", class_="kpi-sub"),
                            class_="kpi",
                        ),
                        ui.div(
                            ui.div("평균 단가(원/kWh)", class_="kpi-title"),
                            ui.div(ui.output_text("kpi_unit"), class_="kpi-value"),
                            ui.div("전기요금/사용량", class_="kpi-sub"),
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
                        class_="kpi-row",
                    ),
                    class_="billx-panel",
                ),

                # 월 달력 · 일일 요금
                ui.div(
                    ui.div("월 달력 · 일일 요금", class_="billx-panel-title"),
                    ui.output_ui("month_calendar"),
                    ui.div({"class": "small-muted mt-2"}, "※ 날짜 밑 수치는 일일 요금 합계(원)입니다."),
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
        top = (mdf["작업유형"]
               .value_counts()
               .head(3)
               .to_dict())
        # 예: Light(120) · Normal(95) …
        parts = [f"{k}({v:,})" for k, v in top.items()]
        return " · ".join(parts) if parts else "—"

    @output
    @render.text
    def issue_info():
        today = dt.date.today().strftime("%Y-%m-%d")
        return f"발행일 {today} · 공장 전력 데이터 기반 자동 생성"

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

        # 선택 월의 일자별 요금 합계
        mdf = df_month()
        daily = {}
        if not mdf.empty and NUM_COLS_COST in mdf:
            tmp = mdf.groupby(mdf[COL_TS].dt.date)[NUM_COLS_COST].sum()
            daily = tmp.to_dict()

        header = ui.tags.div({"class": "cal-weekdays"}, *[ui.tags.div(x) for x in WEEK_LABELS])

        cells = []
        for _ in range(offset):
            cells.append(ui.tags.div({"class": "cal-cell empty"}))

        today = dt.date.today()
        for d in range(1, ndays + 1):
            date_obj = dt.date(first.year, first.month, d)
            col = (offset + (d - 1)) % 7
            cls = "cal-cell" + (" sun" if col == 0 else " sat" if col == 6 else "")
            if date_obj == today:
                cls += " today"
            val = daily.get(date_obj)
            text = f"{val:,.0f}원" if isinstance(val, (int, float)) else "—"
            cells.append(
                ui.tags.div(
                    {"class": cls},
                    ui.tags.div(str(d), {"class": "date"}),
                    ui.tags.div(text, {"class": "cost"}),
                )
            )

        grid = ui.tags.div({"class": "cal-grid"}, *cells)
        return ui.tags.div({"class": "billx-cal"}, header, grid)

    # ===== 시각화: MoM / 2024년 추이 (viz 분리) =====
    @output
    @render.ui
    def mom_chart():
        # 월별 합계 DF와 현재 선택 월 전달
        return mom_bar_chart(monthly_df_all, month_key())

    @output
    @render.ui
    def year_chart():
        return yearly_trend_chart(monthly_df_all, month_key())
