# modules/page_report.py

from shiny import ui, render, reactive
import datetime as dt, calendar, io, tempfile, subprocess
import pandas as pd
from pathlib import Path
import os, shutil

from shared import report_df
from viz.report_plots import mom_bar_chart, yearly_trend_chart

# Word 템플릿/이미지 삽입에 필요한 것들
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
# 차트 만들기 직전
import plotly.graph_objects as go          # ← 추가
from plotly.subplots import make_subplots  # ← 추가

# docxtpl 사용 가능 플래그
try:
    import docxtpl  # 단순 존재 확인
    _DOCXTPL_OK = True
except Exception:
    _DOCXTPL_OK = False

TEMPLATE_PATH = Path(r"C:\Users\LS\Desktop\electricity-project\data\electricity_bill_template_.docx")

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
                    ui.download_button("btn_export_pdf", "PDF 저장", class_="btn btn-primary"),
                    ui.download_button("btn_export_csv", "CSV 내보내기", class_="btn btn-outline-primary"),
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
    
    ### 버튼 코드 추가 - PDF 연결

# ... report_server 내부 맨 아래쪽쯤에 추가 ...

    @output
    @render.download(
        filename=lambda: f"전기요금청구서_{month_key()}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    def btn_export_docx():
        # PDF용에서 doc.render(context)까지 동일하게 생성했다고 가정
        doc = DocxTemplate(str(TEMPLATE_PATH))
        # ... context 채우고 이미지 넣고 ...
        # (이미 btn_export_pdf에서 사용 중인 context/이미지 생성 로직 재사용)
        # ↓ DOCX만 저장/반환
        word_path = tempfile.NamedTemporaryFile(suffix=".docx", delete=False).name
        doc.render(context)
        doc.save(word_path)
        with open(word_path, "rb") as f:
            data = f.read()
        Path(word_path).unlink(missing_ok=True)
        return io.BytesIO(data)


 # ===== CSV 다운로드 =====
    @output
    @render.download(filename="electricity_data.csv")
    def btn_export_csv():
        mdf = df_month()
        return io.StringIO(mdf.to_csv(index=False, encoding="utf-8-sig"))

    # ===== PDF 다운로드 (Word → PDF 변환) =====
    @output
    @render.download(
    filename=lambda: f"전기요금청구서_{month_key()}.pdf",
    media_type="application/pdf",   # ✅ MIME 타입 지정
    )
    def btn_export_pdf():
        if not _DOCXTPL_OK:
            raise RuntimeError("docxtpl 라이브러리가 설치되지 않았습니다.")
        
        # 1) 선택한 월 데이터
        ym = month_key()
        y, m = _ym_to_year_month(ym)
        mdf = df_month()
        
        if mdf.empty:
            raise ValueError(f"{ym} 데이터가 없습니다.")
        
        # 2) 전월 데이터
        prev_start = (pd.to_datetime(f"{ym}-01") - pd.offsets.MonthEnd(1)).replace(day=1)
        prev_end = prev_start + pd.offsets.MonthEnd(0)
        prev_df = df_2024[
            (df_2024[COL_TS] >= prev_start) &
            (df_2024[COL_TS] <= prev_end)
        ]
        
        # 3) 요금 계산 (간단한 예시 - 실제로는 더 복잡한 계산 필요)
        total_cost = mdf[NUM_COLS_COST].sum()
        total_kwh = mdf[NUM_COLS_KWH].sum()
        
        # 기본요금 (예: 총 요금의 20%)
        basic_fee = int(total_cost * 0.2)
        # 전력량요금 (예: 총 요금의 70%)
        energy_fee = int(total_cost * 0.7)
        # 할인 (예: 5%)
        discount_fee = int(total_cost * 0.05)
        # VAT (10%)
        vat = int((basic_fee + energy_fee - discount_fee) * 0.1)
        # 기타
        etc_fee = int(total_cost - basic_fee - energy_fee + discount_fee - vat)
        
        daily_sum = mdf.groupby(mdf[COL_TS].dt.date)[NUM_COLS_COST].sum() if NUM_COLS_COST in mdf else pd.Series(dtype=float)
        max_daily_cost = int(daily_sum.max()) if not daily_sum.empty else 0

        total_co2 = float(mdf["탄소배출량(tCO2)"].sum()) if "탄소배출량(tCO2)" in mdf else 0.0
        # 최종 청구액
        total_amount = basic_fee + energy_fee - discount_fee + vat + etc_fee
        
        # 4) 차트 생성 및 저장
        # === 대시보드와 맞춘 색상 ===
        COLOR_COST = "#2F67FF"   # 요금(파랑)
        COLOR_KWH  = "#22C55E"   # 사용량(초록)
        # ===== Graph 1: 전월 대비 비교 (막대 / 좌:요금, 우:사용량) =====
        # monthly_df_all 에서 당월/전월 값 추출
        current_month = month_key()
        cur_row = monthly_df_all[monthly_df_all["ym"] == current_month]

        yy, mm = _ym_to_year_month(current_month)
        prev_ym = f"{yy-1}-12" if mm == 1 else f"{yy}-{mm-1:02d}"
        prev_row = monthly_df_all[monthly_df_all["ym"] == prev_ym]

        if not cur_row.empty:
            cur_cost = float(cur_row[NUM_COLS_COST].iloc[0])
            cur_kwh  = float(cur_row[NUM_COLS_KWH].iloc[0])
        else:
            cur_cost = float(total_cost)
            cur_kwh  = float(total_kwh)

        if not prev_row.empty:
            prev_cost = float(prev_row[NUM_COLS_COST].iloc[0])
            prev_kwh  = float(prev_row[NUM_COLS_KWH].iloc[0])
        else:
            prev_cost = float(prev_df[NUM_COLS_COST].sum()) if not prev_df.empty else 0.0
            prev_kwh  = float(prev_df[NUM_COLS_KWH].sum()) if not prev_df.empty else 0.0

        # ===== Graph 1: 전월/당월 전력사용량(kWh) 막대 2개 =====
        COLOR_PREV = "#93c5fd"   # 전월 = 아주 연한 파랑(blue-300)
        COLOR_CURR = "#ef4444"   # 당월: 빨강
                # ===== Graph 1: 전월/당월 전력사용량 (가로 막대) =====
        fig_mom = go.Figure()

        # 전월(파랑)
        fig_mom.add_trace(go.Bar(
            name="전월",
            y=["전월"],                     # 카테고리축(세로)
            x=[prev_kwh],                  # 값축(가로)
            orientation="h",
            text=[f"{prev_kwh:,.0f} kWh"],
            textposition="outside",        # 막대 밖 오른쪽
            textfont=dict(size=12),
            marker_color=COLOR_PREV,
            cliponaxis=False               # 텍스트 잘림 방지
        ))

        # 당월(빨강)
        fig_mom.add_trace(go.Bar(
            name="당월",
            y=["당월"],
            x=[cur_kwh],
            orientation="h",
            text=[f"{cur_kwh:,.0f} kWh"],
            textposition="outside",
            textfont=dict(size=12),
            marker_color=COLOR_CURR,
            cliponaxis=False
        ))

        max_kwh = max(prev_kwh, cur_kwh)

        fig_mom.update_layout(
            title=dict(text="전월/당월 전력사용량 비교"),
            barmode="group",
            xaxis=dict(title="전력사용량(kWh)", showgrid=True, gridcolor="lightgray"),
            yaxis=dict(title=""),
            height=240, width=480,
            margin=dict(l=90, r=140, t=60, b=40),   # ← r를 140으로 키움
            showlegend=False,
            plot_bgcolor="white"
        )
        # 막대 끝 오른쪽에 텍스트 들어갈 공간 확보
        fig_mom.update_xaxes(range=[0, max_kwh * 1.12], automargin=True)
        fig_mom.update_yaxes(automargin=True)

        # ===== Graph 2: 2024년 추이 (선 / 좌:요금, 우:사용량) =====
        ms = monthly_df_all.sort_values("ym")
        fig_year = go.Figure()
        fig_year.add_trace(go.Scatter(
            x=ms["ym"], y=ms[NUM_COLS_COST], mode='lines+markers',
            name='전기요금(원)', yaxis='y',
            line=dict(color=COLOR_COST, width=3),   # ★ 색 지정
            marker=dict(size=7)
        ))
        fig_year.add_trace(go.Scatter(
            x=ms["ym"], y=ms[NUM_COLS_KWH], mode='lines+markers',
            name='전력사용량(kWh)', yaxis='y2',
            line=dict(color=COLOR_KWH, width=3, dash='dot'),  # ★ 색 지정
            marker=dict(size=7)
        ))
        fig_year.update_layout(
            title=dict(text="2024년도 사용/요금 추이"),
            xaxis=dict(title="월", tickangle=-45),
            yaxis=dict(title="전기요금(원)", side='left', showgrid=True, gridcolor='lightgray'),
            yaxis2=dict(title="전력사용량(kWh)", side='right', overlaying='y', showgrid=False),
            height=350, width=900, margin=dict(l=60, r=60, t=40, b=60),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2),
            plot_bgcolor='white'
        )

        # 이미지로 저장 (kaleido 필요)
        img1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        fig_mom.write_image(img1, width=480, height=240, scale=2)
        fig_year.write_image(img2, width=900, height=350, scale=2)
        
        # 5) Word 문서 생성
        doc = DocxTemplate(str(TEMPLATE_PATH))
        
        context = {
            "customer_name": "구미래",
            "billing_month": f"{m:02d}",
            "customer_id": "LS202405-01",
            "total_cost": f"{int(total_cost):,}",
            "usage_period": f"{mdf[COL_TS].min():%Y-%m-%d} ~ {mdf[COL_TS].max():%Y-%m-%d}",
            "main_work_type": mdf["작업유형"].mode().iloc[0] if "작업유형" in mdf else "—",
            "previous_month": f"{prev_start.month:02d}",
            "current_usage": f"{total_kwh:,.1f}",
            "previous_usage": f"{prev_df[NUM_COLS_KWH].sum():,.1f}" if not prev_df.empty else "0",
            "address": "충청북도 청주시 흥덕구...",
            "previous_total_cost": f"₩{prev_df[NUM_COLS_COST].sum():,.0f}" if not prev_df.empty else "₩0",
            "contract_type": "일반용 저압",
            "total_amount": f"{total_amount:,}",
            "notice_month": f"{m:02d}",
            "notice_day": f"{dt.date.today().day:02d}",
            "generation_date": dt.datetime.now().strftime("%Y-%m-%d"),
            "max_daily_cost": f"{max_daily_cost:,}",
            "total_co2": f"{total_co2:,.3f}",
        }
        
        # 이미지 삽입
        context["graph1"] = InlineImage(doc, img1, width=Mm(100), height=Mm(55))
        context["graph2"] = InlineImage(doc, img2, width=Mm(100), height=Mm(55))

        doc.render(context)
        
        # Word 파일 저장
        word_path = tempfile.NamedTemporaryFile(suffix=".docx", delete=False).name
        doc.save(word_path)
        
        # 6) Word → PDF 변환 (outdir=TEMP에 맞춰 읽기 경로도 TEMP로!)
        temp_dir = Path(tempfile.gettempdir())
        pdf_name = Path(word_path).with_suffix(".pdf").name
        pdf_out  = temp_dir / pdf_name  # ← 실제로 LibreOffice가 만들어줄 경로

        # soffice(.com) 경로 자동 탐색
        soffice_cmd = shutil.which("soffice.com") or shutil.which("soffice")
        if not soffice_cmd:
            for cand in [
                r"C:\Program Files\LibreOffice\program\soffice.com",
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.com",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            ]:
                if os.path.exists(cand):
                    soffice_cmd = cand
                    break

        try:
            if not soffice_cmd:
                raise FileNotFoundError("LibreOffice(soffice) 실행 파일을 찾을 수 없습니다.")

            # 변환 실행
            subprocess.run(
                [soffice_cmd, "--headless", "--convert-to", "pdf", "--outdir", str(temp_dir), word_path],
                check=True,
                timeout=60,
            )

            # TEMP에 생성된 PDF를 읽어서 반환
            with open(pdf_out, "rb") as f:
                pdf_data = f.read()

            # 임시파일 정리
            Path(word_path).unlink(missing_ok=True)
            pdf_out.unlink(missing_ok=True)
            Path(img1).unlink(missing_ok=True)
            Path(img2).unlink(missing_ok=True)

            return io.BytesIO(pdf_data)

        except Exception as e:
            print("PDF 변환 실패, DOCX로 대체 반환:", repr(e))
            with open(word_path, "rb") as f:
                word_data = f.read()

            Path(word_path).unlink(missing_ok=True)
            Path(img1).unlink(missing_ok=True)
            Path(img2).unlink(missing_ok=True)
            return io.BytesIO(word_data)