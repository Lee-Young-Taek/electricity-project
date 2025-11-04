# modules/page_report.py  (2018 라벨링 뷰: 실제 2024 데이터를 2018로 보여주기)

from shiny import ui, render, reactive
import datetime as dt, calendar, io, tempfile
import pandas as pd
from pathlib import Path
import os

from shared import report_df, TEMPLATE_PATH
from viz.report_plots import mom_bar_chart, yearly_trend_chart

from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
import plotly.graph_objects as go

try:
    import docxtpl
    _DOCXTPL_OK = True
except Exception:
    _DOCXTPL_OK = False

# ===== 연도 매핑 설정 (보이는 연도 = 2018, 실제 데이터 연도 = 2024) =====
SOURCE_YEAR = 2024   # 실제 report_df가 가진 연도
ALIAS_YEAR  = 2018   # 화면/다운로드에서 보여줄 연도
YEAR_SHIFT  = SOURCE_YEAR - ALIAS_YEAR  # 6년

# 셀렉트박스: 2018-01 ~ 2018-11
MONTH_CHOICES = {f"{ALIAS_YEAR}-{m:02d}": f"{ALIAS_YEAR}년 {m}월" for m in range(1, 12)}
DEFAULT_MONTH = f"{ALIAS_YEAR}-11"

NUM_COLS_COST = "전기요금(원)"
NUM_COLS_KWH  = "전력사용량(kWh)"
COL_TS        = "측정일시"

# ===================== 공통 유틸 =====================

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[COL_TS] = pd.to_datetime(out[COL_TS], errors="coerce")
    out = out.dropna(subset=[COL_TS]).sort_values(COL_TS).reset_index(drop=True)
    return out

def _alias_year_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    2024 데이터를 그대로 쓰되, 화면에서는 2018로 보이도록 '측정일시'의 연도만 2018로 치환.
    윤년(2/29)은 pandas DateOffset이 2/28로 안전 변환.
    """
    out = _ensure_datetime(df)
    mask = out[COL_TS].dt.year == SOURCE_YEAR
    # 2024 -> 2018 (6년 빼기)
    out.loc[mask, COL_TS] = out.loc[mask, COL_TS] - pd.DateOffset(years=YEAR_SHIFT)
    return out

def _ym_to_year_month(ym: str) -> tuple[int, int]:
    y, m = ym.split("-")
    return int(y), int(m)

# ===== PDF 대체 경로 (Word/LibreOffice 미설치 대응) =====
def _try_msword_pdf(docx_path: str) -> bytes | None:
    try:
        import win32com.client  # pywin32
        wdFormatPDF = 17
        pdf_path = str(Path(docx_path).with_suffix(".pdf"))
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(docx_path)
        try:
            doc.ExportAsFixedFormat(pdf_path, wdFormatPDF)
        except Exception:
            doc.SaveAs(pdf_path, FileFormat=wdFormatPDF)
        finally:
            doc.Close(False); word.Quit()
        with open(pdf_path, "rb") as f:
            data = f.read()
        Path(pdf_path).unlink(missing_ok=True)
        return data
    except Exception:
        return None

def _try_docx2pdf(docx_path: str) -> bytes | None:
    try:
        from docx2pdf import convert
        pdf_path = str(Path(docx_path).with_suffix(".pdf"))
        convert(docx_path, pdf_path)
        with open(pdf_path, "rb") as f:
            data = f.read()
        Path(pdf_path).unlink(missing_ok=True)
        return data
    except Exception:
        return None

def _try_reportlab_pdf(context: dict, img1: str|None=None, img2: str|None=None) -> bytes|None:
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        try:
            malgun = r"C:\Windows\Fonts\malgun.ttf"
            if os.path.exists(malgun):
                pdfmetrics.registerFont(TTFont("Malgun", malgun)); FONT = "Malgun"
            else:
                FONT = "Helvetica"
        except Exception:
            FONT = "Helvetica"
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        W, H = A4

        c.setFont(FONT, 18); c.drawString(25*mm, H-25*mm, "전기요금 분석 고지서")
        c.setFont(FONT, 11)
        c.drawString(25*mm, H-33*mm, f"발행일: {context.get('generation_date', dt.datetime.now().strftime('%Y-%m-%d'))}")
        c.drawString(25*mm, H-40*mm, f"청구월: {context.get('billing_month','--')}월  ·  기간: {context.get('usage_period','--')}")

        y0 = H - 55*mm; line_h = 7.2*mm
        kvs = [
            ("총 전기요금(원)", context.get("total_cost", "-")),
            ("총 전력사용량(kWh)", context.get("current_usage", "-")),
            ("전월 전기요금(원)", context.get("previous_total_cost", "-")),
            ("전월 전력사용량(kWh)", context.get("previous_usage", "-")),
            ("최대 일일 요금(원)", context.get("max_daily_cost", "-")),
            ("총 탄소배출량(tCO₂)", context.get("total_co2", "-")),
        ]
        c.setFont(FONT, 12); c.drawString(25*mm, y0+6*mm, "요약")
        c.setFont(FONT, 10)
        for i, (k,v) in enumerate(kvs):
            c.drawString(25*mm, y0 - i*line_h, f"• {k}: {v}")

        img_w, img_h = 80*mm, 45*mm
        xL, xR = 25*mm, 110*mm
        y_img_top = y0 - len(kvs)*line_h - 10*mm
        if img1 and os.path.exists(img1):
            c.drawImage(img1, xL,  y_img_top - img_h, width=img_w, height=img_h, preserveAspectRatio=True, anchor='nw')
        if img2 and os.path.exists(img2):
            c.drawImage(img2, xR,  y_img_top - img_h, width=img_w, height=img_h, preserveAspectRatio=True, anchor='nw')

        c.setFont(FONT, 8); c.drawString(25*mm, 12*mm, "※ 본 문서는 공장 전력 데이터 기반 자동 생성 요약본입니다.")
        c.showPage(); c.save()
        return buf.getvalue()
    except Exception:
        return None

def convert_to_pdf_without_libreoffice(docx_path: str, context: dict, img1: str|None, img2: str|None) -> bytes|None:
    for fn in (_try_msword_pdf, _try_docx2pdf):
        data = fn(docx_path)
        if data: return data
    data = _try_reportlab_pdf(context, img1, img2)
    return data

def _build_context_and_charts(ym: str, mdf: pd.DataFrame, prev_df: pd.DataFrame, monthly_df_all: pd.DataFrame) -> tuple[dict, str, str]:
    y, m = _ym_to_year_month(ym)

    total_cost = float(mdf[NUM_COLS_COST].sum()) if NUM_COLS_COST in mdf else 0.0
    total_kwh  = float(mdf[NUM_COLS_KWH].sum()) if NUM_COLS_KWH  in mdf else 0.0

    daily_sum = mdf.groupby(mdf[COL_TS].dt.date)[NUM_COLS_COST].sum() if NUM_COLS_COST in mdf else pd.Series(dtype=float)
    max_daily_cost = int(daily_sum.max()) if not daily_sum.empty else 0
    total_co2 = float(mdf["탄소배출량(tCO2)"].sum()) if "탄소배출량(tCO2)" in mdf else 0.0

    # 전월/당월 비교 값
    current_month = ym
    cur_row = monthly_df_all[monthly_df_all["ym"] == current_month]
    yy, mm = _ym_to_year_month(current_month)
    prev_ym = f"{yy-1}-12" if mm == 1 else f"{yy}-{mm-1:02d}"
    prev_row = monthly_df_all[monthly_df_all["ym"] == prev_ym]

    if not cur_row.empty:
        cur_kwh  = float(cur_row[NUM_COLS_KWH].iloc[0])
    else:
        cur_kwh  = float(total_kwh)

    if not prev_row.empty:
        prev_kwh = float(prev_row[NUM_COLS_KWH].iloc[0])
    else:
        prev_kwh = float(prev_df[NUM_COLS_KWH].sum()) if (not prev_df.empty and NUM_COLS_KWH in prev_df) else 0.0

    # 색상
    COLOR_PREV = "#93c5fd"; COLOR_CURR = "#ef4444"
    COLOR_COST = "#2F67FF"; COLOR_KWH = "#22C55E"

    # Graph 1: 전월/당월 전력사용량
    fig_mom = go.Figure()
    fig_mom.add_trace(go.Bar(name="전월", y=["전월"], x=[prev_kwh], orientation="h",
                             text=[f"{prev_kwh:,.0f} kWh"], textposition="outside",
                             textfont=dict(size=12), marker_color=COLOR_PREV, cliponaxis=False))
    fig_mom.add_trace(go.Bar(name="당월", y=["당월"], x=[cur_kwh], orientation="h",
                             text=[f"{cur_kwh:,.0f} kWh"], textposition="outside",
                             textfont=dict(size=12), marker_color=COLOR_CURR, cliponaxis=False))
    max_kwh = max(prev_kwh, cur_kwh)
    fig_mom.update_layout(title=dict(text="전월/당월 전력사용량 비교"), barmode="group",
                          xaxis=dict(title="전력사용량(kWh)", showgrid=True, gridcolor="lightgray"),
                          yaxis=dict(title=""), height=240, width=480,
                          margin=dict(l=90, r=140, t=60, b=40), showlegend=False, plot_bgcolor="white")
    fig_mom.update_xaxes(range=[0, max_kwh * 1.12], automargin=True)
    fig_mom.update_yaxes(automargin=True)

    # Graph 2: 2018년(표시연도) 추이
    ms = monthly_df_all.sort_values("ym")
    fig_year = go.Figure()
    fig_year.add_trace(go.Scatter(x=ms["ym"], y=ms[NUM_COLS_COST], mode='lines+markers', name='전기요금(원)', yaxis='y',
                                  line=dict(color=COLOR_COST, width=3), marker=dict(size=7)))
    fig_year.add_trace(go.Scatter(x=ms["ym"], y=ms[NUM_COLS_KWH], mode='lines+markers', name='전력사용량(kWh)', yaxis='y2',
                                  line=dict(color=COLOR_KWH, width=3, dash='dot'), marker=dict(size=7)))
    fig_year.update_layout(title=dict(text=f"{ALIAS_YEAR}년도 사용/요금 추이"),
                           xaxis=dict(title="월", tickangle=-45),
                           yaxis=dict(title="전기요금(원)", side='left', showgrid=True, gridcolor='lightgray'),
                           yaxis2=dict(title="전력사용량(kWh)", side='right', overlaying='y', showgrid=False),
                           height=350, width=900, margin=dict(l=60, r=60, t=40, b=60),
                           legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2), plot_bgcolor='white')

    img1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    img2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    fig_mom.write_image(img1, width=480, height=240, scale=2)
    fig_year.write_image(img2, width=900, height=350, scale=2)

    context = {
        "customer_name": "구미래",
        "billing_month": f"{m:02d}",
        "customer_id": "LS202405-01",
        "total_cost": f"{int(total_cost):,}",
        "usage_period": f"{mdf[COL_TS].min():%Y-%m-%d} ~ {mdf[COL_TS].max():%Y-%m-%d}",
        "main_work_type": mdf["작업유형"].mode().iloc[0] if "작업유형" in mdf and not mdf.empty else "—",
        "previous_month": f"{(m-1 if m>1 else 12):02d}",
        "current_usage": f"{total_kwh:,.1f}",
        "previous_usage": f"{prev_df[NUM_COLS_KWH].sum():,.1f}" if (not prev_df.empty and NUM_COLS_KWH in prev_df) else "0",
        "address": "충청북도 청주시 흥덕구...",
        "previous_total_cost": f"₩{prev_df[NUM_COLS_COST].sum():,.0f}" if (not prev_df.empty and NUM_COLS_COST in prev_df) else "₩0",
        "contract_type": "산업용 고압",
        "total_amount": f"{int(total_cost):,}",
        "notice_month": f"{m:02d}",
        "notice_day": f"{dt.date.today().day:02d}",
        "generation_date": dt.datetime.now().strftime("%Y-%m-%d"),
        "max_daily_cost": f"{max_daily_cost:,}",
        "total_co2": f"{total_co2:,.3f}",
    }
    return context, img1, img2

# ===================== UI =====================

def report_ui():
    return ui.page_fluid(
        ui.tags.link(rel="stylesheet", href="report.css"),
        ui.div(
            ui.div(
                ui.div(ui.span("전기요금 분석 고지서", class_="billx-title"),
                       ui.span(" · 생산설비 전력 사용 고지", class_="billx-sub"),
                       class_="billx-titlebox"),
                ui.div(ui.input_select("rep_month", "청구월", choices=MONTH_CHOICES, selected=DEFAULT_MONTH),
                       class_="billx-month"),
                ui.div(ui.div("청구금액", class_="billx-amt-label"),
                       ui.div(ui.output_text("amt_due"), class_="billx-amt-value"),
                       class_="billx-amount-pill"),
                class_="billx-ribbon",
            ),
            class_="billx",
        ),
        ui.div(
            ui.div(
                ui.div(
                    ui.div("청구 요약", class_="billx-panel-title"),
                    ui.div(
                        ui.div(ui.div("전력사용량(kWh)", class_="kpi-title"),
                               ui.div(ui.output_text("kpi_kwh"), class_="kpi-value"),
                               ui.div("당월 합계", class_="kpi-sub"),
                               ui.div(ui.output_ui("mom_kwh"), class_="kpi-delta"),
                               class_="kpi"),
                        ui.div(ui.div("전기요금(원)", class_="kpi-title"),
                               ui.div(ui.output_text("kpi_cost"), class_="kpi-value"),
                               ui.div("합계", class_="kpi-sub"),
                               ui.div(ui.output_ui("mom_cost"), class_="kpi-delta"),
                               class_="kpi"),
                        ui.div(ui.div("평균 단가(원/kWh)", class_="kpi-title"),
                               ui.div(ui.output_text("kpi_unit"), class_="kpi-value"),
                               ui.div("전기요금/사용량", class_="kpi-sub"),
                               ui.div(ui.output_ui("mom_unit"), class_="kpi-delta"),
                               class_="kpi"),
                        class_="kpi-row",
                    ),
                    class_="billx-panel",
                ),
                ui.div(
                    ui.div("전력 품질 평균 요약", class_="billx-panel-title"),
                    ui.tags.ul({"class": "billx-list"},
                               ui.tags.li(ui.tags.b("지상역률(%)"), ui.span(ui.output_text("pf_lg"))),
                               ui.tags.li(ui.tags.b("진상역률(%)"), ui.span(ui.output_text("pf_ld"))),
                               ui.tags.li(ui.tags.b("지상무효전력량(kVarh)"), ui.span(ui.output_text("q_lg"))),
                               ui.tags.li(ui.tags.b("진상무효전력량(kVarh)"), ui.span(ui.output_text("q_ld")))),
                    ui.div({"class": "billx-note"}, "※ 역률 저하는 요금 가산/설비 효율 저하로 이어질 수 있습니다."),
                    class_="billx-panel",
                ),
                ui.div(
                    ui.div("탄소·환경", class_="billx-panel-title"),
                    ui.div(
                        ui.div(ui.div("총 탄소배출량(tCO₂)", class_="kpi-title"),
                               ui.div(ui.output_text("kpi_co2"), class_="kpi-value"),
                               class_="kpi"),
                        ui.div(ui.div("배출강도(kgCO₂/kWh)", class_="kpi-title"),
                               ui.div(ui.output_text("kpi_co2_intensity"), class_="kpi-value"),
                               ui.div("= (tCO₂×1000) / kWh", class_="kpi-sub"),
                               class_="kpi"),
                        ui.div(ui.div("일평균 배출량(kgCO₂/일)", class_="kpi-title"),
                               ui.div(ui.output_text("kpi_co2_daily"), class_="kpi-value"),
                               ui.div("당월 기준", class_="kpi-sub"),
                               class_="kpi"),
                        class_="kpi-row",
                    ),
                    class_="billx-panel",
                ),
                ui.div(
                    ui.div(ui.span("월 달력 · 일일 요금", class_="billx-panel-title"),
                           ui.input_radio_buttons("cal_metric", None, {"cost": "요금", "kwh": "사용량"},
                                                  selected="cost", inline=True),
                           class_="cal-metric-toggle"),
                    ui.output_ui("month_calendar"),
                    ui.div({"class": "small-muted mt-2"}, "※ 날짜 칸 수치는 선택 지표의 일일 합계입니다. ● = 월내 상위 10% 피크"),
                    class_="billx-panel",
                ),
                class_="billx-col",
            ),
            ui.div(
                ui.div(
                    ui.div("검침/사용 정보", class_="billx-panel-title"),
                    ui.tags.ul({"class": "billx-list"},
                               ui.tags.li(ui.tags.b("검침기간"), ui.span(ui.output_text("period"))),
                               ui.tags.li(ui.tags.b("데이터 건수"), ui.span(ui.output_text("rows"))),
                               ui.tags.li(ui.tags.b("주요 작업유형"), ui.span(ui.output_text("worktypes")))),
                    class_="billx-panel",
                ),
                ui.div(ui.div("전월 대비 비교", class_="billx-panel-title"),
                       ui.output_ui("mom_chart"), class_="billx-panel"),
                ui.div(ui.div(f"{ALIAS_YEAR}년 추이", class_="billx-panel-title"),
                       ui.output_ui("year_chart"), class_="billx-panel"),
                class_="billx-col",
            ),
            class_="billx-grid",
        ),
        ui.div(
            ui.div(
                ui.span(ui.output_text("issue_info"), class_="billx-issue"),
                ui.div(ui.download_button("btn_export_pdf", "PDF 저장", class_="btn btn-primary"),
                       ui.download_button("btn_export_csv", "CSV 내보내기", class_="btn btn-outline-primary"),
                       class_="billx-actions"),
                class_="billx-footer-inner",
            ),
            class_="billx-footer",
        ),
    )

# ===================== SERVER =====================

def report_server(input, output, session):
    # 1) 원본 로드 → 2) 2018 라벨 뷰로 변환 → 3) 2018만 사용
    df_all = _alias_year_view(report_df)
    df_view = df_all[df_all[COL_TS].dt.year == ALIAS_YEAR].copy()

    df_view["date"] = df_view[COL_TS].dt.date
    df_view["ym"]   = df_view[COL_TS].dt.strftime("%Y-%m")

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

    monthly_df_all = monthly_agg(df_view)

    @reactive.calc
    def month_key():
        return input.rep_month() or DEFAULT_MONTH

    @reactive.calc
    def df_month():
        ym = month_key()
        y, m = _ym_to_year_month(ym)
        return df_view[(df_view[COL_TS].dt.year == y) & (df_view[COL_TS].dt.month == m)].copy()

    @output
    @render.text
    def amt_due():
        mdf = df_month()
        if mdf.empty or NUM_COLS_COST not in mdf: return "— 원"
        return f"{int(mdf[NUM_COLS_COST].sum()):,}원"

    @output
    @render.text
    def kpi_kwh():
        mdf = df_month()
        if mdf.empty or NUM_COLS_KWH not in mdf: return "—"
        return f"{mdf[NUM_COLS_KWH].sum():,.2f}"

    @output
    @render.text
    def kpi_cost():
        mdf = df_month()
        if mdf.empty or NUM_COLS_COST not in mdf: return "—"
        return f"{int(mdf[NUM_COLS_COST].sum()):,}"

    @output
    @render.text
    def kpi_unit():
        mdf = df_month()
        if mdf.empty or (NUM_COLS_COST not in mdf) or (NUM_COLS_KWH not in mdf): return "—"
        kwh = mdf[NUM_COLS_KWH].sum(); cost = mdf[NUM_COLS_COST].sum()
        return f"{(cost / kwh):,.1f}" if kwh > 0 else "—"

    @output
    @render.text
    def pf_lg():
        mdf = df_month()
        if mdf.empty or "지상역률(%)" not in mdf: return "—"
        return f"{mdf['지상역률(%)'].mean():.1f}"

    @output
    @render.text
    def pf_ld():
        mdf = df_month()
        if mdf.empty or "진상역률(%)" not in mdf: return "—"
        return f"{mdf['진상역률(%)'].mean():.1f}"

    @output
    @render.text
    def q_lg():
        mdf = df_month(); col = "지상무효전력량(kVarh)"
        if mdf.empty or col not in mdf: return "—"
        return f"{mdf[col].mean():,.3f}"

    @output
    @render.text
    def q_ld():
        mdf = df_month(); col = "진상무효전력량(kVarh)"
        if mdf.empty or col not in mdf: return "—"
        return f"{mdf[col].mean():,.3f}"

    @output
    @render.text
    def kpi_co2():
        mdf = df_month(); col = "탄소배출량(tCO2)"
        if mdf.empty or col not in mdf: return "—"
        return f"{mdf[col].sum():,.3f}"

    @output
    @render.text
    def period():
        mdf = df_month()
        if mdf.empty: return "—"
        start = mdf[COL_TS].min().date(); end = mdf[COL_TS].max().date()
        return f"{start} ~ {end}"

    @output
    @render.text
    def rows():
        return f"{len(df_month()):,}"

    @output
    @render.text
    def worktypes():
        mdf = df_month()
        if mdf.empty or "작업유형" not in mdf: return "—"
        def _ko(name: str) -> str:
            key = str(name).strip().lower().replace(" ", "_")
            MAP = {"light_load": "경부하", "medium_load": "중간부하", "maximum_load": "최대부하"}
            return MAP.get(key, str(name))
        vc = mdf["작업유형"].value_counts().head(3)
        parts = [f"{_ko(k)}({v:,})" for k, v in vc.items()]
        return " · ".join(parts) if parts else "—"

    @output
    @render.text
    def issue_info():
        today = dt.date.today().strftime("%Y-%m-%d")
        return f"발행일 {today} · 공장 전력 데이터(표시연도 {ALIAS_YEAR}) 기반 자동 생성"

    def _prev_key(ym: str) -> str|None:
        try:
            y, m = _ym_to_year_month(ym)
            py, pm = (y, m-1) if m > 1 else (y-1, 12)
            return f"{py:04d}-{pm:02d}"
        except Exception:
            return None

    def _mom_fmt(cur: float|None, prev: float|None):
        if cur is None or prev is None or prev == 0: return "—", "na"
        delta = (cur - prev) / prev * 100.0
        sign = "+" if delta >= 0 else ""
        cls = "pos" if delta > 0 else ("neg" if delta < 0 else "na")
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
            prv_df = df_view[(df_view[COL_TS].dt.year==y)&(df_view[COL_TS].dt.month==m)]
            prv = float(prv_df[NUM_COLS_KWH].sum()) if not prv_df.empty and NUM_COLS_KWH in prv_df else None
        txt, cls = _mom_fmt(cur, prv); return ui.span(txt, class_=cls)

    @output
    @render.ui
    def mom_cost():
        ym = month_key(); prev = _prev_key(ym)
        cur_df = df_month()
        cur = float(cur_df[NUM_COLS_COST].sum()) if not cur_df.empty and NUM_COLS_COST in cur_df else None
        prv = None
        if prev:
            y, m = _ym_to_year_month(prev)
            prv_df = df_view[(df_view[COL_TS].dt.year==y)&(df_view[COL_TS].dt.month==m)]
            prv = float(prv_df[NUM_COLS_COST].sum()) if not prv_df.empty and NUM_COLS_COST in prv_df else None
        txt, cls = _mom_fmt(cur, prv); return ui.span(txt, class_=cls)

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
            prv_df = df_view[(df_view[COL_TS].dt.year==y)&(df_view[COL_TS].dt.month==m)]
            if not prv_df.empty and NUM_COLS_KWH in prv_df and NUM_COLS_COST in prv_df:
                prv_kwh  = float(prv_df[NUM_COLS_KWH].sum())
                prv_cost = float(prv_df[NUM_COLS_COST].sum())
                prv = (prv_cost/prv_kwh) if prv_kwh>0 else None
        txt, cls = _mom_fmt(cur, prv); return ui.span(txt, class_=cls)

    WEEK_LABELS = ["일","월","화","수","목","금","토"]

    def _first_meta_from_str(ym: str):
        y, m = _ym_to_year_month(ym)
        first = dt.date(y, m, 1)
        first_weekday, ndays = calendar.monthrange(y, m)
        offset = (first_weekday + 1) % 7
        return first, ndays, offset

    @output
    @render.ui
    def month_calendar():
        ym = month_key()
        first, ndays, offset = _first_meta_from_str(ym)
        metric = input.cal_metric() if hasattr(input, "cal_metric") else "cost"
        mdf = df_month()
        if not mdf.empty:
            g = mdf.groupby(mdf[COL_TS].dt.date)
            daily_cost = g[NUM_COLS_COST].sum()
            daily_kwh  = g[NUM_COLS_KWH].sum() if NUM_COLS_KWH in mdf else None
        else:
            daily_cost, daily_kwh = pd.Series(dtype=float), None
        s_disp = daily_cost if metric == "cost" else (daily_kwh if daily_kwh is not None else pd.Series(dtype=float))
        try:
            thresh = s_disp.quantile(0.9) if len(s_disp) else None
        except Exception:
            thresh = None

        header = ui.tags.div({"class": "cal-weekdays"}, *[ui.tags.div(x) for x in WEEK_LABELS])
        cells = []
        for _ in range(offset):
            cells.append(ui.tags.div({"class": "cal-cell empty"}))

        weeks = calendar.monthcalendar(first.year, first.month)
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
            tip_cost = daily_cost.get(date_obj, None)
            tip_kwh  = daily_kwh.get(date_obj, None) if daily_kwh is not None else None
            tip_unit = (tip_cost / tip_kwh) if (tip_cost is not None and tip_kwh not in (None, 0)) else None
            tooltip = f"{date_obj} · 요금 {tip_cost:,.0f}원" if isinstance(tip_cost, (int,float)) else f"{date_obj}"
            if isinstance(tip_kwh, (int,float)):  tooltip += f" · 사용 {tip_kwh:,.0f} kWh"
            if isinstance(tip_unit, (int,float)): tooltip += f" · 단가 {tip_unit:,.1f} 원/kWh"
            cells.append(ui.tags.div({"class": " ".join(classes), "title": tooltip},
                                     ui.tags.div(str(d), {"class": "date"}),
                                     ui.tags.div(txt, {"class": "cost"})))
        week_pills = []
        for i, wk in enumerate(weeks, start=1):
            day_objs = [dt.date(first.year, first.month, dd) for dd in wk if dd != 0]
            tot = float(sum([s_disp.get(d, 0.0) for d in day_objs])) if len(day_objs) else 0.0
            label = "원" if metric == "cost" else " kWh"
            week_pills.append(ui.tags.span(f"W{i}: {tot:,.0f}{label}", class_="pill"))
        grid = ui.tags.div({"class": "cal-grid"}, *cells)
        footer = ui.tags.div({"class": "cal-week-summary"}, *week_pills)
        return ui.tags.div({"class": "billx-cal"}, header, grid, footer)

    @output
    @render.ui
    def mom_chart():
        return mom_bar_chart(monthly_df_all, month_key())

    @output
    @render.ui
    def year_chart():
        return yearly_trend_chart(monthly_df_all, month_key())

    @output
    @render.text
    def kpi_co2_intensity():
        mdf = df_month()
        if mdf.empty or ("탄소배출량(tCO2)" not in mdf) or (NUM_COLS_KWH not in mdf): return "—"
        co2_t  = float(mdf["탄소배출량(tCO2)"].sum())
        kwh    = float(mdf[NUM_COLS_KWH].sum())
        if kwh <= 0: return "—"
        val = (co2_t * 1000.0) / kwh
        return f"{val:,.2f}"

    @output
    @render.text
    def kpi_co2_daily():
        mdf = df_month()
        if mdf.empty or "탄소배출량(tCO2)" not in mdf: return "—"
        days = mdf[COL_TS].dt.date.nunique()
        if days == 0: return "—"
        kg = float(mdf["탄소배출량(tCO2)"].sum() * 1000.0)
        return f"{(kg / days):,.1f}"

    # ===================== 다운로드 =====================

    @output
    @render.download(
        filename=lambda: f"전기요금청구서_{month_key()}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    def btn_export_docx():
        ym = month_key(); y, m = _ym_to_year_month(ym)
        mdf = df_month()
        if mdf.empty: raise ValueError(f"{ym} 데이터가 없습니다.")
        prev_start = (pd.to_datetime(f"{ym}-01") - pd.offsets.MonthEnd(1)).replace(day=1)
        prev_end = prev_start + pd.offsets.MonthEnd(0)
        prev_df = df_view[(df_view[COL_TS] >= prev_start) & (df_view[COL_TS] <= prev_end)]
        context, img1, img2 = _build_context_and_charts(ym, mdf, prev_df, monthly_df_all)
        doc = DocxTemplate(str(TEMPLATE_PATH))
        context["graph1"] = InlineImage(doc, img1, width=Mm(100), height=Mm(55))
        context["graph2"] = InlineImage(doc, img2, width=Mm(100), height=Mm(55))
        doc.render(context)
        word_path = tempfile.NamedTemporaryFile(suffix=".docx", delete=False).name
        doc.save(word_path)
        with open(word_path, "rb") as f:
            data = f.read()
        Path(word_path).unlink(missing_ok=True); Path(img1).unlink(missing_ok=True); Path(img2).unlink(missing_ok=True)
        return io.BytesIO(data)

    @output
    @render.download(
        filename=lambda: f"electricity_data_{month_key()}.csv",
        media_type="text/csv",
    )
    def btn_export_csv():
        mdf = df_month().copy()
        for col in ("date", "ym"):
            if col in mdf.columns: mdf.drop(columns=[col], inplace=True)
        # cp949 우선, 실패 시 UTF-8 BOM
        try:
            csv_bytes = mdf.to_csv(index=False).encode("cp949")
        except Exception:
            csv_bytes = mdf.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        return io.BytesIO(csv_bytes)

    @output
    @render.download(
        filename=lambda: f"전기요금청구서_{month_key()}.pdf",
        media_type="application/pdf",
    )
    def btn_export_pdf():
        if not _DOCXTPL_OK:
            raise RuntimeError("docxtpl 라이브러리가 설치되지 않았습니다.")
        ym = month_key(); y, m = _ym_to_year_month(ym)
        mdf = df_month()
        if mdf.empty: raise ValueError(f"{ym} 데이터가 없습니다.")
        prev_start = (pd.to_datetime(f"{ym}-01") - pd.offsets.MonthEnd(1)).replace(day=1)
        prev_end = prev_start + pd.offsets.MonthEnd(0)
        prev_df = df_view[(df_view[COL_TS] >= prev_start) & (df_view[COL_TS] <= prev_end)]
        context, img1, img2 = _build_context_and_charts(ym, mdf, prev_df, monthly_df_all)
        doc = DocxTemplate(str(TEMPLATE_PATH))
        context["graph1"] = InlineImage(doc, img1, width=Mm(90), height=Mm(55))
        context["graph2"] = InlineImage(doc, img2, width=Mm(90), height=Mm(50))
        doc.render(context)
        word_path = tempfile.NamedTemporaryFile(suffix=".docx", delete=False).name
        doc.save(word_path)
        data = convert_to_pdf_without_libreoffice(word_path, context, img1, img2)
        if data is None:
            data = _try_reportlab_pdf(context, img1, img2)
            if data is None:
                raise RuntimeError("PDF 생성에 실패했습니다. 'DOCX 저장' 버튼을 사용해 주세요.")
        Path(word_path).unlink(missing_ok=True); Path(img1).unlink(missing_ok=True); Path(img2).unlink(missing_ok=True)
        return io.BytesIO(data)
