# viz/report_plots.py
import json
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from shiny import ui
import pandas as pd


def _fig_to_div(fig: go.Figure, div_id: str) -> ui.Tag:
    payload = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    return ui.tags.div(
        ui.tags.div(id=div_id, style="height: 360px;"),
        ui.tags.script(
            f"""
            (function(){{
                var el = document.getElementById("{div_id}");
                if(!el) return;
                var data = {json.dumps(payload.get("data", []))};
                var layout = {json.dumps(payload.get("layout", {}))};
                Plotly.newPlot(el, data, layout, {{displayModeBar:false, responsive:true}});
            }})()
            """
        ),
    )


def mom_bar_chart(monthly_df_all: pd.DataFrame, selected_ym: str) -> ui.Tag:
    df = monthly_df_all.copy()
    if df.empty or "ym" not in df:
        return ui.div({"class": "placeholder"}, "데이터가 없습니다")

    # 전월 key
    try:
        y, m = map(int, selected_ym.split("-"))
        prev_y, prev_m = (y, m - 1) if m > 1 else (y - 1, 12)
        prev_key = f"{prev_y:04d}-{prev_m:02d}"
    except Exception:
        prev_key = None

    cur = df[df["ym"] == selected_ym]
    prv = df[df["ym"] == prev_key] if prev_key else df.iloc[0:0]

    cost_cur = float(cur["전기요금(원)"].iloc[0]) if not cur.empty else 0.0
    kwh_cur  = float(cur["전력사용량(kWh)"].iloc[0]) if not cur.empty else 0.0
    cost_prv = float(prv["전기요금(원)"].iloc[0]) if not prv.empty else 0.0
    kwh_prv  = float(prv["전력사용량(kWh)"].iloc[0]) if not prv.empty else 0.0

    # 색상: 요금=파랑 / 사용량=초록, 전월은 연하게
    C_COST_CUR  = "rgba(37,99,235,1.0)"
    C_COST_PREV = "rgba(37,99,235,0.35)"
    C_KWH_CUR   = "rgba(16,185,129,1.0)"
    C_KWH_PREV  = "rgba(16,185,129,0.35)"

    fig = go.Figure()

    # 전기요금(좌축)
    fig.add_bar(
        name=f"{prev_key or '전월 없음'} · 요금",
        x=["전기요금(원)"], y=[cost_prv],
        marker=dict(color=C_COST_PREV),
        width=0.4, offsetgroup="prev",
        hovertemplate="전월 %{x}<br>%{y:,.0f}원<extra></extra>",
    )
    fig.add_bar(
        name=f"{selected_ym} · 요금",
        x=["전기요금(원)"], y=[cost_cur],
        marker=dict(color=C_COST_CUR),
        width=0.4, offsetgroup="curr",
        hovertemplate="선택월 %{x}<br>%{y:,.0f}원<extra></extra>",
    )

    # 전력사용량(우축)
    fig.add_bar(
        name=f"{prev_key or '전월 없음'} · 사용량",
        x=["전력사용량(kWh)"], y=[kwh_prv],
        marker=dict(color=C_KWH_PREV),
        width=0.4, offsetgroup="prev", yaxis="y2",
        hovertemplate="전월 %{x}<br>%{y:,.0f} kWh<extra></extra>",
    )
    fig.add_bar(
        name=f"{selected_ym} · 사용량",
        x=["전력사용량(kWh)"], y=[kwh_cur],
        marker=dict(color=C_KWH_CUR),
        width=0.4, offsetgroup="curr", yaxis="y2",
        hovertemplate="선택월 %{x}<br>%{y:,.0f} kWh<extra></extra>",
    )

    fig.update_layout(
        barmode="group", bargap=0.28,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white", height=360,
        hovermode="x unified",
        xaxis=dict(title="지표"),
        yaxis=dict(
            title=dict(text="<b>요금(원)</b>", font=dict(color=C_COST_CUR)),
            tickfont=dict(color=C_COST_CUR),
            tickformat=",.0f",
            gridcolor="rgba(148,163,184,0.2)",
        ),
        yaxis2=dict(
            title=dict(text="<b>사용량(kWh)</b>", font=dict(color=C_KWH_CUR)),
            tickfont=dict(color=C_KWH_CUR),
            overlaying="y", side="right",
            tickformat=",.0f",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            itemclick="toggle", itemdoubleclick="toggleothers",
        ),
    )
    return _fig_to_div(fig, "mom_bar_chart_div")


def yearly_trend_chart(monthly_df_all: pd.DataFrame, selected_ym: str) -> ui.Tag:
    df = monthly_df_all.copy()
    if df.empty or "ym" not in df:
        return ui.div({"class": "placeholder"}, "데이터가 없습니다")

    # ⬇️ 여기만 수정: 선택한 연도의 접두어로 필터
    try:
        sel_year = int(str(selected_ym).split("-")[0])
    except Exception:
        sel_year = None

    if sel_year is not None:
        df = df[df["ym"].astype(str).str.startswith(f"{sel_year}-")].copy()
    else:
        df = df.iloc[0:0].copy()  # 연도 파싱 실패 시 빈 프레임

    if df.empty:
        return ui.div({"class": "placeholder"}, f"{sel_year}년 데이터가 없습니다")

    df["xdate"] = pd.to_datetime(df["ym"] + "-01")
    df = df.sort_values("xdate")

    x = df["xdate"]
    cost = df["전기요금(원)"].astype(float)
    kwh  = df["전력사용량(kWh)"].astype(float)

    # --- 네가 지정해둔 컬러 그대로 사용 ---
    COST = "rgba(37,99,235,1)"
    COST_FADE = "rgba(37,99,235,0.35)"
    KWH = "rgba(16,185,129,1)"
    KWH_FADE = "rgba(16,185,129,0.35)"

    fig = go.Figure()

    # 전체 라인(연하게)
    fig.add_scatter(
        x=x, y=cost, mode="lines+markers", name="전기요금(좌)",
        line=dict(color=COST_FADE, width=2),
        marker=dict(size=6, color=COST_FADE, symbol="circle"),
        hovertemplate="%{x|%b %Y}<br>요금: %{y:,}원<extra></extra>",
        legendgroup="cost",
    )
    fig.add_scatter(
        x=x, y=kwh, mode="lines+markers", name="전력사용량(우)", yaxis="y2",
        line=dict(color=KWH_FADE, width=2, dash="dot"),
        marker=dict(size=6, color=KWH_FADE, symbol="diamond"),
        hovertemplate="%{x|%b %Y}<br>사용량: %{y:,.0f} kWh<extra></extra>",
        legendgroup="kwh",
    )

    # 선택 월 강조 (그대로)
    try:
        sel_date = pd.to_datetime(selected_ym + "-01")
        sel = df.loc[df["xdate"] == sel_date]
        if not sel.empty:
            sc = float(sel["전기요금(원)"].iloc[0])
            sk = float(sel["전력사용량(kWh)"].iloc[0])

            fig.add_scatter(
                x=[sel_date], y=[sc], mode="markers+text", showlegend=False,
                marker=dict(size=13, color=COST, line=dict(width=2, color="white"), symbol="circle"),
                text=[f"{int(sc):,}원"], textposition="top center",
                hovertemplate="%{x|%b %Y}<br>요금(선택): %{y:,}원<extra></extra>",
            )
            fig.add_scatter(
                x=[sel_date], y=[sk], mode="markers+text", showlegend=False, yaxis="y2",
                marker=dict(size=13, color=KWH, line=dict(width=2, color="white"), symbol="diamond"),
                text=[f"{sk:,.0f} kWh"], textposition="bottom center",
                hovertemplate="%{x|%b %Y}<br>사용량(선택): %{y:,.0f} kWh<extra></extra>",
            )
            fig.add_vrect(
                x0=sel_date - pd.Timedelta(days=15),
                x1=sel_date + pd.Timedelta(days=15),
                fillcolor="rgba(37,99,235,0.08)", line_width=0, layer="below"
            )
    except Exception:
        pass

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            title="월",
            type="date",
            tickformat="%b %Y",
            dtick="M1",
            showgrid=True, gridcolor="rgba(148,163,184,0.2)",
        ),
        yaxis=dict(
            title=dict(text="<b>요금(원)</b>", font=dict(color=COST)),
            tickfont=dict(color=COST),
            tickformat=",.0f",
            gridcolor="rgba(148,163,184,0.2)",
        ),
        yaxis2=dict(
            title=dict(text="<b>사용량(kWh)</b>", font=dict(color=KWH)),
            tickfont=dict(color=KWH),
            overlaying="y", side="right",
            tickformat=",.0f",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            itemclick="toggle", itemdoubleclick="toggleothers"
        ),
        template="plotly_white",
        height=360,
        hovermode="x unified",
    )

    return _fig_to_div(fig, "year_trend_chart_div")

