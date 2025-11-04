# modules/page_predict.py
from __future__ import annotations

import pandas as pd
from datetime import datetime
from shiny import ui, render, reactive
from shinywidgets import output_widget, render_plotly
import plotly.graph_objects as go

from utils.ui_components import kpi
from viz.plot_placeholders import hourly_prediction
from shared import df as reactive_db_df  # 최신 스냅샷(오래→최신 정렬 가정)

# ===== 설정 =====
STREAM_TICK_SEC = 3.0   # 초 단위: 1초마다 한 줄씩 소비
WINDOW_POINTS   = 32    # 최근 30개 포인트만 그래프에 유지


# ========================
# UI
# ========================
def predict_ui():
    return ui.page_fluid(
        ui.tags.link(rel="stylesheet", href="predict.css"),

        # ===== 헤더 리본 =====
        ui.div(
            # 좌: 타이틀
            ui.div(
                ui.h4("실시간 전력 예측", class_="pred-title"),
                ui.span("Streaming 기반 모니터링", class_="pred-sub"),
                class_="pred-titlebox",
            ),
            # 중: 측정일시 칩
            ui.div(
                ui.div(
                    ui.span("측정일시", class_="pred-time-label"),
                    ui.span(ui.output_text("toolbar_time"), class_="pred-time-value"),
                    class_="pred-chip pred-timebox",
                ),
                class_="pred-center",
            ),
            # 우: 상태칩 + 컨트롤
            ui.div(
                ui.output_ui("stream_notice"),
                ui.input_action_button("btn_start", "시작", class_="btn btn-primary pred-btn"),
                ui.input_action_button("btn_stop",  "멈춤", class_="btn btn-outline pred-btn"),
                ui.input_action_button("btn_reset", "리셋", class_="btn btn-outline pred-btn"),
                class_="pred-actions",
            ),
            class_="pred-ribbon",
        ),

        # ===== KPI =====
        ui.div(
            ui.div("예측 지표", class_="pred-panel-title"),
            ui.div(
                kpi("실시간 예측 사용량", "— kWh"),
                kpi("실시간 예측 요금",   "— 원"),
                kpi("누적 예측 요금",     "— 원"),
                kpi("작업 유형",          ui.output_text("worktype_text")),
                class_="kpi-row",
            ),
            class_="pred-panel",
        ),

        # ===== 예시 차트 스택(자리용) =====
        ui.div(
            ui.div(
                ui.div("시간대별 요금 예측", class_="pred-panel-title"),
                ui.div(hourly_prediction(), class_="pred-chart"),
                class_="pred-panel",
            ),
            ui.div(
                ui.div("예측 누적 사용량 비교", class_="pred-panel-title"),
                ui.div(hourly_prediction(), class_="pred-chart"),
                class_="pred-panel",
            ),
            class_="pred-stack",
        ),

        # ===== 실시간 Plotly: 최근 30개 포인트 =====
        ui.div(
            ui.div("최근 30개 이벤트 (측정일시당 y=1)", class_="pred-panel-title"),
            output_widget("ts_plot"),
            class_="pred-panel",
        ),
    )


# ========================
# Server
# ========================
def predict_server(input, output, session):
    # ---------- 상태 ----------
    running        = reactive.Value(False)          # 재생 여부
    cursor_idx     = reactive.Value(0)              # 다음 소비 인덱스
    source_df      = reactive.Value(pd.DataFrame()) # START 시점 스냅샷(오래→최신)
    latest_ts      = reactive.Value(None)           # 표시용 측정일시
    worktype_state = reactive.Value("—")            # 표시용 작업유형
    status_msg     = reactive.Value("대기 중")
    status_kind    = reactive.Value("info")         # info/warn/success

    # ⭐ NEW: 리셋 시 그래프 위젯을 새로 만들기 위한 시드
    plot_seed      = reactive.Value(0)

    # ---------- 스냅샷 준비 ----------
    def _prepare_snapshot(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["id", "측정일시", "작업유형"])
        snap = df.copy()

        # 정렬: id 우선, 없으면 측정일시
        if "id" in snap.columns:
            snap = snap.sort_values("id")
        elif "측정일시" in snap.columns:
            tmp = pd.to_datetime(snap["측정일시"], errors="coerce")
            snap = snap.loc[tmp.argsort(kind="mergesort")]
        snap = snap.reset_index(drop=True)

        # 필수 컬럼 보정
        for col in ["측정일시", "작업유형"]:
            if col not in snap.columns:
                snap[col] = pd.NA

        cols = ["측정일시", "작업유형"]
        if "id" in snap.columns:
            cols = ["id"] + cols
        return snap[cols]

    # ---------- Plotly Figure 생성 ----------
    def make_event_widget(title: str) -> go.FigureWidget:
        fig = go.FigureWidget(
            data=[go.Scatter(x=[], y=[], mode="lines+markers", name="events")],
            layout=go.Layout(
                template="simple_white",
                xaxis=dict(title="측정일시", tickangle=0),
                yaxis=dict(title="count", range=[0, 1.2], fixedrange=True),
                hovermode="x unified",
                margin=dict(t=40, r=20, b=40, l=50),
                title=title,
            ),
        )
        return fig

    # ⭐ NEW: plot_seed를 의존시켜, seed가 바뀌면 완전히 새 위젯 생성
    @output
    @render_plotly
    def ts_plot():
        _ = plot_seed()  # 의존성
        return make_event_widget("Events over time (최근 30개)")

    # ---------- 버튼 ----------
    @reactive.effect
    @reactive.event(input.btn_start)
    def _start():
        # 이미 스냅샷이 있으면 그대로 이어서 진행(커서/그래프 유지)
        if source_df().empty:
            try:
                snap = _prepare_snapshot(reactive_db_df())
            except Exception:
                running.set(False)
                status_msg.set("DB 스냅샷 읽기 실패")
                status_kind.set("warn")
                return

            source_df.set(snap)
            cursor_idx.set(0)

            if snap.empty:
                running.set(False)
                status_msg.set("스트리밍할 데이터 없음")
                status_kind.set("warn")
                return

        # 재생만 ON (그래프/커서 보존)
        running.set(True)
        status_msg.set("스트리밍 진행중")
        status_kind.set("info")

    @reactive.effect
    @reactive.event(input.btn_stop)
    def _stop():
        # 일시정지: 데이터/커서/그래프 모두 보존
        running.set(False)
        status_msg.set("일시정지됨")
        status_kind.set("info")

    @reactive.effect
    @reactive.event(input.btn_reset)
    def _reset():
        # 완전 초기화
        running.set(False)
        cursor_idx.set(0)
        latest_ts.set(None)
        worktype_state.set("—")
        status_msg.set("리셋됨 — 대기 중")
        status_kind.set("info")
        source_df.set(pd.DataFrame())  # 다음 시작 때 스냅샷 새로 읽게

        # ⭐ NEW: 그래프 위젯 자체를 재생성(빈 그래프 보장)
        plot_seed.set(plot_seed() + 1)

    # ---------- 포인트 1개 추가 + 최근 30개 유지 ----------
    def _append_point_keep_window(fw: go.FigureWidget, t: datetime, v: float = 1.0):
        if len(fw.data) == 0:
            fw.add_scatter(x=[], y=[], mode="lines+markers", name="events")

        x = list(fw.data[0].x or [])
        y = list(fw.data[0].y or [])

        x.append(t)
        y.append(v)

        if len(x) > WINDOW_POINTS:
            x = x[-WINDOW_POINTS:]
            y = y[-WINDOW_POINTS:]

        # 데이터 갱신
        fw.data[0].x = x
        fw.data[0].y = y

        # X축을 최근 구간으로 고정
        try:
            xmin = x[0]
            xmax = x[-1]
            fw.update_xaxes(range=[xmin, xmax])
        except Exception:
            pass

    # ---------- 틱 루프: 초당 1행 ----------
    @reactive.effect
    def _tick():
        reactive.invalidate_later(STREAM_TICK_SEC)  # 초 단위!

        if not running():
            return

        with reactive.isolate():
            snap = source_df()
            i = cursor_idx()

            if snap.empty or i >= len(snap):
                running.set(False)
                status_msg.set("스트리밍 완료")
                status_kind.set("success")
                return

            row = snap.iloc[i]
            ts_raw = row.get("측정일시")
            wt     = str(row.get("작업유형", "—")) or "—"

            # KPI 업데이트용
            ts_parsed = pd.to_datetime(ts_raw, errors="coerce")
            latest_ts.set(
                ts_parsed.to_pydatetime() if pd.notna(ts_parsed) and hasattr(ts_parsed, "to_pydatetime")
                else (ts_parsed if pd.notna(ts_parsed) else ts_raw)
            )
            worktype_state.set(wt)

            # 그래프에 점 1개 추가 (최근 30개 유지)
            ts_for_plot = ts_parsed if pd.notna(ts_parsed) else datetime.now()
            try:
                # ❗ seed가 바뀌면 widget 객체도 새로워지므로 매 틱마다 현재 widget 참조
                fw = ts_plot.widget
                with fw.batch_animate():
                    _append_point_keep_window(fw, ts_for_plot, 1.0)
            except Exception:
                pass  # 다음 틱에서 자연 복구

            # 다음 인덱스로
            cursor_idx.set(i + 1)

    # ---------- 출력 ----------
    @output
    @render.ui
    def stream_notice():
        kind = (status_kind() or "info").lower()
        text = status_msg() or "대기 중"
        cls = {
            "info": "pred-status--info",
            "warn": "pred-status--warn",
            "success": "pred-status--success",
        }.get(kind, "pred-status--info")
        return ui.div(
            ui.span("상태", class_="pred-time-label"),
            ui.span(text, class_="pred-time-value"),
            class_=f"pred-chip pred-statusbox {cls}",
        )

    @output
    @render.text
    def toolbar_time():
        ts = latest_ts()
        if hasattr(ts, "strftime"):
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        return str(ts) if ts else "—"

    @output
    @render.text
    def worktype_text():
        return worktype_state() or "—"
