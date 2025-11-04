from __future__ import annotations

import pandas as pd
from datetime import datetime
from shiny import ui, render, reactive
from shinywidgets import output_widget, render_plotly

from utils.ui_components import kpi
from shared import df as reactive_db_df  # 최신 스냅샷(오래→최신 정렬 가정)
from viz.predict_plots import (
    make_dual_widget,
    clear_dual_widget,
    append_point_keep_window_dual,
)

# ===== 설정 =====
STREAM_TICK_SEC = 3.0   # 초 단위: 3초마다 한 줄씩 소비
WINDOW_POINTS   = 32    # 최근 32개 포인트만 그래프에 유지


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
                    ui.span(ui.output_text("pred_toolbar_time"), class_="pred-time-value"),
                    class_="pred-chip pred-timebox",
                ),
                class_="pred-center",
            ),
            # 우: 상태칩 + 컨트롤
            ui.div(
                ui.output_ui("pred_stream_notice"),
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
                kpi("실시간 예측 사용량", ui.output_text("pred_kpi_kwh")),
                kpi("실시간 예측 요금",   ui.output_text("pred_kpi_bill")),
                kpi("누적 예측 요금",     ui.output_text("pred_kpi_cum_bill")),
                kpi("작업 유형",          ui.output_text("pred_worktype_text")),
                class_="kpi-row",
            ),
            class_="pred-panel",
        ),

        # ===== 실시간 Plotly: 최근 N개 포인트 (이중축) =====
        ui.div(
            ui.div(f"전력사용량·전기요금 — 최근 {WINDOW_POINTS}개", class_="pred-panel-title"),
            ui.div(
                output_widget("pred_ts_plot"),
                style="width:100%;",
            ),
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

    # KPI 상태
    kwh_now        = reactive.Value(None)
    bill_now       = reactive.Value(None)
    bill_cum       = reactive.Value(0.0)

    # 🔸 하드 리셋용: plotly 위젯을 완전히 새로 마운트하기 위한 seed
    plot_seed      = reactive.Value(0)

    # ---------- 스냅샷 준비 ----------
    def _prepare_snapshot(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["id", "측정일시", "전력사용량(kWh)", "작업유형", "전기요금(원)"])
        snap = df.copy()

        # 정렬: id 우선, 없으면 측정일시
        if "id" in snap.columns:
            snap = snap.sort_values("id")
        elif "측정일시" in snap.columns:
            tmp = pd.to_datetime(snap["측정일시"], errors="coerce")
            snap = snap.loc[tmp.argsort(kind="mergesort")]
        snap = snap.reset_index(drop=True)

        # 필수 컬럼 보정
        for col in ["측정일시", "전력사용량(kWh)", "작업유형", "전기요금(원)"]:
            if col not in snap.columns:
                snap[col] = pd.NA

        cols = ["측정일시", "전력사용량(kWh)", "작업유형", "전기요금(원)"]
        if "id" in snap.columns:
            cols = ["id"] + cols
        return snap[cols]

    # ---------- Plotly Figure 생성 (dual) ----------
    @output
    @render_plotly
    def pred_ts_plot():
        _ = plot_seed()  # seed 의존 → 바뀌면 완전히 새 위젯 생성
        return make_dual_widget(title=f"전력사용량·전기요금 — 최근 {WINDOW_POINTS}개", height=520)

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
            bill_cum.set(0.0)

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
        kwh_now.set(None)
        bill_now.set(None)
        bill_cum.set(0.0)

        # 그래프 위젯 자체를 리마운트 (겹침 방지의 확실한 방법)
        plot_seed.set(plot_seed() + 1)

        # 혹시 남아있는 경우에도 안전망으로 클리어
        try:
            clear_dual_widget(pred_ts_plot.widget, title=f"전력사용량·전기요금 — 최근 {WINDOW_POINTS}개", autorange=True)
        except Exception:
            pass

    # ---------- 틱 루프 ----------
    @reactive.effect
    def _tick():
        reactive.invalidate_later(STREAM_TICK_SEC)

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
            kwh    = row.get("전력사용량(kWh)")
            bill   = row.get("전기요금(원)")
            wt     = str(row.get("작업유형", "—")) or "—"

            # KPI 업데이트용
            ts_parsed = pd.to_datetime(ts_raw, errors="coerce")
            latest_ts.set(
                ts_parsed.to_pydatetime() if pd.notna(ts_parsed) and hasattr(ts_parsed, "to_pydatetime")
                else (ts_parsed if pd.notna(ts_parsed) else ts_raw)
            )
            worktype_state.set(wt)

            try:
                kwh_val = float(kwh) if pd.notna(kwh) else None
            except Exception:
                kwh_val = None
            try:
                bill_val = float(bill) if pd.notna(bill) else None
            except Exception:
                bill_val = None

            kwh_now.set(kwh_val)
            bill_now.set(bill_val)
            if bill_val is not None:
                bill_cum.set(float(bill_cum()) + bill_val)

            # 그래프에 점 1개 추가 (최근 WINDOW_POINTS개 유지)
            ts_for_plot = ts_parsed if pd.notna(ts_parsed) else datetime.now()
            try:
                fw = pred_ts_plot.widget
                with fw.batch_animate():
                    append_point_keep_window_dual(
                        fw,
                        t=ts_for_plot,
                        y1=(kwh_val or 0.0),
                        y2=(bill_val or 0.0),
                        window_points=WINDOW_POINTS,
                    )
            except Exception:
                pass  # 다음 틱에서 자연 복구

            # 다음 인덱스로
            cursor_idx.set(i + 1)

    # ---------- 출력 ----------
    @output
    @render.ui
    def pred_stream_notice():
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
    def pred_toolbar_time():
        ts = latest_ts()
        if hasattr(ts, "strftime"):
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        return str(ts) if ts else "—"

    @output
    @render.text
    def pred_worktype_text():
        return worktype_state() or "—"

    @output
    @render.text
    def pred_kpi_kwh():
        v = kwh_now()
        return f"{v:,.4f} kWh" if v is not None else "— kWh"

    @output
    @render.text
    def pred_kpi_bill():
        v = bill_now()
        return f"{v:,.2f} 원" if v is not None else "— 원"

    @output
    @render.text
    def pred_kpi_cum_bill():
        v = bill_cum()
        return f"{v:,.0f} 원" if v is not None else "— 원"
