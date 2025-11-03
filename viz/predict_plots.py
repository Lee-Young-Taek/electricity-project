# viz/stream_plot.py
from __future__ import annotations

from datetime import datetime
import plotly.graph_objects as go


def make_event_widget(
    *,
    title: str = "Events over time (최근 30개)",
    y_title: str = "count",
) -> go.FigureWidget:
    """빈 FigureWidget 생성 (autorange 기본)"""
    fig = go.FigureWidget(
        data=[go.Scatter(x=[], y=[], mode="lines+markers", name="events")],
        layout=go.Layout(
            template="simple_white",
            xaxis=dict(title="측정일시", tickangle=0, autorange=True),
            yaxis=dict(title=y_title, range=[0, 1.2], fixedrange=True),
            hovermode="x unified",
            margin=dict(t=40, r=20, b=40, l=50),
            title=title,
        ),
    )
    return fig


def clear_event_widget(
    fw: go.FigureWidget,
    *,
    title: str = "Events over time (최근 30개)",
):
    """그래프 완전 초기화 (데이터/축/오토레인지)"""
    if fw is None:
        return
    fw.data = (go.Scatter(x=[], y=[], mode="lines+markers", name="events"),)
    fw.update_layout(title=title, xaxis=dict(autorange=True, range=None))


def append_point_keep_window(
    fw: go.FigureWidget,
    *,
    t: datetime,
    v: float = 1.0,
    window_points: int = 30,
):
    """점 1개 추가 + 최근 window_points개만 유지, X축은 최근 구간으로 고정"""
    if fw is None:
        return

    if len(fw.data) == 0:
        fw.add_scatter(x=[], y=[], mode="lines+markers", name="events")

    x = list(fw.data[0].x or [])
    y = list(fw.data[0].y or [])

    x.append(t)
    y.append(v)

    if window_points is not None and window_points > 0 and len(x) > window_points:
        x = x[-window_points:]
        y = y[-window_points:]

    # 데이터 갱신
    fw.data[0].x = x
    fw.data[0].y = y

    # X축 범위를 최신 구간으로 고정 (포인트 1개면 autorange 유지)
    if len(x) >= 2:
        fw.update_xaxes(autorange=False, range=[x[0], x[-1]])
    else:
        fw.update_xaxes(autorange=True, range=None)
