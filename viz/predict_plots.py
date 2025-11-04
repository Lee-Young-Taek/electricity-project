from __future__ import annotations

from datetime import datetime, timedelta
import plotly.graph_objects as go
from uuid import uuid4

# =========================
# Dual-axis Plotly helpers (final):
# - Start X axis as LINEAR + fully hidden → never shows 2000 default
# - On first data point: switch X axis to DATE + show (ticks/grid/line on)
# - Draw a vertical guide line at each point; keep only the latest N (window)
# =========================

def make_dual_widget(
    *,
    title: str = "전력사용량·전기요금 — 최근 30개",
    y1_title: str = "전력사용량(kWh)",
    y2_title: str = "전기요금(원)",
    height: int = 520,
) -> go.FigureWidget:
    fig = go.FigureWidget(
        data=[
            go.Scatter(x=[], y=[], mode="lines+markers", name=y1_title, yaxis="y"),
            go.Scatter(x=[], y=[], mode="lines+markers", name=y2_title, yaxis="y2"),
        ],
        layout=go.Layout(
            template="simple_white",
            xaxis=dict(
                title="측정일시",
                type="linear",         # start as linear (not date/category)
                visible=False,          # fully hidden until first point
                showticklabels=False,
                ticks="",
                showgrid=False,
                showline=False,
                zeroline=False,
                autorange=True,
                range=None,
            ),
            yaxis=dict(title=y1_title, autorange=True),
            yaxis2=dict(title=y2_title, overlaying="y", side="right", showgrid=False, autorange=True),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=70, r=80, b=50, l=70),
            title=title,
            height=height,
            uirevision=None,
            shapes=(),  # vertical guide lines live here
        ),
    )
    return fig


def clear_dual_widget(fw: go.FigureWidget, *, title: str = "전력사용량·전기요금 — 최근 30개"):
    """Hard reset: traces, axes, shapes, uirevision."""
    if fw is None:
        return
    fw.data = (
        go.Scatter(x=[], y=[], mode="lines+markers", name="전력사용량(kWh)", yaxis="y"),
        go.Scatter(x=[], y=[], mode="lines+markers", name="전기요금(원)", yaxis="y2"),
    )
    fw.update_layout(title=title)
    fw.update_xaxes(
        type="linear",
        visible=False,
        showticklabels=False,
        ticks="",
        showgrid=False,
        showline=False,
        zeroline=False,
        autorange=True,
        range=None,
    )
    fw.update_yaxes(autorange=True, range=None)
    if "yaxis2" in fw.layout:
        fw.layout["yaxis2"].update(autorange=True, range=None)
    # remove all shapes (vertical lines)
    fw.layout.shapes = ()
    fw.layout.uirevision = str(uuid4())


def _ensure_datetime(t: datetime) -> datetime:
    """Best-effort convert to Python datetime."""
    # pandas.Timestamp
    if hasattr(t, "to_pydatetime"):
        try:
            return t.to_pydatetime()
        except Exception:
            pass
    return t


def _sync_vlines(fw: go.FigureWidget, xs: list[datetime], window_points: int, color: str = "rgba(0,0,0,0.08)"):
    """Sync vertical guide lines with x values (paper-relative 0→1)."""
    keep = xs[-window_points:] if window_points and len(xs) > window_points else xs
    shapes = []
    for x in keep:
        xdt = _ensure_datetime(x)
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=xdt, x1=xdt,
                y0=0, y1=1,
                line=dict(width=1, color=color),
                layer="below",
            )
        )
    fw.layout.shapes = tuple(shapes)


def append_point_keep_window_dual(
    fw: go.FigureWidget,
    *,
    t: datetime,
    y1: float,
    y2: float,
    window_points: int = 30,
):
    """Append one point to both traces and keep only the last N points.
    Also draws a vertical guide line at each point.
    """
    if fw is None:
        return

    # trace guard
    if len(fw.data) < 1:
        fw.add_scatter(x=[], y=[], mode="lines+markers", name="전력사용량(kWh)", yaxis="y")
    if len(fw.data) < 2:
        fw.add_scatter(x=[], y=[], mode="lines+markers", name="전기요금(원)", yaxis="y2")

    x1 = list(fw.data[0].x or [])
    y1v = list(fw.data[0].y or [])
    x2 = list(fw.data[1].x or [])
    y2v = list(fw.data[1].y or [])

    t = _ensure_datetime(t)

    # sync x for both traces
    x1.append(t); x2.append(t)
    y1v.append(y1); y2v.append(y2)

    # windowing
    if window_points and len(x1) > window_points:
        x1 = x1[-window_points:]; y1v = y1v[-window_points:]
        x2 = x2[-window_points:]; y2v = y2v[-window_points:]

    # commit
    fw.data[0].x = x1; fw.data[0].y = y1v
    fw.data[1].x = x2; fw.data[1].y = y2v

    # vertical guide lines synced with latest window
    _sync_vlines(fw, x1, window_points)

    n = len(x1)
    # first point → flip to DATE axis & show with full ticks/grid/line
    if n == 1:
        # small padded window to stabilize autorange
        lo = t - timedelta(minutes=1)
        hi = t + timedelta(minutes=1)
        fw.update_xaxes(
            type="date",
            visible=True,
            showticklabels=True,
            ticks="outside",
            showgrid=True,
            showline=True,
            zeroline=False,
            autorange=True,
            range=[lo, hi],
        )
    elif n >= 3:
        fw.update_xaxes(autorange=False, range=[x1[0], x1[-1]])
    else:
        fw.update_xaxes(autorange=True, range=None)
