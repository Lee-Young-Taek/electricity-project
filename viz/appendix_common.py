# =====================================================================
# viz/appendix_common.py  (NEW — shared palette & datetime utils)
# =====================================================================
from __future__ import annotations
import numpy as np, pandas as pd
import plotly.graph_objects as go

_PALETTE = {
    "primary": "#3B82F6",   # blue-500
    "accent":  "#10B981",   # emerald-500
    "warn":    "#F59E0B",   # amber-500
    "danger":  "#EF4444",   # red-500
    "muted":   "#6B7280",   # gray-500
    "line":    "#111827",   # near-black
}


def apply_layout(fig: go.Figure, title: str = "", height: int = 420):
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Noto Sans KR, Inter, Arial", size=12, color=_PALETTE["line"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def to_dt(s) -> pd.Series:
    if isinstance(s, (pd.DatetimeIndex, pd.Index)):
        s = pd.Series(s)
    else:
        s = pd.Series(s)
    return pd.to_datetime(s, errors="coerce")


def safe_replace_year(dt_like, year: int) -> pd.Series:
    s = to_dt(dt_like)
    out = []
    for ts in s:
        if pd.isna(ts):
            out.append(pd.NaT); continue
        m, d, h, mi, se = ts.month, ts.day, ts.hour, ts.minute, ts.second
        if m == 2 and d == 29:
            d = 28
        try:
            out.append(pd.Timestamp(year, m, d, h, mi, se))
        except Exception:
            last = pd.Timestamp(year, m, 1) + pd.offsets.MonthEnd(0)
            out.append(pd.Timestamp(year, m, last.day, h, mi, se))
    return pd.Series(pd.to_datetime(out))


def weekend_flag(dt_like) -> pd.Series:
    s = to_dt(dt_like)
    return (s.dt.dayofweek >= 5).astype(int)


def season_of_month(m: int) -> str:
    return {12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"}.get(m, "unknown")
