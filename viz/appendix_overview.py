# =====================================================================
# viz/appendix_overview.py  (Tab: 개요)
# =====================================================================
from __future__ import annotations
import pandas as pd
from shiny import ui


def render_data_schema():
    schema = pd.DataFrame({
        "컬럼명": [
            "id", "측정일시", "전력사용량(kWh)", "지상무효전력량(kVarh)",
            "진상무효전력량(kVarh)", "탄소배출량(tCO2)", "지상역률(%)",
            "진상역률(%)", "작업유형", "전기요금(원)"
        ],
        "타입": [
            "int", "datetime", "float", "float", "float",
            "float", "float", "float", "object", "float"
        ],
        "설명": [
            "고유 식별자",
            "측정 시각 (15분 간격)",
            "전력 사용량",
            "무효 전력량(지상)",
            "무효 전력량(진상)",
            "탄소 배출량",
            "역률(지상)",
            "역률(진상)",
            "부하 유형",
            "예측 대상 값"
        ],
    })
    html = schema.to_html(classes="table table-sm table-hover", index=False, border=0)
    return ui.HTML(f'<div style="max-height:500px; overflow-y:auto;">{html}</div>')


def render_data_head(df: pd.DataFrame, n: int = 10):
    d = df.head(n).copy()
    html = d.to_html(classes="table table-sm table-striped table-bordered", index=False, border=0)
    return ui.HTML(f'<div style="overflow-x:auto; overflow-y:auto;">{html}</div>')