# =====================================================================
# viz/appendix_modeling.py  (Tab: 모델링)
# =====================================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import plotly.express as px
from plotly.io import to_html
from shiny import ui


def _ph(text: str = "여기에 표/그래프가 표시됩니다.", h: int = 260):
    return ui.div(
        text,
        class_="placeholder d-flex align-items-center justify-content-center small-muted",
        style=f"height:{h}px; font-size: 0.98rem;"
    )


def render_leaderboard():
    return _ph("모델 리더보드 (RMSE/MAE/R²/Latency)", 260)


def _output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "output"


def _best_params_path() -> Path:
    return _output_dir() / "best_model_params.csv"


def _load_best_params() -> pd.DataFrame | None:
    path = _best_params_path()
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if df.empty:
        return df

    df = df.copy()
    df["model"] = df["model"].fillna("-")
    df["mae"] = df["mae"].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "-")

    def make_pairs(raw: str) -> Iterable[Tuple[str, str]]:
        if pd.isna(raw):
            return []
        try:
            parsed: Dict[str, object] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [("raw", str(raw))]

        def fmt(val: object) -> str:
            if isinstance(val, float):
                return f"{val:.6g}"
            return str(val)

        return [(k, fmt(parsed[k])) for k in sorted(parsed.keys())]

    df["param_pairs"] = df["params"].apply(make_pairs)
    return df[["model", "mae", "param_pairs"]]


def _render_param_table(df: pd.DataFrame) -> ui.HTML:
    rows = []
    for _, row in df.iterrows():
        params_list = row["param_pairs"]
        if not params_list:
            params_html = "<div class='text-muted'>-</div>"
        else:
            items = "".join(
                f"<li><code>{name}</code>: <span class='text-muted'>{value}</span></li>"
                for name, value in params_list
            )
            params_html = f"<ul class='list-unstyled mb-0 small'>{items}</ul>"

        rows.append(
            f"<tr>"
            f"<td class='fw-semibold'>{row['model']}</td>"
            f"<td>{row['mae']}</td>"
            f"<td>{params_html}</td>"
            f"</tr>"
        )

    table_html = f"""
    <div class="table-responsive">
      <table class="table table-sm table-hover align-middle mb-0" style="font-size: 0.92rem;">
        <thead class="table-light">
          <tr>
            <th style="width: 16%">Model</th>
            <th style="width: 10%">Holdout&nbsp;MAE</th>
            <th>Best Parameters</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """
    return ui.HTML(table_html)


def render_model_params():
    df = _load_best_params()
    if df is None:
        return ui.div(
            "best_model_params.csv 파일을 찾을 수 없습니다.",
            class_="alert alert-warning mb-0 small"
        )
    if df.empty:
        return ui.div("베스트 파라미터 기록이 비어 있습니다.", class_="alert alert-info mb-0 small")
    return _render_param_table(df)


CURVE_FILES = {
    "LGBM (RAW)": "lgb_raw_learning_curve.csv",
    "XGBoost": "xgb_learning_curve.csv",
    "HGBR": "hgbr_learning_curve.csv",
    "LGBM (LOG)": "lgb_log_learning_curve.csv",
}


def _load_learning_curves() -> Dict[str, pd.DataFrame]:
    train_frames, valid_frames = [], []
    curve_dir = _output_dir()

    for label, filename in CURVE_FILES.items():
        path = curve_dir / filename
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if df.empty or "iteration" not in df.columns:
            continue

        train_col = next((c for c in df.columns if c.lower().startswith("train")), None)
        valid_col = next((c for c in df.columns if c.lower().startswith("valid")), None)

        if train_col is not None:
            train_values = pd.to_numeric(df[train_col], errors="coerce")
            train_frames.append(
                pd.DataFrame({
                    "iteration": df["iteration"],
                    "value": train_values,
                    "model": label
                })
            )

        if valid_col is not None:
            valid_values = pd.to_numeric(df[valid_col], errors="coerce")
            valid_frames.append(
                pd.DataFrame({
                    "iteration": df["iteration"],
                    "value": valid_values,
                    "model": label
                })
            )

    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame(columns=["iteration", "value", "model"])
    valid_df = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame(columns=["iteration", "value", "model"])

    return {"train": train_df, "valid": valid_df}


def _make_curve_html(data: pd.DataFrame, title: str) -> ui.HTML:
    if data.empty:
        return ui.div(
            f"{title} 데이터가 없습니다.",
            class_="alert alert-info mb-0 small"
        )

    data = data.dropna(subset=["value"]).sort_values(["model", "iteration"])
    fig = px.line(
        data,
        x="iteration",
        y="value",
        color="model",
        markers=False,
        title=title,
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        xaxis_title="Iteration",
        yaxis_title="MAE",
        hovermode="x unified",
        template="plotly_white",
    )
    fig.update_traces(mode="lines")

    html = to_html(fig, include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False})
    return ui.HTML(html)


def render_train_curve():
    curves = _load_learning_curves()
    return _make_curve_html(curves["train"], "Train Learning Curve")


def render_val_curve():
    curves = _load_learning_curves()
    return _make_curve_html(curves["valid"], "Validation Learning Curve")
