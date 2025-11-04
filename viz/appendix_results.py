# =====================================================================
# viz/appendix_results.py  (Tab: 결과/검증)
# - render_metrics_table
# - render_residual_plot
# - render_shap_summary
# - render_shap_bar
# - render_deploy_checklist
# =====================================================================
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from shiny import ui
import plotly.graph_objects as go
from viz.appendix_common import apply_layout, _PALETTE


def _ph(text: str = "여기에 표/그래프가 표시됩니다.", h: int = 260):
    """統一 placeholder (톤앤매너 유지)"""
    return ui.div(
        text,
        class_="placeholder d-flex align-items-center justify-content-center small-muted",
        style=f"height:{h}px; font-size: 0.98rem;",
    )


# ---------------------------------------------------------------------
# 1) 평가 지표 표
# ---------------------------------------------------------------------
def render_metrics_table(
    metrics: Optional[pd.DataFrame] = None,
    order: Optional[list[str]] = None,
):
    """
    평가 지표 표를 렌더링.
    - metrics가 None이면 샘플 스켈레톤 테이블을 표시.
    - metrics 포맷 예시:
        pd.DataFrame({
            "Metric": ["RMSE","MAE","MAPE(%)","R²"],
            "Value":  [123.4, 98.7, 12.3, 0.87],
            "Note":   ["holdout","holdout","holdout","holdout"]
        })
    """
    if metrics is None or not isinstance(metrics, pd.DataFrame) or metrics.empty:
        sample = pd.DataFrame({
            "Metric": ["RMSE", "MAE", "MAPE(%)", "R²"],
            "Value":  ["—", "—", "—", "—"],
            "Note":   ["모델 학습 후 반영", "모델 학습 후 반영", "모델 학습 후 반영", "모델 학습 후 반영"],
        })
        html = sample.to_html(classes="table table-sm table-striped", index=False, border=0)
        return ui.HTML(
            f"""
            <div class="p-2">
              <div class="alert alert-warning mb-2">
                현재 표시할 지표가 없어요. 학습 후 반환된 지표 DataFrame을 <code>render_metrics_table(metrics=...)</code>로 넘기면 표가 채워집니다.
              </div>
              {html}
            </div>
            """
        )

    df = metrics.copy()
    if order:
        df = df.set_index("Metric").reindex(order).reset_index()

    # 수치 반올림
    def _fmt(v: Any):
        try:
            if isinstance(v, (int, np.integer)):
                return f"{int(v):,}"
            if isinstance(v, (float, np.floating)):
                return f"{v:,.4f}"
        except Exception:
            pass
        return v

    if "Value" in df.columns:
        df["Value"] = df["Value"].map(_fmt)

    html = df.to_html(classes="table table-sm table-striped", index=False, border=0)
    return ui.HTML(
        f"""
        <div class="p-2">
          {html}
          <div class="small-muted mt-2">※ 기본 스코어는 홀드아웃 기준을 권장합니다. (교차검증 평균/표준편차는 부가로 표시)</div>
        </div>
        """
    )


# ---------------------------------------------------------------------
# 2) 잔차/에러 분포
# ---------------------------------------------------------------------
def render_residual_plot(
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    title: str = "잔차(Actual - Pred) 분포 및 추이"
):
    """
    잔차 플롯 (상단: 시퀀스, 하단: 히스토그램).
    - y_true, y_pred가 없으면 플레이스홀더 안내.
    """
    if y_true is None or y_pred is None:
        return _ph("잔차 플롯은 학습/예측 후 표시됩니다. (y_true, y_pred 전달)", 300)

    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return _ph("잔차 플롯: 입력 길이가 0입니다.", 300)

    y_true, y_pred = y_true[:n], y_pred[:n]
    resid = y_true - y_pred
    x = np.arange(n)

    # 상단: 잔차 시퀀스
    fig1 = go.Figure()
    fig1.add_scatter(x=x, y=resid, mode="lines", name="Residual", line=dict(width=1.2, color=_PALETTE["primary"]))
    fig1.add_hline(y=0.0, line=dict(color=_PALETTE["muted"], width=1, dash="dot"))
    apply_layout(fig1, title=f"{title} — 시퀀스", height=320)
    fig1.update_xaxes(title_text="Index")
    fig1.update_yaxes(title_text="Residual")

    # 하단: 히스토그램
    fig2 = go.Figure()
    fig2.add_histogram(x=resid, nbinsx=50, marker_color=_PALETTE["accent"], name="Residual Hist")
    apply_layout(fig2, title=f"{title} — 분포", height=320)
    fig2.update_xaxes(title_text="Residual")
    fig2.update_yaxes(title_text="Count")

    html = (
        '<div class="billx-panel">'
        + fig1.to_html(include_plotlyjs='cdn', full_html=False)
        + '</div><div class="billx-panel">'
        + fig2.to_html(include_plotlyjs='cdn', full_html=False)
        + '</div>'
    )
    return ui.HTML(html)


# ---------------------------------------------------------------------
# 3) SHAP Summary
# ---------------------------------------------------------------------
def render_shap_summary(
    shap_values: Optional[np.ndarray] = None,
    feature_names: Optional[list[str]] = None,
    max_features: int = 20,
):
    """
    SHAP Summary(벌레떼 플롯 대체용 간이 시각화).
    - 실제 shap plot은 JS 연동/이미지 임베드가 필요하니, 여기서는
      '|mean(|SHAP|)|' 상위 n개 바차트로 대체. (톤 통일)
    - shap_values: (n_samples, n_features)
    """
    if shap_values is None or feature_names is None:
        return _ph("SHAP Summary는 학습 후 shap_values, feature_names를 전달하면 그려집니다.", 300)

    sv = np.asarray(shap_values, dtype=float)
    if sv.ndim != 2 or sv.shape[1] != len(feature_names):
        return _ph("SHAP Summary: shap_values shape가 feature_names와 맞지 않습니다.", 300)

    mean_abs = np.abs(sv).mean(axis=0)
    idx = np.argsort(-mean_abs)[:max_features]
    top_imp = mean_abs[idx]
    top_feat = [feature_names[i] for i in idx]

    fig = go.Figure()
    fig.add_bar(
        x=top_imp[::-1],
        y=top_feat[::-1],
        orientation="h",
        marker_color=_PALETTE["primary"],
        name="|mean(|SHAP|)|"
    )
    apply_layout(fig, title="SHAP Summary (Top Features by |mean(|SHAP|)|)", height=420)
    fig.update_xaxes(title_text="|mean(|SHAP|)|")
    fig.update_yaxes(title_text="Feature")

    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))


# ---------------------------------------------------------------------
# 4) SHAP Bar (특정 샘플/집단 평균의 feature 영향 Top-K)
# ---------------------------------------------------------------------
def render_shap_bar(
    contrib: Optional[Dict[str, float]] = None,
    top_k: int = 15,
    title: str = "상위 피처 영향 (SHAP Bar)"
):
    """
    단일 벡터 형태의 SHAP 기여도 dict를 받아 상위 K개 수평 바차트로 표현.
    - contrib 예시: {"oof_kwh_lag24h": 0.12, "how_resid_kwh": -0.08, ...}
    - 값의 절댓값 기준으로 정렬, 색은 양수/음수 구분.
    """
    if not contrib:
        return _ph("SHAP Bar: contrib dict를 전달하면 상위 기여도 바를 표시합니다.", 300)

    items = sorted(contrib.items(), key=lambda x: abs(float(x[1])), reverse=True)[:top_k]
    names = [k for k, _ in items][::-1]
    vals = np.array([float(v) for _, v in items][::-1])

    colors = np.where(vals >= 0, _PALETTE["accent"], _PALETTE["danger"]).tolist()

    fig = go.Figure()
    fig.add_bar(x=vals, y=names, orientation="h", marker_color=colors, name="SHAP contrib")
    apply_layout(fig, title=title, height=420)
    fig.update_xaxes(title_text="Contribution (±)")
    fig.update_yaxes(title_text="Feature")

    return ui.HTML(fig.to_html(include_plotlyjs='cdn', full_html=False))


# ---------------------------------------------------------------------
# 5) 배포/모니터링 체크리스트
# ---------------------------------------------------------------------
def render_deploy_checklist():
    html = """
    <div class="p-3" style="font-size: 0.98rem;">
      <div class="alert alert-primary">
        <b>배포/모니터링 체크리스트</b>
      </div>
      <ul class="mb-3">
        <li><b>피처 일관성</b>: 학습/추론 파이프라인 동일(결측 처리·스케일·라벨링·캘린더 기준연도)</li>
        <li><b>입력 검증</b>: 스키마/범위(이상치·음수·시간 역전)/00:00 롤오버 보정 여부</li>
        <li><b>드리프트 감시</b>: 데이터/타겟/에러(예: MAPE/MAE의 주간 이동평균), 경보 임계치</li>
        <li><b>재학습 정책</b>: 주기/트리거(성능 하락·분포 변화·설비 변경 등)와 모델 버저닝</li>
        <li><b>성능 추적</b>: Holdout/Online A/B, 예측·실측 대시보드(주말/공휴일 분리)</li>
        <li><b>로깅</b>: 입력/출력/특성량/지표/추론시간, 실패 재처리 전략</li>
        <li><b>보안/권한</b>: 환경변수, 자격증명, 민감 데이터 마스킹</li>
        <li><b>비상 플랜</b>: 장애 시 폴백(룰기반/평균), 롤백 절차</li>
      </ul>
      <div class="small-muted">※ 운영 모니터링 보드에서는 ‘주말/공휴일’과 ‘평일’을 분리해 추세를 비교하세요.</div>
    </div>
    """
    return ui.HTML(html)
