# =====================================================================
# viz/appendix_modeling.py  (Tab: 모델링 — placeholders)
# =====================================================================
from __future__ import annotations
from shiny import ui


def _ph(text: str = "여기에 표/그래프가 표시됩니다.", h: int = 260):
    return ui.div(text, class_="placeholder d-flex align-items-center justify-content-center small-muted", style=f"height:{h}px; font-size: 0.98rem;")


def render_leaderboard():
    return _ph("모델 리더보드 (RMSE/MAE/R²/Latency)", 260)


def render_model_params():
    return _ph("최종 모델 하이퍼파라미터", 220)


def render_train_curve():
    return _ph("학습 곡선(Train)", 300)


def render_val_curve():
    return _ph("검증 곡선(Validation)", 300)
