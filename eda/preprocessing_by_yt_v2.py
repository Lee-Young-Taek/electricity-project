# -*- coding: utf-8 -*-
"""
Knowledge Distillation for Power Bill Forecast
- Teacher: rich features (all available in train)
- Student: limited features (only those available in test)
- OOF + blended target for student
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_random_state


# -----------------------------
# Config
# -----------------------------
SEED = 42
ALPHA = 0.5  # y_blend = ALPHA * y_true + (1-ALPHA) * y_teacher_oof
N_SPLITS = 5  # time-series expanding OOF folds
BASE_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "data" / "processed" / "yt"
TRAIN_PATH = DATA_DIR / "v2_train_full.csv"
TEST_PATH  = DATA_DIR / "v2_test_processed.csv"
OUT_DIR = DATA_DIR
PRED_PATH = OUT_DIR / "v2_test_pred_distilled.csv"


# -----------------------------
# Utils
# -----------------------------
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 보통 CSV 로드 시 공백/이상 문자 등이 섞이는 걸 방지
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def build_feature_sets(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Teacher: train에서 타깃/날짜/id 등을 제외한 모든 열 사용
    Student: test에도 존재하는 공통 열만 사용(타깃/날짜/id 제외)
    """
    drop_cols = {"id", "전기요금(원)", "측정일시"}  # 확실히 제거할 것들

    # teacher features: train의 모든 열 - drop_cols
    teacher_features = [c for c in train.columns if c not in drop_cols]

    # student features: 교집합(= test에 존재) - drop_cols
    common = set(train.columns).intersection(set(test.columns))
    student_features = [c for c in sorted(common) if c not in drop_cols]

    return teacher_features, student_features

def split_time_series_indices(dt_series: pd.Series, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window 방식의 간단한 시계열 OOF 분할.
    dt_series: datetime 시리즈 (정렬 전 가능)
    반환: (train_idx, valid_idx) 리스트
    """
    # 시간 기준 정렬 인덱스
    order = np.argsort(dt_series.values.astype("datetime64[ns]"))
    n = len(order)
    # 컷 포인트를 균등 비율로 잡되, 마지막 검증을 너무 작게 만들지 않기
    cut_fracs = np.linspace(0.6, 0.95, n_splits)  # 시작은 60% 학습, 마지막은 95% 시점까지
    folds = []

    prev_cut = int(0.5 * n)  # 첫 학습 최소 50% 보장
    for frac in cut_fracs:
        cut = int(frac * n)
        tr_idx_sorted = order[:prev_cut]
        val_idx_sorted = order[prev_cut:cut]
        # 유효성 체크: 비어있으면 스킵
        if len(tr_idx_sorted) > 0 and len(val_idx_sorted) > 0:
            folds.append((np.sort(tr_idx_sorted), np.sort(val_idx_sorted)))
        prev_cut = cut

    # 마지막 잔여 구간도 검증에 포함(선택): 필요 없으면 주석
    if prev_cut < n:
        tr_idx_sorted = order[:prev_cut]
        val_idx_sorted = order[prev_cut:n]
        if len(tr_idx_sorted) > 0 and len(val_idx_sorted) > 0:
            folds.append((np.sort(tr_idx_sorted), np.sort(val_idx_sorted)))

    return folds

def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    """
    문자열/범주형은 One-Hot, 나머지는 통과
    HistGB는 결측 허용, 스케일 불요
    """
    X = df[feature_cols]
    cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre

def make_model(random_state: int = SEED) -> HistGradientBoostingRegressor:
    # 기본 세팅: 빠르고 튼튼한 설정 (필요시 튜닝)
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_depth=None,
        max_iter=800,
        l2_regularization=0.0,
        min_samples_leaf=20,
        random_state=random_state,
    )


# -----------------------------
# Load
# -----------------------------
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

train = _clean_columns(train)
test  = _clean_columns(test)

# datetime 변환
if not np.issubdtype(train["측정일시"].dtype, np.datetime64):
    train["측정일시"] = pd.to_datetime(train["측정일시"])
if not np.issubdtype(test["측정일시"].dtype, np.datetime64):
    test["측정일시"] = pd.to_datetime(test["측정일시"])

y = train["전기요금(원)"].values

teacher_feats, student_feats = build_feature_sets(train, test)
print(f"[INFO] #teacher_feats={len(teacher_feats)}, #student_feats={len(student_feats)}")
print(f"[INFO] teacher_feats (head): {teacher_feats[:10]}")
print(f"[INFO] student_feats (head): {student_feats[:10]}")

# -----------------------------
# Teacher: OOF predictions
# -----------------------------
folds = split_time_series_indices(train["측정일시"], n_splits=N_SPLITS)
oof_pred = np.full(len(train), np.nan, dtype=float)

teacher_maes = []
for i, (tr_idx, val_idx) in enumerate(folds, start=1):
    tr_df = train.iloc[tr_idx]
    val_df = train.iloc[val_idx]

    pre = build_preprocessor(train, teacher_feats)
    model = make_model(SEED + i)

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(tr_df[teacher_feats], y[tr_idx])

    pred_val = pipe.predict(val_df[teacher_feats])
    oof_pred[val_idx] = pred_val

    mae = mean_absolute_error(y[val_idx], pred_val)
    teacher_maes.append(mae)
    print(f"[TEACHER] Fold {i} MAE: {mae:.4f}")

# 확인: OOF가 모두 채워졌는지
if np.isnan(oof_pred).any():
    # 남은 NaN은 마지막으로 학습한 파이프로 대체 예측(드문 케이스)
    print("[WARN] Some OOF entries are NaN; filling with last fold's model predictions.")
    last_pipe = pipe  # from last fold
    nan_idx = np.where(np.isnan(oof_pred))[0]
    oof_pred[nan_idx] = last_pipe.predict(train.iloc[nan_idx][teacher_feats])

print(f"[TEACHER] OOF MAE: {mean_absolute_error(y, oof_pred):.4f}")
print(f"[TEACHER] Fold MAEs: {teacher_maes}")

# -----------------------------
# Student: train on limited features with blended target
# -----------------------------
y_blend = ALPHA * y + (1.0 - ALPHA) * oof_pred

pre_student = build_preprocessor(train, student_feats)
student_model = make_model(SEED + 100)

student_pipe = Pipeline(steps=[("pre", pre_student), ("model", student_model)])
student_pipe.fit(train[student_feats], y_blend)

# 내부 검증(옵션): 11월 검증 등 특정 기간 평가를 원하면 여기서 필터링해 점검 가능
# 예시:
# valid_mask = (train["month"] == 11) if "month" in train.columns else np.zeros(len(train), dtype=bool)
# if valid_mask.any():
#     valid_mae = mean_absolute_error(train.loc[valid_mask, "전기요금(원)"],
#                                     student_pipe.predict(train.loc[valid_mask, student_feats]))
#     print(f"[STUDENT] Nov-only MAE (vs true y): {valid_mae:.4f}")

# -----------------------------
# Predict on test (student only)
# -----------------------------
test_pred = student_pipe.predict(test[student_feats])

# 저장
sub = pd.DataFrame({
    "id": test["id"] if "id" in test.columns else np.arange(len(test)),
    "측정일시": test["측정일시"],
    "작업유형": test["작업유형"] if "작업유형" in test.columns else None,
    "전기요금(원)_pred": test_pred
})
sub.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")
print(f"[DONE] Saved predictions to: {PRED_PATH}")

# -----------------------------
# Tips
# -----------------------------
# - ALPHA(0.3~0.7) 탐색, 모델 파라미터 튜닝으로 성능 향상 가능
# - student_feats는 test에 실제로 존재하는 열만 사용하므로 누수 안전
# - 필요 시 teacher를 더 강한 모델(예: LightGBM/XGBoost/CatBoost)로 교체 가능
# - 시계열 폴드 방식을 month 경계 기반으로 커스터마이징 해도 좋음
