# -*- coding: utf-8 -*-
"""
Direct BILL forecasting with N-HiTS (DARTS, univariate) + model saving
- Target: 전기요금(원), 15분 시계열
- Split: 1~10월 학습 → 11월 다중스텝 예측(VAL) → 1~11월 재학습 → 12월 예측(제출)
- No tariff, no exogenous
- Saves:
  - models/nhits_bill_1_10.pth.tar, models/nhits_bill_scaler_1_10.pkl
  - models/nhits_bill_1_11.pth.tar, models/nhits_bill_scaler_1_11.pkl
  - november_validation_predictions.csv
  - december_predictions.csv
  - submission.csv
"""

import os, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# ---------------- paths / columns ----------------
def _first(paths):
    for p in paths:
        if os.path.exists(p): return p
    return paths[0]

TRAIN_PATH = _first(["train.csv","./data/train.csv","/mnt/data/train.csv"])
TEST_PATH  = _first(["test.csv","./data/test.csv","/mnt/data/test.csv"])

DT_COL   = "측정일시"
BILL_COL = "전기요금(원)"
FREQ = "15min"

# ---------------- model save dir ----------------
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)

# ---------------- import DARTS ----------------
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel

np.random.seed(42)

# ---------------- load & align ----------------
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

for df in (train, test):
    df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce")

train = train.sort_values(DT_COL).set_index(DT_COL)
test  = test.sort_values(DT_COL).set_index(DT_COL)

# 15분 정합 + 타깃 결측 보수 처리
train = train.asfreq(FREQ)
if train[BILL_COL].isna().any():
    train[BILL_COL] = train[BILL_COL].ffill().fillna(0.0)

year = train.index.max().year
nov_start = pd.Timestamp(year=year, month=11, day=1,  hour=0, minute=0)
nov_end   = pd.Timestamp(year=year, month=11, day=30, hour=23, minute=45)
nov_index = pd.date_range(nov_start, nov_end, freq=FREQ)

# 12월 구간은 test의 실제 인덱스 사용(유실 방지)
dec_index = test.index.sort_values()

# ---------------- helpers ----------------
def ts_from(idx, values):
    arr = np.asarray(values, dtype="float32")
    if arr.ndim == 1:
        return TimeSeries.from_times_and_values(idx, arr)
    return TimeSeries.from_times_and_values(idx, arr.reshape(-1,1))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(mean_squared_error(y_true[m], y_pred[m])))

def make_nhits(cpu=True):
    return NHiTSModel(
        input_chunk_length=96*7,     # 7일
        output_chunk_length=96,      # 1일
        n_epochs=150,                # CPU면 100~150 선
        random_state=42,
        dropout=0.1,
        batch_size=128,              # CPU: 64~128 권장
        pl_trainer_kwargs={
            "accelerator": "cpu" if cpu else "gpu",
            **({"devices":1, "precision":16} if not cpu else {}),
            "enable_progress_bar": True,
            "log_every_n_steps": 50,
        },
    )

# ---------------- series build ----------------
y_full = ts_from(train.index, train[BILL_COL].values)
scaler = Scaler()

# ---------------- 1) 1~10월 학습 → 11월 예측(VAL) ----------------
# slice(None, ts) 대신 split_before(ts) 사용
y_1_10, _ = y_full.split_before(nov_start)  # nov_start 이전까지 포함
y_1_10_s = scaler.fit_transform(y_1_10)

model_1_10 = make_nhits(cpu=not torch.cuda.is_available())
print("[INFO] Fit N-HiTS on 1~10 months")
model_1_10.fit(series=y_1_10_s, verbose=False)

# --- SAVE (1~10) ---
(model_1_10.save if hasattr(model_1_10, "save") else model_1_10.model.save)(
    str(MODEL_DIR / "nhits_bill_1_10.pth.tar")
)
joblib.dump(scaler, MODEL_DIR / "nhits_bill_scaler_1_10.pkl")

print("[INFO] Predict November (multi-step)")
yhat_nov_s = model_1_10.predict(n=len(nov_index), series=y_1_10_s, verbose=False)
yhat_nov   = scaler.inverse_transform(yhat_nov_s)
pred_nov   = pd.Series(yhat_nov.values().ravel(), index=nov_index)
true_nov   = train[BILL_COL].reindex(nov_index)

print("==== November VALID (Bill) ====", {
    "MAE": mean_absolute_error(true_nov, pred_nov),
    "RMSE": rmse(true_nov, pred_nov)
})

pd.DataFrame({
    "측정일시": nov_index,
    "true_bill": true_nov.values,
    "pred_bill": pred_nov.values
}).to_csv("november_validation_predictions.csv", index=False, encoding="utf-8-sig")
print("[saved] november_validation_predictions.csv")

# ---------------- 2) 1~11월 재학습 → 12월 예측(TEST) ----------------
# 11월 마지막 다음 스텝(=12월 시작) 시각
dec_start_ts = nov_end + pd.Timedelta(minutes=15)

# dec_start_ts가 train 범위를 벗어나면 전체(1~11월)를 그대로 사용
if hasattr(y_full, "end_time") and dec_start_ts > y_full.end_time():
    y_1_11 = y_full
else:
    y_1_11, _ = y_full.split_before(dec_start_ts)

y_1_11_s = scaler.fit_transform(y_1_11)

model_1_11 = make_nhits(cpu=not torch.cuda.is_available())
print("[INFO] Refit N-HiTS on 1~11 months")
model_1_11.fit(series=y_1_11_s, verbose=False)

# --- SAVE (1~11) ---
(model_1_11.save if hasattr(model_1_11, "save") else model_1_11.model.save)(
    str(MODEL_DIR / "nhits_bill_1_11.pth.tar")
)
joblib.dump(scaler, MODEL_DIR / "nhits_bill_scaler_1_11.pkl")

print("[INFO] Predict December (multi-step)")
yhat_dec_s = model_1_11.predict(n=len(dec_index), series=y_1_11_s, verbose=False)
yhat_dec   = scaler.inverse_transform(yhat_dec_s)
pred_dec   = pd.Series(yhat_dec.values().ravel(), index=dec_index)

# 저장
pd.DataFrame({
    "측정일시": dec_index,
    "pred_bill": pred_dec.values
}).to_csv("december_predictions.csv", index=False, encoding="utf-8-sig")
print("[saved] december_predictions.csv")

# ---------------- 3) submission ----------------
sub = (test.reset_index()
          .rename(columns={DT_COL:"측정일시"})
          .merge(pd.DataFrame({"측정일시": dec_index, "pred_bill": pred_dec.values})
                   .rename(columns={"pred_bill":"target"}),
                 on="측정일시", how="left"))[["id","target"]]
sub["target"] = sub["target"].fillna(0.0)
sub = sub.sort_values("id").reset_index(drop=True)
sub.to_csv("submission.csv", index=False, encoding="utf-8-sig")
print(f"[saved] submission.csv rows={len(sub)}")
print(sub.head())
