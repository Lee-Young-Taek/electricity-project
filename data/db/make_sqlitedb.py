# csv_to_sqlite_power.py
import sqlite3
import pandas as pd
from pathlib import Path

# =========================
# 경로/테이블 설정
# =========================
CSV_PATH = "./data/specific_columns_output.csv"   # ⬅️ CSV 경로
DB_PATH  = "./data/db/data.sqlite"                # ⬅️ SQLite 파일 경로
TABLE    = "submission"                           # ⬅️ 테이블명

# =========================
# 스키마/인덱스 정의
# =========================
COLUMNS = [
    ("id", "INTEGER"),
    ("측정일시", "TEXT"),
    ("전력사용량(kWh)", "REAL"),
    ("작업유형", "TEXT"),
    ("전기요금(원)", "REAL"),
]
INDEX_COLUMNS = ["측정일시", "작업유형"]

# =========================
# 출력 폴더 생성
# =========================
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# =========================
# CSV 읽기 (인코딩 자동 판별)
# =========================
try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# 헤더 공백 정리
df.rename(columns=lambda c: c.strip(), inplace=True)

# =========================
# 컬럼 리네임/값 매핑
# - oof_kwh → 전력사용량(kWh)
# - 작업유형: Light_Load/Medium_Load/Maximum_Load → 경부하/중간부하/최대부하
# =========================
df.rename(columns={"oof_kwh": "전력사용량(kWh)"}, inplace=True)

worktype_map = {
    "Light_Load": "경부하",
    "Medium_Load": "중간부하",
    "Maximum_Load": "최대부하",
}
if "작업유형" in df.columns:
    df["작업유형"] = (
        df["작업유형"]
        .astype(str)
        .str.strip()
        .replace(worktype_map)
    )

# =========================
# 필수 컬럼 확인
# =========================
expected_cols = [c[0] for c in COLUMNS]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

# =========================
# 타입 정리
# =========================
df["측정일시"] = df["측정일시"].astype(str)
df["전력사용량(kWh)"] = pd.to_numeric(df["전력사용량(kWh)"], errors="coerce")
df["전기요금(원)"] = pd.to_numeric(df["전기요금(원)"], errors="coerce")
df["id"] = pd.to_numeric(df["id"], errors="coerce")

# 스키마 순서로 정렬
df = df[expected_cols]

# =========================
# DB 쓰기
# =========================
create_sql_cols = [f'"{name}" {typ}' for name, typ in COLUMNS]
create_table_sql = f'''
CREATE TABLE "{TABLE}" (
    {", ".join(create_sql_cols)}
)
'''

with sqlite3.connect(DB_PATH, timeout=30) as con:
    con.execute("PRAGMA busy_timeout=30000;")
    con.execute("PRAGMA journal_mode=WAL;")

    # 테이블 재생성
    con.execute(f'DROP TABLE IF EXISTS "{TABLE}"')
    con.execute(create_table_sql)

    # 대량 INSERT
    df.to_sql(TABLE, con, if_exists="append", index=False, chunksize=100_000, method="multi")

    # 인덱스 생성
    for col in INDEX_COLUMNS:
        con.execute(f'CREATE INDEX IF NOT EXISTS "idx_{TABLE}_{col}" ON "{TABLE}"("{col}");')

print(f"✅ 완료: {DB_PATH} / 테이블 '{TABLE}' 로 적재")
print(f"   - 총 행수: {len(df)}")
print(f"   - 컬럼: {', '.join(expected_cols)}")
