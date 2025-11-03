# csv_to_sqlite_power.py
import sqlite3
import pandas as pd
from pathlib import Path

# =========================
# 1) 경로/테이블 설정
# =========================
CSV_PATH = "./data/submission_electricity.csv"                 # ⬅️ CSV 경로
DB_PATH  = "./data/db/data.sqlite"    # ⬅️ SQLite 파일 경로
TABLE    = "submission"                     # ⬅️ 테이블명

# =========================
# 2) 타깃 스키마 정의
#    - CSV의 "예측된 전력사용량(kWh)" → DB의 "전력사용량(kWh)" 로 저장
# =========================
COLUMNS = [
    ("id", "INTEGER"),
    ("측정일시", "TEXT"),
    ("전력사용량(kWh)", "REAL"),
    ("전기요금(원)", "REAL"),
]

# 기본키(없으면 None): id가 100% 유일하지 않다면 None 유지 권장
PRIMARY_KEY = None  # 예: PRIMARY_KEY = "id"

# =========================
# 3) 인덱스 (선택)
# =========================
INDEX_COLUMNS = [
    "측정일시",
]

# =========================
# 유틸
# =========================
def build_create_table_sql(table: str, columns: list[tuple[str, str]], primary_key: str | None):
    if not columns:
        raise ValueError("COLUMNS 가 비어 있습니다.")
    defs = []
    for name, typ in columns:
        defs.append(f'"{name}" {typ}')
    if primary_key:
        col_names = [c[0] for c in columns]
        if primary_key not in col_names:
            raise ValueError(f"PRIMARY_KEY '{primary_key}' 가 COLUMNS에 없습니다.")
        defs.append(f'PRIMARY KEY ("{primary_key}")')
    return f'''
        CREATE TABLE "{table}" (
            {", ".join(defs)}
        )
    '''

def read_csv_safely(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")

def main():
    # 출력 폴더
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    # CSV 읽기
    df = read_csv_safely(CSV_PATH)

    # 1) 헤더 공백 정리(혹시 모를 스페이스/탭 제거)
    df.rename(columns=lambda c: c.strip(), inplace=True)

    # 2) 요청한 컬럼 리네임: "예측된 전력사용량(kWh)" → "전력사용량(kWh)"
    rename_map = {
        "예측된 전력사용량(kWh)": "전력사용량(kWh)"
    }
    df.rename(columns=rename_map, inplace=True)

    # 3) 필수 컬럼 존재 확인
    expected = [c[0] for c in COLUMNS]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

    # 4) 타입 정리
    # - 측정일시는 문자열(TEXT)로 고정
    df["측정일시"] = df["측정일시"].astype(str)
    # - 수치형은 안전하게 숫자로 변환 (문자 섞여있으면 NaN → 그대로 insert됨)
    for col in ["전력사용량(kWh)", "전기요금(원)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # - id도 가능하면 정수로
    if "id" in df.columns:
        # float로 읽힌 경우를 대비해 NaN 처리 후 정수 변환 시도
        df["id"] = pd.to_numeric(df["id"], errors="coerce")

    # 5) 스키마 순서로 컬럼 정렬
    df = df[expected]

    # 6) DB 쓰기
    with sqlite3.connect(DB_PATH, timeout=30) as con:
        con.execute("PRAGMA busy_timeout=30000;")
        con.execute("PRAGMA journal_mode=WAL;")

        # 완전 교체
        con.execute(f'DROP TABLE IF EXISTS "{TABLE}"')
        con.execute(build_create_table_sql(TABLE, COLUMNS, PRIMARY_KEY))

        # INSERT
        df.to_sql(TABLE, con, if_exists="append", index=False, chunksize=100_000, method="multi")

        # 인덱스
        for col in INDEX_COLUMNS:
            con.execute(f'CREATE INDEX IF NOT EXISTS "idx_{TABLE}_{col}" ON "{TABLE}"("{col}");')

    print(f"✅ 완료: {DB_PATH} / 테이블 '{TABLE}' 로 적재")
    print(f"   - 총 행수: {len(df)}")
    print(f"   - 컬럼: {', '.join(expected)}")

if __name__ == "__main__":
    main()
