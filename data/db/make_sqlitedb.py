
# convert_csv_to_sqlite_preserve_korean.py
import sqlite3
import pandas as pd
from pathlib import Path

CSV_PATH = "./data/test.csv"       # ⬅️ 네 CSV 경로
DB_PATH  = "./data/db/data.sqlite"      # ⬅️ 만들 SQLite 파일(.sqlite)
TABLE    = "test"                    # ⬅️ 테이블명(원하는 이름으로)

# 출력 폴더 확보
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# 1) CSV 읽기 (BOM 이슈 대비해서 encoding 옵션은 상황 따라 조정)
df = pd.read_csv(CSV_PATH, encoding="utf-8")

# 2) 컬럼 존재 확인 (필요시 에러 메시지)
required = ["id", "측정일시", "작업유형"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

# 3) '측정일시'를 TEXT로 저장하기 위해 명시적으로 문자열화 (가공/파싱 X)
df["측정일시"] = df["측정일시"].astype(str)
# 작업유형도 문자열로 고정(권장)
df["작업유형"] = df["작업유형"].astype(str)

# 4) DB에 쓰기 (테이블 교체)
with sqlite3.connect(DB_PATH, timeout=30) as con:
    # 잠금 대기 설정(락 충돌 줄이기)
    con.execute("PRAGMA busy_timeout=30000;")

    # 스키마를 TEXT로 명확히 고정 (id는 정수/문자 선택 가능: 여기선 정수)
    con.execute(f'''
        CREATE TABLE IF NOT EXISTS "{TABLE}" (
            "id"        INTEGER,
            "측정일시"  TEXT,
            "작업유형"  TEXT
        )
    ''')

    # 기존 테이블을 완전히 교체하고 싶으면 DROP 후 새로 생성
    con.execute(f'DROP TABLE IF EXISTS "{TABLE}"')
    con.execute(f'''
        CREATE TABLE "{TABLE}" (
            "id"        INTEGER,
            "측정일시"  TEXT,
            "작업유형"  TEXT
        )
    ''')

    # pandas → SQLite (스키마를 우리가 만들었으니 그대로 append)
    df[["id", "측정일시", "작업유형"]].to_sql(
        TABLE, con, if_exists="append", index=False, chunksize=100_000, method="multi"
    )

    # 조회 최적화를 위한 인덱스(선택)
    con.execute(f'CREATE INDEX IF NOT EXISTS "idx_{TABLE}_측정일시" ON "{TABLE}"("측정일시");')
    con.execute(f'CREATE INDEX IF NOT EXISTS "idx_{TABLE}_작업유형" ON "{TABLE}"("작업유형");')

print(f"✅ 완료: {DB_PATH} / 테이블 '{TABLE}' (컬럼명 그대로, 측정일시=TEXT)")
