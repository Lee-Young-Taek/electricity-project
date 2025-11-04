import sqlite3
from shiny import reactive
from pathlib import Path
import pandas as pd

# ===== 기본 경로 =====
APP_DIR  = Path(__file__).resolve().parent
DATA_DIR = (APP_DIR / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE  = (DATA_DIR / "./db/data.sqlite").resolve()

# ===== 보고용 CSV (옵션) =====
REPORT_CSV = DATA_DIR / "train.csv"
try:
    report_df: pd.DataFrame = pd.read_csv(REPORT_CSV)
except Exception:
    report_df = pd.DataFrame()

TEMPLATE_PATH = Path(DATA_DIR / "electricity_bill_template_.docx")

# ===== 설정 =====
POLL_MS    = 1000         # DB 폴링 주기(ms)
TABLE      = "submission" # ⬅️ 최신 스키마 테이블명 (id, 측정일시, 전력사용량(kWh), 작업유형, 전기요금(원))
READ_LIMIT = 5000         # 최근 N행만 읽기

# ===== SQLite 연결 =====
def _open_readonly() -> sqlite3.Connection:
    uri = f"file:{DB_FILE.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=10)
    con.execute("PRAGMA busy_timeout=30000;")
    return con

def _open_rw() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_FILE), check_same_thread=False, timeout=10)
    con.execute("PRAGMA busy_timeout=30000;")
    return con

try:
    con = _open_readonly() if DB_FILE.exists() else _open_rw()
except sqlite3.OperationalError:
    con = _open_rw()

# ===== 변경 감지(값싼 쿼리) =====
def last_modified():
    try:
        cur = con.execute(f'SELECT MAX("id") FROM "{TABLE}"')
        return cur.fetchone()[0]
    except Exception:
        return None

# ===== 리액티브 최신 스냅샷(df) =====
@reactive.calc
def df() -> pd.DataFrame:
    # 타이머 폴링
    reactive.invalidate_later(POLL_MS)

    if last_modified() is None:
        return pd.DataFrame(columns=[
            "id", "측정일시", "전력사용량(kWh)", "작업유형", "전기요금(원)", "측정일시_dt",
        ])

    try:
        q = (
            f'SELECT "id","측정일시","전력사용량(kWh)","작업유형","전기요금(원)" '
            f'FROM "{TABLE}" ORDER BY "id" DESC LIMIT ?'
        )
        tbl = pd.read_sql(q, con, params=[READ_LIMIT])
    except Exception:
        return pd.DataFrame(columns=[
            "id", "측정일시", "전력사용량(kWh)", "작업유형", "전기요금(원)", "측정일시_dt",
        ])

    if tbl.empty:
        tbl["측정일시_dt"] = pd.NaT
        return tbl

    # 오래된 → 최신으로 뒤집기
    tbl = tbl.iloc[::-1].reset_index(drop=True)

    # 타입 보정
    tbl["측정일시_dt"] = pd.to_datetime(tbl["측정일시"], errors="coerce")
    # 숫자형 안정 변환 (혹시 문자열/NaN 섞여도 안전)
    for col in ["전력사용량(kWh)", "전기요금(원)"]:
        tbl[col] = pd.to_numeric(tbl[col], errors="coerce")

    return tbl

# ===== 스트리밍 초기 스냅샷(정지형) =====
def _initial_snapshot() -> pd.DataFrame:
    try:
        snap = pd.read_sql(
            f'SELECT "id","측정일시","전력사용량(kWh)","작업유형","전기요금(원)" '
            f'FROM "{TABLE}" ORDER BY "id" ASC',
            con,
        )
        return snap
    except Exception:
        return pd.DataFrame(columns=[
            "id", "측정일시", "전력사용량(kWh)", "작업유형", "전기요금(원)",
        ])

streaming_df: pd.DataFrame = _initial_snapshot()

# ===== 연결 리프레시 =====
def refresh_connection(readonly_preferred: bool = True):
    global con
    try:
        con.close()
    except Exception:
        pass
    try:
        con = _open_readonly() if (readonly_preferred and DB_FILE.exists()) else _open_rw()
    except sqlite3.OperationalError:
        con = _open_rw()
