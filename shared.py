# shared.py
import sqlite3
from shiny import reactive
from pathlib import Path
import pandas as pd

app_dir = Path(__file__).parent
data_dir = app_dir / "data"
# streaming_df = pd.read_csv(data_dir / "test.csv")
report_df = pd.read_csv(data_dir / "train.csv")


# ===== 설정 =====
POLL_MS    = 1000       # DB 폴링 주기(ms)
TABLE      = "test"     # 테이블명
READ_LIMIT = 5000       # 최근 N행만 읽기

# ===== 경로 =====
APP_DIR  = Path(__file__).resolve().parent
DATA_DIR = (APP_DIR / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE  = (DATA_DIR / "./db/data.sqlite").resolve()

# (선택) 보고용 CSV (없어도 무방)
REPORT_CSV = DATA_DIR / "train.csv"
try:
    report_df: pd.DataFrame = pd.read_csv(REPORT_CSV)
except Exception:
    report_df = pd.DataFrame()

# ===== SQLite 연결 =====
def _open_readonly() -> sqlite3.Connection:
    # Windows에서도 안전한 URI
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
    # 버전 호환: 수동 타이머 폴링
    reactive.invalidate_later(POLL_MS)

    if last_modified() is None:
        return pd.DataFrame(columns=["id", "측정일시", "작업유형", "측정일시_dt"])

    try:
        q = f'SELECT "id","측정일시","작업유형" FROM "{TABLE}" ORDER BY "id" DESC LIMIT ?'
        tbl = pd.read_sql(q, con, params=[READ_LIMIT])
    except Exception:
        return pd.DataFrame(columns=["id", "측정일시", "작업유형", "측정일시_dt"])

    if tbl.empty:
        tbl["측정일시_dt"] = pd.NaT
        return tbl

    # 오래된 → 최신
    tbl = tbl.iloc[::-1].reset_index(drop=True)
    # 파생 datetime(표시/집계용). 원본 TEXT는 보존
    tbl["측정일시_dt"] = pd.to_datetime(tbl["측정일시"], errors="coerce")

    return tbl

# ===== 스트리밍 초기 스냅샷(정지형) =====
def _initial_snapshot() -> pd.DataFrame:
    try:
        snap = pd.read_sql(
            f'SELECT "id","측정일시","작업유형" FROM "{TABLE}" ORDER BY "id" ASC',
            con,
        )
        return snap
    except Exception:
        return pd.DataFrame(columns=["id", "측정일시", "작업유형"])

streaming_df: pd.DataFrame = _initial_snapshot()

# (선택) 연결 리프레시
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