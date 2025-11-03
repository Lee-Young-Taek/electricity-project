# utils/time_streamer.py
import pandas as pd

class RealTimeStreamer:
    """DataFrame을 n행씩 순차 제공하는 간단 스트리머"""
    def __init__(self, source: pd.DataFrame):
        self.set_source(source)

    def set_source(self, source: pd.DataFrame):
        self._src = source.copy() if isinstance(source, pd.DataFrame) else pd.DataFrame()
        self._i = 0

    def reset_stream(self):
        self._i = 0

    def get_next_batch(self, n: int = 1) -> pd.DataFrame | None:
        if self._src is None or self._src.empty or self._i >= len(self._src):
            return None
        j = min(self._i + n, len(self._src))
        out = self._src.iloc[self._i:j].copy()
        self._i = j
        return out
