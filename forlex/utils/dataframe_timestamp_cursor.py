from datetime import datetime


class DataFrameTimeStampCursor:
    def __init__(self, df):
        self.rows = iter(df.itertuples(index=False))
        self.current = next(self.rows, None)
        self.next = next(self.rows, None)

    def get_latest(self, timestamp: datetime):
        while self.next is not None and self.next.timestamp <= timestamp:
            self.current = self.next
            self.next = next(self.rows, None)

        if self.current is None or self.current.timestamp > timestamp:
            return None

        return self.current
