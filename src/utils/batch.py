from sqlmodel import Session


class DatabaseBatchCommit:
    """Commits to the database the entries added in batchs of max_size. Best
    way to use this object is with a context manager"""

    def __init__(self, db_engine, max_size=300):
        self.db_engine = db_engine
        self.max_size = max_size
        self.batch = []

    def _commit_batch(self):
        with Session(self.db_engine) as session:
            for element in self.batch:
                session.add(element)
            session.commit()
        self.batch = []

    def add(self, element):
        self.batch.append(element)
        if len(self.batch) == self.max_size:
            self._commit_batch()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._commit_batch()
