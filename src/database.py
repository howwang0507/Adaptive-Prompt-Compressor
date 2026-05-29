import sqlite3
import pandas as pd
import datetime
import os


class ExperimentDB:
    """
    Handles persistent storage of experiment logs and agent states using SQLite.
    Ensures robust data management beyond simple CSV files.
    """

    def __init__(self, db_path="results/experiments.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Log table for every request
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    mode TEXT,
                    category TEXT,
                    arm INTEGER,
                    reward REAL,
                    token_saved REAL,
                    success_rate INTEGER,
                    semantic_score REAL,
                    latency REAL
                )
            """)
            # Agent state table for persistence (storing A and b matrices as blobs)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_state (
                    arm INTEGER PRIMARY KEY,
                    matrix_A BLOB,
                    vector_b BLOB
                )
            """)
            conn.commit()

    def log_trial(self, data: dict):
        """
        Logs a single trial result to the database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.DataFrame([data])
                df["timestamp"] = datetime.datetime.now().isoformat()
                df.to_sql("logs", conn, if_exists="append", index=False)
        except Exception as e:
            print(f"⚠️ Database Log Error: {e}")

    def get_summary_by_category(self):
        """
        Performs complex SQL analysis on the logs.
        """
        query = """
            SELECT category, 
                   AVG(reward) as avg_reward, 
                   AVG(token_saved) as avg_saving,
                   AVG(success_rate) as success_rate
            FROM logs
            GROUP BY category
            HAVING COUNT(*) > 5
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
