import os
import psycopg2
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()


class DBService:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL")

        if not self.database_url:
            raise ValueError("DATABASE_URL is missing in .env")

    def get_connection(self):
        return psycopg2.connect(self.database_url)

    def run_sql_file(self, sql_file_path: str):
        path = Path(sql_file_path)

        if not path.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_file_path}")

        sql = path.read_text()

        conn = self.get_connection()

        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)

            print(f"Executed SQL file: {sql_file_path}")

        finally:
            conn.close()