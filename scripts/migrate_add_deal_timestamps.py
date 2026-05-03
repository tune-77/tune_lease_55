import sqlite3
from contextlib import closing

from data_cases import DB_PATH


def main() -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        for col in ("registration_date", "estimate_sent_date", "customer_response_date", "final_result_date"):
            try:
                conn.execute(f"ALTER TABLE past_cases ADD COLUMN {col} TEXT")
                print(f"added: {col}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    print(f"skip(existing): {col}")
                else:
                    raise
        conn.commit()


if __name__ == "__main__":
    main()
