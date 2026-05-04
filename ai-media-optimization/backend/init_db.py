from backend.services.db_service import DBService


def main():
    db = DBService()
    db.run_sql_file("database/sql/001_init_tables.sql")


if __name__ == "__main__":
    main()