import os
from databricks import sql

host = os.environ["DATABRICKS_HOST"].replace("https://", "")
http_path = os.environ["DATABRICKS_HTTP_PATH"]
token = os.environ["DATABRICKS_TOKEN"]

def run_sql_file(path: str):
    with open(path, "r") as f:
        query = f.read()
    with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)

def main():
    base = os.path.dirname(os.path.dirname(__file__))
    run_sql_file(os.path.join(base, "sql", "bronze.sql"))
    run_sql_file(os.path.join(base, "sql", "silver.sql"))
    run_sql_file(os.path.join(base, "sql", "gold.sql"))
    print("Bronze → Silver → Gold refresh completed.")

if __name__ == "__main__":
    main()
