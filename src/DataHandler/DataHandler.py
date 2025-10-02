import sqlite3
from pathlib import Path
import pandas as pd

DB  = "paysim.db"
DB_PATH = Path("src\DataBase") / DB
CHUNK   = 200_000

def return_df(query) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            data = pd.read_sql(query, conn)
        return data
    except Exception as e:
        raise sqlite3.Error(e)
