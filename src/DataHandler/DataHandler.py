'''
Handles the interactions with the database.
'''

import sqlite3
from pathlib import Path
import pandas as pd

DB  = "paysim.db"
DB_PATH = Path("src/DataBase") / DB
CHUNK   = 200_000

def return_df(query) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            data = pd.read_sql(query, conn)
        return data
    except Exception as e:
        raise sqlite3.Error(e)

def return_df_with_params(query, params) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            data = pd.read_sql(query, conn, params=params)
        return data
    except Exception as e:
        raise sqlite3.Error(e)

def execute_query_in_chunks(query, params, chunksize) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            for chunk in pd.read_sql(query, conn, params=params, chunksize=chunksize):
                yield chunk
    except Exception as e:
        raise sqlite3.Error(e)
