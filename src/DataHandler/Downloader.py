import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv('.env')

DATASET = "ealaxi/paysim1"
OUT_DB  = "paysim.db"
DB_PATH = Path("src\DataBase") / OUT_DB
TABLE   = "paysim"
CHUNK   = 200_000

def main():
    if not DB_PATH.exists():
        
        import sqlite3
        import zipfile
        import pandas as pd
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print('Downloading dataset...')
        api = KaggleApi()
        api.authenticate()

        dataset = "ealaxi/paysim1"
        api.dataset_download_files(dataset, path="dl", unzip=True)
        
        print('Dataset downloaded.')

        csv_path = "dl/PS_20174392719_1491204439457_log.csv"  # exemplo de nome
        df = pd.read_csv(csv_path)
        os.remove(csv_path)

        # Ensure the database directory exists
        os.makedirs(DB_PATH.parent, exist_ok=True)

        con = sqlite3.connect(DB_PATH)
        df.to_sql("paysim", con, if_exists="replace", index=False,chunksize=CHUNK)
        con.close()
        print('Dataset saved to database.')
    else:
        print('Dataset already downloaded.')

if __name__ == "__main__":
    main()
