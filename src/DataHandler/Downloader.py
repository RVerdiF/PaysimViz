import os
import json
import streamlit as st
from pathlib import Path

from src.LogHandler.SetupLog import setup_logger

logger = setup_logger()

DATASET = "ealaxi/paysim1"
OUT_DB = "paysim.db"
DB_PATH = Path("src/DataBase") / OUT_DB
TABLE = "paysim"
CHUNK = 600_000

def setup_kaggle_api():
    """Set up Kaggle API credentials from Streamlit secrets."""
    kaggle_credentials = {
        "username": st.secrets["kaggle"]["username"],
        "key": st.secrets["kaggle"]["key"]
    }
    
    # Define the path for kaggle.json
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Write the credentials to kaggle.json
    with open(kaggle_dir / 'kaggle.json', 'w') as f:
        json.dump(kaggle_credentials, f)
    
    # Set permissions
    os.chmod(kaggle_dir / 'kaggle.json', 0o600)

def main():
    if not DB_PATH.exists():
        setup_kaggle_api()
        
        import sqlite3
        import zipfile
        import pandas as pd
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        logger.info('Downloading dataset...')
        api = KaggleApi()
        api.authenticate()

        dataset = "ealaxi/paysim1"
        api.dataset_download_files(dataset, path="dl", unzip=True)
        
        logger.info('Dataset downloaded.')

        csv_path = "dl/PS_20174392719_1491204439457_log.csv"

        os.makedirs(DB_PATH.parent, exist_ok=True)
        con = sqlite3.connect(DB_PATH)

        # Process the CSV in chunks to avoid memory errors
        logger.info("Processing CSV in chunks...")
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK)):
            logger.info(f"Processing chunk {i+1}...")
            # For the first chunk, replace the table, for others, append.
            if_exists_strategy = "replace" if i == 0 else "append"
            chunk.to_sql("paysim", con, if_exists=if_exists_strategy, index=False)

        con.close()
        os.remove(csv_path) # Remove the CSV after successful processing
        logger.info("Finished processing all chunks.")
        logger.info('Dataset saved to database.')
    else:
        logger.info('Dataset already downloaded.')

if __name__ == "__main__":
    main()
