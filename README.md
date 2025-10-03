# PaySim Dataset Explorer

A Streamlit application for exploring and analyzing the PaySim synthetic financial dataset. This tool provides insights into transaction patterns, fraud detection, and data anomalies.

## Features

- **Efficient, High-Performance Backend:** Designed to handle the large PaySim dataset without crashing. The app uses a query-based architecture, parallel data loading, and the high-performance Polars data analysis library to ensure stability and speed.
- **Data-Aware Loading:** The app checks if the dataset exists. If not, it presents a simple one-click download button that securely uses your Kaggle credentials from Streamlit's secrets.
- **Home Page:** General statistics about the dataset, including null value analysis, negative value checks, and zero-amount transaction reports, all generated through efficient, direct-to-database queries.
- **Data Exploration:** In-depth analysis of the dataset, including:
    - Transaction distribution over time (Hourly, Daily, Weekly).
    - Transaction type overview (Pie chart and summary table).
    - Analysis of the `isFlaggedFraud` feature's performance.
    - Insights into transaction amounts and descriptive statistics.
    - "Mule account" identification and analysis, powered by Polars for high-performance aggregation.
    - Analysis of balance-draining fraudulent transactions.

## Architecture and Design

This application has been refactored for high performance and memory safety to handle the multi-gigabyte PaySim dataset.

- **SQL-Centric Backend:** Instead of loading the entire dataset into memory, the application leverages a **SQLite** database backend. Each chart and analysis on the UI is powered by a specific, targeted SQL query that only pulls the necessary aggregated data.
- **Parallel Execution:** To ensure a fast and responsive user experience, all independent database queries on a given page are executed in **parallel** using a `concurrent.futures.ThreadPoolExecutor`.
- **High-Performance Aggregation with Polars:** For complex aggregations that are not suitable for SQL (like the "Top Mule Accounts Overview"), the application uses the **Polars** library. It employs a memory-safe streaming strategy: data is queried from the database in chunks, processed, and aggregated without ever holding the complete intermediate dataset in memory.

## Dataset

This application uses the PaySim dataset from Kaggle. You can find more information here: [PaySim Synthetic Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/data)

## Getting Started

### 1. Prerequisites

- Python 3.8+
- A Kaggle account and an API token.

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd KrakenInterview
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Kaggle API Credentials:**
    - Create a file at `.streamlit/secrets.toml`.
    - Add your Kaggle username and API key to this file in the following format:
      ```toml
      [kaggle]
      username = "YOUR_KAGGLE_USERNAME"
      key = "YOUR_KAGGLE_KEY"
      ```

### 3. Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Download the Dataset:**
    - On the first launch, the app will display a "Download Dataset" button.
    - Click this button to automatically download the data from Kaggle and set up the local SQLite database. This process is memory-safe and handles the large CSV in chunks.
    - After the download is complete, the app will proceed to the main data explorer.

## Project Structure
```
├── .gitignore
├── app.py              # Main Streamlit application file
├── LICENSE
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── secrets.toml    # Streamlit secrets for Kaggle credentials
├── dl/                 # Directory for downloaded data
└── src/
    ├── DataBase/       # Stores the final SQLite database
    ├── DataHandler/    # Modules for handling data (loading, downloading)
    ├── LogHandler/     # Module for logging setup
    ├── notebooks/      # Jupyter notebooks for exploration
    └── Utils/          # Utility scripts (e.g., SQL queries)
```