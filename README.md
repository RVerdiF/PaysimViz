# PaySim Dataset Explorer

A Streamlit application for exploring and analyzing the PaySim synthetic financial dataset. This tool provides insights into transaction patterns, fraud detection, and data anomalies.

## Features

- **Home Page:** General statistics about the dataset, including null value analysis, negative value checks, and zero-amount transaction reports.
- **Data Exploration:** In-depth analysis of the dataset, including:
    - Transaction distribution over time (Hourly, Daily, Weekly).
    - Transaction type overview (Pie chart and summary table).
    - Analysis of the `isFlaggedFraud` feature's performance.
    - Insights into transaction amounts and descriptive statistics.
    - "Mule account" identification and analysis.
    - Analysis of balance-draining fraudulent transactions.

## Dataset

This application uses the PaySim dataset, which is a synthetic dataset generated using the PaySim simulator. The dataset is designed to be a realistic simulation of mobile money transactions and is intended for fraud detection research.

You can find more information about the dataset on Kaggle: [PaySim Synthetic Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/data)

## Installation

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

## Usage

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run app.py
```

The application will open in your web browser.

## Project Structure
```
├── .gitignore
├── app.py              # Main Streamlit application file
├── LICENSE
├── requirements.txt    # Python dependencies
├── dl/                 # Directory for downloaded data
└── src/
    ├── DataHandler/    # Modules for handling data (loading, downloading)
    ├── LogHandler/     # Module for logging setup
    ├── notebooks/      # Jupyter notebooks for exploration
    └── Utils/          # Utility scripts (e.g., SQL queries)
```
