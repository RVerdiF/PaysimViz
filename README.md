# PaySim Dataset Explorer

A Streamlit application for exploring and analyzing the PaySim synthetic financial dataset. This tool provides insights into transaction patterns, fraud detection, and data anomalies.

## Features

- **Data-Aware Loading:** The app checks if the dataset exists. If not, it presents a simple one-click download button.
- **Home Page:** General statistics about the dataset, including null value analysis, negative value checks, and zero-amount transaction reports.
- **Data Exploration:** In-depth analysis of the dataset, including:
    - Transaction distribution over time (Hourly, Daily, Weekly).
    - Transaction type overview (Pie chart and summary table).
    - Analysis of the `isFlaggedFraud` feature's performance.
    - Insights into transaction amounts and descriptive statistics.
    - "Mule account" identification and analysis.
    - Analysis of balance-draining fraudulent transactions.

## Dataset

This application uses the PaySim dataset from Kaggle. You can find more information here: [PaySim Synthetic Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/data)

## Getting Started

### 1. Prerequisites

- Python 3.8+
- A Kaggle account and an API token (`kaggle.json`).

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd KrakenInterview
    ```

2.  **Set up Kaggle API Credentials:**
    - Download your `kaggle.json` API token from your Kaggle account settings.
    - Place the `kaggle.json` file in the expected directory. For most systems, this is `~/.kaggle/kaggle.json`. The application needs this file to download the dataset.

3.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

4.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Download the Dataset:**
    - On the first launch, the app will display a "Download Dataset" button.
    - Click this button to automatically download the data from Kaggle and set up the local database.
    - After the download is complete, the app will proceed to the main data explorer.

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