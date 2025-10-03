
import streamlit as st
import pandas as pd
import plotly.express as px
import polars as pl
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from streamlit_scroll_to_top import scroll_to_here

from src.DataHandler.Downloader import main as setup_database
from src.DataHandler.DataHandler import return_df, DB_PATH, return_df_with_params, execute_query_in_chunks
from src.Utils.queries import (
    all_data,
    dataframe_metrics,
    time_data,
    transaction_type_analysis,
    fraud_flagging_analysis,
    home_page_data,
    descriptive_stats_query,
    zero_amount_by_type_query,
    zero_amount_by_fraud_query,
    negative_value_query,
    get_fraud_transactions_query,
    confusion_matrix_query,
    draining_transaction_stats_query,
    draining_behavior_by_type_query
)
from src.LogHandler.SetupLog import setup_logger

# Setup logger
logger = setup_logger()
logger.info("Application started.")

st.set_page_config(layout="wide")

# Custom CSS for st.info with purple theme
st.markdown(
    """
    <style>
    div[data-testid="stAlertContainer"] {
        background-color: #b494e6;
        border-left: 5px solid #7434f3;
        color: #262730;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def download_wrapper():
    """Wrapper to call the setup function and provide UI feedback."""
    with st.spinner("Downloading dataset... See console for progress."):
        try:
            setup_database()
            st.success("Download complete!")
            st.balloons()
        except Exception as e:
            logger.error(f"An error occurred during dataset download: {e}")
            st.error(f"An error occurred: {e}. Please check console logs and ensure kaggle.json is set up.")

@st.cache_data(show_spinner=False)
def load_home_page_data() -> pd.DataFrame:
    logger.info("Loading home page data.")
    df = return_df(home_page_data)
    logger.info("Home page data loaded successfully.")
    return df

@st.cache_data(show_spinner=False)
def load_descriptive_stats() -> pd.DataFrame:
    logger.info("Loading descriptive stats.")
    df = return_df(descriptive_stats_query)
    logger.info("Descriptive stats loaded successfully.")
    return df

@st.cache_data(show_spinner=False)
def load_zero_amount_by_type() -> pd.DataFrame:
    logger.info("Loading zero amount by type.")
    df = return_df(zero_amount_by_type_query)
    logger.info("Zero amount by type loaded successfully.")
    return df

@st.cache_data(show_spinner=False)
def load_zero_amount_by_fraud() -> pd.DataFrame:
    logger.info("Loading zero amount by fraud.")
    df = return_df(zero_amount_by_fraud_query)
    logger.info("Zero amount by fraud loaded successfully.")
    return df

@st.cache_data(show_spinner=False)
def load_negative_value_data() -> pd.DataFrame:
    logger.info("Loading negative value data.")
    df = return_df(negative_value_query)
    logger.info("Negative value data loaded successfully.")
    return df

@st.cache_data(show_spinner=False)
def load_fraud_transactions() -> pd.DataFrame:
    logger.info("Loading fraud transactions.")
    df = return_df(get_fraud_transactions_query)
    logger.info("Fraud transactions loaded successfully.")
    return df

@st.cache_data(show_spinner=False)
def load_confusion_matrix() -> pd.DataFrame:
    logger.info("Loading confusion matrix data.")
    df = return_df(confusion_matrix_query)
    logger.info("Confusion matrix data loaded successfully.")
    return df

@st.cache_data(show_spinner=False)
def load_draining_transaction_stats() -> pd.DataFrame:
    logger.info("Loading draining transaction stats.")
    df = return_df(draining_transaction_stats_query)
    logger.info("Draining transaction stats loaded successfully.")
    return df

@st.cache_data(show_spinner=False)
def load_draining_behavior_by_type() -> pd.DataFrame:
    logger.info("Loading draining behavior by type.")
    df = return_df(draining_behavior_by_type_query)
    logger.info("Draining behavior by type loaded successfully.")
    return df

@st.cache_data(show_spinner=False)
def load_main_data() -> pd.DataFrame:
    logger.info("Loading main data.")
    df = return_df(all_data)
    logger.info("Main data loaded successfully.")
    return df

def home():

    logger.info("Displaying Home page.")
    
    if not DB_PATH.exists():
        top_of_page_container = st.container()
        top_of_page_container.key = 'top_of_page'
        scroll_to_here(delay=10,key='top_of_page')
        st.title("Hi!")
        st.write("The PaySim dataset is not yet downloaded. Click the button below to start the download.")
        st.info("Note: This requires your Kaggle API credentials to be set up in Streamlit's secrets. Please add your Kaggle username and key to the secrets file.")

        if st.button("Download Dataset", type="primary"):
            download_wrapper()
            st.rerun()
    else:
        top_of_page_container = st.container()
        top_of_page_container.key = 'top_of_page'
        scroll_to_here(delay=10,key='top_of_page')
        title = st.title("Hi!")
        st.write("Welcome to the PaySim dataset explorer!")
        st.write("This app allows you to explore the PaySim dataset (https://www.kaggle.com/datasets/ealaxi/paysim1/data) and visualize its data.")  
        st.write("Please select a page from the sidebar to get started.")

        with st.spinner("Loading data..."):
            with ThreadPoolExecutor() as executor:
                future_home_df = executor.submit(load_home_page_data)
                future_descriptive_stats_df = executor.submit(load_descriptive_stats)
                future_zero_amount_by_type_df = executor.submit(load_zero_amount_by_type)
                future_zero_amount_by_fraud_df = executor.submit(load_zero_amount_by_fraud)
                future_negative_value_df = executor.submit(load_negative_value_data)

                home_df = future_home_df.result()
                descriptive_stats_df = future_descriptive_stats_df.result()
                zero_amount_by_type_df = future_zero_amount_by_type_df.result()
                zero_amount_by_fraud_df = future_zero_amount_by_fraud_df.result()
                negative_value_df = future_negative_value_df.result()

        st.subheader("General DataFrame Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Rows", f"{home_df['total_rows'].iloc[0]:,}")
        with col2:
            st.metric("Number of Columns", len(home_df.columns))
        with col3:
            st.metric("Total Null Values", f"{home_df.iloc[0, 1:].sum():,}")

        st.subheader("Null Values per Column")
        nulls = home_df.iloc[0, 1:].T.to_frame().rename(columns={0: 'count'})
        nulls = nulls[nulls['count'] > 0]
        if not nulls.empty:
            st.dataframe(nulls, width='stretch')
        else:
            st.write("No null values found.")

        st.subheader("Negative Value Analysis")
        negative_counts = negative_value_df.T.rename(columns={0: 'Count of Negative Values'})
        negative_counts = negative_counts[negative_counts['Count of Negative Values'] > 0]

        if not negative_counts.empty:
            st.warning("Negative values were found in the following columns:")
            st.dataframe(negative_counts, width='stretch')
        else:
            st.success("No negative values found in any of the numeric columns.")

        st.subheader("Descriptive Statistics")
        descriptive_stats_df = descriptive_stats_df.set_index('metric').T
        descriptive_stats_df.index.name = 'Column'
        st.dataframe(descriptive_stats_df, width='stretch')

        st.subheader("Zero Amount Transaction Analysis")
        zero_amount_count = zero_amount_by_type_df['count'].sum()
        
        if zero_amount_count > 0:
            st.warning(f"Found {zero_amount_count:,} transactions with an amount of 0.")
            
            st.write("Breakdown by transaction type:")
            st.dataframe(zero_amount_by_type_df.set_index('type'), width='stretch')
            
            st.write("Breakdown by fraud status:")
            zero_amount_by_fraud_df.index = zero_amount_by_fraud_df.isFraud.map({0: 'Not Fraud', 1: 'Fraud'})
            zero_amount_by_fraud_df = zero_amount_by_fraud_df.drop(columns=['isFraud'])
            st.dataframe(zero_amount_by_fraud_df, width='stretch')

        else:
            st.success("No transactions with an amount of 0 found.")

@st.cache_data(show_spinner=False)
def load_transaction_data() -> tuple[pd.DataFrame,pd.DataFrame]:
    logger.info("Loading transaction data.")
    time_data_df = return_df(time_data)
    time_data_df['date'] = pd.to_datetime(time_data_df['date'])
    transaction_type_analysis_df = return_df(transaction_type_analysis)
    transposed = transaction_type_analysis_df.T
    transposed.columns = transposed.iloc[0]
    transposed = transposed.drop(transposed.index[0])
    transposed.index.name = 'METRIC'
    dataframe_metrics_df = return_df(dataframe_metrics).melt(var_name='metric', value_name='value')
    logger.info("Transaction data loaded successfully.")
    return time_data_df, transposed, transaction_type_analysis_df, dataframe_metrics_df

def data_analysis():
    top_of_page_container = st.container()
    top_of_page_container.key = 'top_of_page'
    scroll_to_here(delay=10,key='top_of_page')
    st.title("Paysim Data Analysis")
    logger.info("Displaying Data Analysis page.")

    if not DB_PATH.exists():
        st.warning("The database has not been downloaded yet. Please go to the 'Home' page to download the dataset.")
        return # Stop execution of the function

    with ThreadPoolExecutor() as executor:
        future_transaction_data = executor.submit(load_transaction_data)
        future_fraud_df = executor.submit(load_fraud_transactions)
        future_confusion_matrix_df = executor.submit(load_confusion_matrix)
        future_draining_stats_df = executor.submit(load_draining_transaction_stats)
        future_draining_behavior_df = executor.submit(load_draining_behavior_by_type)

        time_data_df, transaction_type_analysis_df_T, transaction_type_analysis_df, dataframe_metrics_df = future_transaction_data.result()
        fraud_df = future_fraud_df.result()
        confusion_matrix_df = future_confusion_matrix_df.result()
        draining_stats_df = future_draining_stats_df.result()
        draining_behavior_df = future_draining_behavior_df.result()
    
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", int(dataframe_metrics_df[dataframe_metrics_df['metric']=='count']['value'].iloc[0]))
    with col2:
        amount = dataframe_metrics_df[dataframe_metrics_df['metric']=='sum_amount']['value'].iloc[0]
        st.metric("Total Amount", f"${amount/1_000_000:,.0f}M")
    with col3:
        st.metric("Total Fraud Count", int(dataframe_metrics_df[dataframe_metrics_df['metric']=='fraud_count']['value'].iloc[0]))
    with col4:
        amount = dataframe_metrics_df[dataframe_metrics_df['metric']=='fraud_amount']['value'].iloc[0]
        st.metric("Total Fraud Amount", f"${amount/1_000_000:,.0f}M")

    st.subheader("Transactions distribution over time")
    st.info("For this exercise, let's assume the final step (1) is 2025-09-30 23:00:00. From that, each step is subtracted from the start date, ending on 2025-08-31 01:00:00" )
    period = st.radio("Select the time range", ["Hourly", "Daily", "Weekly"], key="time_range", horizontal=True)

    time_data_df_filtered = time_data_df.copy(deep=True)
    if period == "Hourly":
        time_data_df_filtered['date'] = time_data_df_filtered['date'].dt.to_period('h').dt.to_timestamp()
        time_data_df_filtered = time_data_df_filtered.groupby('date').sum().reset_index()
    elif period == "Daily":
        time_data_df_filtered['date'] = time_data_df_filtered['date'].dt.to_period('D').dt.to_timestamp()
        time_data_df_filtered = time_data_df_filtered.groupby('date').sum().reset_index()
    elif period == "Weekly":
        time_data_df_filtered['date'] = time_data_df_filtered['date'].dt.to_period('W').dt.to_timestamp()
        time_data_df_filtered = time_data_df_filtered.groupby('date').sum().reset_index()
    
    st.plotly_chart(px.line(time_data_df_filtered, x='date', y='count', color_discrete_sequence=["#7434f3", "#b494e6", "#bc91f7"]), use_container_width=True)
    st.markdown('---')
    st.subheader("Transaction types overview")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(transaction_type_analysis_df, title='Count of transactions by type', names='type', values='count', color_discrete_sequence=["#7434f3", "#b494e6", "#bc91f7"]), use_container_width=True)
    with col2:
        fraud_by_type = fraud_df['type'].value_counts().reset_index()
        fraud_by_type.columns = ['type', 'count']
        st.plotly_chart(px.pie(fraud_by_type, title='Fraud distribution by transaction type', names='type', values='count', color_discrete_sequence=["#7434f3", "#b494e6", "#bc91f7"]), use_container_width=True)
    st.dataframe(transaction_type_analysis_df_T)

    st.info('''By analyzing the charts above, we can see that the fraudulent transactions are located primarily on transfers and cash outs.''')
    
    st.markdown('---')
    st.header("Fraud & Amount Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("`isFlaggedFraud` Performance")
        
        confusion_matrix = confusion_matrix_df.pivot_table(index='isFraud', columns='isFlaggedFraud', values='count').fillna(0)
        
        fig = px.imshow(confusion_matrix, text_auto=True, labels=dict(x="Flagged as Fraud", y="Actual Fraud", color="Count"), color_continuous_scale=["#b494e6", "#7434f3"])
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Transaction Amount Insights")
        
        st.write("**Descriptive Statistics for Amount:**")
        descriptive_stats = fraud_df['amount'].describe().to_frame()
        descriptive_stats_display = descriptive_stats.copy()
        descriptive_stats_display['amount'] = descriptive_stats_display['amount'].apply(lambda x: f'${x:,.2f}')
        descriptive_stats_display.loc['count', 'amount'] = f"{int(descriptive_stats.loc['count', 'amount']):,}"
        
        large_fraud_transactions = fraud_df[fraud_df['amount'] > 200000]
        total_fraud_transactions = len(fraud_df)
        large_fraud_count = len(large_fraud_transactions)
        percentage = 100 - ((large_fraud_count / total_fraud_transactions) * 100) if total_fraud_transactions > 0 else 0
        value=200000
        new_row = pd.DataFrame({'amount': [f'${value:,.2f}'] }, index=[f'{percentage:.0f}%'])
        descriptive_stats_display = pd.concat([descriptive_stats_display, new_row]).sort_index(ascending=False)
        descriptive_stats_display.index.name = 'metric'
        st.dataframe(descriptive_stats_display)

    st.info("The current system operates on a single, rigid rule: it flags any transfer exceeding $200,000 in a single transaction.\n\n"\
            "Perfect Precision: This rule is 100% precise. Every transaction it flags as fraudulent (isFlaggedFraud = 1) is, in fact, actual fraud (isFraud = 1). It produces zero false positives.\n\n"\
            "Extremely Low Recall: However, the system has critically low recall, meaning it fails to identify the vast majority of fraudulent activities. It gets even worse when we find that that over 67% of actual fraudulent transactions are over the $200k threshold.\n\n"\
            "In short, the flag correctly identifies a very specific type of high-value fraud but allows most fraudulent transactions to go completely undetected.")

    st.markdown('---')
    st.header("Mule Account Analysis")
    st.info("Accounts that repeatedly appear in fraudulent transactions may be 'mule accounts,' used to concentrate and launder money.")

    fraudulent_accounts = pd.concat([fraud_df['nameOrig'], fraud_df['nameDest']]).unique()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Fraudulent Origin Accounts")
        top_origin = fraud_df['nameOrig'].value_counts().nlargest(10).to_frame()
        top_origin.rename(columns={'count': 'Number of Frauds'}, inplace=True)
        st.dataframe(top_origin)

    with col2:
        st.subheader("Top Fraudulent Destination Accounts")
        top_dest = fraud_df['nameDest'].value_counts().nlargest(10).to_frame()
        top_dest.rename(columns={'count': 'Number of Frauds'}, inplace=True)
        st.dataframe(top_dest)
        
    st.subheader("Potential Mule Accounts Transactional Overview")

    with st.spinner("Analyzing potential mule accounts... This may take a moment."):
        fraudulent_accounts_list = list(fraudulent_accounts)
        
        # --- Efficiently process mule account transactions in chunks --- #
        placeholders = ', '.join('?' for _ in fraudulent_accounts_list)
        query = f"SELECT nameOrig, nameDest, amount, isFraud FROM paysim WHERE nameOrig IN ({placeholders}) OR nameDest IN ({placeholders})"
        params = fraudulent_accounts_list * 2
        
        polars_dfs = []
        for df_chunk in execute_query_in_chunks(query, params, chunksize=100_000):
            polars_dfs.append(pl.from_pandas(df_chunk))

        if polars_dfs:
            mule_transactions_pl = pl.concat(polars_dfs)

            melted_lf = mule_transactions_pl.lazy().unpivot(
                index=["isFraud", "amount"],
                on=["nameOrig", "nameDest"],
                value_name="Account"
            )

            account_metrics_lf = melted_lf.group_by("Account").agg(
                pl.len().alias("Total_Transactions"),
                pl.sum("amount").alias("Total_Amount"),
                pl.sum("isFraud").alias("Fraudulent_Transactions"),
                pl.when(pl.col("isFraud") == 1).then(pl.col("amount")).otherwise(0).sum().alias("Fraudulent_Amount")
            )

            final_metrics_lf = account_metrics_lf.filter(
                pl.col("Account").is_in(fraudulent_accounts_list)
            ).sort(
                by=["Fraudulent_Transactions", "Total_Transactions"],
                descending=True
            )

            account_metrics_df = final_metrics_lf.collect().to_pandas()
            account_metrics_df['Fraudulent_Amount'] = account_metrics_df['Fraudulent_Amount'].fillna(0)
            
            st.dataframe(account_metrics_df, width='stretch', hide_index=True)
        else:
            st.write("No transactions found for the identified mule accounts.")
        
    st.info("Even though only some transactions are flagged as fraud, it's important to check if the other transactions related to fraudsters are not contaminated as well.")

    st.subheader("Explore All Transactions of Potentially Fraudulent Accounts")
    
    selected_accounts = st.multiselect(
        "Select accounts to view all their transactions:",
        options=fraudulent_accounts,
        default=list(fraudulent_accounts[:5])
    )
    
    if selected_accounts:
        placeholders = ', '.join('?' for _ in selected_accounts)
        query = f"SELECT * FROM paysim WHERE nameOrig IN ({placeholders}) OR nameDest IN ({placeholders})"
        params = selected_accounts * 2
        mule_account_transactions_selection = return_df_with_params(query, params)
        st.dataframe(mule_account_transactions_selection, width='stretch', hide_index=True)

    st.markdown('---')
    st.header("Fraudulent Transaction Balance Analysis")
    
    st.subheader("Distribution of Origin Account Balance After Fraudulent Transactions")
    balance_counts = fraud_df['newbalanceOrig'].value_counts().reset_index()
    balance_counts.columns = ['newbalanceOrig', 'count']
    fig = px.scatter(balance_counts, 
                     x='newbalanceOrig', 
                     y='count', 
                     title='Number of Fraudulent Transactions per Origin Account Balance', 
                     color_discrete_sequence=["#7434f3", "#b494e6", "#bc91f7"])
    st.plotly_chart(fig, use_container_width=True)

    color_map = {
        'Draining Frauds': '#7434f3', 
        'Other Frauds': '#b494e6',
        'Draining Transactions': '#7434f3',
        'Other Transactions': '#b494e6'
    }
    col1, col2= st.columns(2)
    with col1:
        st.subheader("Analysis of Balance-Draining Frauds")
        draining_frauds_count = fraud_df[fraud_df['amount'] == fraud_df['oldbalanceOrg']].shape[0]
        total_fraud_count = len(fraud_df)

        st.metric(
            "Fraudulent Transactions that Drain the Origin Account",
            f"{draining_frauds_count}"
        )

        pie_data = pd.DataFrame({
            'Category': ['Draining Frauds', 'Other Frauds'],
            'Count': [draining_frauds_count, total_fraud_count - draining_frauds_count]
        })
        fig_pie = px.pie(pie_data, names='Category', values='Count', title='Proportion of Balance-Draining Frauds', color='Category', color_discrete_map=color_map)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Analysis of Balance-Draining Transactions (All)")
        draining_transactions_all_count = draining_stats_df['draining_count'].iloc[0]
        total_transactions_all = draining_stats_df['total_count'].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "All Transactions that Drain the Origin Account",
                f"{draining_transactions_all_count}"
            )
        with col2:
            st.metric(
                "Total Transactions",
                f"{total_transactions_all}"
            )
        pie_data_all = pd.DataFrame({
            'Category': ['Draining Transactions', 'Other Transactions'],
            'Count': [draining_transactions_all_count, total_transactions_all - draining_transactions_all_count]
        })
        fig_pie_all = px.pie(pie_data_all, names='Category', values='Count', title='Proportion of Balance-Draining Transactions (All)', color='Category', color_discrete_map=color_map)
        st.plotly_chart(fig_pie_all, use_container_width=True)

    st.info("A critical insight emerges from balance analysis: while over 90% of fraudulent transactions completely empty the source account, this behavior is extremely rare among genuine customers, occurring in only 0.12% of their transactions.")

    st.subheader("Balance-Draining Behavior by Transaction Type")

    summary = draining_behavior_df.pivot_table(index='type', columns='isFraud', values='draining_percentage').fillna(0)
    summary.rename(columns={0: '% Draining (Not Fraud)', 1: '% Draining (Fraud)'}, inplace=True)

    # Ensure both columns exist
    if '% Draining (Not Fraud)' not in summary.columns:
        summary['% Draining (Not Fraud)'] = 0
    if '% Draining (Fraud)' not in summary.columns:
        summary['% Draining (Fraud)'] = 0

    st.dataframe(summary[['% Draining (Not Fraud)', '% Draining (Fraud)']].style.format('{:.2f}%'), width='stretch')

    st.warning("Proposed Rule: Flag 'TRANSFER' and 'CASH_OUT' type transactions that result in a zero balance in the origin account for manual review.")

PAGES = {
    "Home": home,
    "Data Analysis": data_analysis,
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
logger.info(f"Navigating to {selection} page.")

page = PAGES[selection]
page()
