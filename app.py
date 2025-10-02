'''A simple streamlit app for the Kraken Interview.'''
import streamlit as st
import pandas as pd
import plotly.express as px

from src.DataHandler.Downloader import main as setup_database
from src.DataHandler.DataHandler import return_df, DB_PATH, return_df_with_params
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
        st.title("Welcome!")
        st.write("The PaySim dataset is not yet downloaded. Click the button below to start the download.")
        st.info("Note: This requires your Kaggle API credentials to be set up in Streamlit's secrets. Please add your Kaggle username and key to the secrets file.")

        if st.button("Download Dataset", type="primary"):
            download_wrapper()
    else:
        st.title("Home")
        st.write("Welcome to the PaySim dataset explorer!")
        st.write("This app allows you to explore the PaySim dataset (https://www.kaggle.com/datasets/ealaxi/paysim1/data) and visualize its data.")  
        
        with st.spinner("Loading data..."):
            home_df = load_home_page_data()
            descriptive_stats_df = load_descriptive_stats()
            zero_amount_by_type_df = load_zero_amount_by_type()
            zero_amount_by_fraud_df = load_zero_amount_by_fraud()
            negative_value_df = load_negative_value_data()

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
    dataframe_metrics_df = return_df(dataframe_metrics).melt(var_name='metric', value_name='value')
    logger.info("Transaction data loaded successfully.")
    return time_data_df, transposed, transaction_type_analysis_df, dataframe_metrics_df

def data_exploration():
    logger.info("Displaying Data Exploration page.")

    if not DB_PATH.exists():
        st.warning("The database has not been downloaded yet. Please go to the 'Home' page to download the dataset.")
        return # Stop execution of the function

    time_data_df, transaction_type_analysis_df_T, transaction_type_analysis_df, dataframe_metrics_df = load_transaction_data()
    fraud_df = load_fraud_transactions()
    confusion_matrix_df = load_confusion_matrix()
    draining_stats_df = load_draining_transaction_stats()
    draining_behavior_df = load_draining_behavior_by_type()
    
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", dataframe_metrics_df[dataframe_metrics_df['metric']=='count']['value'].iloc[0])
    with col2:
        amount = dataframe_metrics_df[dataframe_metrics_df['metric']=='sum_amount']['value'].iloc[0]
        st.metric("Total Amount", f"${amount:,.2f}")
    with col3:
        st.metric("Total Fraud Count", dataframe_metrics_df[dataframe_metrics_df['metric']=='fraud_count']['value'].iloc[0])
    with col4:
        amount = dataframe_metrics_df[dataframe_metrics_df['metric']=='fraud_amount']['value'].iloc[0]
        st.metric("Total Fraud Amount", f"${amount:,.2f}")


    st.subheader("Transactions distribution over time")
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
    
    st.line_chart(data=time_data_df_filtered, x='date', y='count')
    st.markdown('---')
    st.subheader("Transaction types overview")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(transaction_type_analysis_df, names='type', values='count'), use_container_width=True)
    with col2:
        st.dataframe(transaction_type_analysis_df_T)
    
    st.markdown('---')
    st.header("Fraud & Amount Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("`isFlaggedFraud` Performance")
        
        confusion_matrix = confusion_matrix_df.pivot_table(index='isFraud', columns='isFlaggedFraud', values='count').fillna(0)
        
        fig = px.imshow(confusion_matrix, text_auto=True, labels=dict(x="Flagged as Fraud", y="Actual Fraud", color="Count"))
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
        st.dataframe(mule_account_transactions_selection, width='stretch')

    st.markdown('---')
    st.header("Fraudulent Transaction Balance Analysis")
    
    st.subheader("Distribution of Origin Account Balance After Fraudulent Transactions")
    balance_counts = fraud_df['newbalanceOrig'].value_counts().reset_index()
    balance_counts.columns = ['newbalanceOrig', 'count']
    fig = px.scatter(balance_counts, x='newbalanceOrig', y='count', title='Number of Fraudulent Transactions per Origin Account Balance')
    st.plotly_chart(fig, use_container_width=True)

    color_map = {
        'Draining Frauds': 'crimson', 
        'Other Frauds': 'lightgrey',
        'Draining Transactions': 'crimson',
        'Other Transactions': 'lightgrey'
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
                "All Transactions",
                f"{total_transactions_all}"
            )
        pie_data_all = pd.DataFrame({
            'Category': ['Draining Transactions', 'Other Transactions'],
            'Count': [draining_transactions_all_count, total_transactions_all - draining_transactions_all_count]
        })
        fig_pie_all = px.pie(pie_data_all, names='Category', values='Count', title='Proportion of Balance-Draining Transactions (All)', color='Category', color_discrete_map=color_map)
        st.plotly_chart(fig_pie_all, use_container_width=True)

    st.info("Insight: A critical insight emerges from balance analysis: while over 90% of fraudulent transactions completely empty the source account, this behavior is extremely rare among genuine customers, occurring in only 0.12% of their transactions.")

    st.subheader("Balance-Draining Behavior by Transaction Type")

    summary = draining_behavior_df.pivot_table(index='type', columns='isFraud', values='draining_percentage').fillna(0)
    summary.rename(columns={0: '% Draining (Not Fraud)', 1: '% Draining (Fraud)'}, inplace=True)

    # Ensure both columns exist
    if '% Draining (Not Fraud)' not in summary.columns:
        summary['% Draining (Not Fraud)'] = 0
    if '% Draining (Fraud)' not in summary.columns:
        summary['% Draining (Fraud)'] = 0

    st.dataframe(summary[['% Draining (Not Fraud)', '% Draining (Fraud)']].style.format('{:.2f}%'), width='stretch')

    st.warning("Proposed Rule: Flag 'TRANSFER' type transactions that result in a zero balance in the origin account for manual review.")

        
def model_performance():
    st.title("Current Model Performance")
    st.write("This page will briefily show the performance of the current fraud detection model.")

def fraud_detection():
    st.title("Fraud Detection")
    st.write("This page will allow you to interact with the fraud detection model.")

PAGES = {
    "Home": home,
    "Data Exploration": data_exploration,
}

# The data is expected to be downloaded and prepared before running the app.
# Run `python src/DataHandler/Downloader.py` manually to do this.
logger.info("Skipping data setup. Database is expected to exist.")

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
logger.info(f"Navigating to {selection} page.")
page = PAGES[selection]
page()
