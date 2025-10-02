'''A simple streamlit app for the Kraken Interview.'''
import streamlit as st
import pandas as pd
import plotly.express as px

from src.DataHandler.Downloader import main
from src.DataHandler.DataHandler import return_df
from src.Utils.queries import (
    all_data,
    dataframe_metrics,
    time_data,
    transaction_type_analysis,
    fraud_flagging_analysis
)
from src.LogHandler.SetupLog import setup_logger

# Setup logger
logger = setup_logger()
logger.info("Application started.")

st.set_page_config(layout="wide")

@st.cache_resource(show_spinner=False)
def load_main_data() -> pd.DataFrame:
    logger.info("Loading main data.")
    df = return_df(all_data)
    logger.info("Main data loaded successfully.")
    return df

def home():
    logger.info("Displaying Home page.")
    st.title("Home")
    st.write("Welcome to the PaySim dataset explorer!")
    st.write("This app allows you to explore the PaySim dataset (https://www.kaggle.com/datasets/ealaxi/paysim1/data) and visualize its data.")  
    
    with st.spinner("Loading data..."):
        df = load_main_data()

    st.subheader("General DataFrame Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Number of Columns", df.shape[1])
    with col3:
        st.metric("Total Null Values", f"{df.isnull().sum().sum():,}")

    st.subheader("Null Values per Column")
    nulls = df.isnull().sum().to_frame().rename(columns={0: 'count'})
    nulls = nulls[nulls['count'] > 0]
    if not nulls.empty:
        st.dataframe(nulls, use_container_width=True)
    else:
        st.write("No null values found.")

    st.subheader("Negative Value Analysis")
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    negative_counts = {}
    for col in numeric_cols:
        count = (df[col] < 0).sum()
        if count > 0:
            negative_counts[col] = count
            
    if negative_counts:
        st.warning("Negative values were found in the following columns:")
        neg_df = pd.DataFrame(list(negative_counts.items()), columns=['Column', 'Count of Negative Values'])
        st.dataframe(neg_df, use_container_width=True)
    else:
        st.success("No negative values found in any of the numeric columns.")

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Zero Amount Transaction Analysis")
    zero_amount_transactions = df[df['amount'] == 0]
    zero_amount_count = len(zero_amount_transactions)
    
    if zero_amount_count > 0:
        st.warning(f"Found {zero_amount_count:,} transactions with an amount of 0.")
        
        st.write("Breakdown by transaction type:")
        zero_amount_by_type = zero_amount_transactions['type'].value_counts().to_frame()
        st.dataframe(zero_amount_by_type, use_container_width=True)
        
        st.write("Breakdown by fraud status:")
        zero_amount_by_fraud = zero_amount_transactions['isFraud'].value_counts().to_frame()
        zero_amount_by_fraud.index = zero_amount_by_fraud.index.map({0: 'Not Fraud', 1: 'Fraud'})
        st.dataframe(zero_amount_by_fraud, use_container_width=True)

    else:
        st.success("No transactions with an amount of 0 found.")
    
    
@st.cache_resource(show_spinner=False)
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
    time_data_df, transaction_type_analysis_df_T, transaction_type_analysis_df, dataframe_metrics_df = load_transaction_data()
    df = load_main_data()
    
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
        time_data_df_filtered['date'] = time_data_df_filtered['date'].dt.to_period('H').dt.to_timestamp()
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
        
        confusion_matrix = pd.crosstab(df['isFraud'], df['isFlaggedFraud'], rownames=['Actual Fraud'], colnames=['Flagged Fraud'])
        
        fig = px.imshow(confusion_matrix, text_auto=True, labels=dict(x="Flagged as Fraud", y="Actual Fraud", color="Count"))
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Transaction Amount Insights")
        
        st.write("**Descriptive Statistics for Amount:**")
        descriptive_stats = df[df['isFraud']==1]['amount'].describe().to_frame()
        descriptive_stats_display = descriptive_stats.copy()
        descriptive_stats_display['amount'] = descriptive_stats_display['amount'].apply(lambda x: f'${x:,.2f}')
        descriptive_stats_display.loc['count', 'amount'] = f"{int(descriptive_stats.loc['count', 'amount']):,}"
        
        fraud_df = df[df['isFraud'] == 1]
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

    fraud_df = df[df['isFraud'] == 1]
    
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
        
    st.subheader("Top Mule Accounts Overview")
    mule_account_transactions = df[df['nameOrig'].isin(fraudulent_accounts) | df['nameDest'].isin(fraudulent_accounts)]
    
    # Melt the dataframe to have one account per row
    melted_df = pd.melt(mule_account_transactions, id_vars=['isFraud', 'amount'], value_vars=['nameOrig', 'nameDest'], value_name='Account')
    
    # Calculate total transactions and amount
    total_metrics = melted_df.groupby('Account').agg(
        Total_Transactions=('Account', 'size'),
        Total_Amount=('amount', 'sum')
    ).reset_index()
    
    # Calculate fraudulent transactions and amount
    fraud_metrics = melted_df[melted_df['isFraud'] == 1].groupby('Account').agg(
        Fraudulent_Transactions=('isFraud', 'size'),
        Fraudulent_Amount=('amount', 'sum')
    ).reset_index()
    
    # Merge the metrics
    account_metrics = pd.merge(total_metrics, fraud_metrics, on='Account', how='left').fillna(0)
    
    # Filter for fraudulent accounts and sort
    account_metrics = account_metrics[account_metrics['Account'].isin(fraudulent_accounts)]
    account_metrics['Fraudulent_Transactions'] = account_metrics['Fraudulent_Transactions'].astype(int)
    account_metrics = account_metrics.sort_values(by=['Fraudulent_Transactions','Total_Transactions'], ascending=False).reset_index(drop=True).head(10)
    
    st.dataframe(account_metrics, use_container_width=True)

    st.subheader("Explore All Transactions of Potentially Fraudulent Accounts")
    
    selected_accounts = st.multiselect(
        "Select accounts to view all their transactions:",
        options=fraudulent_accounts,
        default=list(fraudulent_accounts[:5])
    )
    
    if selected_accounts:
        mule_account_transactions_selection = df[df['nameOrig'].isin(selected_accounts) | df['nameDest'].isin(selected_accounts)]
        st.dataframe(mule_account_transactions_selection, use_container_width=True)

    st.markdown('---')
    st.header("Fraudulent Transaction Balance Analysis")
    
    fraud_transactions = df[df['isFraud'] == 1]
    
    st.subheader("Distribution of Origin Account Balance After Fraudulent Transactions")
    balance_counts = fraud_transactions['newbalanceOrig'].value_counts().reset_index()
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
        draining_frauds_count = fraud_transactions[fraud_transactions['amount'] == fraud_transactions['oldbalanceOrg']].shape[0]
        total_fraud_count = fraud_transactions.shape[0]

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
        draining_transactions_all_count = df[df['amount'] == df['oldbalanceOrg']].shape[0]
        total_transactions_all = df.shape[0]
        
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

    # Create a boolean column for draining transactions
    df['isDraining'] = df['amount'] == df['oldbalanceOrg']
    
    # Group by type and fraud status, then calculate the percentage of draining transactions
    summary = df.groupby(['type', 'isFraud'])['isDraining'].value_counts(normalize=True).unstack().fillna(0)
    if True in summary.columns:
        summary = summary.loc[:, True] * 100 # Get the percentage for isDraining=True
    else: # Handle case where no draining transactions exist
        summary[True] = 0
        summary = summary.loc[:, True] * 100

    summary = summary.unstack(level='isFraud').fillna(0)
    
    # Rename columns
    if 0 in summary.columns:
        summary.rename(columns={0: '% Draining (Not Fraud)'}, inplace=True)
    if 1 in summary.columns:
        summary.rename(columns={1: '% Draining (Fraud)'}, inplace=True)

    # Ensure both columns exist
    if '% Draining (Not Fraud)' not in summary.columns:
        summary['% Draining (Not Fraud)'] = 0
    if '% Draining (Fraud)' not in summary.columns:
        summary['% Draining (Fraud)'] = 0

    st.dataframe(summary[['% Draining (Not Fraud)', '% Draining (Fraud)']].style.format('{:.2f}%'), use_container_width=True)

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

main()
logger.info("Data setup main() executed.")

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
logger.info(f"Navigating to {selection} page.")
page = PAGES[selection]
page()
