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

st.set_page_config(layout="wide")

@st.cache_resource(show_spinner=False)
def load_main_data() -> pd.DataFrame:
    df = return_df(all_data)
    return df

def home():
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

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
@st.cache_resource(show_spinner=False)
def load_transaction_data() -> tuple[pd.DataFrame,pd.DataFrame]:
    time_data_df = return_df(time_data)
    time_data_df['date'] = pd.to_datetime(time_data_df['date'])
    transaction_type_analysis_df = return_df(transaction_type_analysis)
    transposed = transaction_type_analysis_df.T
    transposed.columns = transposed.iloc[0]
    transposed = transposed.drop(transposed.index[0])
    dataframe_metrics_df = return_df(dataframe_metrics).melt(var_name='metric', value_name='value')
    return time_data_df, transposed, transaction_type_analysis_df, dataframe_metrics_df

def data_exploration():
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

    st.write("**High-Value Transactions (> $200k):**")
    amount_over_200k = df[df['amount'] > 200000]
    fraud_over_200k = amount_over_200k[amount_over_200k['isFraud'] == 1]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Count of Transactions > $200k", f"{len(amount_over_200k):,}")
    with col2:
        st.metric("Count of Fraudulent Transactions > $200k", f"{len(fraud_over_200k):,}")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("`isFlaggedFraud` Performance")
        # st.info("`isFlaggedFraud` is a basic heuristic from the dataset creators that flags transactions over 200k. This section analyzes its performance as a baseline model.")

        confusion_matrix = pd.crosstab(df['isFraud'], df['isFlaggedFraud'], rownames=['Actual Fraud'], colnames=['Flagged Fraud'])
        
        fig = px.imshow(confusion_matrix, text_auto=True, labels=dict(x="Flagged as Fraud", y="Actual Fraud", color="Count"))
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Transaction Amount Insights")
        
        st.write("**Descriptive Statistics for Amount:**")
        descriptive_stats = df['amount'].describe().to_frame()
        descriptive_stats_display = descriptive_stats.copy()
        descriptive_stats_display['amount'] = descriptive_stats_display['amount'].apply(lambda x: f'${x:,.2f}')
        descriptive_stats_display.loc['count', 'amount'] = f"{int(descriptive_stats.loc['count', 'amount']):,}"
        st.dataframe(descriptive_stats_display)

    # st.write("**Amount Quantiles:**")
    # st.dataframe(df['amount'].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).to_frame())

        
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

if __name__ == "__main__":
    main()
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page()
