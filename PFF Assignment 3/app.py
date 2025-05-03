# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import os

# Page config
st.set_page_config(page_title="FINML", layout="wide", page_icon="📈")

# Custom CSS Styling
st.markdown("""
    <style>
        .main { background-color: #f0f2f6; }
        h1 { color: #003366; }
        .stButton>button { background-color: #003366; color: white; border-radius: 10px; padding: 10px 20px; }
        .stSidebar { background-color: #e6ecf2; }
    </style>
""", unsafe_allow_html=True)

# Welcome section
st.markdown("""
    <h1 style='text-align: center;'>📊 FINML App</h1>
""", unsafe_allow_html=True)

st.markdown("Welcome to FINML App! Upload data or fetch stock data and step through the ML pipeline.")

# Sidebar
st.sidebar.header("📂 Load Data")
data_source = st.sidebar.radio("Choose Data Source:", ["Kragle Dataset", "Fetch from Yahoo Finance"])

# Resolve base directory path
base_path = os.path.dirname(__file__)

# Load Kragle datasets using local relative paths
kragle_paths = {
    "Finance_data.csv": os.path.join(base_path, "kragle_data", "Finance_data.csv"),
    "Original_data.csv": os.path.join(base_path, "kragle_data", "Original_data.csv")
}

# Data container
df = None

if data_source == "Kragle Dataset":
    dataset_choice = st.sidebar.selectbox("Select Kragle Dataset:", list(kragle_paths.keys()))
    if dataset_choice:
        df_path = kragle_paths[dataset_choice]
        try:
            df = pd.read_csv(df_path)
            st.success(f"✅ {dataset_choice} loaded!")
        except FileNotFoundError:
            st.error(f"❌ File not found: {df_path}")
elif data_source == "Fetch from Yahoo Finance":
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    if st.sidebar.button("Fetch Data"):
        if ticker:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                st.success(f"✅ Fetched data for {ticker}!")
            else:
                st.error("No data returned. Check ticker symbol and date range.")
        else:
            st.warning("Please enter a valid stock ticker.")

# Step-by-step ML pipeline
if df is not None:
    st.subheader("📈 Data Preview")
    st.dataframe(df.head())

    # Step 1: Preprocessing
    if st.button("Step 1: Preprocess Data"):
        missing_before = df.isnull().sum().sum()
        df.dropna(inplace=True)
        missing_after = df.isnull().sum().sum()
        st.info(f"Missing values removed. Before: {missing_before}, After: {missing_after}")
        st.success("✅ Preprocessing completed!")

    # Step 2: Feature Engineering
    if st.button("Step 2: Feature Engineering"):
        if 'Close' in df.columns:
            df['Return'] = df['Close'].pct_change().fillna(0)
            st.success("✅ Added 'Return' feature!")
            st.line_chart(df['Return'])
        else:
            st.warning("'Close' column not found in dataset.")

    # Step 3: Train/Test Split
    if st.button("Step 3: Train/Test Split"):
        if 'Return' in df.columns:
            X = df[['Return']].shift(1).dropna()
            y = df['Return'][1:]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            st.success("✅ Split data into training and testing sets!")
            fig = px.pie(names=["Train", "Test"], values=[len(X_train), len(X_test)], title="Train/Test Split")
            st.plotly_chart(fig)
        else:
            st.warning("Please run feature engineering first.")

    # Step 4: Model Training
    if st.button("Step 4: Train Model"):
        if 'Return' in df.columns:
            model = LinearRegression()
            model.fit(X_train, y_train)
            st.success("✅ Model trained successfully!")
        else:
            st.warning("Please complete previous steps before training.")

    # Step 5: Evaluate Model
    if st.button("Step 5: Evaluate Model"):
        if 'Return' in df.columns:
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            st.metric("Mean Squared Error", round(mse, 6))
            st.metric("R² Score", round(r2, 4))
            fig = px.line(x=y_test.index, y=predictions, labels={'x': 'Date', 'y': 'Predicted Return'}, title="Predicted Returns")
            st.plotly_chart(fig)
        else:
            st.warning("Please train the model first.")

# Footer
st.markdown("---")
st.caption("Developed by Ibrahim Aziz")
