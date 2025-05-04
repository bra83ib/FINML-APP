import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="FINML Pro Dashboard",
    layout="wide",
    page_icon="💹",
    initial_sidebar_state="expanded"
)

# --------------------------
# Custom Styling
# --------------------------
st.markdown("""
<style>
    :root {
        --primary: #2a3f5f;
        --secondary: #4e79a7;
        --accent: #ff7f0e;
        --background: #f9f9f9;
        --card: #ffffff;
    }
    
    .main {
        background-color: var(--background);
    }
    
    .stApp {
        background-image: linear-gradient(to bottom, #f0f2f6, #ffffff);
    }
    
    .sidebar .sidebar-content {
        background-image: linear-gradient(to bottom, #2a3f5f, #4e79a7);
        color: white;
    }
    
    h1, h2, h3 {
        color: var(--primary);
        font-family: 'Arial', sans-serif;
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: var(--accent);
        transform: scale(1.02);
    }
    
    .metric-card {
        background-color: var(--card);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .progress-container {
        margin: 30px 0;
    }
    
    .progress-step {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .progress-icon {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background-color: #e0e0e0;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-weight: bold;
    }
    
    .progress-icon.completed {
        background-color: var(--accent);
        color: white;
    }
    
    .progress-text {
        flex-grow: 1;
    }
    
    .progress-text.completed {
        color: var(--primary);
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Initialize Session State
# --------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'feature_engineering_done' not in st.session_state:
    st.session_state.feature_engineering_done = False
if 'train_test_split_done' not in st.session_state:
    st.session_state.train_test_split_done = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'feature_description' not in st.session_state:
    st.session_state.feature_description = ""

# --------------------------
# Header Section
# --------------------------
st.markdown("""
<div style="background: linear-gradient(to right, #2a3f5f, #4e79a7); 
            padding: 30px; 
            border-radius: 10px; 
            color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 30px;">
    <h1 style="color: white; text-align: center; margin: 0;">💰 FINML Pro Dashboard</h1>
    <p style="text-align: center; margin: 10px 0 0; font-size: 16px;">
        Advanced Financial Machine Learning Pipeline with Interactive Visualizations
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Sidebar - Data Loading
# --------------------------
with st.sidebar:
    st.markdown("""
    <div style="color: white; margin-bottom: 30px;">
        <h2 style="color: white;">📂 Data Source</h2>
        <p>Load your financial data to begin analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    data_source = st.radio("Select Data Source:", 
                         ["Fetch from Yahoo Finance", "Upload CSV"],
                         label_visibility="collapsed")
    
    if data_source == "Fetch from Yahoo Finance":
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime.today())
        
        if st.button("📊 Fetch Market Data", key="fetch_data"):
            with st.spinner("Downloading market data..."):
                try:
                    df = yf.download(ticker, start=start_date, end=end_date)
                    if not df.empty:
                        st.session_state.df = df
                        st.session_state.feature_engineering_done = False
                        st.session_state.train_test_split_done = False
                        st.session_state.model_trained = False
                        st.session_state.feature_description = ""
                        st.success(f"✅ Successfully fetched {ticker} data!")
                    else:
                        st.error("No data returned. Check ticker and date range.")
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
    
    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.feature_engineering_done = False
                st.session_state.train_test_split_done = False
                st.session_state.model_trained = False
                st.session_state.feature_description = ""
                st.success("✅ File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

# --------------------------
# Pipeline Progress Tracker
# --------------------------
st.markdown("### 🔧 Pipeline Progress")
progress_cols = st.columns(5)

with progress_cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="progress-step">
            <div class="progress-icon {'completed' if st.session_state.df is not None else ''}">1</div>
            <div class="progress-text {'completed' if st.session_state.df is not None else ''}">Data Loaded</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with progress_cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="progress-step">
            <div class="progress-icon {'completed' if st.session_state.feature_engineering_done else ''}">2</div>
            <div class="progress-text {'completed' if st.session_state.feature_engineering_done else ''}">Feature Engineering</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with progress_cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="progress-step">
            <div class="progress-icon {'completed' if st.session_state.train_test_split_done else ''}">3</div>
            <div class="progress-text {'completed' if st.session_state.train_test_split_done else ''}">Train/Test Split</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with progress_cols[3]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="progress-step">
            <div class="progress-icon {'completed' if st.session_state.model_trained else ''}">4</div>
            <div class="progress-text {'completed' if st.session_state.model_trained else ''}">Model Training</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with progress_cols[4]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="progress-step">
            <div class="progress-icon">5</div>
            <div class="progress-text">Evaluation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------
# Main Content Area
# --------------------------
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    
    # Data Preview Section
    with st.expander("📊 Data Preview (Click to Expand)", expanded=True):
        st.dataframe(df.style.background_gradient(cmap='Blues'), height=300)
        
        # Basic Statistics
        st.markdown("### 📈 Basic Statistics")
        stats_cols = st.columns(4)
        stats_cols[0].metric("Total Records", len(df))
        stats_cols[1].metric("Columns", len(df.columns))
        stats_cols[2].metric("Start Date", df.index.min().strftime('%Y-%m-%d') if hasattr(df.index, 'strftime') else "N/A")
        stats_cols[3].metric("End Date", df.index.max().strftime('%Y-%m-%d') if hasattr(df.index, 'strftime') else "N/A")
        
        # Quick Visualization
        st.markdown("### 📉 Price Trend")
        selected_column = st.selectbox("Select column to visualize:", df.select_dtypes(include=np.number).columns)
        fig = px.line(df, y=selected_column, title=f"{selected_column} Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------
    # Step 1: Preprocessing
    # --------------------------
    st.markdown("---")
    st.markdown("## 🧹 Step 1: Data Preprocessing")
    
    if st.button("Run Data Preprocessing", key="preprocess"):
        with st.spinner("Processing data..."):
            missing_before = df.isnull().sum().sum()
            df.dropna(inplace=True)
            missing_after = df.isnull().sum().sum()
            st.session_state.df = df
            
            st.success(f"✅ Removed {missing_before - missing_after} missing values!")
            st.session_state.feature_engineering_done = False  # Reset downstream steps
            st.session_state.feature_description = ""
    
    # --------------------------
    # Step 2: Feature Engineering
    # --------------------------
    st.markdown("---")
    st.markdown("## 🛠️ Step 2: Feature Engineering")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox(
            "Select numeric column for feature engineering:", 
            numeric_cols,
            key='feature_col_select'
        )
        
        calculation_type = st.radio(
            "Select calculation type:",
            ["Daily Returns", "Daily Differences", "Log Returns"],
            horizontal=True
        )
        
        if st.button("Calculate Features", key="calc_features"):
            with st.spinner("Calculating features..."):
                try:
                    # Perform selected calculation
                    if calculation_type == "Daily Returns":
                        df['Engineered_Feature'] = df[selected_col].pct_change()
                    elif calculation_type == "Daily Differences":
                        df['Engineered_Feature'] = df[selected_col].diff()
                    elif calculation_type == "Log Returns":
                        df['Engineered_Feature'] = np.log(df[selected_col] / df[selected_col].shift(1))
                    
                    # Clean results
                    initial_count = len(df)
                    df.dropna(subset=['Engineered_Feature'], inplace=True)
                    final_count = len(df)
                    
                    # Update session state
                    st.session_state.df = df
                    st.session_state.feature_engineering_done = True
                    st.session_state.feature_description = f"{calculation_type} of {selected_col}"
                    
                    # Reset downstream states
                    st.session_state.train_test_split_done = False
                    st.session_state.model_trained = False
                    
                    # Show results
                    st.success(f"""
                    ✅ {calculation_type} calculated successfully from {selected_col}!
                    - Removed {initial_count - final_count} NA values
                    - New feature column: 'Engineered_Feature'
                    """)
                    
                    # Visualize results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Feature Statistics:**")
                        st.dataframe(df['Engineered_Feature'].describe().to_frame().T)
                    
                    with col2:
                        st.markdown("**Feature Distribution:**")
                        fig_dist = px.histogram(df, x='Engineered_Feature', nbins=50)
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    st.markdown("**Feature Over Time:**")
                    fig_line = px.line(df, y='Engineered_Feature')
                    st.plotly_chart(fig_line, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error in calculation: {str(e)}")
    else:
        st.warning("No numeric columns found for feature engineering.")

    # --------------------------
    # Step 3: Train/Test Split
    # --------------------------
    st.markdown("---")
    st.markdown("## ✂️ Step 3: Train/Test Split")
    
    if st.button("Split Data", disabled=not st.session_state.feature_engineering_done, key="split"):
        if 'Engineered_Feature' in df.columns:
            with st.spinner("Splitting data..."):
                df['Prev_Feature'] = df['Engineered_Feature'].shift(1)
                df.dropna(inplace=True)
                
                X = df[['Prev_Feature']]
                y = df['Engineered_Feature']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                st.session_state.update({
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'train_test_split_done': True,
                    'model_trained': False
                })
                
                st.success("✅ Data split into training and testing sets!")
                
                # Visualize split
                st.markdown("### 📊 Data Split Visualization")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=X_train.index, y=y_train,
                    mode='lines',
                    name='Training Data',
                    line=dict(color='#2a3f5f')
                ))
                fig.add_trace(go.Scatter(
                    x=X_test.index, y=y_test,
                    mode='lines',
                    name='Test Data',
                    line=dict(color='#ff7f0e')
                ))
                fig.update_layout(
                    title="Train/Test Split Timeline",
                    xaxis_title="Date",
                    yaxis_title="Feature Value"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
                st.write(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        else:
            st.warning("Please complete feature engineering first.")

    # --------------------------
    # Step 4: Model Training
    # --------------------------
    st.markdown("---")
    st.markdown("## 🧠 Step 4: Model Training")
    
    if st.button("Train Model", disabled=not st.session_state.train_test_split_done, key="train"):
        with st.spinner("Training model..."):
            model = LinearRegression()
            model.fit(st.session_state.X_train, st.session_state.y_train)
            
            st.session_state.model = model
            st.session_state.model_trained = True
            
            st.success("✅ Model trained successfully!")
            
            # Display model coefficients
            st.markdown("### 📐 Model Coefficients")
            coef_col1, coef_col2 = st.columns(2)
            coef_col1.metric("Intercept", f"{model.intercept_:.6f}")
            coef_col2.metric("Coefficient", f"{model.coef_[0]:.6f}")
            
            # Training performance
            train_pred = model.predict(st.session_state.X_train)
            train_r2 = r2_score(st.session_state.y_train, train_pred)
            st.markdown(f"**Training R² Score:** {train_r2:.4f}")

    # --------------------------
    # Step 5: Model Evaluation
    # --------------------------
    st.markdown("---")
    st.markdown("## 📊 Step 5: Model Evaluation")
    
    if st.button("Evaluate Model", disabled=not st.session_state.model_trained, key="evaluate"):
        if all(key in st.session_state for key in ['model', 'X_test', 'y_test']):
            with st.spinner("Evaluating model..."):
                model = st.session_state.model
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                predictions = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                corr = np.corrcoef(y_test, predictions)[0,1]
                
                # Metrics display
                st.markdown(f"### 📈 Evaluating: {st.session_state.feature_description}")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Mean Squared Error", f"{mse:.6f}")
                metric_col2.metric("R² Score", f"{r2:.4f}")
                metric_col3.metric("Correlation", f"{corr:.4f}")
                
                # Visualization tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Predictions Timeline", 
                    "🔍 Actual vs Predicted", 
                    "📊 Error Analysis",
                    "💰 Cumulative Values"
                ])
                
                with tab1:
                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Scatter(
                        x=y_test.index, y=y_test,
                        name='Actual Values',
                        line=dict(color='#2a3f5f')
                    ))
                    fig_timeline.add_trace(go.Scatter(
                        x=y_test.index, y=predictions,
                        name='Predicted Values',
                        line=dict(color='#ff7f0e')
                    ))
                    fig_timeline.update_layout(
                        title="Actual vs Predicted Values Over Time",
                        xaxis_title="Date",
                        yaxis_title="Feature Value",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
                
                with tab2:
                    fig_scatter = px.scatter(
                        x=y_test, y=predictions,
                        trendline="ols",
                        title="Actual vs Predicted Values",
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        color_discrete_sequence=['#4e79a7']
                    )
                    fig_scatter.add_shape(
                        type="line", line=dict(dash='dash'),
                        x0=y_test.min(), y0=y_test.min(),
                        x1=y_test.max(), y1=y_test.max()
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with tab3:
                    errors = y_test - predictions
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_hist = px.histogram(
                            x=errors, 
                            title="Prediction Error Distribution",
                            labels={'x': 'Prediction Error'},
                            color_discrete_sequence=['#cc3300'],
                            nbins=50
                        )
                        fig_hist.add_vline(x=0, line_dash="dash")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        fig_box = px.box(
                            y=errors,
                            title="Error Distribution Summary",
                            labels={'y': 'Prediction Error'},
                            color_discrete_sequence=['#cc3300']
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                
                with tab4:
                    cumulative_actual = (y_test + 1).cumprod() - 1
                    cumulative_pred = (pd.Series(predictions, index=y_test.index) + 1).cumprod() - 1
                    
                    fig_cumulative = go.Figure()
                    fig_cumulative.add_trace(go.Scatter(
                        x=cumulative_actual.index,
                        y=cumulative_actual,
                        name='Actual Cumulative',
                        line=dict(color='#2a3f5f', width=3)
                    ))
                    fig_cumulative.add_trace(go.Scatter(
                        x=cumulative_pred.index,
                        y=cumulative_pred,
                        name='Predicted Cumulative',
                        line=dict(color='#ff7f0e', width=3)
                    ))
                    fig_cumulative.update_layout(
                        title="Cumulative Values Comparison",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Value",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_cumulative, use_container_width=True)
        else:
            st.warning("Please complete all previous steps first.")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px; margin-top: 50px;">
    <p>Developed  by  Ibrahim Aziz</p>
    <p>© 2023 Financial Machine Learning Dashboard</p>
</div>
""", unsafe_allow_html=True)
