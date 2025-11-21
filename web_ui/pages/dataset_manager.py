"""Dataset upload and management page"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def show():
    """Display dataset manager page"""

    st.header("📊 Dataset Manager")

    # Session state initialization
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None

    # File upload section
    st.subheader("Upload Dataset")

    upload_method = st.radio(
        "Choose upload method:",
        ["Upload File", "Load Example", "Load from Path"]
    )

    if upload_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your time-series data file"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state.dataset = df
                st.session_state.dataset_name = uploaded_file.name
                st.success(f"✅ Loaded {uploaded_file.name} successfully!")

            except Exception as e:
                st.error(f"Error loading file: {e}")

    elif upload_method == "Load Example":
        example = st.selectbox(
            "Select example dataset:",
            ["Synthetic Stock Data", "Synthetic Sensor Data"]
        )

        if st.button("Load Example"):
            if example == "Synthetic Stock Data":
                dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
                np.random.seed(42)
                trend = np.linspace(100, 200, len(dates))
                noise = np.random.normal(0, 10, len(dates))
                df = pd.DataFrame({
                    'Date': dates,
                    'Close': trend + noise,
                    'Volume': np.random.randint(1000000, 10000000, len(dates)),
                })
            else:
                dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
                np.random.seed(42)
                df = pd.DataFrame({
                    'Timestamp': dates,
                    'Temperature': 20 + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.normal(0, 2, len(dates)),
                    'Humidity': 50 + 10 * np.cos(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.normal(0, 5, len(dates)),
                })

            st.session_state.dataset = df
            st.session_state.dataset_name = example
            st.success(f"✅ Loaded {example}!")

    else:  # Load from Path
        file_path = st.text_input("Enter file path:")
        if st.button("Load") and file_path:
            try:
                path = Path(file_path)
                if path.suffix == '.csv':
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)

                st.session_state.dataset = df
                st.session_state.dataset_name = path.name
                st.success(f"✅ Loaded {path.name} successfully!")

            except Exception as e:
                st.error(f"Error loading file: {e}")

    # Display dataset if loaded
    if st.session_state.dataset is not None:
        df = st.session_state.dataset

        st.markdown("---")
        st.subheader(f"Dataset: {st.session_state.dataset_name}")

        # Dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            missing = df.isnull().sum().sum()
            st.metric("Missing Values", missing)
        with col4:
            memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory", f"{memory:.2f} MB")

        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(100), use_container_width=True)

        # Statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        # Column selection for visualization
        st.subheader("Data Visualization")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Try to auto-detect date column
        if not date_cols:
            for col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                    break
                except Exception:
                    pass

        if numeric_cols:
            col1, col2 = st.columns(2)

            with col1:
                x_col = st.selectbox("X-axis (time/index):", date_cols + ['Index'] if date_cols else ['Index'])
                y_cols = st.multiselect("Y-axis (values):", numeric_cols, default=numeric_cols[:1])

            with col2:
                chart_type = st.selectbox("Chart Type:", ["Line", "Scatter", "Histogram", "Box"])

            if y_cols:
                if x_col == 'Index':
                    x_data = df.index
                else:
                    x_data = df[x_col]

                if chart_type == "Line":
                    fig = go.Figure()
                    for col in y_cols:
                        fig.add_trace(go.Scatter(x=x_data, y=df[col], mode='lines', name=col))
                    fig.update_layout(
                        title="Time Series Plot",
                        xaxis_title=x_col,
                        yaxis_title="Value",
                        hovermode='x unified'
                    )

                elif chart_type == "Scatter":
                    if len(y_cols) >= 2:
                        fig = px.scatter(df, x=y_cols[0], y=y_cols[1], title="Scatter Plot")
                    else:
                        fig = px.scatter(df, x=x_data, y=y_cols[0], title="Scatter Plot")

                elif chart_type == "Histogram":
                    fig = go.Figure()
                    for col in y_cols:
                        fig.add_trace(go.Histogram(x=df[col], name=col, opacity=0.7))
                    fig.update_layout(title="Distribution", barmode='overlay')

                else:  # Box
                    fig = go.Figure()
                    for col in y_cols:
                        fig.add_trace(go.Box(y=df[col], name=col))
                    fig.update_layout(title="Box Plot")

                st.plotly_chart(fig, use_container_width=True)

        # Missing values analysis
        if df.isnull().sum().sum() > 0:
            st.subheader("Missing Values Analysis")

            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = px.bar(missing_df, x='Column', y='Missing Count', title="Missing Values by Column")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(missing_df, use_container_width=True)

            # Handling options
            st.subheader("Handle Missing Values")
            method = st.selectbox(
                "Select method:",
                ["None", "Forward Fill", "Backward Fill", "Drop Rows", "Fill with Zero", "Fill with Mean"]
            )

            if method != "None" and st.button("Apply"):
                if method == "Forward Fill":
                    st.session_state.dataset = df.fillna(method='ffill')
                elif method == "Backward Fill":
                    st.session_state.dataset = df.fillna(method='bfill')
                elif method == "Drop Rows":
                    st.session_state.dataset = df.dropna()
                elif method == "Fill with Zero":
                    st.session_state.dataset = df.fillna(0)
                elif method == "Fill with Mean":
                    st.session_state.dataset = df.fillna(df.mean())

                st.success(f"✅ Applied {method}")
                st.rerun()

        # Export options
        st.markdown("---")
        st.subheader("Export Dataset")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save to Session"):
                st.success("✅ Dataset saved to session (available for model training)")

        with col2:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
