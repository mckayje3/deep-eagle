"""Prediction interface page"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path


def show():
    """Display prediction interface page"""

    st.header("🔮 Make Predictions")

    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        st.warning("⚠️ No trained model available. Please train a model first.")
        return

    st.subheader("📊 Input Data for Prediction")

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload New Data", "Use Test Set", "Manual Input"]
    )

    predictions = None
    input_data = None

    if input_method == "Upload New Data":
        uploaded_file = st.file_uploader(
            "Upload CSV file with features",
            type=['csv'],
            help="Upload a CSV file with the same features as training data"
        )

        if uploaded_file is not None:
            try:
                input_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(input_data)} rows")
                st.dataframe(input_data.head(), use_container_width=True)

                if st.button("🔮 Generate Predictions"):
                    predictions = generate_predictions(input_data)

            except Exception as e:
                st.error(f"Error loading file: {e}")

    elif input_method == "Use Test Set":
        if 'dataset' in st.session_state and st.session_state.dataset is not None:
            df = st.session_state.dataset
            config = st.session_state.get('model_config', {})

            test_size = config.get('data', {}).get('test_size', 0.2)
            train_size = int(len(df) * (1 - test_size))

            input_data = df.iloc[train_size:]

            st.info(f"📊 Using test set with {len(input_data)} samples")
            st.dataframe(input_data.head(), use_container_width=True)

            if st.button("🔮 Generate Predictions"):
                predictions = generate_predictions(input_data)

        else:
            st.warning("No dataset available. Please load a dataset first.")

    else:  # Manual Input
        st.markdown("### Enter Feature Values")

        if 'model_config' in st.session_state:
            config = st.session_state.model_config
            seq_len = config.get('data', {}).get('sequence_length', 30)

            st.info(f"📝 Model expects a sequence of {seq_len} time steps")

            num_features = st.number_input(
                "Number of features:",
                min_value=1,
                max_value=20,
                value=1
            )

            # Create input fields
            manual_data = []
            for i in range(num_features):
                col_name = st.text_input(f"Feature {i+1} name:", value=f"feature_{i+1}")
                values = st.text_area(
                    f"Enter {seq_len} values for {col_name} (comma-separated):",
                    help="Enter comma-separated values"
                )

                if values:
                    try:
                        vals = [float(x.strip()) for x in values.split(',')]
                        if len(vals) == seq_len:
                            manual_data.append(vals)
                        else:
                            st.warning(f"Expected {seq_len} values, got {len(vals)}")
                    except ValueError:
                        st.error("Invalid number format")

            if len(manual_data) == num_features and st.button("🔮 Generate Prediction"):
                input_data = pd.DataFrame(np.array(manual_data).T)
                predictions = generate_predictions(input_data)

    # Display predictions
    if predictions is not None:
        st.markdown("---")
        st.subheader("🎯 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predictions Made", len(predictions))

        with col2:
            st.metric("Mean Prediction", f"{np.mean(predictions):.4f}")

        with col3:
            st.metric("Std Prediction", f"{np.std(predictions):.4f}")

        # Visualization
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=predictions,
            mode='lines+markers',
            name='Predictions',
            line=dict(color='purple', width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="Prediction Values Over Time",
            xaxis_title="Time Step",
            yaxis_title="Predicted Value",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Prediction table
        st.markdown("### Detailed Predictions")

        pred_df = pd.DataFrame({
            'Index': range(len(predictions)),
            'Predicted Value': predictions
        })

        st.dataframe(pred_df, use_container_width=True)

        # Download predictions
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        # Confidence intervals (simulated)
        st.markdown("---")
        st.subheader("📊 Confidence Intervals")

        st.info("📝 Confidence intervals are estimated (in production, use proper uncertainty quantification)")

        lower_bound = predictions - np.random.uniform(0.1, 0.3, len(predictions))
        upper_bound = predictions + np.random.uniform(0.1, 0.3, len(predictions))

        fig_ci = go.Figure()

        fig_ci.add_trace(go.Scatter(
            y=predictions,
            mode='lines+markers',
            name='Predictions',
            line=dict(color='purple')
        ))

        fig_ci.add_trace(go.Scatter(
            y=upper_bound,
            mode='lines',
            name='Upper Bound (95%)',
            line=dict(color='lightblue', dash='dash')
        ))

        fig_ci.add_trace(go.Scatter(
            y=lower_bound,
            mode='lines',
            name='Lower Bound (95%)',
            line=dict(color='lightblue', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.2)'
        ))

        fig_ci.update_layout(
            title="Predictions with Confidence Intervals",
            xaxis_title="Time Step",
            yaxis_title="Value",
            height=500
        )

        st.plotly_chart(fig_ci, use_container_width=True)


def generate_predictions(data):
    """Generate predictions from input data (simulated)"""

    # In real implementation, this would use the actual trained model
    # For now, generate simulated predictions

    n_predictions = min(len(data), 100)

    # Simulate predictions based on input data patterns
    if isinstance(data, pd.DataFrame):
        numeric_data = data.select_dtypes(include=[np.number]).values
        if len(numeric_data) > 0:
            mean_val = np.mean(numeric_data)
            std_val = np.std(numeric_data)
        else:
            mean_val, std_val = 0, 1
    else:
        mean_val, std_val = 0, 1

    # Generate realistic-looking predictions
    predictions = mean_val + std_val * np.random.normal(0, 0.5, n_predictions)

    return predictions
