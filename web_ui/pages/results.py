"""Results visualization and evaluation page"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


def show():
    """Display results and evaluation page"""

    st.header("📈 Results & Evaluation")

    if 'training_history' not in st.session_state or st.session_state.training_history is None:
        st.warning("⚠️ No training results available. Please train a model first.")
        return

    history = st.session_state.training_history

    # Model Performance Overview
    st.subheader("📊 Model Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        final_train = history['train_loss'][-1]
        st.metric("Final Train Loss", f"{final_train:.4f}")

    with col2:
        final_val = history['val_loss'][-1]
        st.metric("Final Val Loss", f"{final_val:.4f}")

    with col3:
        best_val = min(history['val_loss'])
        best_epoch = history['val_loss'].index(best_val) + 1
        st.metric("Best Val Loss", f"{best_val:.4f}", delta=f"Epoch {best_epoch}")

    with col4:
        improvement = (history['val_loss'][0] - history['val_loss'][-1]) / history['val_loss'][0] * 100
        st.metric("Improvement", f"{improvement:.1f}%")

    st.markdown("---")

    # Training Curves
    st.subheader("📉 Training Curves")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history['epoch'],
        y=history['train_loss'],
        mode='lines+markers',
        name='Train Loss',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=history['epoch'],
        y=history['val_loss'],
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=6)
    ))

    # Add best epoch marker
    best_val_idx = history['val_loss'].index(min(history['val_loss']))
    fig.add_trace(go.Scatter(
        x=[history['epoch'][best_val_idx]],
        y=[history['val_loss'][best_val_idx]],
        mode='markers',
        name='Best Epoch',
        marker=dict(size=15, color='red', symbol='star')
    ))

    fig.update_layout(
        title="Loss Over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        height=500,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Loss Distribution
    st.subheader("📊 Loss Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=history['train_loss'],
            name='Train Loss',
            opacity=0.7,
            marker_color='blue'
        ))
        fig_hist.add_trace(go.Histogram(
            x=history['val_loss'],
            name='Val Loss',
            opacity=0.7,
            marker_color='orange'
        ))
        fig_hist.update_layout(
            title="Loss Distribution",
            xaxis_title="Loss Value",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Calculate statistics
        train_stats = pd.DataFrame({
            'Metric': ['Mean', 'Std', 'Min', 'Max', 'Median'],
            'Train Loss': [
                np.mean(history['train_loss']),
                np.std(history['train_loss']),
                np.min(history['train_loss']),
                np.max(history['train_loss']),
                np.median(history['train_loss'])
            ],
            'Val Loss': [
                np.mean(history['val_loss']),
                np.std(history['val_loss']),
                np.min(history['val_loss']),
                np.max(history['val_loss']),
                np.median(history['val_loss'])
            ]
        })

        st.markdown("**Loss Statistics**")
        st.dataframe(train_stats.style.format({
            'Train Loss': '{:.4f}',
            'Val Loss': '{:.4f}'
        }), use_container_width=True)

    st.markdown("---")

    # Predictions vs Actual (Simulated)
    st.subheader("🎯 Predictions vs Actual")

    st.info("📝 Note: This is simulated data for demonstration. "
            "In a real scenario, this would show actual model predictions.")

    # Generate simulated predictions
    n_points = 100
    actual = np.sin(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 0.1, n_points)
    predicted = actual + np.random.normal(0, 0.15, n_points)

    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(
        x=list(range(n_points)),
        y=actual,
        mode='lines+markers',
        name='Actual',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ))

    fig_pred.add_trace(go.Scatter(
        x=list(range(n_points)),
        y=predicted,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='purple', width=2, dash='dash'),
        marker=dict(size=4)
    ))

    fig_pred.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Time Step",
        yaxis_title="Value",
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig_pred, use_container_width=True)

    # Error analysis
    col1, col2 = st.columns(2)

    with col1:
        errors = predicted - actual
        fig_error = go.Figure()
        fig_error.add_trace(go.Scatter(
            x=list(range(n_points)),
            y=errors,
            mode='lines',
            name='Prediction Error',
            line=dict(color='red')
        ))
        fig_error.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_error.update_layout(
            title="Prediction Errors Over Time",
            xaxis_title="Time Step",
            yaxis_title="Error",
            height=400
        )
        st.plotly_chart(fig_error, use_container_width=True)

    with col2:
        fig_scatter = px.scatter(
            x=actual,
            y=predicted,
            labels={'x': 'Actual', 'y': 'Predicted'},
            title="Actual vs Predicted Scatter"
        )
        # Add perfect prediction line
        min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='gray', dash='dash')
        ))
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Performance Metrics
    st.subheader("📐 Performance Metrics")

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / (actual + 1e-8))) * 100

    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE'],
        'Value': [mse, rmse, mae, mape],
        'Unit': ['squared', 'same as target', 'same as target', '%']
    })

    st.dataframe(metrics_df.style.format({'Value': '{:.4f}'}), use_container_width=True)

    st.markdown("---")

    # Export Results
    st.subheader("💾 Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export training history
        history_df = pd.DataFrame(history)
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Training History",
            data=csv,
            file_name="training_history.csv",
            mime="text/csv"
        )

    with col2:
        # Export predictions
        predictions_df = pd.DataFrame({
            'actual': actual,
            'predicted': predicted,
            'error': errors
        })
        pred_csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=pred_csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

    with col3:
        # Export metrics
        metrics_csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics",
            data=metrics_csv,
            file_name="metrics.csv",
            mime="text/csv"
        )
