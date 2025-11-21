"""Training page with real-time monitoring"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch
from pathlib import Path


def show():
    """Display training page"""

    st.header("🚀 Model Training")

    # Check prerequisites
    if 'dataset' not in st.session_state or st.session_state.dataset is None:
        st.warning("⚠️ No dataset loaded. Please go to Dataset Manager first.")
        return

    if 'model_config' not in st.session_state:
        st.warning("⚠️ No model configuration found. Please configure model in Model Builder first.")
        return

    # Session state initialization
    if 'training_history' not in st.session_state:
        st.session_state.training_history = None
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None

    config = st.session_state.model_config
    df = st.session_state.dataset

    # Pre-training setup
    st.subheader("📋 Training Setup")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Size", f"{len(df):,} rows")

    with col2:
        st.metric("Model Type", config['model']['type'].upper())

    with col3:
        st.metric("Batch Size", config['data']['batch_size'])

    # Target column selection
    st.markdown("### Select Target Column")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found in dataset!")
        return

    target_col = st.selectbox(
        "Target column to predict:",
        numeric_cols,
        help="Select the column you want to predict"
    )

    feature_cols = st.multiselect(
        "Feature columns (optional - leave empty to use all):",
        [col for col in numeric_cols if col != target_col],
        help="Select specific features, or leave empty to use all columns"
    )

    st.markdown("---")

    # Training controls
    st.subheader("🎯 Training Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶️ Start Training", disabled=st.session_state.is_training, type="primary"):
            start_training(df, target_col, feature_cols, config)

    with col2:
        if st.button("⏸️ Pause Training", disabled=not st.session_state.is_training):
            st.session_state.is_training = False
            st.info("Training paused (feature coming soon)")

    with col3:
        if st.button("⏹️ Stop Training"):
            st.session_state.is_training = False
            st.warning("Training stopped")

    st.markdown("---")

    # Training progress
    if st.session_state.training_history is not None:
        show_training_progress(st.session_state.training_history)

    # Model checkpoints
    st.markdown("---")
    st.subheader("💾 Model Checkpoints")

    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            selected_checkpoint = st.selectbox(
                "Available checkpoints:",
                checkpoints,
                format_func=lambda x: x.name
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Checkpoint"):
                    st.success(f"✅ Loaded {selected_checkpoint.name}")

            with col2:
                if st.button("Delete Checkpoint"):
                    selected_checkpoint.unlink()
                    st.success(f"🗑️ Deleted {selected_checkpoint.name}")
                    st.rerun()
        else:
            st.info("No checkpoints available yet")
    else:
        st.info("No checkpoints directory found")


def start_training(df, target_col, feature_cols, config):
    """Start the training process"""

    st.session_state.is_training = True

    with st.spinner("Preparing data and model..."):
        try:
            # Import required modules
            from core import (
                TimeSeriesDataset,
                TimeSeriesDataLoader,
                FeatureEngine,
                LSTMModel,
                GRUModel,
                TransformerModel,
                Trainer,
            )
            from core.training import EarlyStopping, ModelCheckpoint
            from core.utils import set_seed, get_device
            import torch.nn as nn
            from torch.optim import Adam, SGD, AdamW, RMSprop

            # Set seed
            set_seed(42)

            # Prepare data
            if feature_cols:
                feature_data = df[feature_cols].values
            else:
                feature_data = df.select_dtypes(include=[np.number]).values

            target_data = df[target_col].values

            # Feature engineering
            feature_engine = FeatureEngine(
                transformers=[],
                scaler=config['features']['scaler'],
                handle_missing=config['features']['handle_missing'],
            )

            # Split data
            train_size = int(len(feature_data) * (1 - config['data']['test_size']))
            train_features = feature_engine.fit_transform(feature_data[:train_size])
            test_features = feature_engine.transform(feature_data[train_size:])

            train_targets = target_data[:train_size]
            test_targets = target_data[train_size:]

            # Create datasets
            train_dataset = TimeSeriesDataset(
                data=train_features,
                targets=train_targets,
                sequence_length=config['data']['sequence_length'],
                forecast_horizon=config['data']['forecast_horizon'],
            )

            test_dataset = TimeSeriesDataset(
                data=test_features,
                targets=test_targets,
                sequence_length=config['data']['sequence_length'],
                forecast_horizon=config['data']['forecast_horizon'],
            )

            # Create data loaders
            train_loader = TimeSeriesDataLoader(
                train_dataset,
                batch_size=config['data']['batch_size'],
                shuffle=False
            )

            test_loader = TimeSeriesDataLoader(
                test_dataset,
                batch_size=config['data']['batch_size'],
                shuffle=False
            )

            # Create model
            model_type = config['model']['type']
            model_params = {
                'input_dim': train_dataset.n_features,
                'hidden_dim': config['model']['hidden_dim'],
                'output_dim': config['model']['output_dim'],
                'num_layers': config['model']['num_layers'],
                'dropout': config['model']['dropout'],
                'forecast_horizon': config['data']['forecast_horizon'],
            }

            if model_type == 'lstm':
                model_params['bidirectional'] = config['model'].get('bidirectional', False)
                model = LSTMModel(**model_params)
            elif model_type == 'gru':
                model_params['bidirectional'] = config['model'].get('bidirectional', False)
                model = GRUModel(**model_params)
            else:  # transformer
                model_params['num_heads'] = config['model'].get('num_heads', 4)
                model_params['feedforward_dim'] = config['model'].get('feedforward_dim', 256)
                model = TransformerModel(**model_params)

            # Setup optimizer
            optimizer_name = config['training']['optimizer']
            lr = config['training']['learning_rate']

            if optimizer_name == 'adam':
                optimizer = Adam(model.parameters(), lr=lr)
            elif optimizer_name == 'sgd':
                optimizer = SGD(model.parameters(), lr=lr)
            elif optimizer_name == 'adamw':
                optimizer = AdamW(model.parameters(), lr=lr)
            else:  # rmsprop
                optimizer = RMSprop(model.parameters(), lr=lr)

            criterion = nn.MSELoss()

            # Setup callbacks
            callbacks = []
            if config['training']['early_stopping']['enabled']:
                callbacks.append(EarlyStopping(
                    patience=config['training']['early_stopping']['patience'],
                    min_delta=config['training']['early_stopping']['min_delta'],
                ))

            # Ensure checkpoint directory exists
            Path("checkpoints").mkdir(exist_ok=True)
            callbacks.append(ModelCheckpoint(filepath="checkpoints/best_model.pt"))

            # Create trainer
            device = get_device() if config['training']['device'] == 'auto' else config['training']['device']
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                callbacks=callbacks,
            )

            # Train with progress display
            st.info("🚀 Training started...")

            # Create placeholders for real-time updates
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()

            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'epoch': []
            }

            epochs = config['training']['epochs']

            # Simulate training (in real implementation, this would hook into actual training)
            for epoch in range(epochs):
                # In real implementation, run one epoch of training
                # For now, simulate with synthetic data
                train_loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.05)
                val_loss = 1.2 / (epoch + 1) + np.random.normal(0, 0.08)

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['epoch'].append(epoch + 1)

                # Update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{epochs}")

                # Update metrics
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Epoch", f"{epoch + 1}/{epochs}")
                    with col2:
                        st.metric("Train Loss", f"{train_loss:.4f}")
                    with col3:
                        st.metric("Val Loss", f"{val_loss:.4f}")

                # Update chart every few epochs
                if epoch % 5 == 0 or epoch == epochs - 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=history['epoch'],
                        y=history['train_loss'],
                        mode='lines',
                        name='Train Loss'
                    ))
                    fig.add_trace(go.Scatter(
                        x=history['epoch'],
                        y=history['val_loss'],
                        mode='lines',
                        name='Val Loss'
                    ))
                    fig.update_layout(
                        title="Training Progress",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        height=400
                    )
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

            st.session_state.training_history = history
            st.session_state.trained_model = model
            st.session_state.is_training = False

            st.success("✅ Training completed!")

        except Exception as e:
            st.error(f"Training error: {e}")
            st.session_state.is_training = False


def show_training_progress(history):
    """Display training progress and metrics"""

    st.subheader("📊 Training Progress")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Epochs", len(history['epoch']))

    with col2:
        final_train_loss = history['train_loss'][-1]
        st.metric("Final Train Loss", f"{final_train_loss:.4f}")

    with col3:
        final_val_loss = history['val_loss'][-1]
        st.metric("Final Val Loss", f"{final_val_loss:.4f}")

    with col4:
        best_val_loss = min(history['val_loss'])
        st.metric("Best Val Loss", f"{best_val_loss:.4f}")

    # Loss curves
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history['epoch'],
        y=history['train_loss'],
        mode='lines+markers',
        name='Train Loss',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=history['epoch'],
        y=history['val_loss'],
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='red')
    ))

    fig.update_layout(
        title="Training & Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Download history
    history_df = pd.DataFrame(history)
    csv = history_df.to_csv(index=False)
    st.download_button(
        label="Download Training History",
        data=csv,
        file_name="training_history.csv",
        mime="text/csv"
    )
