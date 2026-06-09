"""Training page with real-time monitoring"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def show():
    """Display training page"""

    st.header("🚀 Model Training")

    # Check prerequisites
    if "dataset" not in st.session_state or st.session_state.dataset is None:
        st.warning("⚠️ No dataset loaded. Please go to Dataset Manager first.")
        return

    if "model_config" not in st.session_state:
        st.warning("⚠️ No model configuration found. Please configure model in Model Builder first.")
        return

    # Session state initialization
    if "training_history" not in st.session_state:
        st.session_state.training_history = None
    if "is_training" not in st.session_state:
        st.session_state.is_training = False
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None

    config = st.session_state.model_config
    df = st.session_state.dataset

    # Pre-training setup
    st.subheader("📋 Training Setup")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Size", f"{len(df):,} rows")

    with col2:
        st.metric("Model Type", config["model"]["type"].upper())

    with col3:
        st.metric("Batch Size", config["data"]["batch_size"])

    # Target column selection
    st.markdown("### Select Target Column")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found in dataset!")
        return

    target_col = st.selectbox(
        "Target column to predict:", numeric_cols, help="Select the column you want to predict"
    )

    feature_cols = st.multiselect(
        "Feature columns (optional - leave empty to use all):",
        [col for col in numeric_cols if col != target_col],
        help="Select specific features, or leave empty to use all columns",
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
                "Available checkpoints:", checkpoints, format_func=lambda x: x.name
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


class _EpochProgressCallback:
    """Adapt Trainer's on_epoch_end into an arbitrary per-epoch callback."""

    def __init__(self, on_update):
        self.on_update = on_update

    def on_epoch_end(self, epoch, trainer, train_results, val_results=None):
        train_loss = train_results.get("loss") if train_results else None
        val_loss = val_results.get("loss") if val_results else None
        self.on_update(epoch, train_loss, val_loss)
        return False


def _build_model(config, input_dim):
    """Construct the configured model from a config dict (no Streamlit)."""
    from core import GRUModel, LSTMModel, TransformerModel

    model_type = config["model"]["type"]
    params = {
        "input_dim": input_dim,
        "hidden_dim": config["model"]["hidden_dim"],
        "output_dim": config["model"]["output_dim"],
        "num_layers": config["model"]["num_layers"],
        "dropout": config["model"]["dropout"],
        "forecast_horizon": config["data"]["forecast_horizon"],
    }
    if model_type == "lstm":
        params["bidirectional"] = config["model"].get("bidirectional", False)
        return LSTMModel(**params)
    if model_type == "gru":
        params["bidirectional"] = config["model"].get("bidirectional", False)
        return GRUModel(**params)
    # transformer — the constructor parameter is dim_feedforward, not feedforward_dim
    params["num_heads"] = config["model"].get("num_heads", 4)
    params["dim_feedforward"] = config["model"].get("feedforward_dim", 256)
    return TransformerModel(**params)


def _build_optimizer(config, model):
    """Construct the configured optimizer."""
    from torch.optim import SGD, Adam, AdamW, RMSprop

    optimizers = {"adam": Adam, "sgd": SGD, "adamw": AdamW, "rmsprop": RMSprop}
    optimizer_cls = optimizers.get(config["training"]["optimizer"], Adam)
    return optimizer_cls(model.parameters(), lr=config["training"]["learning_rate"])


def run_training(df, target_col, feature_cols, config, progress_callback=None):
    """
    Run real training end-to-end and return ``(model, history)``.

    Free of Streamlit so it can be unit-tested. Uses a chronological (non-shuffled)
    train/validation split and fits the scaler on the training slice only, so there
    is no lookahead leakage. Pass ``progress_callback(epoch, train_loss, val_loss)``
    to receive per-epoch updates.
    """
    import torch.nn as nn

    from core import FeatureEngine, TimeSeriesDataLoader, TimeSeriesDataset, Trainer
    from core.training import EarlyStopping, ModelCheckpoint
    from core.utils import get_device, set_seed

    set_seed(42)

    if feature_cols:
        feature_data = df[feature_cols].values
    else:
        feature_data = df.select_dtypes(include=[np.number]).values
    target_data = df[target_col].values

    # Chronological split; scaler fit on train only (no leakage)
    feature_engine = FeatureEngine(
        transformers=[],
        scaler=config["features"]["scaler"],
        handle_missing=config["features"]["handle_missing"],
    )
    train_size = int(len(feature_data) * (1 - config["data"]["test_size"]))
    train_features = feature_engine.fit_transform(feature_data[:train_size])
    test_features = feature_engine.transform(feature_data[train_size:])

    seq = config["data"]["sequence_length"]
    horizon = config["data"]["forecast_horizon"]
    batch = config["data"]["batch_size"]

    train_dataset = TimeSeriesDataset(
        data=train_features,
        targets=target_data[:train_size],
        sequence_length=seq,
        forecast_horizon=horizon,
    )
    # The holdout may be too small to form even one window; train without a
    # validation set in that case rather than crashing.
    try:
        test_dataset = TimeSeriesDataset(
            data=test_features,
            targets=target_data[train_size:],
            sequence_length=seq,
            forecast_horizon=horizon,
        )
        has_val = len(test_dataset) > 0
    except ValueError:
        has_val = False

    train_loader = TimeSeriesDataLoader(train_dataset, batch_size=batch, shuffle=False)
    val_loader = (
        TimeSeriesDataLoader(test_dataset, batch_size=batch, shuffle=False) if has_val else None
    )

    model = _build_model(config, train_dataset.n_features)
    optimizer = _build_optimizer(config, model)

    callbacks = []
    early = config["training"].get("early_stopping", {"enabled": False})
    if early.get("enabled"):
        callbacks.append(
            EarlyStopping(
                patience=early.get("patience", 10),
                min_delta=early.get("min_delta", 0.0001),
            )
        )
    Path("checkpoints").mkdir(exist_ok=True)
    callbacks.append(ModelCheckpoint(filepath="checkpoints/best_model.pt", verbose=False))
    if progress_callback is not None:
        callbacks.append(_EpochProgressCallback(progress_callback))

    device = (
        get_device() if config["training"]["device"] == "auto" else config["training"]["device"]
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        device=device,
        callbacks=callbacks,
    )
    trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        verbose=False,
    )
    return model, trainer.history


def start_training(df, target_col, feature_cols, config):
    """Run real training and stream per-epoch progress into the Streamlit UI."""
    st.session_state.is_training = True

    st.info("🚀 Training started...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()

    history = {"train_loss": [], "val_loss": [], "epoch": []}
    epochs = config["training"]["epochs"]

    def on_epoch(epoch, train_loss, val_loss):
        # Fall back to train loss for display when there is no validation split
        shown_val = val_loss if val_loss is not None else train_loss
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(shown_val)

        progress_bar.progress(min((epoch + 1) / epochs, 1.0))
        status_text.text(f"Epoch {epoch + 1}/{epochs}")
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Epoch", f"{epoch + 1}/{epochs}")
            col2.metric("Train Loss", f"{train_loss:.4f}")
            col3.metric("Val Loss", f"{shown_val:.4f}")

        if epoch % 5 == 0 or epoch == epochs - 1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=history["epoch"], y=history["train_loss"], mode="lines", name="Train Loss"
                )
            )
            fig.add_trace(
                go.Scatter(x=history["epoch"], y=history["val_loss"], mode="lines", name="Val Loss")
            )
            fig.update_layout(
                title="Training Progress", xaxis_title="Epoch", yaxis_title="Loss", height=400
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

    with st.spinner("Preparing data and training model..."):
        try:
            model, _ = run_training(
                df, target_col, feature_cols, config, progress_callback=on_epoch
            )
            st.session_state.training_history = history
            st.session_state.trained_model = model
            st.success("✅ Training completed!")
        except Exception as e:
            st.error(f"Training error: {e}")
        finally:
            st.session_state.is_training = False


def show_training_progress(history):
    """Display training progress and metrics"""

    st.subheader("📊 Training Progress")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Epochs", len(history["epoch"]))

    with col2:
        final_train_loss = history["train_loss"][-1]
        st.metric("Final Train Loss", f"{final_train_loss:.4f}")

    with col3:
        final_val_loss = history["val_loss"][-1]
        st.metric("Final Val Loss", f"{final_val_loss:.4f}")

    with col4:
        best_val_loss = min(history["val_loss"])
        st.metric("Best Val Loss", f"{best_val_loss:.4f}")

    # Loss curves
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=history["train_loss"],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=history["val_loss"],
            mode="lines+markers",
            name="Validation Loss",
            line=dict(color="red"),
        )
    )

    fig.update_layout(
        title="Training & Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Download history
    history_df = pd.DataFrame(history)
    csv = history_df.to_csv(index=False)
    st.download_button(
        label="Download Training History",
        data=csv,
        file_name="training_history.csv",
        mime="text/csv",
    )
