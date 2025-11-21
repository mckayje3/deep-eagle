"""Visual model builder interface"""

import streamlit as st
import json
from pathlib import Path


def show():
    """Display model builder page"""

    st.header("🏗️ Model Builder")

    # Session state initialization
    if 'model_config' not in st.session_state:
        st.session_state.model_config = {
            'model': {
                'type': 'lstm',
                'hidden_dim': 64,
                'output_dim': 1,
                'num_layers': 2,
                'dropout': 0.1,
                'bidirectional': False,
            },
            'data': {
                'sequence_length': 30,
                'forecast_horizon': 1,
                'batch_size': 32,
                'test_size': 0.2,
            },
            'training': {
                'epochs': 100,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'device': 'auto',
            },
            'features': {
                'scaler': 'standard',
                'handle_missing': 'ffill',
            }
        }

    config = st.session_state.model_config

    # Model Architecture Section
    st.subheader("📐 Model Architecture")

    col1, col2 = st.columns([2, 1])

    with col1:
        model_type = st.selectbox(
            "Model Type",
            ['lstm', 'gru', 'transformer'],
            index=['lstm', 'gru', 'transformer'].index(config['model']['type']),
            help="Choose the neural network architecture"
        )
        config['model']['type'] = model_type

        # Show architecture diagram
        if model_type == 'lstm':
            st.info("""
            **LSTM (Long Short-Term Memory)**
            - Good for capturing long-term dependencies
            - Handles vanishing gradient problem
            - Best for: General time-series tasks
            """)
        elif model_type == 'gru':
            st.info("""
            **GRU (Gated Recurrent Unit)**
            - Simpler than LSTM, faster to train
            - Fewer parameters
            - Best for: When training speed is important
            """)
        else:
            st.info("""
            **Transformer**
            - Attention-based mechanism
            - Parallel processing
            - Best for: Complex patterns, long sequences
            """)

    with col2:
        st.markdown("### Quick Presets")
        if st.button("🚀 Fast Prototype"):
            config['model'].update({
                'hidden_dim': 32,
                'num_layers': 1,
                'dropout': 0.0,
            })
            st.success("Applied Fast Prototype preset")

        if st.button("⚖️ Balanced"):
            config['model'].update({
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.1,
            })
            st.success("Applied Balanced preset")

        if st.button("💪 High Capacity"):
            config['model'].update({
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.2,
            })
            st.success("Applied High Capacity preset")

    # Model Parameters
    st.markdown("### Model Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        config['model']['hidden_dim'] = st.slider(
            "Hidden Dimension",
            min_value=16,
            max_value=512,
            value=config['model']['hidden_dim'],
            step=16,
            help="Size of hidden layers (larger = more capacity)"
        )

    with col2:
        config['model']['num_layers'] = st.slider(
            "Number of Layers",
            min_value=1,
            max_value=5,
            value=config['model']['num_layers'],
            help="Depth of the network"
        )

    with col3:
        config['model']['dropout'] = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=0.5,
            value=config['model']['dropout'],
            step=0.05,
            help="Dropout for regularization (prevents overfitting)"
        )

    # Additional model-specific parameters
    if model_type in ['lstm', 'gru']:
        config['model']['bidirectional'] = st.checkbox(
            "Bidirectional",
            value=config['model'].get('bidirectional', False),
            help="Process sequences in both directions"
        )
    elif model_type == 'transformer':
        col1, col2 = st.columns(2)
        with col1:
            config['model']['num_heads'] = st.slider(
                "Attention Heads",
                min_value=1,
                max_value=8,
                value=config['model'].get('num_heads', 4),
                help="Number of attention heads"
            )
        with col2:
            config['model']['feedforward_dim'] = st.slider(
                "Feedforward Dimension",
                min_value=64,
                max_value=512,
                value=config['model'].get('feedforward_dim', 256),
                step=64
            )

    # Output dimension
    config['model']['output_dim'] = st.number_input(
        "Output Dimension",
        min_value=1,
        max_value=10,
        value=config['model']['output_dim'],
        help="Number of values to predict"
    )

    st.markdown("---")

    # Data Configuration
    st.subheader("📊 Data Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        config['data']['sequence_length'] = st.number_input(
            "Sequence Length",
            min_value=5,
            max_value=200,
            value=config['data']['sequence_length'],
            help="Number of past time steps to look at"
        )

    with col2:
        config['data']['forecast_horizon'] = st.number_input(
            "Forecast Horizon",
            min_value=1,
            max_value=50,
            value=config['data']['forecast_horizon'],
            help="Number of future steps to predict"
        )

    with col3:
        config['data']['batch_size'] = st.selectbox(
            "Batch Size",
            [16, 32, 64, 128, 256],
            index=[16, 32, 64, 128, 256].index(config['data']['batch_size']),
            help="Number of samples per training batch"
        )

    with col4:
        config['data']['test_size'] = st.slider(
            "Test Split %",
            min_value=0.1,
            max_value=0.4,
            value=config['data']['test_size'],
            step=0.05,
            help="Percentage of data for testing"
        )

    st.markdown("---")

    # Training Configuration
    st.subheader("🚀 Training Configuration")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        config['training']['epochs'] = st.number_input(
            "Epochs",
            min_value=10,
            max_value=1000,
            value=config['training']['epochs'],
            step=10,
            help="Number of training epochs"
        )

    with col2:
        config['training']['learning_rate'] = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=config['training']['learning_rate'],
            help="Optimizer learning rate"
        )

    with col3:
        config['training']['optimizer'] = st.selectbox(
            "Optimizer",
            ['adam', 'sgd', 'adamw', 'rmsprop'],
            index=['adam', 'sgd', 'adamw', 'rmsprop'].index(config['training']['optimizer'])
        )

    with col4:
        config['training']['device'] = st.selectbox(
            "Device",
            ['auto', 'cuda', 'cpu'],
            index=['auto', 'cuda', 'cpu'].index(config['training']['device']),
            help="Training device (auto detects GPU)"
        )

    # Early stopping
    st.markdown("### Early Stopping")
    col1, col2, col3 = st.columns(3)

    with col1:
        early_stopping_enabled = st.checkbox(
            "Enable Early Stopping",
            value=True,
            help="Stop training when validation loss stops improving"
        )

    if early_stopping_enabled:
        with col2:
            patience = st.number_input(
                "Patience",
                min_value=5,
                max_value=50,
                value=10,
                help="Epochs to wait before stopping"
            )

        with col3:
            min_delta = st.number_input(
                "Min Delta",
                min_value=0.0,
                max_value=0.01,
                value=0.0001,
                step=0.0001,
                format="%.4f",
                help="Minimum improvement required"
            )

        config['training']['early_stopping'] = {
            'enabled': True,
            'patience': patience,
            'min_delta': min_delta,
        }
    else:
        config['training']['early_stopping'] = {'enabled': False}

    st.markdown("---")

    # Feature Engineering
    st.subheader("🔧 Feature Engineering")

    col1, col2 = st.columns(2)

    with col1:
        config['features']['scaler'] = st.selectbox(
            "Feature Scaler",
            ['standard', 'minmax', 'robust'],
            index=['standard', 'minmax', 'robust'].index(config['features']['scaler']),
            help="Method for scaling features"
        )

    with col2:
        config['features']['handle_missing'] = st.selectbox(
            "Handle Missing Values",
            ['ffill', 'bfill', 'drop', 'zero'],
            index=['ffill', 'bfill', 'drop', 'zero'].index(config['features']['handle_missing']),
            help="How to handle missing data"
        )

    # Technical indicators
    st.markdown("### Technical Indicators (Optional)")

    tech_indicators = st.checkbox("Enable Technical Indicators")

    if tech_indicators:
        col1, col2, col3 = st.columns(3)

        with col1:
            include_rsi = st.checkbox("RSI (Relative Strength Index)", value=True)

        with col2:
            include_macd = st.checkbox("MACD (Moving Average Convergence Divergence)", value=True)

        with col3:
            include_bollinger = st.checkbox("Bollinger Bands", value=True)

        config['features']['technical_indicators'] = {
            'enabled': True,
            'include_rsi': include_rsi,
            'include_macd': include_macd,
            'include_bollinger': include_bollinger,
        }
    else:
        config['features']['technical_indicators'] = {'enabled': False}

    st.markdown("---")

    # Model Summary
    st.subheader("📋 Configuration Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model:**")
        st.json({
            'type': config['model']['type'],
            'hidden_dim': config['model']['hidden_dim'],
            'num_layers': config['model']['num_layers'],
            'dropout': config['model']['dropout'],
        })

    with col2:
        st.markdown("**Training:**")
        st.json({
            'epochs': config['training']['epochs'],
            'learning_rate': config['training']['learning_rate'],
            'batch_size': config['data']['batch_size'],
            'optimizer': config['training']['optimizer'],
        })

    # Parameter estimation
    estimated_params = estimate_parameters(config)
    st.info(f"📊 Estimated Model Parameters: **{estimated_params:,}**")

    st.markdown("---")

    # Save/Load Configuration
    st.subheader("💾 Save/Load Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Save Configuration"):
            config_json = json.dumps(config, indent=2)
            st.download_button(
                label="Download config.json",
                data=config_json,
                file_name="model_config.json",
                mime="application/json"
            )

    with col2:
        uploaded_config = st.file_uploader("Load Configuration", type=['json'])
        if uploaded_config is not None:
            try:
                loaded_config = json.load(uploaded_config)
                st.session_state.model_config = loaded_config
                st.success("✅ Configuration loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading config: {e}")

    with col3:
        if st.button("Reset to Defaults"):
            # Will reset on next rerun
            del st.session_state.model_config
            st.success("✅ Reset to defaults!")
            st.rerun()


def estimate_parameters(config):
    """Estimate number of model parameters"""
    model_type = config['model']['type']
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model']['num_layers']

    # Rough estimation
    if model_type == 'lstm':
        # LSTM has 4 gates
        params_per_layer = 4 * (hidden_dim * hidden_dim + hidden_dim)
        total = params_per_layer * num_layers
    elif model_type == 'gru':
        # GRU has 3 gates
        params_per_layer = 3 * (hidden_dim * hidden_dim + hidden_dim)
        total = params_per_layer * num_layers
    else:  # transformer
        num_heads = config['model'].get('num_heads', 4)
        ff_dim = config['model'].get('feedforward_dim', 256)
        params_per_layer = (hidden_dim * hidden_dim * 4) + (hidden_dim * ff_dim * 2)
        total = params_per_layer * num_layers

    return int(total)
