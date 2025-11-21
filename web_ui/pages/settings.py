"""Settings and configuration page"""

import streamlit as st
import json
from pathlib import Path


def show():
    """Display settings page"""

    st.header("⚙️ Settings")

    # Session state for settings
    if 'app_settings' not in st.session_state:
        st.session_state.app_settings = load_default_settings()

    settings = st.session_state.app_settings

    # General Settings
    st.subheader("🎨 General Settings")

    col1, col2 = st.columns(2)

    with col1:
        settings['theme'] = st.selectbox(
            "Theme",
            ['Light', 'Dark', 'Auto'],
            index=['Light', 'Dark', 'Auto'].index(settings.get('theme', 'Auto'))
        )

    with col2:
        settings['auto_save'] = st.checkbox(
            "Auto-save configurations",
            value=settings.get('auto_save', True)
        )

    st.markdown("---")

    # Paths Configuration
    st.subheader("📂 Paths Configuration")

    settings['checkpoint_dir'] = st.text_input(
        "Checkpoints Directory",
        value=settings.get('checkpoint_dir', 'checkpoints'),
        help="Directory to save model checkpoints"
    )

    settings['data_dir'] = st.text_input(
        "Default Data Directory",
        value=settings.get('data_dir', 'data'),
        help="Default directory for datasets"
    )

    settings['results_dir'] = st.text_input(
        "Results Directory",
        value=settings.get('results_dir', 'results'),
        help="Directory to save training results"
    )

    st.markdown("---")

    # Training Defaults
    st.subheader("🚀 Training Defaults")

    col1, col2, col3 = st.columns(3)

    with col1:
        settings['default_device'] = st.selectbox(
            "Default Device",
            ['auto', 'cuda', 'cpu'],
            index=['auto', 'cuda', 'cpu'].index(settings.get('default_device', 'auto'))
        )

    with col2:
        settings['default_batch_size'] = st.selectbox(
            "Default Batch Size",
            [16, 32, 64, 128, 256],
            index=[16, 32, 64, 128, 256].index(settings.get('default_batch_size', 32))
        )

    with col3:
        settings['default_epochs'] = st.number_input(
            "Default Epochs",
            min_value=10,
            max_value=1000,
            value=settings.get('default_epochs', 100),
            step=10
        )

    st.markdown("---")

    # Usage Analytics Settings
    st.subheader("📊 Usage Analytics")

    col1, col2 = st.columns(2)

    with col1:
        enable_analytics = st.checkbox(
            "Enable usage tracking",
            value=settings.get('enable_analytics', True),
            help="Track feature usage across projects"
        )
        settings['enable_analytics'] = enable_analytics

    with col2:
        if enable_analytics:
            if st.button("Clear Usage Statistics"):
                try:
                    from core import clear_usage_stats
                    clear_usage_stats()
                    st.success("✅ Usage statistics cleared!")
                except Exception as e:
                    st.error(f"Error clearing stats: {e}")

    st.markdown("---")

    # UI Preferences
    st.subheader("🎛️ UI Preferences")

    col1, col2 = st.columns(2)

    with col1:
        settings['show_code_snippets'] = st.checkbox(
            "Show code snippets",
            value=settings.get('show_code_snippets', True),
            help="Display code examples in the UI"
        )

    with col2:
        settings['detailed_errors'] = st.checkbox(
            "Show detailed errors",
            value=settings.get('detailed_errors', True),
            help="Display full error tracebacks"
        )

    settings['max_data_preview'] = st.slider(
        "Max rows in data preview",
        min_value=10,
        max_value=1000,
        value=settings.get('max_data_preview', 100),
        step=10
    )

    st.markdown("---")

    # Export/Import Settings
    st.subheader("💾 Settings Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Save Settings"):
            save_settings(settings)
            st.success("✅ Settings saved!")

    with col2:
        settings_json = json.dumps(settings, indent=2)
        st.download_button(
            label="📥 Export Settings",
            data=settings_json,
            file_name="app_settings.json",
            mime="application/json"
        )

    with col3:
        uploaded_settings = st.file_uploader(
            "Import Settings",
            type=['json'],
            key='settings_upload'
        )

        if uploaded_settings is not None:
            try:
                loaded_settings = json.load(uploaded_settings)
                st.session_state.app_settings = loaded_settings
                st.success("✅ Settings imported!")
                st.rerun()
            except Exception as e:
                st.error(f"Error importing settings: {e}")

    st.markdown("---")

    # User Management
    st.subheader("👤 User Management")

    from auth import show_user_management
    show_user_management()

    st.markdown("---")

    # System Information
    st.subheader("ℹ️ System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Framework Version:**")
        try:
            from core import __version__
            st.text(__version__)
        except ImportError:
            st.text("Unknown")

    with col2:
        st.markdown("**Python Version:**")
        import sys
        st.text(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    st.markdown("**PyTorch:**")
    try:
        import torch
        st.text(f"Version: {torch.__version__}")
        st.text(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.text(f"CUDA Version: {torch.version.cuda}")
            st.text(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        st.text("Not installed")

    st.markdown("---")

    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        st.markdown("### Developer Options")

        settings['debug_mode'] = st.checkbox(
            "Debug mode",
            value=settings.get('debug_mode', False)
        )

        settings['cache_size'] = st.number_input(
            "Cache size (MB)",
            min_value=100,
            max_value=10000,
            value=settings.get('cache_size', 1000),
            step=100
        )

        settings['log_level'] = st.selectbox(
            "Log level",
            ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            index=['DEBUG', 'INFO', 'WARNING', 'ERROR'].index(settings.get('log_level', 'INFO'))
        )

    # Reset to defaults
    st.markdown("---")
    if st.button("🔄 Reset All Settings to Defaults", type="secondary"):
        st.session_state.app_settings = load_default_settings()
        st.success("✅ Settings reset to defaults!")
        st.rerun()


def load_default_settings():
    """Load default settings"""
    return {
        'theme': 'Auto',
        'auto_save': True,
        'checkpoint_dir': 'checkpoints',
        'data_dir': 'data',
        'results_dir': 'results',
        'default_device': 'auto',
        'default_batch_size': 32,
        'default_epochs': 100,
        'enable_analytics': True,
        'show_code_snippets': True,
        'detailed_errors': True,
        'max_data_preview': 100,
        'debug_mode': False,
        'cache_size': 1000,
        'log_level': 'INFO',
    }


def save_settings(settings):
    """Save settings to file"""
    try:
        settings_dir = Path.home() / '.deep-timeseries'
        settings_dir.mkdir(exist_ok=True)

        settings_file = settings_dir / 'ui_settings.json'

        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)

    except Exception as e:
        st.error(f"Error saving settings: {e}")
