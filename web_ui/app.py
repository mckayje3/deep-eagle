"""
Deep-TimeSeries Web UI
Main application entry point with navigation
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for local imports
parent_dir = Path(__file__).parent.parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

st.set_page_config(
    page_title="Deep-TimeSeries Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Authentication
from auth import require_authentication, logout

username = require_authentication()
if username is None:
    st.stop()

# Main title
st.title("📈 Deep-TimeSeries Dashboard")
st.markdown("*A visual interface for time-series deep learning*")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown(f"👤 **User:** {username}")
if st.sidebar.button("🚪 Logout"):
    logout()
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dataset Manager",
        "🏗️ Model Builder",
        "🚀 Training",
        "📈 Results & Evaluation",
        "🔮 Prediction",
        "🔍 Project Scanner",
        "⚙️ Settings",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Deep-TimeSeries is a modular PyTorch framework "
    "for time-series analysis and forecasting."
)

# Version info
try:
    from core import __version__
    st.sidebar.text(f"Version: {__version__}")
except ImportError:
    st.sidebar.text("Version: Unknown")

# Page routing
if page == "🏠 Home":
    from pages import home
    home.show()
elif page == "📊 Dataset Manager":
    from pages import dataset_manager
    dataset_manager.show()
elif page == "🏗️ Model Builder":
    from pages import model_builder
    model_builder.show()
elif page == "🚀 Training":
    from pages import training
    training.show()
elif page == "📈 Results & Evaluation":
    from pages import results
    results.show()
elif page == "🔮 Prediction":
    from pages import prediction
    prediction.show()
elif page == "🔍 Project Scanner":
    from pages import project_scanner
    project_scanner.show()
elif page == "⚙️ Settings":
    from pages import settings
    settings.show()
