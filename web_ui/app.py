"""
Deep-TimeSeries Web UI
Main application entry point with navigation
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for local imports
parent_dir = Path(__file__).parent.parent.resolve()
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

st.set_page_config(
    page_title="Deep Eagle Dashboard",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Authentication — imported after sys.path setup and st.set_page_config (must be first)
from auth import logout, require_authentication  # noqa: E402

username = require_authentication()
if username is None:
    st.stop()

# Main title with logo
col1, col2 = st.columns([1, 5])
with col1:
    try:
        logo_path = Path(__file__).parent / "assets" / "eagleeye.jpg"
        st.image(str(logo_path), width=100)
    except Exception:
        st.write("🦅")
with col2:
    st.title("Deep Eagle Dashboard")
    st.markdown("*A visual interface for time-series deep learning*")

# Sidebar navigation
st.sidebar.title("📈 Navigation")

page = st.sidebar.radio(
    "Select Page",
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
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Deep Eagle is a modular PyTorch framework for time-series analysis and forecasting."
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"👤 **User:** {username}")
if st.sidebar.button("🚪 Logout"):
    logout()

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
