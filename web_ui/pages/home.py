"""Home page for Deep-TimeSeries Dashboard"""

import streamlit as st

from core import get_usage_stats


def show():
    """Display home page"""

    st.header("Welcome to Deep-TimeSeries Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Framework Version", "0.1.0")

    with col2:
        try:
            stats = get_usage_stats()
            total_uses = sum(stats.values()) if stats else 0
            st.metric("Total Feature Uses", f"{total_uses:,}")
        except Exception:
            st.metric("Total Feature Uses", "N/A")

    with col3:
        st.metric("Status", "Ready", delta="Online")

    st.markdown("---")

    # Quick start guide
    st.subheader("🚀 Quick Start")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### For New Users

        1. **📊 Upload Dataset** - Go to Dataset Manager to upload your time-series data
        2. **🏗️ Build Model** - Configure your model architecture
        3. **🚀 Train** - Start training with real-time monitoring
        4. **📈 Evaluate** - View results and model performance
        5. **🔮 Predict** - Make predictions on new data
        """)

    with col2:
        st.markdown("""
        ### For Existing Projects

        1. **🔍 Scan Projects** - Analyze your existing deep-timeseries projects
        2. **📊 Load Data** - Import data from your projects
        3. **⚙️ Settings** - Configure preferences and paths
        4. **🚀 Resume Training** - Continue from checkpoints
        """)

    st.markdown("---")

    # Features overview
    st.subheader("✨ Features")

    features_col1, features_col2, features_col3 = st.columns(3)

    with features_col1:
        st.markdown("""
        **📊 Data Management**
        - CSV/Excel upload
        - Data preview & statistics
        - Missing value handling
        - Train/test splitting
        """)

    with features_col2:
        st.markdown("""
        **🏗️ Model Building**
        - LSTM, GRU, Transformer
        - Visual architecture builder
        - Hyperparameter tuning
        - Configuration export
        """)

    with features_col3:
        st.markdown("""
        **🚀 Training & Evaluation**
        - Real-time metrics
        - Training visualization
        - Early stopping
        - Model checkpointing
        """)

    st.markdown("---")

    # Usage statistics
    st.subheader("📊 Usage Analytics")

    try:
        stats = get_usage_stats()
        if stats:
            import pandas as pd

            df = pd.DataFrame(
                [(k, v) for k, v in sorted(stats.items(), key=lambda x: -x[1])],
                columns=["Feature", "Uses"],
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No usage data collected yet. Start using the framework to see statistics!")
    except Exception as e:
        st.warning(f"Could not load usage statistics: {e}")

    st.markdown("---")

    # Documentation links
    st.subheader("📚 Documentation")

    doc_col1, doc_col2, doc_col3 = st.columns(3)

    with doc_col1:
        st.markdown("[📖 README](../README.md)")

    with doc_col2:
        st.markdown("[🔧 API Documentation](../docs/)")

    with doc_col3:
        st.markdown("[💡 Examples](../examples/)")
