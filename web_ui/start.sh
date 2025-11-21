#!/bin/bash

echo "Starting Deep-TimeSeries Dashboard..."
echo ""

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "Opening dashboard at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
