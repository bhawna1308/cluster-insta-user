#!/bin/bash

echo "============================"
echo "🚀 Starting ML Project"
echo "============================"

# 1. Activate virtual environment (optional)
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate   # for Linux/Mac
    # venv\Scripts\activate    # (uncomment if on Windows PowerShell)
fi

# 2. Run pipeline
echo "📥 Running main pipeline..."
python main.py

# 3. Launch Streamlit
echo "🌐 Launching Streamlit App..."
streamlit run app.py
