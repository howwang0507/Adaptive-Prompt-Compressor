#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Starting the Streamlit application..."
streamlit run app.py