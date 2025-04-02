#!/bin/bash

# Ensure the virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install matplotlib numpy
else
    source venv/bin/activate
fi

# Create necessary directories
mkdir -p logs
mkdir -p reports

# Run the main script
echo "Running MST algorithm experiments..."
python3 src/main.py

# Deactivate the virtual environment
deactivate

echo "Program execution completed. Logs are available in logs/ directory."
echo "You can view the report by opening reports/mst_analysis_report.html in your browser." 