#!/bin/bash

# Script to run the AI Colors main Python script
# Usage: ./run.sh [arguments]
#
# Examples:
#   ./run.sh train training_images --epochs 50
#   ./run.sh api --port 8080
#   ./run.sh test red blue --count 6
#   ./run.sh extract image.jpg --colors 5
#   ./run.sh process training_images
#   ./run.sh test-colors

# Check if Python 3 is available
if ! command -v python3 &> /dev/null
then
    echo "❌ python3 could not be found. Please install Python 3."
    exit 1
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in current directory"
    exit 1
fi

# Run the main Python script with all passed arguments
echo "🎨 Running AI Colors..."
python3 main.py "$@"
