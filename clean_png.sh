#!/bin/bash

echo "Cleaning generated png files and best model..."

# Remove all PNG files in current directory
if ls *.png 1> /dev/null 2>&1; then
    echo "Removing PNG files..."
    rm -f *.png
else
    echo "No PNG files found."
fi
# Remove trained model
if [ -f "best_model.pth" ]; then
    echo "Removing best_model.pth..."
    rm -f best_model.pth
else
    echo "best_model.pth not found."
fi
echo "Cleanup complete."