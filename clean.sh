#!/bin/bash

echo "Cleaning generated artifacts..."

# Remove processed data folder
if [ -d "data_v3" ]; then
    echo "Removing data_v3 folder..."
    rm -rf data_v3
else
    echo "data_v3 folder not found."
fi

# Remove trained model
if [ -f "best_model.pth" ]; then
    echo "Removing best_model.pth..."
    rm -f best_model.pth
else
    echo "best_model.pth not found."
fi

echo "Cleanup complete."
