#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define kernel name (default to 'custom-kernel' if not provided as argument)
KERNEL_NAME=${1:-$(basename "$PWD")}

# Create and activate a new environment using uv
echo "Setting up Python environment with uv..."
uv venv .venv --allow-existing

source .venv/bin/activate

# Install necessary packages
echo "Installing Jupyter and ipykernel..."
uv pip install jupyter ipykernel

# Add the new kernel
echo "Adding kernel: $KERNEL_NAME"
uv run python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"

echo "Kernel setup complete! You can now use it in Jupyter. Run: jupyter notebook"
