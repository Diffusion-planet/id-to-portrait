#!/bin/bash
# FastFace Mac MPS Setup Script
# For MacBook with Apple Silicon (M1/M2/M3)

set -e

echo "=== FastFace Mac MPS Setup ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Detected Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv_mps
source venv_mps/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing MPS-compatible dependencies..."
pip install -r requirements_mps.txt

# Create models cache directory
echo ""
echo "Creating models cache directory..."
mkdir -p models_cache

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv_mps/bin/activate"
echo ""
echo "To run inference:"
echo "  python simple_inference.py \\"
echo "    --config_dir=configs/fastface/am1_and_dcg.json \\"
echo "    --model_name=hyper \\"
echo "    --id_img_path=<your_face_image.jpg> \\"
echo "    --prompt=\"your prompt here\" \\"
echo "    --device=mps"
echo ""
echo "Note: insightface face detection will run on CPU (ONNX doesn't support MPS)"
echo "      but the main diffusion model will use MPS acceleration."
