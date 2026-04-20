#!/bin/bash
# Setup script for GIFT-Eval benchmark on Princeton PLI cluster
# Run this ONCE on a login node with internet access

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIFTEVAL_DIR="/scratch/gpfs/EHAZAN/jh1161/gifteval"
DATA_DIR="${GIFTEVAL_DIR}/data"

echo "=== GIFT-Eval Setup Script ==="
echo "Installation directory: ${GIFTEVAL_DIR}"

# Create directories
mkdir -p "${GIFTEVAL_DIR}"
mkdir -p "${DATA_DIR}"

cd "${GIFTEVAL_DIR}"

# Clone gift-eval repository if not exists
if [ ! -d "${GIFTEVAL_DIR}/gift-eval" ]; then
    echo "Cloning gift-eval repository..."
    git clone https://github.com/SalesforceAIResearch/gift-eval.git
else
    echo "gift-eval repository already exists, pulling latest..."
    cd gift-eval && git pull && cd ..
fi

# Activate the uni2ts environment
echo "Activating uni2ts environment..."
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

# Install gift-eval in the existing uni2ts environment
echo "Installing gift-eval..."
cd "${GIFTEVAL_DIR}/gift-eval"
pip install -e .

# Download GIFT-Eval dataset from HuggingFace
echo "Downloading GIFT-Eval dataset (this may take a while)..."
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir "${DATA_DIR}"

# Add GIFT_EVAL to .env file
ENV_FILE="/scratch/gpfs/EHAZAN/jh1161/uni2ts/.env"
if grep -q "GIFT_EVAL=" "${ENV_FILE}" 2>/dev/null; then
    echo "GIFT_EVAL already in .env, updating..."
    sed -i "s|GIFT_EVAL=.*|GIFT_EVAL=${DATA_DIR}|" "${ENV_FILE}"
else
    echo "Adding GIFT_EVAL to .env..."
    echo "GIFT_EVAL=${DATA_DIR}" >> "${ENV_FILE}"
fi

echo ""
echo "=== Setup Complete ==="
echo "GIFT-Eval data location: ${DATA_DIR}"
echo "Environment variable GIFT_EVAL set in: ${ENV_FILE}"
echo ""
echo "To run evaluation:"
echo "  sbatch ${GIFTEVAL_DIR}/eval_gifteval.slurm"
echo ""
echo "Or for a quick subset evaluation:"
echo "  sbatch ${GIFTEVAL_DIR}/eval_gifteval_quick.slurm"
