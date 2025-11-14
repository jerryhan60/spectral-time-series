#!/bin/bash
# Compare training runs in TensorBoard
# This will show both runs in the same graph

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

# Activate virtual environment
source venv/bin/activate

# Start TensorBoard with both runs
# TensorBoard will automatically group runs by parent directory
tensorboard --logdir_spec \
  "non-precond:outputs/pretrain/moirai_small/lotsa_v1_unweighted/pretrain_run_20251020_205126/logs,precond:outputs/pretrain/moirai_small_precond/lotsa_v1_unweighted/precond_default_20251102_102511/logs" \
  --port 6006 \
  --bind_all

echo ""
echo "TensorBoard is running!"
echo "Access it at: http://localhost:6006"
echo ""
echo "If running on a cluster:"
echo "  1. On your local machine, run: ssh -L 6006:localhost:6006 <your-cluster-login>"
echo "  2. Then open: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
