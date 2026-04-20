#!/bin/bash
# Launch TensorBoard to monitor training

cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate

echo "Starting TensorBoard..."
echo "Access at: http://localhost:6006"
echo ""
echo "If on cluster, use SSH tunnel:"
echo "  ssh -L 6006:localhost:6006 della.princeton.edu"
echo ""

tensorboard --logdir=outputs/pretrain --port=6006 --bind_all
