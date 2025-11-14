# spectral-time-series


# Helpful commands

To request gpu from pli:
salloc --nodes=1 --ntasks=1 --mem=128G --time=03:01:00 --gres=gpu:1 --partition=pli --mail-type=begin --account=eladgroup

Module loads:
module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6

source uni2ts/venv/bin/activate

User|Account|Partition|QOS
jh1161|eladgroup||pli-low
jh1161|hazan_intern||pli-low
jh1161|spectralssmtorch||pli-low
jh1161|ehazan||gpu-long,gpu-medium,gpu-short,gpu-test,medium,short,test,vlong