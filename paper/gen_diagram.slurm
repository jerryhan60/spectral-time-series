#!/bin/bash
#SBATCH --job-name=diagram
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/diagram_%j.out
module load anaconda3/2024.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
python3 << 'PY'
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
FIG='/scratch/gpfs/EHAZAN/jh1161/paper/figures'
fig,axes=plt.subplots(1,2,figsize=(11,5.5))
ax1,ax2=axes
def bx(a,x,y,w,h,t,c='#e8e8e8',f=8,b=False):
 a.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.08',fc=c,ec='k',lw=0.8))
 a.text(x,y,t,ha='center',va='center',fontsize=f,fontweight='bold' if b else 'normal')
def ar(a,x1,y1,x2,y2):a.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.2))
for a in[ax1,ax2]:a.set_xlim(-3,3);a.set_ylim(0,10);a.axis('off')
ax1.set_title('(a) Baseline',fontsize=11)
bx(ax1,0,9.5,3.5,.5,'x[t]','#f0f0f0',9);ar(ax1,0,9.2,0,8.8);bx(ax1,0,8.5,3,.5,'Normalize','#dce6f1');ar(ax1,0,8.2,0,7.8);bx(ax1,0,7.5,3,.5,'Patch (P=16)','#dce6f1');ar(ax1,0,7.2,0,6.5);bx(ax1,0,6.2,3.5,.5,'Linear(16, 384)','#b8d4e3');ar(ax1,0,5.9,0,5.2);bx(ax1,0,4.5,3.5,1.2,'Transformer\n6 layers, d=384','#a8c8a8');ar(ax1,0,3.85,0,3.4);bx(ax1,0,3.1,3,.5,'Output projection','#dce6f1');ar(ax1,0,2.8,0,2.4);bx(ax1,0,2.1,3,.5,'Forecasts','#f0f0f0')
ax2.set_title('(b) With polynomial hint',fontsize=11)
bx(ax2,0,9.5,3.5,.5,'x[t]','#f0f0f0',9);ar(ax2,-.3,9.2,-1.2,8.8);ar(ax2,.3,9.2,1.2,8.8);bx(ax2,-1.2,8.5,2.2,.5,'Normalize','#dce6f1',8);bx(ax2,1.2,8.5,2.2,.5,'Chebyshev FIR','#f5d0a9',8,True);ar(ax2,-1.2,8.2,-1.2,7.8);ar(ax2,1.2,8.2,1.2,7.8);bx(ax2,-1.2,7.5,2.2,.5,'Patch','#dce6f1',8);bx(ax2,1.2,7.5,2.2,.5,'Patch','#f5d0a9',8);ar(ax2,-.7,7.2,0,6.55);ar(ax2,.7,7.2,0,6.55);ax2.text(0,6.85,'concat',ha='center',fontsize=7,style='italic');bx(ax2,0,6.2,3.8,.5,'Linear(32, 384)  [+12K]','#f5d0a9',8,True);ar(ax2,0,5.9,0,5.2);bx(ax2,0,4.5,3.5,1.2,'Transformer\n6 layers, d=384\n(identical)','#a8c8a8');ar(ax2,0,3.85,0,3.4);bx(ax2,0,3.1,3,.5,'Output projection','#dce6f1');ar(ax2,0,2.8,0,2.4);bx(ax2,0,2.1,3,.5,'Forecasts','#f0f0f0');ax2.text(2.8,1.5,'Orange = new\n(+0.11% params)',fontsize=7,ha='right',color='#cc7722',style='italic')
plt.tight_layout();fig.savefig(f'{FIG}/fig_architecture.pdf',bbox_inches='tight');fig.savefig(f'{FIG}/fig_architecture.png',bbox_inches='tight');plt.close();print('DONE')
PY
