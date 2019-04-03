source /etc/profile.d/modules.sh
module load cuda/9.0.176 cudnn nccl/2.2.13
export PATH=$PATH:/apps/t3/sles12sp2/cuda/9.1.85//bin:/apps/t3/sles12sp2/free/gcc/5.5.0:/apps/t3/sles12sp2/cuda/9.1.85//bin:/apps/t3/sles12sp2/free/gcc/5.5.0:/home/3/17R17067/anaconda3/bin:/opt/sgij/bin:/apps/t3/sles12sp2/hpe/ptl/user/bin:/apps/t3/sles12sp2/hpe/ptl/admin/bin:/apps/t3/sles12sp2/uge/latest/bin/lx-amd64:/home/3/17R17067/bin:/usr/local/bin:/usr/bin:/bin:/usr/bin/X11:/usr/games:/home/3/17R17067/Software/bin/:/home/3/17R17067/Software/clang+llvm-7.0.1-x86_64-linux-sles11.3/bin:/home/3/17R17067/Software/sox-14.4.2/bin:/home/3/17R17067/Software/bin/:/home/3/17R17067/Software/clang+llvm-7.0.1-x86_64-linux-sles11.3/bin:/home/3/17R17067/Software/sox-14.4.2/bin:/home/3/17R17067/.local/bin:/home/3/17R17067/bin
source activate LY

python3 cyclegan.py
