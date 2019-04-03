log_file=~/logs/kaiming_normal.log
qsub -l q_node=1 -l h_rt=5:00:00 -g tga-tslab -e ${log_file} -o ${log_file} ./run.sh
