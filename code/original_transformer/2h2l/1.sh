#! /bin/bash
#SBATCH -J tyx
#SBATCH -p intel
#SBATCH -N 1 -n 1
#SBATCH -o job-%j.log
#SBATCH -e job-%j.err
##SBATCH -w qhcn651

echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

source /home/chinachky_group/liujie/data/soft/miniconda3/etc/profile.d/conda.sh
source /home/chinachky_group/liujie/data/soft/miniconda3/bin/activate trans

i=2h2l
t1=$(date +%s)
python nn.py
t2=$(date +%s)
echo $i time is $(($t2 - $t1))

