#!/bin/ksh 
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N ihc4bc_train
cd $WORKDIR
source /beegfs/data/work/imvia/in156281/ihc4bc/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/ihc4bc/venv/lib/python3.9/site-packages:$PYTHONPATH
python /beegfs/data/work/imvia/in156281/ia2hl/train.py