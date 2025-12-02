#!/bin/ksh 
#$ -q gpu
#$ -N ia2hl_train
cd $WORKDIR
source /beegfs/data/work/imvia/in156281/ihc4bc/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/ihc4bc/venv/lib/python3.9/site-packages:$PYTHONPATH
cd $WORKDIR/ia2hl
python train.py