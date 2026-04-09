#!/bin/ksh
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N ia2hl_infer
#$ -cwd

cd $WORKDIR
source /beegfs/data/work/imvia/in156281/ia2hl/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/ia2hl/venv/lib/python3.9/site-packages:$PYTHONPATH
export TORCH_HOME=/beegfs/data/work/imvia/in156281/ia2hl/torch_cache
cd $WORKDIR/ia2hl

python infer.py