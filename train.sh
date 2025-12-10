#!/bin/ksh 
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N ia2hl_train
cd $WORKDIR
source /beegfs/data/work/imvia/in156281/ia2hl/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/ia2hl/venv/lib/python3.9/site-packages:$PYTHONPATH
export TORCH_HOME=/beegfs/data/work/imvia/in156281/ia2hl/torch_cache
cd $WORKDIR/ia2hl
python train.py \
    --img_width 256 \
    --img_height 256 \
    --lrg 0.0002 \
    --lrd 0.0002 \
    --batch_size 4 \
    --epochs 10 \
    --lambda_l1 10 \
    --model_scale 0.75