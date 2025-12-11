#!/bin/ksh
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N ia2hl_train
#$ -cwd

# Arguments passés par qsub
IMG_W=$1
IMG_H=$2
LRG=$3
LRD=$4
BATCH=$5
EPOCHS=$6
LAMBDA_L1=$7
SCALE=$8

cd $WORKDIR
source /beegfs/data/work/imvia/in156281/ia2hl/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/ia2hl/venv/lib/python3.9/site-packages:$PYTHONPATH
export TORCH_HOME=/beegfs/data/work/imvia/in156281/ia2hl/torch_cache
cd $WORKDIR/ia2hl

python train.py \
    --img_width "$IMG_W" \
    --img_height "$IMG_H" \
    --lrg "$LRG" \
    --lrd "$LRD" \
    --batch_size "$BATCH" \
    --epochs "$EPOCHS" \
    --lambda_l1 "$LAMBDA_L1" \
    --model_scale "$SCALE"
