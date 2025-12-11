python infer.py \
    --checkpoint_path best_models/batch-32-scale-1.0-lambda-10.0-width-256-lrg-0.0001-lrd-0.0001.pth \
    --img_width 256 \
    --img_height 256 \
    --batch_size 32 \
    --model_scale 1 \
    --base_hes_path dataset_v2/HES \
    --base_ihc_path dataset_v2/CD30