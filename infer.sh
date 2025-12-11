python infer.py \
    --checkpoint_path best_models/best_model.pth \
    --img_width 256 \
    --img_height 256 \
    --batch_size 8 \
    --model_scale 0.75 \
    --base_hes_path dataset_v2/HES \
    --base_ihc_path dataset_v2/CD30