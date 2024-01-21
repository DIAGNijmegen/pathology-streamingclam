cd /data/pathology/projects/pathology-bigpicture-uncertainty/github/pathology-streamingclam
FOLD="0"

python3 main.py \
    --image_path=/home/data/images \
    --mask_path=/home/data/images_tissue_masks \
    --fold="${FOLD}" \
    --train_csv="/data/pathology/projects/pathology-bigpicture-streamingclam/streaming_experiments/camelyon/data_splits/train_${FOLD}.csv" \
    --val_csv="/data/pathology/projects/pathology-bigpicture-streamingclam/streaming_experiments/camelyon/data_splits/val_${FOLD}.csv" \
    --test_csv="/data/pathology/projects/pathology-bigpicture-streamingclam/streaming_experiments/camelyon/data_splits/test.csv" \
    --mask_suffix="_tissue" \
    --mode="fit" \
    --unfreeze_streaming_layers_at_epoch=20 \
    --num_epochs=40 \
    --strategy="ddp_find_unused_parameters_true" \
    --default_save_dir="/data/pathology/projects/pathology-bigpicture-uncertainty/ckp" \
    --ckp_path="" \
    --grad_batches=1 \
    --num_gpus=1 \
    --precision="bf16-mixed" \
    --encoder="resnet39" \
    --branch="sb" \
    --pooling_layer="avgpool" \
    --pooling_kernel=16 \
    --num_classes=2 \
    --learning_rate=1e-4 \
    --tile_size=9600 \
    --tile_size_finetune=6400 \
    --image_size=65536 \
    --filetype=".tif" \
    --read_level=1 \
    --num_workers=3
