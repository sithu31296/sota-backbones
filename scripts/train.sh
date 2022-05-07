MODEL=VAN
VARIANT=S
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/distributed_train.sh 8 /path/to/imagenet \
    --model $MODEL --variant $VARIANT -b 128 --lr 1e-3