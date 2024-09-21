python scripts/train.py \
    --layer_scale --local \
    --mode classifier \
    --warm_epoch 3 --epoch 10  --lr 3e-5  --batch 256 \
    --lr_factor 8 \
    --stop_epoch 3

    #    --wd 0.0001
    # --mode classifier --fine_tune \


