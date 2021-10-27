SOURCE_MODELS='ViT-B_16-224
               ViT-L_16-224
               ViT-augreg_B_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res-224
               ViT-augreg_L_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res-224
               mixer_b16_224
               mixer_l16_224
               swsl_resnet18
               swsl_resnet50
               resnet18
               resnet50'

for sm in $SOURCE_MODELS; do 
    python attack_fb.py \
        --source-model ${sm} \
        --target-model ${sm} \
        --dataset neurips \
        --batch-size 20 \
        --attack-variant cw_l2 \
        --cw-binary-search-steps 9 \
        --cw-steps 10000 \
        --cw-stepsize 0.01 \
        --cw-confidence 0. \
        --cw-initial-constant 0.001 \
        --cw-abort-early True \
        --subfolder cw \
        --postfix _${sm}
done