SOURCE_MODELS='ViT-B_16-224
               ViT-L_16-224
               mixer_b16_224
               mixer_l16_224
               swsl_resnet18
               swsl_resnet50
               resnet18
               resnet50'

for sm in $SOURCE_MODELS; do 
    python attack_uap.py \
        --source-model $sm \
        --target-model ViT-B_16-224 ViT-L_16-224 mixer_b16_224 mixer_l16_224 swsl_resnet18 swsl_resnet50 resnet18 resnet50 \
        --dataset neurips \
        --batch-size 20 \
        --attack-iterations 2000 \
        --attack-loss-fn ce-untargeted \
        --attack-lr 0.005 \
        --attack-epsilon 10/255 \
        --subfolder 'uap/10_255' \
        --postfix _$sm
done