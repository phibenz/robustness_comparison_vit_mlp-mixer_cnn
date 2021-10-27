SOURCE_MODELS='ViT-B_16-224
               ViT-L_16-224
               mixer_b16_224
               mixer_l16_224
               swsl_resnet18
               swsl_resnet50
               resnet18
               resnet50'

EPSILON='16/255'
ITERATIONS=7

for sm in $SOURCE_MODELS; do 
    python attack.py \
        --source-model $sm \
        --target-model ViT-B_16-224 ViT-L_16-224 mixer_b16_224 mixer_l16_224 swsl_resnet18 swsl_resnet50 resnet18 resnet50 \
        --dataset neurips \
        --batch-size 20 \
        --attack-iterations $ITERATIONS \
        --attack-loss-fn ce-untargeted \
        --attack-step-size 20./7/255 \
        --attack-epsilon $EPSILON \
        --attack-log-interval 1 \
        --subfolder 'transferability/16_255' \
        --postfix _$sm
done