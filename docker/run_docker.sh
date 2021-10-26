docker run \
    -it --rm \
    --runtime=nvidia \
    --shm-size=32G \
    --volume /home/philipp/data_local:/workspace/data_local \
    --volume /home/philipp/Projects/robustness_comparison_vit_mlp-mixer_cnn:/workspace/Projects/robustness_comparison_vit_mlp-mixer_cnn \
    phibenz/robustness_comparison