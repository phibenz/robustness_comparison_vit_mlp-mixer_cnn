# Adversarial Robustness Comparison of Vision Transformer and MLP-Mixer to CNNs [ArXiv](https://arxiv.org/pdf/2110.02797.pdf)

## Abstract 
Convolutional Neural Networks (CNNs) have become the de facto gold standard in computer vision applications in the past years. Recently, however, new model architectures have been proposed challenging the status quo. The Vision Transformer (ViT) relies solely on attention modules, while the MLP-Mixer architecture substitutes the self-attention modules with Multi-Layer Perceptrons (MLPs). Despite their great success, CNNs have been widely known to be vulnerable to adversarial attacks, causing serious concerns for security-sensitive applications. Thus, it is critical for the community to know whether the newly proposed ViT and MLP-Mixer are also vulnerable to adversarial attacks. To this end, we empirically evaluate their adversarial robustness under several adversarial attack setups and benchmark them against the widely used CNNs. Overall, we find that the two architectures, especially ViT, are more robust than their CNN models. Using a toy example, we also provide empirical evidence that the lower adversarial robustness of CNNs can be partially attributed to their shift-invariant property. Our frequency analysis suggests that the most robust ViT architectures tend to rely more on low-frequency features compared with CNNs. Additionally, we have an intriguing finding that MLP-Mixer is extremely vulnerable to universal adversarial perturbations.

## Setup 
### Set Paths
Set the paths in `./config.py` according to your system and environment.

### Download ViT Checkpoints
Run `bash ./download_checkpoints.sh`

### NeurIPS dataset
We are providing the NeurIPS adversarial challenge dataset together with this repository. The images are stored in `./images` together with the data sheet in `./images.csv`

## Evaluate Models
As a sanity check you can evaluate the models on the NeurIPS dataset and check if the numbers match Table 1 of the paper with `bash ./experiments/eval_models.sh`

## White-box attack
For the white-box attacks you can run the corresponding script.

### PGD attack
`bash ./experiments/attack_pgd.sh`

### FGSM attack
`bash ./experiments/attack_fgsm.sh`

### C&W
`bash ./experiments/attack_cw.sh`

### DeepFool
`bash ./experiments/attack_deepfool.sh`



## Black-box attack
* Query-based
* Transfer-based
### Boundary attack
### Bandits(TD) attack
### Transferability with I-FGSM
`bash ./experiments/transferability.sh`

## Universal Adversarial Attack
Run `bash ./experiments/attack_uap.sh`

## Docker
We provide a Dockerfile to get better reproducibility of the results presented in the paper. Have a look in the docker folder.

## Credits
We would like to credit the following resources, which helped tremendously in our development-process.
 * [Timm](https://github.com/rwightman/pytorch-image-models)
 * [Foolbox](https://github.com/bethgelab/foolbox)
 * [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
 * [vision_transformer](https://github.com/google-research/vision_transformer)

## Citation
```
@article{benz2021adversarial,
  title={Adversarial Robustness Comparison of Vision Transformer and MLP-Mixer to CNNs},
  author={Benz, Philipp and Ham, Soomin and Zhang, Chaoning and Karjauv, Adil and Kweon, In So},
  journal={arXiv preprint arXiv:2110.02797},
  year={2021}
}
```
