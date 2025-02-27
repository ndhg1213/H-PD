# [CVPR 2025)] H-PD

This is the official implementation of paper **Hierarchical Features Matter: A Deep Exploration of Progressive Parameterization Method for Dataset Distillation (CVPR2025)** .

The repository is based on [GLaD](https://github.com/georgecazenavette/glad). Please cite their papers if you use the code. 


### Distillation by Gradient Matching
The following command will distill imagenet-birds down to 1 image per class using StyleGAN:
```bash
python h_glad_dc.py --dataset=imagenet-birds --space=wp --ipc=1 --data_path={path_to_dataset}
```

### Distillation by Distribution Matching
The following command will distill imagenet-fruit down to 1 image per class using StyleGAN:
```bash
python h_glad_dm.py --dataset=imagenet-fruits --space=wp --ipc=1 --data_path={path_to_dataset}
```

### Distillation by Trajectory Matching
First you will need to create the expert trajectories.
```bash
python buffer_mtt.py --dataset=imagenet-b --train_epochs=15 --data_path={path_to_dataset}
```

The following command will then use the buffers we just generated to distill imagenet-b down to 1 image per class using StyleGAN:
```bash
python h_glad_mtt.py --dataset=imagenet-b --space=wp --ipc=1 --data_path={path_to_dataset}
```

### Extra Options
Adding ```--rand_f``` will initialize the f-latents with Gaussian noise.

Adding ```--special_gan=ffhq``` or ```--special_gan=pokemon``` will use a StyleGAN trained on FFHQ or Pokémon instead of ImageNet.

Adding ```--learn_g``` will allow the weights of the StyleGAN to be updated along with the latent codes.

Adding ```--avg_w``` will initialize the w-latents with the average w for the respective class. 
(Do not do this if attempting to distill multiple images per class.)

If you use the repo, please consider citing:
```
@article{zhong2024hierarchical,
  title={Hierarchical Features Matter: A Deep Exploration of GAN Priors for Improved Dataset Distillation},
  author={Zhong, Xinhao and Fang, Hao and Chen, Bin and Gu, Xulin and Dai, Tao and Qiu, Meikang and Xia, Shu-Tao},
  journal={arXiv preprint arXiv:2406.05704},
  year={2024}
}
```
