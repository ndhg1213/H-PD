# H-GLaD: Hierarchical Features Matter: A Deep Exploration of GAN Priors for Improved Dataset Distillation

[Paper](https://arxiv.org/abs/2406.05704)
<br>

<!-- This repo contains code for training expert trajectories and distilling synthetic data from our GLaD paper (CVPR 2023). Please see our [project page](https://georgecazenavette.github.io/glad) for more results.


> [**Dataset Distillation by Matching Training Trajectories**](https://georgecazenavette.github.io/mtt-distillation/)<br>
> [George Cazenavette](https://georgecazenavette.github.io/), [Tongzhou Wang](https://ssnl.github.io/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)<br>
> MIT, UC Berkeley, CMU<br>
> CVPR 2023 -->

H-GLaD utilizes hierarchical features to enhance the GAN-based parameterization dataset distillation method.

<!-- ![method image](resources/method.svg)

Please see our [Project Page](https://georgecazenavette.github.io/glad) for more visualizations. -->

<!-- ## Getting Started -->

<!-- First, download our repo:
```bash
git clone https://github.com/GeorgeCazenavette/glad.git
cd glad
```

To setup an environment, please run

```bash
conda env create -n glad python=3.9
conda activate glad
pip install -r requirements.txt
``` -->

## Usage
Below are some example commands to run each method.

Using the default hyper-parameters, you should be able to comfortable run each method on a 24GB GPU.

### Distillation by Gradient Matching
The following command will then use the buffers we just generated to distill imagenet-birds down to 1 image per class using StyleGAN:
```bash
python h_glad_dc.py --dataset=imagenet-birds --space=wp --ipc=1 --data_path={path_to_dataset}
```

### Distillation by Distribution Matching
The following command will then use the buffers we just generated to distill imagenet-fruit down to 1 image per class using StyleGAN:
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

# Reference
If you find our code useful for your research, please cite our paper.
```
@article{zhong2024hierarchical,
  title={Hierarchical Features Matter: A Deep Exploration of GAN Priors for Improved Dataset Distillation},
  author={Zhong, Xinhao and Fang, Hao and Chen, Bin and Gu, Xulin and Dai, Tao and Qiu, Meikang and Xia, Shu-Tao},
  journal={arXiv preprint arXiv:2406.05704},
  year={2024}
}
```
