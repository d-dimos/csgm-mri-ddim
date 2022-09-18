# Dimploma Thesis - NTUA (2021-2022)

This repository hosts the code for replicating the experiments and plots included in my diploma thesis **Accelerating Compressed Sensing MRI with Score-based Implicit Model**.

<p align="center">
  <img width="579" height="193"  alt="animated" src="https://github.com/d-dimos/thesis_ntua_sbim/blob/main/assets/32stepsMRI_comp.gif">
</p>

## Sample Experiments
A comparison between **SBIM**, **SBIM-PC** and **Langevin Dynamics** when used to reconstruct MRIs accross different acceleration factors R.
 
<p float="left">
  <img src="https://github.com/d-dimos/thesis_ntua_sbim/blob/main/assets/ssim_vert.png" width="400" />
  <img src="https://github.com/d-dimos/thesis_ntua_sbim/blob/main/assets/psnr_vert.png" width="400" /> 
</p>

**Masked SSIM**
| R | 32 steps (SBIM) | 3990 steps (Langevin) | 256 steps (SBIM \w corr=3) |
|:---------:|:---------------:|:---------------------:|:--------------------------:|
|     2     |      0.92       |         0.91          |          **0.93**          |
|     3     |      0.92       |         0.90          |          **0.93**          |
|     6     |      0.89       |         0.89          |          **0.92**          |
|     8     |      0.84       |         0.89          |          **0.90**          |
|    12     |      0.83       |         0.89          |          **0.90**          |

**Masked PSNR**
| R | 32 steps (SBIM) | 3990 steps (Langevin) | 256 steps (SBIM \w corr=3) |
|:---------:|:---------------:|:---------------------:|:--------------------------:|
|     2     |      36.4       |         35.9          |          **37.1**          |
|     3     |      35.9       |         35.8          |          **36.8**          |
|     6     |      31.0       |         **34.1**          |          34.0          |
|     8     |      26.4       |         **32.9**          |          30.8          |
|    12     |      26.0       |         **32.3**          |          30.3          |

**Sample Reconstructions**
<p float="center">
  <img src="https://github.com/d-dimos/thesis_ntua_sbim/blob/main/assets/brains_grid.png" width="400" />
</p>

## Installation Instructions

1. Clone and enter this repo: <br />
  `git clone https://github.com/d-dimos/csgm-mri-ddim.git` <br />
  `cd thesis_ntua_sbim`
2. Install the requirements: `pip install -r requirements.txt`
3. Download a small subset of the test dataset: <br />
  `gdown https://drive.google.com/uc?id=1mpnV1iXid1PG0RaJswM6t9yI76b2IPxc` <br />
  `tar -zxvf datasets.tar.gz`
4. Download the pretrained NCSNv2: <br />
  `gdown https://drive.google.com/uc?id=1vAIXf8n67yEAPmH2I9qiDWzmq9fGKPYL` <br />
  `tar -zxvf checkpoint.tar.gz`
5. For instructions on how to estimate sensitivity maps (apart from the ones downloaded along with the sample data above), please visit [this repo](https://github.com/utcsilab/csgm-mri-langevin).


## Example Commands
1. Sample using the pretrained NCSNv2: `python main.py`. Arguments:
   - `--config` configuration file, in the format of `configs/brain_T2`
   - `--sampler` sampler type ['ddim', 'LD']
   - `--steps` number of DDIM (SBIM) steps
   - `--R` MRI acceleration factor
   - `--orientation` MRI orientation ['vertival', 'horizontal']
   - `--pattern` ['equispaced']
   - `--save_images` if you want to save the reconstructed MRIs
   - `--exp` directory to save results
3. Plot the evaluation metrics: `python make_plot.py`. Arguments:
   - `--exp` directory of saved results
   - `--orientation` MRI orientation
4. Make a grid of reconstructed MRIs: `python make_grid.py`. Arguments:
   - `--exp_dir` directory of saved images

    
## Citations

This code uses prior work from the following papers, which must be cited:
```
@article{jalal2021robust,
  title={Robust Compressed Sensing MRI with Deep Generative Priors},
  author={Jalal, Ajil and Arvinte, Marius and Daras, Giannis and Price, Eric and Dimakis, Alexandros G and Tamir, Jonathan I},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}

@inproceedings{song2019generative,
  title={Generative modeling by estimating gradients of the data distribution},
  author={Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11918--11930},
  year={2019}
}

@article{song2020improved,
  title={Improved Techniques for Training Score-Based Generative Models},
  author={Song, Yang and Ermon, Stefano},
  journal={arXiv preprint arXiv:2006.09011},
  year={2020}
}
```

The data used belongs to the NYU fastMRI dataset, which must also be cited:
```
@inproceedings{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Tullie Murrell and Zhengnan Huang and Matthew J. Muckley and Aaron Defazio and Ruben Stern and Patricia Johnson and Mary Bruno and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and Nafissa Yakubova and James Pinkerton and Duo Wang and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}

@article{knoll2020fastmri,
  title={fastMRI: A publicly available raw k-space and DICOM dataset of knee images for accelerated MR image reconstruction using machine learning},
  author={Knoll, Florian and Zbontar, Jure and Sriram, Anuroop and Muckley, Matthew J and Bruno, Mary and Defazio, Aaron and Parente, Marc and Geras, Krzysztof J and Katsnelson, Joe and Chandarana, Hersh and others},
  journal={Radiology: Artificial Intelligence},
  volume={2},
  number={1},
  pages={e190007},
  year={2020},
  publisher={Radiological Society of North America}
}
```
