# Dimploma Thesis - NTUA (2021-2022)

This repository hosts the code for replicating the experiments and plots included in my diploma thesis **Accelerating Compressed Sensing MRI with Score-based Implicit Model**.

## Installation Instructions

1. Clone and enter this repo: <br />
  `git clone https://github.com/d-dimos/csgm-mri-ddim.git` <br />
  `cd csgm-mri-ddim`
2. Install the requirements: `pip install -r requirements.txt`
3. Download a small subset of the test dataset: <br />
  `gdown https://drive.google.com/uc?id=1mpnV1iXid1PG0RaJswM6t9yI76b2IPxc` <br />
  `tar -zxvf datasets.tar.gz`
 4. Download the pretrained NCSNv2: <br />
  `gdown https://drive.google.com/uc?id=1vAIXf8n67yEAPmH2I9qiDWzmq9fGKPYL` <br />
  `tar -zxvf checkpoint.tar.gz`
  
 ## Example Commands
 
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
