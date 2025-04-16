# fcHMRF-LIS
The fcHMRF-LIS is a stable spatial false discover rate (FDR) control method that utilizes a fully connected hidden Markov random field to model the spatial dependencies. 

![gui_image](https://github.com/kimtae55/fcHMRF-LIS/blob/f004c8272bd890871416c79717d4a6518125b0a0/misc/fdr_fnr_atp_variation.png)

## Table of Contents
* [Installation](#requirements-and-installation)
* [Usage](#usage)

## Installation
This package was developed using Python 3.12 and Pytorch 2.6.0.
The permutohedral lattice based filtering relies on C++ implementation provided by [crfasrnn_pytorch](https://github.com/sadeepj/crfasrnn_pytorch).

To install the package, please run the following lines:
```bash
git clone $fcHMRF-LIS-repo$
cd $PATH_TO_fcHMRF-LIS-repo$
pip install -r requirements.txt
```

## Usage
```bash
python -m src.train --lr 1e-4
                    --e 20
                    --threshold 0.05
                    --labelpath {optional groundtruth if using for simulation}
                    --datapath {input test statistics}
                    --betapath {delta_mu as in the paper}
                    --ppath {p-values of the input test statistics}
                    --savepath {directory path for saving results}
```
