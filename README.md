# fcHMRF-LIS
FDR Control Method for Spatial 3D Data 

## Table of Contents
* [Requirements and Installation](#requirements-and-installation)
* [Usage](#usage)

## Requirements and Installation
This package was developed using Python 3.12 and Pytorch 2.6.0.
The permutohedral lattice based filtering relies on C++ implementation provided by [crfasrnn_pytorch](https://github.com/sadeepj/crfasrnn_pytorch).

To install the package, please run the following lines:
```bash
git clone https://github.com/kimtae55/fcHMRF-LIS
cd $PATH_TO_fcHMRF-LIS$
pip install -r requirements.txt
python setup.py install
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
