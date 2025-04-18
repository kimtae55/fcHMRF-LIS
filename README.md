# FcHMRF-LIS: a stable, spatial FDR control method 

# 🧠 Introduction
 **Why use FDR control?**

In high-dimensional testing (e.g., brain imaging, genomics analyses), we often perform hundreds of thousands of statistical tests (e.g. comparing voxel-level brain glucose metabolism across healthy versus diseased groups using the standardized uptake value ratio (SUVR) from Fluorine-18 fluorodeoxyglucose positron emission tomography (FDG-PET) data). 

When you test many hypotheses, the chance of making false discoveries increases. FDR control limits the expected proportion of false positives among the discoveries, ensuring results are statistically reliable at a desired level, without being overly conservative like family-wise error rate (FWER) methods or false discovery proportion (FDP) control methods. 

The task of FDR control in the field of statistics can be viewed analogously to an unsupervised binary segmentation task. 

# 🌟 What is special about fcHMRF-LIS?

❗ Existing spatial FDR control methods face three fundamental challenges in neuroimaging applications:

  - First, many fail to account for the complex spatial dependencies inherent in neuroimaging data, including distance-related dependence and spatial heterogeneity.
  
  - Second, while state-of-the-art methods focus on controlling the FDR and minimizing the false non-discovery rate (FNR), they often exhibit high variance in FDP and FNP, leading to instability across replications. Figure below illustrates this high variability across 50 replications of real-data based simulation settings. 
  
  - Many spatial FDR methods lack computational scalability, making them impractical for analyzing high-resolution datasets (e.g. neuroimaging data)

🔦 The *fcHMRF-LIS* is a stable, spatial false discover rate (FDR) control method that mitigates the above three issues:

  - It is the first multiple hypothesis testing framework to utilize a *fully connected HMRF* to model voxel-wise spatial dependencies. Our method exhibits exceptional stability across multiple replications.
  
  - We provide an efficient and scalable expectation-maximization (EM) algorithm leveraging the CRF-RNN, where high-dimensional filtering is accelerated on a permutohedral lattice.
  
  - We apply our method to the FDG-PET data from ADNI, to identify brain areas affected along the progression of the Alzheimer's disease (AD).

![gui_image](https://github.com/kimtae55/fcHMRF-LIS/blob/f004c8272bd890871416c79717d4a6518125b0a0/misc/fdr_fnr_atp_variation.png)

## Installation
This package was developed using Python 3.12 and Pytorch 2.6.0.
The permutohedral lattice based filtering relies on C++ implementation provided by [crfasrnn_pytorch](https://github.com/sadeepj/crfasrnn_pytorch).

To install the package, please run the following lines:
```bash
git clone $fcHMRF-LIS-repo$
cd $PATH_TO_fcHMRF-LIS-repo$
python setup.py install 
```

## Usage
```bash
python -m src.train --lr {learning rate}
                    --e {number of EM steps}
                    --threshold {desired FDR threshold}
                    --labelpath {optional groundtruth if using for simulation}
                    --datapath {z-statistics as in the paper}
                    --betapath {delta_mu as in the paper}
                    --ppath {p-values corresponding to the z-statistics, this is just used for initialization}
                    --savepath {directory path for saving results}
```

## Citing our work 
Arxiv coming soon!

## Updates
