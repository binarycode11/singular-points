# Singular Points  

This repository encompasses the implementation designed to reproduce and validate the experiments detailed in the publication:  
**"Optimizing Image Identification: Discriminative Keypoint Detection with Equivariant CNN and SSIM-Based Triplet Loss"**  

## Objective  

The objective of this project is to rigorously replicate and extend the experimental framework outlined in the paper, with a particular emphasis on:  
1. The **repeatability** of detected keypoints—an essential property for robust feature matching, often underexplored in prior research.  
2. Evaluating the efficacy of the proposed method in addressing matching and identification challenges within datasets characterized by high intra-class similarity and medium inter-class variations.  


## Environment Setup
For detailed environment setup instructions, refer to the [Step-by-Step Setup Guide](pages/environment.md)

## Datasets
The following datasets were utilized for the experiments:

  - [Woods Texture](https://drive.google.com/uc?export=download&id=1DzJYC00lcZo-SQWdaRQHylMGTd-Mcz2h)
  - [Flowers](https://drive.google.com/uc?export=download&id=1z4Us0tlRrNEDSHlWwU42IFX57BNj7mcb)
  - [Fibers](https://drive.google.com/file/d/1oDWni9HrX92dl4xszXVoo2IGpUbj0Cs6/view?usp=sharing)

Use the following commands to download and prepare the datasets:
``` bash
# Download the "Woods Texture" dataset
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1DzJYC00lcZo-SQWdaRQHylMGTd-Mcz2h' -O data/woods_texture.zip

# Download the "Flowers" dataset
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1z4Us0tlRrNEDSHlWwU42IFX57BNj7mcb' -O data/flowers.zip

# Download the "Fibers" dataset
wget --no-check-certificate 'https://drive.google.com/file/d/1oDWni9HrX92dl4xszXVoo2IGpUbj0Cs6/view?usp=sharing' -O data/fibers.zip

# Unzip the downloaded files
unzip data/woods_texture.zip -d data/woods_texture
unzip data/flowers.zip -d data/flowers

# Remove the zip files after extraction (optional)
rm data/woods_texture.zip
rm data/flowers.zip
```
## Experimentos

- [Assessing Positional Congruence in Keypoint Detection](kornia-positional-test.ipynb)
- [Evaluating Repeatability vs Matching Accuracy](kornia-matching-test.ipynb)
- [Identification in N:M Search Spaces](grid_avaliacao_local.ipynb)

## Referências:

