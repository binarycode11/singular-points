# Singular Points  

This repository encompasses the implementation designed to reproduce and validate the experiments detailed in the publication:  
**"Optimizing Image Identification: Discriminative Keypoint Detection with Equivariant CNN and SSIM-Based Triplet Loss"**  

## Objective  

The objective of this project is to rigorously replicate and extend the experimental framework outlined in the paper, with a particular emphasis on:  
1. The **repeatability** of detected keypoints—an essential property for robust feature matching, often underexplored in prior research.  
2. Evaluating the efficacy of the proposed method in addressing matching and identification challenges within datasets characterized by high intra-class similarity and medium inter-class variations.  

## Datasets

The project utilizes the following datasets:

*   **Oxford 102 Flowers:**
    *   Number of Images: 8,189
    *   Number of Categories: 102
    *   Class Distribution: Ranges from 40 to 258 images per category.

*   **Wood Textures:**
    *   Number of Images: 8,544
    *   Description: Images of various wood types, exhibiting diverse colors, textures, and patterns.

*   **Banknote Security Fibers:**
    *   Number of Images: 440
    *   Description: Images of fluorescent synthetic fibers found in banknotes, captured under UV light.
      
## Experimentos

- [Assessing Positional Congruence in Keypoint Detection](kornia-positional-test.ipynb)
### Table 1: Positional Congruence Evaluation (% of α = 1)  
| **Method** | **Flowers** | **Woods** | **Fibers** |
| ---------- | ----------- | --------- | ---------- |
| KeyNet     | 11.08       | 13.18     | 10.42      |
| REKD       | 30.29       | 21.70     | 3.62       |
| **Ours**   | **40.16**   | **37.63** | **42.18**  |

- [Evaluating Repeatability vs Matching Accuracy](kornia-matching-test.ipynb)
### Table 2: Descriptor Matching Accuracy (%) at Different α
| **α** | **HardNet** | **SOSNet** | **SIFT**  |
| ----- | ----------- | ---------- | --------- |
| 0.5   | **59.04**   | **58.76**  | **56.24** |
| 1.0   | 46.84       | 45.52      | 44.12     |
| 1.5   | 39.08       | 37.85      | 36.86     |

- [Identification in N:M Search Spaces](grid_avaliacao_local.ipynb)

### Table 3: Identification and Matching in N:M Search Spaces
| **Detector** | **Descriptor** | **Dataset** | **Pr ↑ (%)** | **Re ↑ (%)** | **F1 ↑ (%)** |
| ------------ | -------------- | ----------- | ------------ | ------------ | ------------ |
| KeyNet       | SIFT           | Flowers     | 31           | 45           | 37           |
| KeyNet       | SosNet         | Flowers     | 56           | 51           | 53           |
| REKD         | SosNet         | Flowers     | 65           | 100          | **79**       |
| REKD         | HardNet        | Flowers     | 55           | 99           | 71           |
| Ours         | SosNet         | Flowers     | 65           | 100          | **79**       |
| Ours         | HardNet        | Flowers     | 55           | 100          | 71           |
| KeyNet       | SIFT           | Woods       | 13           | 14           | 13           |
| KeyNet       | SosNet         | Woods       | 52           | 54           | 53           |
| REKD         | SosNet         | Woods       | 48           | 67           | 56           |
| REKD         | HardNet        | Woods       | 45           | 66           | 53           |
| Ours         | SosNet         | Woods       | 77           | 98           | **86**       |
| Ours         | HardNet        | Woods       | 66           | 98           | 79           |
| KeyNet       | SIFT           | Fibers      | 11           | 12           | 11           |
| KeyNet       | SosNet         | Fibers      | 4            | 5            | 4            |
| REKD         | SosNet         | Fibers      | 9            | 1            | 2            |
| REKD         | HardNet        | Fibers      | 11           | 2            | 3            |
| Ours         | SosNet         | Fibers      | 69           | 62           | **65**       |
| Ours         | HardNet        | Fibers      | 56           | 64           | 59           |

## Supplementary Material(samples)

This supplementary material provides an idea of the technical results in the experimental validation of our Equivariant CNN & SSIM method, including some challenging test cases designed to evaluate robustness under transformation variations: [Equivariant CNN & SSIM](data/Equivariant_CNN_and_SSIM_Based_Triplet_Loss__Supplem_Mat_adapted_SIBGRAPI.pdf)

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
