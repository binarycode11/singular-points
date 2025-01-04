# Singular Points - Development Environment

This guide provides instructions to set up the environment required for the **Singular Points** project using Conda and Python.

---

## Prerequisites

- **Conda**: Make sure Conda is installed on your system. If not, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

---

## Step-by-Step Setup

### 1. Create a Conda Environment
Create a Conda environment named `singular-points` with Python 3.9:
```bash
conda create -n singular-points python=3.9 -y
conda activate singular-points
```

### 2. Install Dependencies

Install the required libraries using Conda and pip:
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other project dependencies
pip install kornia e2cnn matplotlib tqdm scikit-image PyQt5==5.15.1 jupyterlab ipywidgets opencv-python kornia_moons

# Upgrade pip
pip install --upgrade pip
```

### 3. Launch Jupyter Lab

To work with interactive notebooks, launch Jupyter Lab:

```bash
jupyter lab
```

### 4. Activate and Deactivate the Environment
```bash
conda activate singular-points
conda deactivate
conda env list
```