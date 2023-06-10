# Enformer_DNA_Diffusion

Enformer DNA Diff is a Python package that combines the DeepMind Enformer model with the DNA Diffusion project.
This implementation was tested under Ubuntu 20.04, Python 3.8, Tensorflow 2.12, and CUDA 11.6. Versions of all dependencies can be found in requirements.txt.

## Enformer

If you find this code useful, please consider reference in your paper:

```
@article{Avsec2021,
author={Avsec, {\v{Z}}iga
and Agarwal, Vikram
and Visentin, Daniel
and Ledsam, Joseph R.
and Grabska-Barwinska, Agnieszka
and Taylor, Kyle R.
and Assael, Yannis
and Jumper, John
and Kohli, Pushmeet
and Kelley, David R.},
title={Effective gene expression prediction from sequence by integrating long-range interactions},
journal={Nature Methods},
year={2021}
}
``` 

## DNA-Diffusion

This package has been developed for an interesting open-source project: [[DNA-Diffusion]](https://github.com/pinellolab/DNA-Diffusion)

## Installation

### 1. Installation with pip

You can install Enformer DNA Diff using pip. Run the following command:

```bash
pip install enformer-dna-diff
```

### 2. Installation via Git clone
Alternatively, you can clone this repository and install the package locally. Follow these steps:

Clone the repository:

```bash
git clone https://github.com/ttunja/Enformer_DNA_Diffusion.git
```

Navigate to the cloned repository:

```bash
cd Enformer_DNA_Diffusion
```

Install the package using pip:

```bash
pip install .
```
## Prepare the dataset

After the installation has been done, one needs to get the data needed for the package to work. Run the following command:
```bash
get_data.py
```

## Example

If one is interested to see the package in action, try example.ipynb.