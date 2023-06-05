# MRI_Sampling_Diffusion
Code for "Optimizing Sampling Patterns for Compressed Sensing MRI with Diffusion Generative Models"

## Setup
First, set up a Conda environment using ```conda env create -f conda_env.yml```.

Download the model checkpoints and fastMRI metadata from: https://drive.google.com/file/d/18n2QUN30qrBbM9rcxS3HIjIWImSbkJ-2/view?usp=sharing

## Structure
- **algorithms**: algorithms for solving inverse problems
- **configs**: yaml config files for running experiments
- **datasets**: PyTorch dataset classes
- **learners**: the main control classes for gradient-based meta-learning
- **problems**: defines forward operators as classes for re-usability
- **utils**: useful functions for experiment logging, metrics, and losses
- ```main.py```: program to invoke for running meta-learning from command line

## How to run
Here is an example command for training and evaluating a sampling mask:

```python3 main.py --config PATH_TO_CONFIG --doc NAME_OF_EXPERIMENT```

Here is a command for evaluating a baseline mask on test data:

```python3 main.py --config PATH_TO_CONFIG --doc NAME_OF_EXPERIMENT --baseline```


## Submodule initialization
```
git submodule update --init --recursive
```
