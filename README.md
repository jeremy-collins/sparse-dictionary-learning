# Sparse Dictionary Learning

This project demonstrates the implementation of sparse dictionary learning for image representation using Python and Numpy.


## Installation

First, clone this repository by running:
`git clone https://github.com/jeremy-collins/sparse-dictionary-learning.git`

## Requirements

To run the code, you will need Python 3.8, and you can install the following packages with pip:

- numpy
- matplotlib
- opencv-python
- PyYAML

You can install these packages using the provided `environment.yaml` file by running:
`conda env create --file environment.yml`


## Usage

To run the code, provide the path to the input image as an argument:
`python sparse_dict.py <path_to_image>`

For example, using the image included in this repo, we can use the following command:
`python sparse_dict.py images/jeremy_and_yann.jpg`

This code will display the original image and reconstructed images using different numbers of atoms specified in `config.yaml`, where you can find additional hyperparameters for this script.

## Report

For a detailed description of the results and discussion, please refer to `report.md`.