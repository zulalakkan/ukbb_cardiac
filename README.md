## Overview

**uk_biobank_cardiac** is a toolbox used for processing and analysing cardiovascular magnetic resonance (CMR) images from the [UK Biobank Imaging Study](http://imaging.ukbiobank.ac.uk/). It consists of several parts:

* pre-processing the original DICOM images, converting them into NIfTI format, which is more convenient for image analysis;
* training fully convolutional networks for short-axis and long-axis CMR image segmentation;
* deploying the networks to new images.

## Installation

The toolbox is developed using [Python](https://www.python.org/downloads/) programming language. Python is usually installed by default on Linux and OSX machines but needs to be installed on Windows machines. Regarding the Python version, I use Python 3. But Python 2 may also work, since I have not used any function specific for Python 3.

The toolbox depends on some external libraries which need to be installed, including:

* tensorflow for deep learning;
* numpy and scipy for numerical computation;
* pandas and python-dateutil for handling spreadsheet;
* pydicom, SimpleITK for handling dicom images
* nibabel for reading and writing nifti images;
* opencv-python for transforming images in data augmentation.

The most convenient way to install these libraries is to use pip3 (or pip for Python 2) by running this command in the terminal:
```
pip3 install tensorflow-gpu numpy scipy pandas python-dateutil pydicom SimpleITK nibabel opencv-python
```

If you have problems in installing tensorflow, you can read this page [Installing TensorFlow](https://www.tensorflow.org/install/) for more information.

## Usage

