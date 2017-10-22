## Overview

**uk_biobank_cardiac** is a toolbox used for processing and analysing cardiovascular magnetic resonance (CMR) images from the [UK Biobank Imaging Study](http://imaging.ukbiobank.ac.uk/). It consists of several parts:

* pre-processing the original DICOM images, converting them into NIfTI format, which is more convenient for image analysis;
* training fully convolutional networks for short-axis and long-axis CMR image segmentation;
* deploying the networks to new images.

## Installation

The toolbox is developed using [Python](https://www.python.org) programming language. Python is usually installed by default on Linux and OSX machines but may need to be installed on Windows machines. Regarding the Python version, I use Python 3. But Python 2 may also work, since I have not used any function specific for Python 3.

The toolbox depends on some external libraries which need to be installed, including:

* [tensorflow](https://www.tensorflow.org) for deep learning;
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

**A quick demo** You can go to the *segmentation* directory and run the demo file there:
```
cd segmentation
python3 demo.py
```
There is one parameter in the script, *CUDA_VISIBLE_DEVICES*, which controls which GPU device to use on your machine. Currently, I set it to 0, which means the first GPU on your machine.

This script will download two exemplar cardiac MR images and a pre-trained network, then segment the images using the network, saving the segmentation results *seg_sa.nii.gz* and also saving the clinical measures in a spreadsheet *clinical_measure.csv*, including the left ventricular end-diastolic volume (LVEDV), end-systolic volume (LVESV), myocardial mass (LVM) and the right ventricular end-diastolic volume (RVEDV), end-systolic volume (RVESV).

**To know more** If you want to know more about how the network works and how it is trained, you can read these following files under the *segmentation* directory:
* network.py, which describes the neural network architecture;
* train_network.py, which trains a network on a dataset with both images and manual annotations;
* deploy_network.py, which deploys the trained network onto new images, i.e. your test set.

You will notice there is another directory named *data*, which contains the scripts for preparing the training dataset. For a machine learning project, data preparation step including acquisition, cleaning, format conversion etc normally takes at least the same amount of your time and headache, if nor more, as the machine learning step. But this is a crucial part, as all the following work (your novel machine learning ideas) needs the data.

In my project, I use imaging data from [the UK Biobank](http://www.ukbiobank.ac.uk/), which is a very large clinical research resource. Its sub-project, [the UK Biobank Imaging Study](http://imaging.ukbiobank.ac.uk/), aims to conduct detailed MRI imaging scans of the vital organs of over 100,000 participants. Researchers can [apply](http://www.ukbiobank.ac.uk/register-apply/) to use the UK Biobank data resource for health-related research in the public interest.

I have written the following scripts for preparing the UK Biobank cardiac imaging data:
* convert_data_ukbb2964.py and prepare_data_ukbb2964.py, which prepare the cardiac images and manual annotation of 5,000 subjects under UK Biobank Application 2964. This is the dataset that I used for training the network.
* download_data_ukbb_general.py, which downloads cardiac MR images and converts the image format from dicom to nifti for a general UK Biobank application. You may adapt this script to prepare your data, if they also come from the UK Biobank.

