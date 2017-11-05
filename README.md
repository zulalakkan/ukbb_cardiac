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

## Usage

**A quick demo** You can go to the *segmentation* directory and run the demo file there:
```
cd segmentation
python3 demo.py
```
There is one parameter in the script, *CUDA_VISIBLE_DEVICES*, which controls which GPU device to use on your machine. Currently, I set it to 0, which means the first GPU on your machine.

This script will download two exemplar short-axis cardiac MR images and a pre-trained network, then segment the left and right ventricles using the network, saving the segmentation results *seg_sa.nii.gz* and also saving the clinical measures in a spreadsheet *clinical_measure.csv*, including the left ventricular end-diastolic volume (LVEDV), end-systolic volume (LVESV), myocardial mass (LVM) and the right ventricular end-diastolic volume (RVEDV), end-systolic volume (RVESV). The script will also download exemplar long-axis cardiac MR images and segment the left and right atria.

**Speed** The speed of image segmentation depends several factors, such as whether to use GPU or CPU, the GPU hardware, the test image size etc. In my case, I use a Nvidia Titan K80 GPU and it takes about 9.5 seconds to segment a full time sequence (50 time frames) for each subject, with the image size to be 198x208x10x50 (i.e. 10 image slices and 50 time frames). If I only need to segment the end-diastolic (ED) and end-systolic (ES) time frames, it is much faster and takes about 2.2 seconds.

**To know more** If you want to know more about how the network works and how it is trained, you can read these following files under the *segmentation* directory:
* network.py, which describes the neural network architecture;
* train_network.py, which trains a network on a dataset with both images and manual annotations;
* deploy_network.py, which deploys the trained network onto new images. If you are interested in deploying the pre-trained network to more UK Biobank cardiac image set, this is the file that you need to read.

**Data preparation** You will notice there is another directory named *data*, which contains the scripts for preparing the training dataset. For a machine learning project, data preparation step including acquisition, cleaning, format conversion etc normally takes at least the same amount of your time and headache, if nor more, as the machine learning step. But this is a crucial part, as all the following work (your novel machine learning ideas) needs the data.

In my project, I use imaging data from [the UK Biobank](http://www.ukbiobank.ac.uk/), which is a very large clinical research resource. Its sub-project, [the UK Biobank Imaging Study](http://imaging.ukbiobank.ac.uk/), aims to conduct detailed MRI imaging scans of the vital organs of over 100,000 participants. Researchers can [apply](http://www.ukbiobank.ac.uk/register-apply/) to use the UK Biobank data resource for health-related research in the public interest.

I have written the following scripts for preparing the UK Biobank cardiac imaging data:
* convert_data_ukbb2964.py and prepare_data_ukbb2964.py, which prepare the cardiac images and manual annotation of 5,000 subjects under UK Biobank Application 2964. This is the dataset that I used for training the network.
* download_data_ukbb_general.py, which shows how to download cardiac MR images and convert the image format from dicom to nifti for a general UK Biobank application. You may adapt this script to prepare your data, if they also come from the UK Biobank.

## References

We would like to thank all the UK Biobank participants and staff who make the CMR imaging dataset possible and also people from Queen Mary's University London and Oxford University who performed the hard work of manual annotation. In case you find the toolbox or a certain part of it useful, please consider giving appropriate credit to it by citing one or some of the papers here, which respectively describes the segmentation method [1] and the manual annotation dataset [2]. Thanks.

[1] W. Bai, et al. Human-level CMR image analysis with deep fully convolutional networks. arXiv:1710.09289. [arxiv](https://arxiv.org/abs/1710.09289)

[2] S. Petersen, et al. Reference ranges for cardiac structure and function using cardiovascular magnetic resonance (CMR) in Caucasians from the UK Biobank population cohort. Journal of Cardiovascular Magnetic Resonance, 19:18, 2017. [doi](https://doi.org/10.1186/s12968-017-0327-9)