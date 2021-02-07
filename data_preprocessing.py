import os
import numpy as np
import nibabel as nib
from ukbb_cardiac.common.cardiac_utils import get_frames

def main():
    subset_dir = os.path.join("../M&Ms", 'train')
    for data in sorted(os.listdir(subset_dir)):
        print(data)
        data_dir = os.path.join(subset_dir, data)
        # Check the existence of the image and label map at ED and ES time frames
        # and add their file names to the list
        image_name = '{0}/{1}_sa.nii.gz'.format(data_dir, data)
        label_name = '{0}/{1}_sa_gt.nii.gz'.format(data_dir, data)
        if os.path.exists(image_name) and os.path.exists(label_name):
            # data_list += [[image_name, label_name, "ED"], [image_name, label_name, "ES"]]
            nim = nib.load(image_name)
            image = nim.get_fdata()
            nil = nib.load(label_name)
            label = nil.get_fdata()
            
            if image.shape != label.shape:
                print('Error: mismatched size, image.shape = {0}, '
                      'label.shape = {1}'.format(image.shape, label.shape))
                print('Skip {0}, {1}'.format(image_name, label_name))
                continue

            if image.max() < 1e-6:
                print('Error: blank image, image.max = {0}'.format(image.max()))
                print('Skip {0} {1}'.format(image_name, label_name))
                continue

            fr = get_frames(label, 'sa')
            for frame in ['ED', 'ES']:
                label_fr = label[:, :, :, fr[frame]] 
                image_fr = image[:, :, :, fr[frame]]

                image_fr = nib.Nifti1Image(image_fr, nim.affine)
                nib.save(image_fr, '{0}/{1}_sa_{2}.nii.gz'.format(data_dir, data, frame))
                
                label_fr = nib.Nifti1Image(label_fr, nil.affine)
                nib.save(label_fr, '{0}/{1}_sa_gt_{2}.nii.gz'.format(data_dir, data, frame))

if __name__ == "__main__":
    main()