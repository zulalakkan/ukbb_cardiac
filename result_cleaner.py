import os

root = './demo_image'
folder_list = sorted(os.listdir(root)) # [1, 2]
images = ['sa.nii.gz', 'la_2ch.nii.gz', 'la_3ch.nii.gz', 'la_4ch.nii.gz', 'sa_gt.nii.gz']
for folder in folder_list:
    folder_dir = os.path.join(root, folder)
    file_list = sorted(os.listdir(folder_dir))
    for _file in file_list:
        if _file not in ['{0}_{1}'.format(folder, image) for image in images]:
            os.remove(os.path.join(folder_dir, _file))
    