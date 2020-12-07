import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from ukbb_cardiac.common.cardiac_utils import get_frames
from ukbb_cardiac.common.image_utils import np_categorical_dice

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_csv', metavar='csv_name', default='DM_EJ_table.csv', required=True)
    args = parser.parse_args()

    print('Creating accuracy spreadsheet file ...')

    if os.path.exists(args.output_csv):
        os.remove(args.output_csv)
        
    # Record ED ES frames to csv
    init = {'Data': [],
            'EDLV': [],
            'EDLVM': [],
            'EDRV': [],
            'ESLV': [],
            'ESLVM': [],
            'ESRV': [],
            }

    df = pd.DataFrame(init)

    root = './demo_image'
    folder_list = sorted(os.listdir(root))

    for folder in folder_list:
        folder_dir = os.path.join(root, folder)
        if os.path.exists('{0}/{1}_seg_sa_ED.nii.gz'.format(folder_dir, folder) and ('{0}/{1}_seg_sa_ES.nii.gz'.format(folder_dir, folder))
                        and ('{0}/{1}_sa_gt.nii.gz'.format(folder_dir, folder))):
            
            seg_sa_ED = '{0}/{1}_seg_sa_ED.nii.gz'.format(folder_dir, folder)
            seg_sa_ES = '{0}/{1}_seg_sa_ES.nii.gz'.format(folder_dir, folder)
            seg_sa_ground_truth = '{0}/{1}_sa_gt.nii.gz'.format(folder_dir, folder)
            ##seg_sa_ED ='{0}/{1}_sa_gt.nii.gz'.format(folder_dir, folder) # To see Dice metric between same segmentations is 1
            
            seg_gt = nib.load(seg_sa_ground_truth).get_fdata()
            #print(seg_gt.shape)
            
            fr = get_frames(seg_gt, 'sa')
            seg_ED_gt = seg_gt[:, :, :, fr['ED']]  # ED frame value 0
            seg_ES_gt = seg_gt[:, :, :, fr['ES']]  # ES frame value A4A8 10, C4E9 ->8
        
            dice_arr = np.zeros(6)
            ind = 0
            
            frames = ['ED','ES']
            segm =  ['LV','LV Myocardium','RV']
            for fr in frames:
                print('\nFor image {0}, Comparison between: {1} \n'.format(folder, fr))

                seg_model = nib.load(seg_sa_ED).get_fdata() if fr == 'ED' else nib.load(seg_sa_ES).get_fdata()
                print(seg_model.shape)
                ##if fr == 'ED' : seg_model = seg_model[:,:,:,0] # To see Dice metric between same segmentations is 1
                
                
                for i in range(1,4): # Loop for all segmentations
                    print('Calculate Dice metric for ',segm[i - 1])
                    
                    total_seg_ED = np.sum(seg_ED_gt == i, axis=(0, 1, 2))
                    print('Seg num (', segm[i-1],') in ground truth ED: ',np.max(total_seg_ED))
                    total_seg_ES = np.sum(seg_ES_gt == i, axis=(0, 1, 2))
                    print('Seg num (', segm[i-1],') in ground truth ES: ',np.max(total_seg_ES))

                    total_seg = np.sum(seg_model == i, axis=(0, 1, 2))
                    print('Seg num in model: ', np.max(total_seg))
                    
                    #denom = seg_ED_gt.shape[0]* seg_ED_gt.shape[1]* seg_ED_gt.shape[2]
                    
                    if fr == 'ED':
                        dice_metric = np_categorical_dice(seg_model, seg_ED_gt, i) if (total_seg + total_seg_ED > 0) else 0 
                    else:
                        dice_metric = np_categorical_dice(seg_model, seg_ES_gt, i) if (total_seg + total_seg_ES > 0) else 0
                    
                    print("Dice metric for {0}: %".format(fr) , dice_metric * 100,'\n')
                    
                    dice_arr[ind] = dice_metric * 100
                    ind += 1
            print('{0} finished'.format(folder))      

            frames_dict = {'Data': [folder],
                        'EDLV': [dice_arr[0]],
                        'EDLVM': [dice_arr[1]],
                        'EDRV': [dice_arr[2]],
                        'ESLV': [dice_arr[3]],
                        'ESLVM': [dice_arr[4]],
                        'ESRV': [dice_arr[5]],
                        }
            df1 = pd.DataFrame(frames_dict)
            df = df.append(df1, ignore_index = True)
        
        else:
            print('Error! Can not find one of the expected files: {0}/{1}_seg_sa_ED.nii.gz or {0}/{1}_sa_gt.nii.gz'.format(folder_dir, folder))

    df.to_csv(args.output_csv)