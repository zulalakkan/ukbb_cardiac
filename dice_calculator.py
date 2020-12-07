import os
import numpy as np
import pandas as pd
import nibabel as nib

print('Creating accuracy spreadsheet file ...')

if os.path.exists('DM_EJ_table.csv'):
    os.remove('DM_EJ_table.csv')
    
# Record ED ES frames to csv
init = {'Data': [],
        'Dice Metric-ED-LV': [],
        'Dice Metric-ED-LV Myocardium': [],
        'Dice Metric-ED-RV': [],
        'Dice Metric-ES-LV': [],
        'Dice Metric-ES-LV Myocardium': [],
        'Dice Metric-ES-RV': [],
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
        
        #TO DO -> ED ES times of ground truth
        seg_ED_gt = seg_gt[:, :, :, 0]  # ED frame value
        seg_ES_gt = seg_gt[:, :, :, 10]  # ES frame value A4A8, C4E9 ->8
    
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

                intersect = 0
                for a in range (seg_ED_gt.shape[0]):
                    for b in range (seg_ED_gt.shape[1]):
                        for c in range (seg_ED_gt.shape[2]):
                            
                            if fr == 'ED':
                                if(seg_model[a,b,c] == i and seg_ED_gt[a,b,c] == i):
                                    intersect += 1
                            else:
                                if(seg_model[a,b,c] == i and seg_ES_gt[a,b,c] == i):
                                    intersect += 1

                total_seg = np.sum(seg_model == i, axis=(0, 1, 2))
                print('Seg num in model: ', np.max(total_seg))
                
                
                print('Intersection: ',intersect)
                
                #denom = seg_ED_gt.shape[0]* seg_ED_gt.shape[1]* seg_ED_gt.shape[2]
                
                if fr == 'ED':
                    dice_metric = 2 *intersect / (total_seg + total_seg_ED) if (total_seg + total_seg_ED > 0) else 0 
                else:
                    dice_metric = 2 *intersect / (total_seg + total_seg_ES) if (total_seg + total_seg_ES > 0) else 0
                
                print("Dice metric for {0}: %".format(fr) , dice_metric * 100,'\n')
                
                dice_arr[ind] = dice_metric * 100
                ind += 1
        print('{0} finished'.format(folder))      

        frames_dict = {'Data': [folder],
                       'Dice Metric-ED-LV': [dice_arr[0]],
                       'Dice Metric-ED-LV Myocardium': [dice_arr[1]],
                       'Dice Metric-ED-RV': [dice_arr[2]],
                       'Dice Metric-ES-LV': [dice_arr[3]],
                       'Dice Metric-ES-LV Myocardium': [dice_arr[4]],
                       'Dice Metric-ES-RV': [dice_arr[5]],
                       }
        df1 = pd.DataFrame(frames_dict)
        df = df.append(df1, ignore_index = True)
    
    else:
        print('Error! Can not find one of the expected files: {0}/{1}_seg_sa_ED.nii.gz or {0}/{1}_sa_gt.nii.gz'.format(folder_dir, folder))

df.to_csv('DM_EJ_table.csv') # relative position