"""
    This script is for testing FCN model with short axis CMR images from M&Ms OpenDataset.
    Calculates dice metric and ejection fraction on model's segmentation results.
    """
import os
import urllib.request
import shutil


if __name__ == '__main__':
    # The GPU device id
    CUDA_VISIBLE_DEVICES = 0

    # Download information spreadsheet
    print('Creating information spreadsheet folder ...')
    if not os.path.exists('demo_csv'):
        os.makedirs('demo_csv')
        
    print('Cleaning workspace ...')
    os.system('python result_cleaner.py')

    # Analyse show-axis images
    print('******************************')
    print('  Short-axis image analysis')
    print('******************************')

    # Deploy the segmentation network
    print('Deploying the segmentation network ...')
    os.system('SET CUDA_VISIBLE_DEVICES={0} & python common/deploy_network.py --seq_name sa --data_dir demo_image '
              '--result_csv demo_csv/result.csv --model_path trained_model/FCN_sa'.format(CUDA_VISIBLE_DEVICES))

    # Evaluate ventricular volumes
    print('Evaluating ventricular volumes ...')
    os.system('python short_axis/eval_ventricular_volume.py --data_dir demo_image '
              '--output_csv demo_csv/table_ventricular_volume.csv --eval_gt')

    # Evaluate dice metric
    print('Evaluating dice scores ...')
    os.system('python dice_calculator.py --output_csv demo_csv/table_dice_score.csv')


    print('Done.')
