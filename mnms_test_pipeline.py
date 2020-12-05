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
        
    # Download trained models
    print('Downloading trained models ...')
    if not os.path.exists('trained_model'):
        os.makedirs('trained_model')
    
    model_name  = 'FCN_sa'
    for f in ['trained_model/{0}.meta'.format(model_name),
                'trained_model/{0}.index'.format(model_name),
                'trained_model/{0}.data-00000-of-00001'.format(model_name)]:
        urllib.request.urlretrieve(URL + f, f)
    
    if not os.path.exists('demo_result'):
            os.makedirs('demo_result')

    # Analyse show-axis images
    print('******************************')
    print('  Short-axis image analysis')
    print('******************************')

    # Deploy the segmentation network
    print('Deploying the segmentation network ...')
    os.system('SET CUDA_VISIBLE_DEVICES={0} & python common/deploy_network.py --seq_name sa --data_dir demo_image '
              '--result_dir demo_result --model_path trained_model/FCN_sa'.format(CUDA_VISIBLE_DEVICES))

    # Evaluate ventricular volumes
    print('Evaluating ventricular volumes ...')
    os.system('python short_axis/eval_ventricular_volume.py --data_dir demo_image '
              '--output_csv demo_csv/table_ventricular_volume.csv')

    # Evaluate wall thickness
    print('Evaluating myocardial wall thickness ...')
    os.system('python short_axis/eval_wall_thickness.py --data_dir demo_image '
              '--output_csv demo_csv/table_wall_thickness.csv')

    # To Do: Calculate dice metric

    # To Do: Calculate ejection fraction

    print('Done.')
