#!/usr/bin/python3
import os
import sys
sys.path.append(os.getcwd())
from image_utils import *
from cardiac_utils import *


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {0} start_idx end_idx'.format(sys.argv[0]))
        exit(1)

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    print('Processing {0} subjects from {1} to {2} ...'.format(end_idx - start_idx, start_idx, end_idx))

    # Data directory
    data_path = '/vol/vipdata/data/biobank/cardiac/Application_18545/data'
    par_dir = '/vol/vipdata/data/biobank/cardiac/Application_18545/par'
    data_list = sorted(os.listdir(data_path))

    for data in data_list[start_idx:end_idx]:
        print(data)
        data_dir = os.path.join(data_path, data)

        # Skip data directories that are already processed
        if os.path.exists('{0}/wall_thickness_ED.csv'.format(data_dir)) \
            and os.path.exists('{0}/cine_2d_strain_sa_radial.csv'.format(data_dir)) \
            and os.path.exists('{0}/cine_2d_strain_sa_circum.csv'.format(data_dir)) \
            and os.path.exists('{0}/cine_2d_strain_la_4ch_longit.csv'.format(data_dir)):
            continue

        # Image segmentation quality control
        seg_sa_name = '{0}/seg_sa_ED.nii.gz'.format(data_dir)
        if not os.path.exists(seg_sa_name):
            continue
        if not sa_pass_quality_control(seg_sa_name):
            continue

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  Step 1: wall thickness analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        thickness_dir = os.path.join(data_dir, 'thickness')
        if not os.path.exists(thickness_dir):
            os.makedirs(thickness_dir)

        # Evaluate myocardial wall thickness
        seg_sa_name = '{0}/seg_sa_ED.nii.gz'.format(data_dir)
        output_name_stem = '{0}/wall_thickness_ED'.format(thickness_dir)
        evaluate_wall_thickness(seg_sa_name, output_name_stem)
        os.system('cp {0}*.csv {1}'.format(output_name_stem, data_dir))

        # Remove intermediate files
        os.system('rm -rf {0}'.format(thickness_dir))

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #  Step 2: cine MR motion and strain analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        motion_dir = os.path.join(data_dir, 'cine_motion')
        if not os.path.exists(motion_dir):
            os.makedirs(motion_dir)

        # Perform motion tracking on short-axis images and calculate the strain
        output_name_stem = '{0}/cine_2d_strain_sa'.format(motion_dir)
        cine_2d_sa_motion_and_strain_analysis(data_dir, par_dir, motion_dir, output_name_stem)
        os.system('cp {0}*.* {1}'.format(output_name_stem, data_dir))

        # Perform motion tracking on long-axis images and calculate the strain
        seg_la_name = '{0}/seg2_la_4ch_ED.nii.gz'.format(data_dir)
        if os.path.exists(seg_la_name):
            if la_pass_quality_control(seg_la_name):
                output_name_stem = '{0}/cine_2d_strain_la_4ch'.format(motion_dir)
                cine_2d_la_motion_and_strain_analysis(data_dir, par_dir, motion_dir, output_name_stem)
                os.system('cp {0}*.* {1}'.format(output_name_stem, data_dir))

        # Remove intermediate files
        os.system('rm -rf {0}'.format(motion_dir))