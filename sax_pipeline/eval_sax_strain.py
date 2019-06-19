# Copyright 2019, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import argparse
import pandas as pd
from ukbb_cardiac.common.cardiac_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='dir_name', default='', required=True)
    parser.add_argument('--output_csv', metavar='csv_name', default='', required=True)
    parser.add_argument('--par_dir', metavar='dir_name', default='', required=True)
    parser.add_argument('--start_idx', metavar='start index', type=int, default=0)
    parser.add_argument('--end_idx', metavar='end index', type=int, default=0)
    args = parser.parse_args()

    data_path = args.data_dir
    data_list = sorted(os.listdir(data_path))
    n_data = len(data_list)
    start_idx = args.start_idx
    end_idx = n_data if args.end_idx == 0 else args.end_idx
    table = []
    processed_list = []
    for data in data_list[start_idx:end_idx]:
        print(data)
        data_dir = os.path.join(data_path, data)

        # Quality control for segmentation at ED
        # If the segmentation quality is low, the following functions may fail.
        seg_sa_name = '{0}/seg_sa_ED.nii.gz'.format(data_dir)
        if not os.path.exists(seg_sa_name):
            continue
        if not sa_pass_quality_control(seg_sa_name):
            continue

        # Intermediate result directory
        motion_dir = os.path.join(data_dir, 'cine_motion')
        if not os.path.exists(motion_dir):
            os.makedirs(motion_dir)

        # Perform motion tracking on short-axis images and calculate the strain
        cine_2d_sa_motion_and_strain_analysis(data_dir,
                                              args.par_dir,
                                              motion_dir,
                                              '{0}/cine_2d_strain_sa'.format(data_dir))

        # Remove intermediate files
        # os.system('rm -rf {0}'.format(motion_dir))