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
# ==============================================================================
import os

# Deploy the segmentation network
os.system('python3 deploy_network.py --seq_name sa '
          '--data_dir /vol/bitbucket/wbai/own_work/ukbb_cardiac_demo '
          '--model_path /homes/wbai/public_html/data/ukbb_cardiac/trained_model/FCN_sa')

# Evaluate ventricular volumes
os.system('python3 eval_ventricular_volume.py '
          '--data_dir /vol/bitbucket/wbai/own_work/ukbb_cardiac_demo '
          '--output_csv table_ventricular_volume.csv')

# Evaluate wall thickness
os.system('python3 eval_wall_thickness.py '
          '--data_dir /vol/bitbucket/wbai/own_work/ukbb_cardiac_demo '
          '--output_csv table_wall_thickness.csv')

# Evaluate strain values
os.system('python3 eval_strain_sax.py '
          '--data_dir /vol/bitbucket/wbai/own_work/ukbb_cardiac_demo '
          '--par_dir /vol/biomedic2/wbai/git/ukbb_cardiac/par '
          '--output_csv table_strain_sax.csv')
