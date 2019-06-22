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
working_dir = os.getcwd()


# Deploy the segmentation network
#  os.system('python3 {0}/../common/deploy_network_ao.py --seq_name ao '
#           '--data_dir /vol/bitbucket/wbai/own_work/ukbb_cardiac_demo '
#           '--model_path /vol/biomedic2/wbai/ukbb_cardiac/UKBB_18545/model/UNet-LSTM_ao_level5_filter16_22222_batch1_iter20000_lr0.001_zscore_tw9_h16_bidir_seq2seq_wR5_wr0.1_joint/UNet-LSTM_ao_level5_filter16_22222_batch1_iter20000_lr0.001_zscore_tw9_h16_bidir_seq2seq_wR5_wr0.1_joint.ckpt-20000'.format(working_dir))

# Evaluate aortic distensibility
os.system('python3 {0}/eval_aortic_area.py '
          '--data_dir /vol/bitbucket/wbai/own_work/ukbb_cardiac_demo/data '
          '--pressure_csv /vol/bitbucket/wbai/own_work/ukbb_cardiac_demo/csv/blood_pressure_info.csv '
          '--output_csv table_aortic_area.csv'.format(working_dir))
