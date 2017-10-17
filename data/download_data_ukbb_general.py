import os, csv, glob, re, time
import pandas as pd
from biobank_utils import *
import dateutil.parser


def repl(m):
    return '{}{}-{}-20{}'.format(m.group(1), m.group(2), m.group(3), m.group(4))


# Read the lines in the manifest.csv file and check whether the date format consists a comma, which needs to be removed
# since it affects parsing the file.
def process_manifest(name, name2):
    with open(name2, 'w') as f2:
        with open(name, 'r') as f:
            for line in f:
                line2 = re.sub('([A-Z])(\w{2}) (\d{1,2}), 20(\d{2})', repl, line)
                f2.write(line2)


# Paths
csv_dir = '/vol/vipdata/data/biobank/cardiac/Application_18545/downloaded'
util_dir = '/vol/vipdata/data/biobank/cardiac/Application_17806/util'
dest_root = '/vol/biomedic/users/wbai/data/biobank/download'
data_root = '/vol/vipdata/data/biobank/cardiac/Application_18545/data'

# The authentication file (application id + password)
ukbkey = '/homes/wbai/ukbkey'

# Parse the spreadsheet, create batch file and download data
csv_file = 'ukb9137_image_subset.csv'
df = pd.read_csv(os.path.join(csv_dir, csv_file), header=1)
# df = pd.read_csv('/vol/vipdata/data/biobank/cardiac/Application_18545/CMR.csv')

start_idx = 5790
end_idx = len(df) #10

for i in range(start_idx, end_idx):
    eid = '{}'.format(df.loc[i, 'eid'])
    print(eid)

    # Destination directories
    dicom_dir = os.path.join(dest_root, eid)
    data_dir = os.path.join(data_root, eid)

    if os.path.exists(data_dir) and os.path.exists(os.path.join(data_dir, 'sa.nii.gz')):
        print('{0}: Already downloaded. Skip ...'.format(i))
        continue

    if not os.path.exists(dicom_dir):
        os.system('mkdir -p {0}'.format(dicom_dir))

    if not os.path.exists(data_dir):
        os.system('mkdir -p {0}'.format(data_dir))

    # Create the batch file for this subject
    batch_file = os.path.join(dest_root, '{0}_batch'.format(eid))
    with open(batch_file, 'w') as f_batch:
        for j in range(20208, 20210):
            field = '{0}-2.0'.format(j)
            # if pd.notnull(df.loc[i, field]):
                # f_batch.write('{0} {1}\n'.format(eid, df.loc[i, field]))
            f_batch.write('{0} {1}_2_0\n'.format(eid, j))

    # Download the data
    ukbfetch = os.path.join(util_dir, 'ukbfetch')
    print('{0}: Downloading data for subject {1} ...'.format(i, eid))
    os.system('{0} -b{1} -a{2}'.format(ukbfetch, batch_file, ukbkey))

    # Unpack the data
    files = glob.glob('{0}_*.zip'.format(eid))
    for f in files:
        os.system('unzip -o {0} -d {1}'.format(f, dicom_dir))

        # Organise the dicom files
        # Process the manifest file
        if os.path.exists(os.path.join(dicom_dir, 'manifest.cvs')):
            os.system('cp {0} {1}'.format(os.path.join(dicom_dir, 'manifest.cvs'),
                                          os.path.join(dicom_dir, 'manifest.csv')))
        process_manifest(os.path.join(dicom_dir, 'manifest.csv'), \
                         os.path.join(dicom_dir, 'manifest2.csv'))
        df2 = pd.read_csv(os.path.join(dicom_dir, 'manifest2.csv'), error_bad_lines=False)

        # Patient ID and acquisition date
        # These information could also be obtained from the Dicom headers
        pid = df2.at[0, 'patientid']
        if not os.path.exists('Patient_ID'):
            os.system('echo {0} > {1}/Patient_ID'.format(pid, data_dir))

        date = dateutil.parser.parse(df2.at[0, 'date'][:11]).date().isoformat()
        if not os.path.exists('Date'):
            os.system('echo {0} > {1}/Date'.format(date, data_dir))

        # Group the files into subdirectories
        for series_name, series_df in df2.groupby('series discription'):
            series_dir = os.path.join(dicom_dir, series_name)
            if not os.path.exists(series_dir):
                os.mkdir(series_dir)
            series_files = [os.path.join(dicom_dir, x) for x in series_df['filename']]
            os.system('mv {0} {1}'.format(' '.join(series_files), series_dir))

    # Convert dicom to nifti
    dset = Biobank_Dataset(dicom_dir)
    dset.read_dicom_images()
    dset.convert_dicom_to_nifti(data_dir)

    # Remove intermediate files
    os.system('rm -rf {0}'.format(dicom_dir))
    os.system('rm -f {0}'.format(batch_file))
    os.system('rm -f {0}_*.zip'.format(eid))


# csv_file = 'ukb7393_image_subset.csv'
# with open(os.path.join(csv_dir, csv_file), 'r') as csv_f:
#     reader = csv.reader(csv_f, delimiter=',')
#     row = next(reader)
#     row = next(reader)
#     d = dict((field_id, i) for i, field_id in enumerate(row))
#     start_time = time.time()
#
#     # For each subject, download the available DICOM images
#     count = 0
#     for row in reader:
#         # Subject ID
#         eid = row[0]
#         print(eid)
#         count += 1
#
#         # Destination directories
#         dicom_dir = os.path.join(dest_root, eid)
#         data_dir = os.path.join(nifti_root, eid)
#
#         if os.path.exists(data_dir) and os.path.exists(os.path.join(data_dir, 'sa.nii.gz')):
#             print('Already downloaded. Skip ...')
#             continue
#
#         if eid == '2182277':
#             continue
#
#         if not os.path.exists(dicom_dir):
#             os.system('mkdir -p {0}'.format(dicom_dir))
#
#         if not os.path.exists(data_dir):
#             os.system('mkdir -p {0}'.format(data_dir))
#
#         # Go to the destination directory
#         pre_dir = os.getcwd()
#         os.chdir(dicom_dir)
#
#         # Create the batch file for this subject
#         batch_file = os.path.join(dest_root, '{0}_batch'.format(eid))
#         with open(batch_file, 'w') as f_batch:
#             #for i in range(20208, 20215):
#             for i in range(20208, 20210):
#                 field = '{0}-2.0'.format(i)
#                 if row[d[field]]:
#                     f_batch.write('{0} {1}\n'.format(eid, row[d[field]]))
#
#         # # Download the data
#         # ukbfetch = os.path.join(util_dir, 'ukbfetch')
#         # print('{0}: Downloading data for subject {1} ...'.format(count, eid))
#         # os.system('{0} -b{1} -a{2}'.format(ukbfetch, batch_file, ukbkey))
#         #
#         # # Unpack the data
#         # files = glob.glob('{0}_*.zip'.format(eid))
#         # for f in files:
#         #     os.system('unzip {0}'.format(f))
#         #
#         #     # Edit the manifest file
#         #     process_manifest('manifest.csv', 'manifest2.csv')
#         #     df = pd.read_csv('manifest2.csv')
#         #
#         #     # Patient ID and acquisition date
#         #     # These information could also be obtained from the Dicom headers
#         #     pid = df.at[0, 'patientid']
#         #     if not os.path.exists('Patient_ID'):
#         #         os.system('echo {0} > Patient_ID'.format(pid))
#         #
#         #     date = dateutil.parser.parse(df.at[0, 'date'][:11]).date().isoformat()
#         #     if not os.path.exists('Date'):
#         #         os.system('echo {0} > Date'.format(date))
#         #
#         #     # Group the files into subdirectories
#         #     for group_name, group_df in df.groupby('series discription'):
#         #         if not os.path.exists(group_name):
#         #             os.mkdir(group_name)
#         #         os.system('mv {0} {1}'.format(' '.join(group_df['filename']), group_name))
#         #
#         #     os.system('rm -f manifest.csv manifest2.csv')
#
#         # Convert dicom to nifti
#         dset = Biobank_Dataset(dicom_dir)
#         dset.read_dicom_images()
#         #dset.read_sax_dicom_images()
#         #dset.read_lax_dicom_images()
#         dset.convert_dicom_to_nifti()
#
#         break
#
#         os.system('mv {0}/Patient_ID {1}'.format(dicom_dir, data_dir))
#         os.system('mv {0}/Date {1}'.format(dicom_dir, data_dir))
#         os.system('mv {0}/*.nii.gz {1}'.format(dicom_dir, data_dir))
#
#         # Remove intermediate files
#         os.system('rm -f {0}'.format(batch_file))
#         os.system('rm -f {0}_*.zip'.format(eid))
#         os.chdir(pre_dir)
#         os.system('rm -rf {0}'.format(dicom_dir))
