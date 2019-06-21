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
# =========================c===================================================
import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib


def pass_quality_control(eid, label, label_dict):
    """ Quality control """
    for l_name, l in label_dict.items():
        # Criterion 1: the atrium does not disappear at some point.
        T = label.shape[3]
        for t in range(T):
            label_t = label[:, :, 0, t]
            area = np.sum(label_t == l)
            if area == 0:
                logging.info('{0}: The area of {1} is 0 at time frame {2}.'.format(
                    eid, l_name, t))
                return False
    return True


# Evaluate the atrial area and length from 2 chamber or 4 chamber view images
def evaluate_area_length(label, nim, long_axis):
    # Area per pixel
    pixdim = nim.header['pixdim'][1:4]
    area_per_pix = pixdim[0] * pixdim[1] * 1e-2  # Unit: cm^2

    # Go through the label class
    L = []
    A = []
    landmarks = []
    labs = np.sort(list(set(np.unique(label)) - set([0])))
    for i in labs:
        # The binary label map
        label_i = (label == i)

        # Get the largest component in case we have a bad segmentation
        label_i = get_largest_cc(label_i)

        # Go through all the points in the atrium,
        # sort them by the distance along the long-axis
        points_label = np.nonzero(label_i)
        points = []
        for j in range(len(points_label[0])):
            x = points_label[0][j]
            y = points_label[1][j]
            points += [[x, y,
                        np.dot(np.dot(nim.affine, np.array([x, y, 0, 1]))[:3], long_axis)]]
        points = np.array(points)
        points = points[points[:, 2].argsort()]

        # The centre at the top part of the atrium (top third)
        n_points = len(points)
        top_points = points[int(2 * n_points / 3):]
        cx, cy, _ = np.mean(top_points, axis=0)

        # The centre at the bottom part of the atrium (bottom third)
        bottom_points = points[:int(n_points / 3)]
        bx, by, _ = np.mean(bottom_points, axis=0)

        # Determine the major axis by connecting the geometric centre and the bottom centre
        # TODO: in the future, determine the major axis using the mitral valve
        major_axis = np.array([cx - bx, cy - by])
        major_axis = major_axis / np.linalg.norm(major_axis)

        # Get the intersection between the major axis and the atrium contour
        px = cx + major_axis[0] * 100
        py = cy + major_axis[1] * 100
        qx = cx - major_axis[0] * 100
        qy = cy - major_axis[1] * 100

        if np.isnan(px) or np.isnan(py) or np.isnan(qx) or np.isnan(qy):
            return -1, -1, -1

        # Note the difference between nifti image index and cv2 image index
        # nifti image index: XY
        # cv2 image index: YX (height, width)
        image_line = np.zeros(label_i.shape)
        cv2.line(image_line, (int(qy), int(qx)), (int(py), int(px)), (1, 0, 0))
        image_line = label_i & (image_line > 0)
        # plt.figure()
        # image_rgb = np.zeros(label_i.shape + (3,))
        # image_rgb[:, :, 0] = label_i
        # image_rgb[:, :, 1] = image_line
        # plt.imshow(np.transpose(image_rgb, (1, 0, 2)), origin='lower')
        # plt.show()

        # Sort the intersection points by the distance along long-axis
        # and calculate the length of the intersection
        points_line = np.nonzero(image_line)
        points = []
        for j in range(len(points_line[0])):
            x = points_line[0][j]
            y = points_line[1][j]
            # World coordinate
            point = np.dot(nim.affine, np.array([x, y, 0, 1]))[:3]
            # Distance along the long-axis
            points += [np.append(point, np.dot(point, long_axis))]
        points = np.array(points)
        if len(points) == 0:
            return -1, -1, -1
        points = points[points[:, 3].argsort(), :3]
        L += [np.linalg.norm(points[-1] - points[0]) * 1e-1]  # Unit: cm

        # Calculate the area
        A += [np.sum(label_i) * area_per_pix]

        # Landmarks of the intersection points
        landmarks += [points[0]]
        landmarks += [points[-1]]
    return A, L, landmarks




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='dir_name', default='', required=True)
    parser.add_argument('--output_csv', metavar='csv_name', default='', required=True)
    args = parser.parse_args()

    data_path = args.data_dir
    data_list = sorted(os.listdir(data_path))
    table = []
    processed_list = []
    for data in data_list:
        data_dir = os.path.join(data_path, str(data))
        image_name = '{0}/sa.nii.gz'.format(data_dir)
        seg_name = '{0}/seg_sa.nii.gz'.format(data_dir)

        if os.path.exists(image_name) and os.path.exists(seg_name):
            print(data)

            # Image
            nim = nib.load(image_name)
            pixdim = nim.header['pixdim'][1:4]
            volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3
            density = 1.05

            # Heart rate
            duration_per_cycle = nim.header['dim'][4] * nim.header['pixdim'][4]
            heart_rate = 60.0 / duration_per_cycle

            # Segmentation
            seg = nib.load(seg_name).get_data()

            frame = {}
            frame['ED'] = 0
            vol_t = np.sum(seg == 1, axis=(0, 1, 2)) * volume_per_pix
            frame['ES'] = np.argmin(vol_t)

            val = {}
            for fr_name, fr in frame.items():
                # Clinical measures
                val['LV{0}V'.format(fr_name)] = np.sum(seg[:, :, :, fr] == 1) * volume_per_pix
                val['LV{0}M'.format(fr_name)] = np.sum(seg[:, :, :, fr] == 2) * volume_per_pix * density
                val['RV{0}V'.format(fr_name)] = np.sum(seg[:, :, :, fr] == 3) * volume_per_pix

            val['LVSV'] = val['LVEDV'] - val['LVESV']
            val['LVCO'] = val['LVSV'] * heart_rate * 1e-3
            val['LVEF'] = val['LVSV'] / val['LVEDV'] * 100

            val['RVSV'] = val['RVEDV'] - val['RVESV']
            val['RVCO'] = val['RVSV'] * heart_rate * 1e-3
            val['RVEF'] = val['RVSV'] / val['RVEDV'] * 100

            line = [val['LVEDV'], val['LVESV'], val['LVSV'], val['LVEF'], val['LVCO'], val['LVEDM'],
                    val['RVEDV'], val['RVESV'], val['RVSV'], val['RVEF']]
            table += [line]
            processed_list += [data]

    df = pd.DataFrame(table, index=processed_list,
                      columns=['LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)', 'LVEF (%)', 'LVCO (L/min)', 'LVM (g)',
                               'RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)', 'RVEF (%)'])
    df.to_csv(args.output_csv)
