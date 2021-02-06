import os
import time
import random
import numpy as np
import nibabel as nib
import tensorflow as tf
from ukbb_cardiac.common.cardiac_utils import get_frames
from ukbb_cardiac.common.image_utils import tf_categorical_accuracy, tf_categorical_dice
from ukbb_cardiac.common.image_utils import crop_image, rescale_intensity, data_augmenter

# modified version of ./common/train_network.py
def get_random_batch(filename_list, batch_size, image_size=192, data_augmentation=False,
                     shift=0.0, rotate=0.0, scale=0.0, intensity=0.0, flip=False):
    # Randomly select batch_size images from filename_list
    n_file = len(filename_list)
    print(n_file)
    n_selected = 0
    images = []
    labels = []
    nims = []
    while n_selected < batch_size:
        rand_index = random.randrange(n_file)
        image_name, label_name, frame = filename_list[rand_index]
        if os.path.exists(image_name) and os.path.exists(label_name):
            print('  Select {0} {1} {2}'.format(image_name, label_name, frame))

            # Read image and label
            nim = nib.load(image_name)
            image = nim.get_fdata()
            nil = nib.load(label_name)
            label = nil.get_fdata()

            fr = get_frames(label, 'sa')
            label = label[:, :, :, fr[frame]] 
            image = image[:, :, :, fr[frame]]
            
            # Handle exceptions
            if image.shape != label.shape:
                print('Error: mismatched size, image.shape = {0}, '
                      'label.shape = {1}'.format(image.shape, label.shape))
                print('Skip {0}, {1}'.format(image_name, label_name))
                continue

            if image.max() < 1e-6:
                print('Error: blank image, image.max = {0}'.format(image.max()))
                print('Skip {0} {1}'.format(image_name, label_name))
                continue

            # Normalise the image size
            X, Y, Z = image.shape
            cx, cy = int(X / 2), int(Y / 2)
            image = crop_image(image, cx, cy, image_size)
            label = crop_image(label, cx, cy, image_size)

            # Intensity rescaling
            image = rescale_intensity(image, (1.0, 99.0))

            print(image.shape)
            # Append the image slices to the batch
            # Use list for appending, which is much faster than numpy array
            for z in range(Z):
                images += [image[:, :, z]]
                labels += [label[:, :, z]]
                nims.append(nim)

            # Increase the counter
            n_selected += 1

    # Convert to a numpy array
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(images.shape)

    # Add the channel dimension
    # tensorflow by default assumes NHWC format
    images = np.expand_dims(images, axis=3)

    # Perform data augmentation
    if data_augmentation:
        images, labels = data_augmenter(images, labels,
                                        shift=shift, rotate=rotate,
                                        scale=scale,
                                        intensity=intensity, flip=flip)
    print(images.shape)
    return images, labels, nims

def main(argv=None):
    TRAIN_ITER = 10000
    data_list = {}
    for k in ['train', 'validation']:
        subset_dir = os.path.join("./dataset", k)
        data_list[k] = []
        for data in sorted(os.listdir(subset_dir)):
            data_dir = os.path.join(subset_dir, data)
            # Check the existence of the image and label map at ED and ES time frames
            # and add their file names to the list
            image_name = '{0}/{1}_sa.nii.gz'.format(data_dir, data)
            label_name = '{0}/{1}_sa_gt.nii.gz'.format(data_dir, data)
            if os.path.exists(image_name) and os.path.exists(label_name):
                data_list[k] += [[image_name, label_name, "ED"], [image_name, label_name, "ES"]]

    
    # print(images.shape,len(nims))
    # for i, im in enumerate(nims):
    #     nim2 = nib.Nifti1Image(images[i], im.affine)
    #     nib.save(nim2, './test/{0}.nii.gz'.format(i))
    #     image = nim2.get_fdata()
    #     print(i, image.shape)


    with tf.compat.v1.Session() as sess:
        print('Start training...')
        start_time = time.time()
        
        # Import the computation graph and restore the variable values
        saver = tf.compat.v1.train.import_meta_graph("./FCN_sa.meta")
        saver.restore(sess, '{0}'.format("./FCN_sa"))

        train_op = tf.compat.v1.get_collection('train_op')[0]
        print(train_op)
        for iteration in range(TRAIN_ITER):
            # For each iteration, we randomly choose a batch of subjects
            print('Iteration {0}: training...'.format(iteration))
            start_time_iter = time.time()
        
            images, labels, _ = get_random_batch(data_list['train'],
                                        batch_size=5,
                                        image_size=192,
                                        data_augmentation=False,
                                        shift=0, rotate=20, scale=0,
                                        intensity=0, flip=False)
            
            # Stochastic optimisation using this batch
            sess.run([train_op],
                    {'image:0': images, 'label:0': labels, 'training:0': True})
              

            print('Iteration {} of {} took {:.3f}s'.format(iteration, TRAIN_ITER,
                                                               time.time() - start_time_iter))

            # Save models after every 1000 iterations (1 epoch)
            # One epoch needs to go through
            #   1000 subjects * 2 time frames = 2000 images = 1000 training iterations
            # if one iteration processes 2 images.
            if iteration % 1000 == 0:
                saver.save(sess, save_path=os.path.join("./trained_model", '{0}.ckpt'.format("FCN_sa2")),
                           global_step=iteration)

        print('Training took {:.3f}s in total.\n'.format(time.time() - start_time))
        print("Training is over")

if __name__ == '__main__':
    tf.compat.v1.app.run()