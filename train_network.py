# Copyright 2017, Wenjia Bai. All Rights Reserved.
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
import os, time, random
import numpy as np, nibabel as nib
import tensorflow as tf
from network import *
from image_utils import *


""" Training parameters """
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_size', 192, 'Image size after cropping.')
tf.app.flags.DEFINE_integer('train_batch_size', 2, 'Number of subjects for each training batch.')
tf.app.flags.DEFINE_integer('validation_batch_size', 2, 'Number of subjects for each validation batch.')
tf.app.flags.DEFINE_integer('train_iteration', 50000, 'Number of training iterations.')
tf.app.flags.DEFINE_integer('num_filter', 16, 'Number of filters for the first convolution layer.')
tf.app.flags.DEFINE_integer('num_level', 5, 'Number of network levels.')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
tf.app.flags._global_parser.add_argument('--seq_name', choices=['sa', 'la_2ch', 'la_4ch'],
                                         default='sa', help='Sequence name for training.')
tf.app.flags._global_parser.add_argument('--model', choices=['FCN', 'FCN_legacy', 'ResNet', 'ResNet_legacy'],
                                         default='FCN', help='Model name.')
tf.app.flags._global_parser.add_argument('--optimizer', choices=['Adam', 'SGD', 'Momentum'],
                                         default='Adam', help='Optimizer.')
tf.app.flags.DEFINE_string('dataset_dir', '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/sa',
                           'Path to the dataset, which is split into training, validation and test.')
tf.app.flags.DEFINE_string('log_dir', '/vol/bitbucket/wbai/cardiac_cnn_tf/Biobank/log',
                           'Path for saving the log file.')
tf.app.flags.DEFINE_string('checkpoint_dir', '/vol/bitbucket/wbai/cardiac_cnn_tf/Biobank/model',
                           'Path for saving the trained model.')


def get_random_batch(filename_list, batch_size, seq_name, image_size=192,
                     data_augmentation=False, shift=0, rotate=0, scale=0, intensity=0, flip=False):
    # TODO: remove seq_name, which should belong to the pre-processing step
    # Randomly select batch_size images from filename_list
    n_file = len(filename_list)
    n_selected = 0
    images = []
    labels = []
    while n_selected < batch_size:
        rand_index = random.randrange(n_file)
        image_name, label_name = filename_list[rand_index]
        if os.path.exists(image_name) and os.path.exists(label_name):
            print('  Select {0} {1}'.format(image_name, label_name))

            # Read image and label
            image = nib.load(image_name).get_data()
            label = nib.load(label_name).get_data()

            # Handle exceptions
            if image.shape != label.shape:
                print('Error: mismatched size, image.shape = {0}, label.shape = {1}'.format(image.shape, label.shape))
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

            # Change the labels for la_2ch and la_4ch
            # la_2ch: LA: 4 -> 1
            # la_4ch: LA: 4 -> 1, RA: 5 -> 2
            if seq_name == 'la_2ch':
                label = (label == 4).astype(np.int32)
            elif seq_name == 'la_4ch':
                label = ((label == 4) + (label == 5) * 2).astype(np.int32)

            # Append the image slices to the batch
            # Use list for appending, which is much faster than numpy array
            for z in range(Z):
                images += [image[:, :, z]]
                labels += [label[:, :, z]]

            # Increase the counter
            n_selected += 1

    # Convert to a numpy array
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Add the channel dimension
    # tensorflow by default assumes NHWC format
    images = np.expand_dims(images, axis=3)

    # Perform data augmentation
    if data_augmentation:
        images, labels = data_augmenter(images, labels,
                                        shift=shift, rotate=rotate, scale=scale,
                                        intensity=intensity, flip=flip)
    return images, labels


def main(argv=None):
    """ Main function """
    # Go through each subset (training, validation, test) under the data directory
    # and list the file names of the subjects
    data_list = {}
    for k in ['train', 'validation', 'test']:
        subset_dir = os.path.join(FLAGS.dataset_dir, k)
        data_list[k] = []

        for data in sorted(os.listdir(subset_dir)):
            data_dir = os.path.join(subset_dir, data)
            # Check the existence of the image and label map at ED and ES time frames
            # and add their file names to the list
            for fr in ['ED', 'ES']:
                image_name = '{0}/{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)
                label_name = '{0}/label_{1}_{2}.nii.gz'.format(data_dir, FLAGS.seq_name, fr)
                if os.path.exists(image_name) and os.path.exists(label_name):
                    data_list[k] += [[image_name, label_name]]

    # Prepare tensors for the image and label map pairs
    # Use int32 for label_pl as tf.one_hot uses int32
    image_pl = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='image')
    label_pl = tf.placeholder(tf.int32, shape=[None, None, None], name='label')

    # Print out the placeholders' names, which will be useful when deploying the network
    print('Placeholder image_pl.name = ' + image_pl.name)
    print('Placeholder label_pl.name = ' + label_pl.name)

    # Determine the number of label classes according to the manual annotation procedure
    # for each image sequence.
    n_class = 0
    if FLAGS.seq_name == 'sa':
        # sa, short-axis images
        # 4 classes (background, LV cavity, LV myocardium, RV cavity)
        n_class = 4
    elif FLAGS.seq_name == 'la_2ch':
        # la_2ch, long-axis 2 chamber view images
        # 2 classes (background, LA cavity)
        n_class = 2
    elif FLAGS.seq_name == 'la_4ch':
        # la_4ch, long-axis 4 chamber views
        # 3 classes (background, LA cavity, RA cavity)
        n_class = 3
    else:
        print('Error: unknown seq_name {0}.'.format(FLAGS.seq_name))
        exit(0)

    # The number of resolution levels
    n_level = FLAGS.num_level

    # The number of filters at each resolution level
    n_filter = []
    for i in range(n_level):
        n_filter += [FLAGS.num_filter * pow(2, i)]
    print(n_filter)

    # Build the neural network, which outputs the logits, i.e. the unscaled values just before
    # the softmax layer, which will normalise the logits into the probabilities.
    if FLAGS.model == 'FCN_legacy':
        n_block = [2, 2, 3, 3, 3]
        logits = build_FCN(image_pl, n_class, n_level=n_level, n_filter=n_filter, n_block=n_block,
                           same_dim=n_filter[0], fc=64)
    if FLAGS.model == 'FCN':
        n_block = [2, 2, 3, 3, 3]
        logits = build_FCN(image_pl, n_class, n_level=n_level, n_filter=n_filter, n_block=n_block,
                           same_dim=32, fc=64)
    elif FLAGS.model == 'ResNet_legacy':
        n_block = [2, 2, 3, 4, 6]
        logits = build_ResNet(image_pl, n_class, n_level=n_level, n_filter=n_filter, n_block=n_block,
                              use_bottleneck=False, same_dim=n_filter[0], fc=64)
    elif FLAGS.model == 'ResNet':
        n_block = [2, 2, 3, 4, 6]
        logits = build_ResNet(image_pl, n_class, n_level=n_level, n_filter=n_filter, n_block=n_block,
                              use_bottleneck=False, same_dim=32, fc=64)
    else:
        print('Error: unknown model {0}.'.format(FLAGS.model))
        exit(0)

    # The softmax probability and the predicted segmentation
    prob = tf.nn.softmax(logits, name='prob')
    pred = tf.cast(tf.argmax(prob, axis=-1), dtype=tf.int32, name='pred')
    print('prob.name = ' + prob.name)
    print('pred.name = ' + pred.name)

    # Loss
    label_1hot = tf.one_hot(indices=label_pl, depth=n_class)
    label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_1hot, logits=logits)
    loss = tf.reduce_mean(label_loss)

    # Evaluation metrics
    accuracy = tf_categorical_accuracy(pred, label_pl)
    dice_lv = tf_categorical_dice(pred, label_pl, 1)
    dice_myo = tf_categorical_dice(pred, label_pl, 2)
    dice_rv = tf_categorical_dice(pred, label_pl, 3)
    dice_la = tf_categorical_dice(pred, label_pl, 1)
    dice_ra = tf_categorical_dice(pred, label_pl, 2)

    # Optimiser
    lr = FLAGS.learning_rate
    if FLAGS.optimizer == 'SGD':
        print('Using SGD optimizer.')
        train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    elif FLAGS.optimizer == 'Adam':
        print('Using Adam optimizer.')
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    elif FLAGS.optimizer == 'Momentum':
        print('Using Momentum optimizer with Nesterov momentum.')
        train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9,
                                              use_nesterov=True).minimize(loss)
    else:
        print('Error: unknown optimizer {0}.'.format(FLAGS.optimizer))
        exit(0)

    # Model name and directory
    model_name = '{0}_{1}_level{2}_filter{3}_{4}_{5}_batch{6}_iter{7}_lr{8}'.format(
        FLAGS.model, FLAGS.seq_name, n_level, n_filter[0], ''.join([str(x) for x in n_block]),
        FLAGS.optimizer, FLAGS.train_batch_size, FLAGS.train_iteration, FLAGS.learning_rate)
    model_dir = os.path.join(FLAGS.checkpoint_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create a logger
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    csv_name = os.path.join(FLAGS.log_dir, '{0}_log.csv'.format(model_name))
    f_log = open(csv_name, 'w')
    if FLAGS.seq_name == 'sa':
        f_log.write('iter,time,train_loss,train_acc,test_loss,test_acc,test_dice_lv,test_dice_myo,test_dice_rv\n')
    elif FLAGS.seq_name == 'la_2ch':
        f_log.write('iter,time,train_loss,train_acc,test_loss,test_acc,test_dice_la\n')
    elif FLAGS.seq_name == 'la_4ch':
        f_log.write('iter,time,train_loss,train_acc,test_loss,test_acc,test_dice_la,test_dice_ra\n')

    # Start the tensorflow session
    with tf.Session() as sess:
        print('Start training...')
        start_time = time.time()

        # Create a saver
        saver = tf.train.Saver(max_to_keep=20)

        # Summary writer
        summary_dir = os.path.join(FLAGS.log_dir, model_name)
        if os.path.exists(summary_dir):
            os.system('rm -rf {0}'.format(summary_dir))
        train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), graph=sess.graph)
        validation_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'validation'), graph=sess.graph)

        # Initialise variables
        sess.run(tf.global_variables_initializer())

        # Iterate
        for iter in range(1, 1 + FLAGS.train_iteration):
            # For each iteration, we randomly choose a batch of subjects for training
            print('Iteration {0}: training...'.format(iter))
            start_time_iter = time.time()

            images, labels = get_random_batch(data_list['train'],
                                              FLAGS.train_batch_size,
                                              FLAGS.seq_name, image_size=FLAGS.image_size,
                                              data_augmentation=True,
                                              shift=10, rotate=10, scale=0.1,
                                              intensity=0.1, flip=False)

            # Stochastic optimisation using this batch
            _, train_loss, train_acc = sess.run([train_op, loss, accuracy],
                                                {image_pl: images, label_pl: labels})

            summary = tf.Summary()
            summary.value.add(tag='loss', simple_value=train_loss)
            summary.value.add(tag='accuracy', simple_value=train_acc)
            train_writer.add_summary(summary, iter)

            # After every ten iterations, we perform validation
            if iter % 10 == 0:
                print('Iteration {0}: validation...'.format(iter))
                images, labels = get_random_batch(data_list['validation'],
                                                  FLAGS.validation_batch_size,
                                                  FLAGS.seq_name, image_size=FLAGS.image_size,
                                                  data_augmentation=False)

                if FLAGS.seq_name == 'sa':
                    validation_loss, validation_acc, validation_dice_lv, validation_dice_myo, validation_dice_rv = \
                        sess.run([loss, accuracy, dice_lv, dice_myo, dice_rv],
                                 {image_pl: images, label_pl: labels})
                elif FLAGS.seq_name == 'la_2ch':
                    validation_loss, validation_acc, validation_dice_la = \
                        sess.run([loss, accuracy, dice_la],
                                 {image_pl: images, label_pl: labels})
                elif FLAGS.seq_name == 'la_4ch':
                    validation_loss, validation_acc, validation_dice_la, validation_dice_ra = \
                        sess.run([loss, accuracy, dice_la, dice_ra],
                                 {image_pl: images, label_pl: labels})

                summary = tf.Summary()
                summary.value.add(tag='loss', simple_value=validation_loss)
                summary.value.add(tag='accuracy', simple_value=validation_acc)
                if FLAGS.seq_name == 'sa':
                    summary.value.add(tag='dice_lv', simple_value=validation_dice_lv)
                    summary.value.add(tag='dice_myo', simple_value=validation_dice_myo)
                    summary.value.add(tag='dice_rv', simple_value=validation_dice_rv)
                elif FLAGS.seq_name == 'la_2ch':
                    summary.value.add(tag='dice_la', simple_value=validation_dice_la)
                elif FLAGS.seq_name == 'la_4ch':
                    summary.value.add(tag='dice_la', simple_value=validation_dice_la)
                    summary.value.add(tag='dice_ra', simple_value=validation_dice_ra)
                validation_writer.add_summary(summary, iter)

                # Print the results for this epoch
                print('Iteration {} of {} took {:.3f}s'.format(iter, FLAGS.train_iteration, time.time() - start_time_iter))
                print('  training loss:\t\t{:.6f}'.format(train_loss))
                print('  training accuracy:\t\t{:.2f}%'.format(train_acc * 100))
                print('  validation loss: \t\t{:.6f}'.format(validation_loss))
                print('  validation accuracy:\t\t{:.2f}%'.format(validation_acc * 100))
                if FLAGS.seq_name == 'sa':
                    print('  validation Dice LV:\t\t{:.6f}'.format(validation_dice_lv))
                    print('  validation Dice Myo:\t\t{:.6f}'.format(validation_dice_myo))
                    print('  validation Dice RV:\t\t{:.6f}\n'.format(validation_dice_rv))
                elif FLAGS.seq_name == 'la_2ch':
                    print('  validation Dice LA:\t\t{:.6f}'.format(validation_dice_la))
                elif FLAGS.seq_name == 'la_4ch':
                    print('  validation Dice LA:\t\t{:.6f}'.format(validation_dice_la))
                    print('  validation Dice RA:\t\t{:.6f}'.format(validation_dice_ra))

                # Log
                if FLAGS.seq_name == 'sa':
                    f_log.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}\n'.format(
                        iter, time.time() - start_time, train_loss, train_acc, validation_loss,
                        validation_acc, validation_dice_lv, validation_dice_myo, validation_dice_rv))
                elif FLAGS.seq_name == 'la_2ch':
                    f_log.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}\n'.format(
                        iter, time.time() - start_time, train_loss, train_acc, validation_loss,
                        validation_acc, validation_dice_la))
                elif FLAGS.seq_name == 'la_4ch':
                    f_log.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}\n'.format(
                        iter, time.time() - start_time, train_loss, train_acc, validation_loss,
                        validation_acc, validation_dice_la, validation_dice_ra))
                f_log.flush()
            else:
                # Print the results for this epoch:
                print('Iteration {} of {} took {:.3f}s'.format(iter, FLAGS.train_iteration, time.time() - start_time_iter))
                print('  training loss:\t\t{:.6f}'.format(train_loss))
                print('  training accuracy:\t\t{:.2f}%'.format(train_acc * 100))

            # Save models after every 1000 iterations (1 epoch)
            # One epoch needs to go through
            #   1000 subjects * 2 time frames = 2000 images = 1000 training iterations
            # if one iteration processes 2 images.
            if iter % 1000 == 0:
                saver.save(sess, save_path=os.path.join(model_dir, '{0}.ckpt'.format(model_name)), global_step=iter)

        # Close the logger and summary writers
        f_log.close()
        train_writer.close()
        validation_writer.close()
        print('Training took {:.3f}s in total.\n'.format(time.time() - start_time))


if __name__ == '__main__':
    tf.app.run()
